# API notes (verified facts only)

Read this before writing code against any external API. Everything here was read from
official docs or source on the date shown.
When a live call teaches us something new, add it under "Learned from live calls".

Docs tip: every page on dev.meta.ai has a Markdown version (append `.md` to the URL), and
`https://dev.meta.ai/llms.txt` lists all pages.

---

## 1. Meta Model API (verified 2026-10-26)

Sources: https://dev.meta.ai/docs/quickstart, /docs/authentication, /docs/api-reference,
/docs/models, /docs/pricing-rate-limits, /docs/error-handling

| Item | Value |
|---|---|
| Base URL | `https://api.meta.ai/v1` |
| Auth | Header `Authorization: Bearer $MODEL_API_KEY` |
| Env var | `MODEL_API_KEY` (keys start with `LLM_`). SDKs do **not** read it automatically, so pass `api_key=` explicitly |
| Get a key | https://dev.meta.ai → API keys → Create API key (shown once) |
| SDK | OpenAI SDK works as-is (`OpenAI(base_url="https://api.meta.ai/v1", api_key=...)`). Anthropic SDK works via `/v1/messages` |
| Endpoints | `POST /v1/responses` (recommended), `POST /v1/chat/completions`, `POST /v1/messages`, `/v1/files`, `GET /v1/models`, `POST /v1/images/generations`, `POST /v1/images/edits`, `GET /v1/status` (no auth) |
| Errors | OpenAI-style `{"error": {"message","type","param","code"}}`. 401 bad key, 402 billing (no credits / payment method), 403 key lacks model access, 404 `model_not_found` (also when the team has no access), 413 payload too large, 429 rate limit (honor `Retry-After`). Don't retry 400/401/402/403/404/413 |
| Rate limits | Per **team**, not per key |
| Data use | "Standard mode" model IDs: our content is not used for training. `-contributor` IDs are cheaper but Meta may train on the content |

### Model IDs and prices (Standard mode, pay-as-you-go)

| Model | ID | Price | Rate limit |
|---|---|---|---|
| Muse Spark 1.3 (latest, recommended) | `muse-spark-1.3` | $1.25 / 1M input, $4.25 / 1M output, $0.15 / 1M cached input | 3,000 RPM, 4M TPM |
| Muse Spark 1.2 / 1.1 | `muse-spark-1.2`, `muse-spark-1.1` | same | same |
| Muse Spark 1.3 Contributor | `muse-spark-1.3-contributor` | $0.10 / 1M input, $0.20 / 1M output | 150 RPM |
| SAM 3.1 | `sam-3.1` | $2.50 per 1,000 images; $0.20 per 1,000 video frames (**min 50 frames billed per video request**) | 3,000 RPM |
| Muse Image | `muse-image-1.0` | $0.01 per generated image | 150 RPM |
| Muse Wonder (not planned) | `muse-wonder-1.0` | $4 / $20 per 1M | 3,000 RPM |

Meta API credits may be available; check your account.

## 2. Muse Spark (verified 2026-10-26)

Sources: /docs/image-understanding, /docs/structured-output, /docs/reasoning, /docs/tool-calling

- Input: text, image, video, audio, PDF. Output: text. Context 1,048,576 tokens.
- **Images, Responses API:** content part `{"type": "input_image", "image_url": "<https URL or data:image/jpeg;base64,...>"}`
  (or `"file_id"`). Read the answer from `response.output_text`.
- **Images, Chat Completions:** `{"type": "image_url", "image_url": {"url": "<URL or data URL>"}}` next to
  `{"type": "text", ...}` in a `user` message. Images only count in `user` messages.
- Limits: **max 50 images per request**, 50 MB per inline image, MIME jpeg/png/gif/webp.
  A ~1280 px image costs about 1,300–1,500 input tokens.
- **Structured JSON output:** Chat Completions `response_format={"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}`;
  Responses API uses `text.format` instead (the other endpoint's parameter is ignored). Decoding is
  constrained to the schema. No recursive schemas; max nesting depth 10. The OpenAI SDK's Pydantic
  helper `beta.chat.completions.parse(response_format=Model)` is documented as working.
- **Reasoning:** `reasoning_effort` (Chat Completions, top level) or `reasoning={"effort": ...}` (Responses):
  `minimal | low | medium | high | xhigh | max`. `"none"` is **not** supported (400). Reasoning tokens
  are billed as output and count toward `max_tokens` / `max_output_tokens`. Use `low` for simple tasks.
- **Grounding:** Muse Spark can return object locations on a normalized **0–1000** grid (x right, y down)
  relative to the image actually sent; clamp values (it may emit 1001).
- **Tool calling** (used by "what if"): Responses API with flat `function` tools; the model may return several
  `function_call` items per turn; `tool_choice` must be `"auto"` (others return 400). Source: /docs/tool-calling.

## 3. SAM 3.1 (verified 2026-10-26)

Sources: /docs/media-segmentation, /docs/sam/overview, /docs/sam/segmenting,
/docs/sam/reading-segmentation, /docs/sam/client-libraries, and Meta's cookbook code
`meta-models/meta-model-cookbook` → `07_sam/01_api_basics/python/sam_api.py`

**Request:** `POST /v1/responses`, `model="sam-3.1"`, one `user` message whose content is one
`input_text` (the prompt) plus one media part:
- image: `{"type": "input_image", "image_url": "<URL or data URL>"}`
- video: `{"type": "input_video", "video_url": "<URL or data URL>"}` or `{"type": "input_video", "file_id": "file-..."}`

Options go in `metadata` (all values are **strings**):
- `"mask_encoding": "one_bit"` gives compact masks (default `lossless`). Box/point prompts always return one_bit.
- `"include_confidence": "true"` adds a 0–1 confidence per box/mask per frame.
- `"propagation_direction": "both" | "forward" | "backward"`, for box/point prompts in video.

**Prompts:**
- Concept prompt = a short noun phrase naming **one** concept ("red mug"). No lists ("mug, book" returns
  nothing); one request per concept. No commands, questions, negation or spatial phrases. Attributes narrow
  the match. Zero matches is a normal result (empty output, `response.completed`).
- Box/point prompts: `<Nf>id<|box;x1=..;y1=..;x2=..;y2=..;w=W;h=H|>` and `<|point;x=..;y=..;w=W;h=H|>`, with
  `-` before a point for "not this". Never mix with a noun phrase. Max 16 prompted objects. `w,h` must be
  the media's displayed size (after EXIF/rotation). Build them with `meta_sam_parser.build_object_prompt`.

**Output:** special-token text in the `output_text` lane, one line per frame (image = `<0f>`):
`<Nf>id<|box;x1;y1;x2;y2;w;h[;c]|><|mask;x=0;y=0[;c];data=H,W,<enc>payload|>,id...`
- Boxes are in source pixels; `x2,y2` inclusive on the wire (the parser gives half-open `right,bottom`).
- **Masks are box-local:** the raster is the box's size and starts at the box's top-left corner
  (confirmed offline: raster 60x27 for box 211..271 x 228..255). Payload is base85, not base64.
- Object ids are stable across frames within one response but **not** dense or 0-based. Frames with no
  objects emit no line, so frame indices can skip. Always key by id and by `<Nf>`, never by position.
- Video concept prompts: lines arrive in ascending frame order. Box/point prompts with `both`: forward
  frames first, then backward.

**Python parser:** `pip install meta-sam-parser` (0.1.0 pinned; needs ≥0.0.6 for confidence).
`parse_responses_stream(async_events, image_segmentation_format())` (or `video_segmentation_format()`),
then `await parsed.final_result()`. Records have `kind` = `box` / `mask`; mask records have `.mask`
(width, height, payload), `.bounds` (half-open, source pixels), `.frame.frame_index`, `.confidence`.
`decode_mask_to_raster(mask)` returns `bytes` of 0/1, row-major, `height*width`. Also
`decode_mask_to_rle` (COCO RLE) and `decode_mask_to_svg_path`.
- **Gotcha (from Meta's cookbook):** the API sends `response.content_part.done` but **not**
  `response.output_text.done`, which the parser needs to close the lane. Convert SDK events to dicts and
  synthesize `output_text.done` from `content_part.done`, as in `finish_text_lanes` in
  `pipeline/scripts/check_meta_api.py`. Use the **async** OpenAI client with `stream=True`.
- Check `result.outcome.status == "completed"` and `diagnostics` with severity `error`.

**Video specifics:**
- Formats: MP4 or MOV with H.264, H.265 or MPEG-4. **Not** VP9/WebM.
- Up to 15,000 frames (~10 min) per request; **video tracks at most 16 objects per frame**.
- Upload once with `POST /v1/files` (`purpose="user_data"`); upload is synchronous (200 = ready).
  Re-encode with `ffmpeg -i in.mp4 -c copy -movflags +faststart out.mp4` to skip a server remux.
  Uploaded files never expire by default (team storage cap 100 GiB); set `expires_after` or delete them.
- Streaming is recommended for video. Results can vary slightly between identical runs.
- Cost example: a 60 s clip at 30 fps = 1,800 frames ≈ $0.36 per concept prompt.

## 4. Muse Image (verified 2026-10-26)

Sources: /docs/image-generation, /docs/api-reference/images/edit-image

- Model `muse-image-1.0`; one model for generation and editing. $0.01 per returned image; 150 RPM.
- **One-shot edit:** `POST /v1/images/edits`. OpenAI SDK: `client.images.edit(model=..., prompt=..., image=<file or list of files>, n=1)`
  (multipart). Raw HTTP alternative: JSON body `{"model", "prompt", "images": [{"image_url": "<URL or data URL>"} | {"file_id": ...}], "n"}`.
- Options: `n` 1–10, `size` `"WxH"` sets **aspect ratio only**, `output_format` `webp` (default) / `png` / `jpeg`,
  `response_format` `b64_json` (default) / `url`.
- **No mask input:** `mask`, `input_fidelity`, `partial_images` return 400. Object removal must be described
  in the prompt, and **output size won't exactly match the input**. For hole filling (stage 7) we must resize
  back and check alignment. Plan for the texture-sampling fallback.
- Conversational editing (multi-turn) is on the Responses API with `previous_response_id`; the image comes back
  as an `image_generation_call` output item with base64 in `result`. It searches the web on its own; turn that
  off with an `image_generation` tool entry (`enable_web_search: false`, `enable_image_search: false`,
  `reasoning_strength: "low"`).

## 5. VGGT (verified 2026-10-26)

Sources: https://github.com/facebookresearch/vggt (README, LICENSE.txt, requirements.txt, pyproject.toml,
`vggt/utils/load_fn.py`; latest commit 2026-05-19), https://huggingface.co/facebook/VGGT-1B

- **License:** code is under the "VGGT License" (v1, 2025-07-29), which allows commercial use except military
  and similar uses in its Acceptable Use Policy, and requires acknowledging VGGT in publications.
  Weights: `facebook/VGGT-1B` is **CC BY-NC 4.0 (non-commercial)**, not gated. `facebook/VGGT-1B-Commercial` is
  licensed for commercial use with similar quality, but **gated**: request access on Hugging Face, then use an
  `HF_TOKEN`. The README says requests are processed automatically, but the HF API reports `gated: manual`,
  so approval may take a while. Files: `model.safetensors`, `vggt_1B_commercial.pt`. Plan: use VGGT-1B-Commercial
  so commercial use is clearly covered.
- **Install:** `pip install` from the git repo (package `vggt`, Python ≥3.10). Repo requirements pin
  `torch==2.3.1`, `torchvision==0.18.1`, `numpy==1.26.1` (pyproject says `numpy<2`). Run it in its own
  Modal image.
- **Usage:** `VGGT.from_pretrained("facebook/VGGT-1B")`; `load_and_preprocess_images(paths, mode="crop")` resizes
  to **518 px wide** (height multiple of 14, center-cropped if taller than 518); `mode="pad"` keeps all pixels
  in a 518x518 square. Run under `torch.cuda.amp.autocast(dtype=bfloat16)` on Ampere or newer, float16 otherwise.
- **Outputs:** camera extrinsics and intrinsics (OpenCV convention, camera-from-world), depth map + confidence,
  point map + confidence, point tracks. README: points from **depth + cameras** (`unproject_depth_map_to_point_map`)
  are usually more accurate than the point-map head. Unwanted pixels (e.g. shiny surfaces) can be masked by
  setting them to 0 or 1.
- **GPU memory (aggregator only, H100, Flash Attention 3; README before May 2026):**

  | Frames | 1 | 10 | 20 | 50 | 100 | 200 |
  |---|---|---|---|---|---|---|
  | Time (s) | 0.04 | 0.14 | 0.31 | 1.04 | 3.12 | 8.75 |
  | Memory (GB) | 1.88 | 3.63 | 5.58 | 11.41 | 21.15 | 40.63 |

  2026-05-15 fix: "VGGT can now run on roughly 2-3x more input frames" with the same memory. So 60–120 frames
  should fit a 24 GB GPU (L4/A10); measured in practice: 26 frames at 392x518 peak at 8.1 GB on an L40S, including the heads.
- `demo_colmap.py` exports COLMAP format (optional bundle adjustment), which gsplat can train from directly
  (README recommends `gsplat==1.3.0`).
- VGGT-Omega (successor, released 2026-05-18) exists. Not evaluated; a candidate for future work.

## 6. Modal (verified 2026-10-26)

Sources: https://modal.com/docs/guide/gpu, https://modal.com/pricing, `modal` 1.6.2 CLI help

- GPU strings: `T4`, `L4`, `A10`, `L40S`, `A100`, `A100-40GB`, `A100-80GB`, `RTX-PRO-6000`, `H100` (`H100!` = no
  auto-upgrade), `H200`, `B200` (`B200+`), `B300`. There is no `A10G` string. Fallback list: `gpu=["L4", "A10"]`.
- Price per hour (derived from per-second): T4 $0.59, L4 $0.80, A10 $1.10, L40S $1.95, A100-40GB $2.10,
  A100-80GB $2.50, H100 $3.95, H200 $4.54, B200 $6.25. Starter plan: $30/month free credits.
- Login: `modal token new` (browser), stored in `~/.modal.toml`. Secrets: `modal secret create <name> --from-dotenv .env`,
  or `modal.Secret.from_dotenv()` in code.
- Always pass `timeout=` to `@app.function` (house rule).

## 7. Gaussian splat viewer library

Not used: objects render as point clouds. Moving groups of splats per rigid body is future work.

## Learned from live calls

**2026-10-26: first runs of `pipeline/scripts/check_meta_api.py`** (913x1280 still-life photo)
- **Muse Spark 1.3** (Chat Completions, one image, `json_schema` output, `reasoning_effort="low"`): 7–8 s and
  about $0.009 per call (1,511 prompt tokens, 1,923 completion tokens, of which 1,405 are reasoning). The JSON
  always matched the schema. It listed 14–17 objects with plausible masses (book 1.2–1.5 kg, orange 100–120 g,
  ceramic bowl 800–900 g). **Labels change between runs** ("tangerine", "mandarin orange"), and it also lists
  minor items (coaster, pile of nuts).
- **SAM 3.1** (image, concept prompt, `one_bit` + confidence): 1–2.5 s, $0.0025 per call.
  - `"object"` gave **0 matches**. A generic prompt does not work, so there is no generic "find every object" pass.
  - `"tangerine"` and `"orange"` each found all 6 oranges as separate objects (touching ones too), with
    confidence 75–89%, and left out the apples and the pear. `"mandarin orange"` gave **0 matches**.
    SAM is very sensitive to wording.
  - Live output confirms masks are box-local (raster size = box size).
- **Implication for segmentation:** don't feed Muse Spark's free-form labels straight to SAM. Better plan to test:
  Muse Spark returns a short label **plus a bounding box** (0–1000 grid) for each object on one keyframe,
  then SAM gets **box prompts** (which don't depend on wording) on that frame and tracks each object through
  the video under our own ids. Fallback: a short common noun per object, retrying with the last word
  ("mandarin orange" → "orange").

**2026-10-26: first batch run (24 Objectron clips, up to 8 jobs at once)**
- About 1 in 6 Muse Spark Chat Completions calls returned **`404 The requested model was not found`** for
  `muse-spark-1.3`, while identical calls before and after succeeded. A direct probe right afterwards
  (24 calls, 12 concurrent) returned no errors. Treated as transient: `meta_api.ask_structured` retries
  404 with exponential backoff (2, 4, 8 s) before failing, despite the docs' "don't retry 404" guidance.
- OpenAI SDK exceptions (`APIStatusError`) can't be unpickled across Modal containers (their constructor
  needs `response` and `body`), so the caller saw only "Could not deserialize remote exception". Stages now
  re-raise as `RuntimeError("HTTP 404: ...")` (`modal_app._run_stage`).
- SAM 3.1 box-prompt video tracking held every object in every keyframe for the two-mug clip
  (26 frames, mean confidence 0.985).
