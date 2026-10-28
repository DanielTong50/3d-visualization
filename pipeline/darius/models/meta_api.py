"""Thin clients for the Meta Model API: Muse Spark (structured JSON) and SAM 3.1 (streamed masks).

Request formats are documented in docs/api-notes.md; nothing here relies on
undocumented behavior except the stream-lane workaround noted in `finish_text_lanes`.
"""

from __future__ import annotations

import base64
import io
import os
import random
import time
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import openai
from meta_sam_parser import (
    FramePrompt,
    PromptBox,
    PromptObject,
    build_object_prompt,
    decode_mask_to_raster,
    image_segmentation_format,
    parse_responses_stream,
    video_segmentation_format,
)
from PIL import Image
from pydantic import BaseModel

BASE_URL = "https://api.meta.ai/v1"
MUSE_SPARK_MODEL = "muse-spark-1.3"
SAM_MODEL = "sam-3.1"
SAM_MAX_OBJECTS = 16

T = TypeVar("T", bound=BaseModel)


def api_key() -> str:
    key = os.environ.get("MODEL_API_KEY", "").strip()
    if not key:
        raise RuntimeError("MODEL_API_KEY is not set (see .env.example)")
    return key


def image_data_url(image: Image.Image, max_side: int = 1280, quality: int = 88) -> str:
    image = image.convert("RGB")
    if max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# ---------- Muse Spark ----------


def ask_structured(
    prompt: str,
    images: Sequence[Image.Image],
    schema: type[T],
    *,
    reasoning_effort: str = "low",
    retries: int = 5,
) -> tuple[T, dict[str, Any]]:
    """One Chat Completions call constrained to `schema`; returns the parsed model and usage.

    Retries connection errors, 5xx, replies that fail validation, and 404s. The docs
    say not to retry 404, but during a 24-clip batch run the API returned
    `404 model not found` for about 1 in 6 calls of a model that answered the
    identical request moments later (docs/api-notes.md, 2026-10-26), so a 404 is
    retried with jittered exponential backoff (about 90 s in total) before it is treated as real.
    """
    client = openai.OpenAI(base_url=BASE_URL, api_key=api_key(), max_retries=3)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": image_data_url(image)}} for image in images]
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MUSE_SPARK_MODEL,
                reasoning_effort=reasoning_effort,
                messages=[{"role": "user", "content": content}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
                },
            )
            parsed = schema.model_validate_json(response.choices[0].message.content or "")
            usage = response.usage.model_dump() if response.usage else {}
            return parsed, usage
        except (openai.APIConnectionError, openai.InternalServerError, openai.NotFoundError, ValueError) as error:
            # ValueError covers a reply that fails Pydantic validation (e.g. out-of-range values).
            last_error = error
            # 404 bursts cluster under concurrency, so back off long enough to outlast one.
            time.sleep(min(30.0, 3.0 * 2**attempt) * random.uniform(0.75, 1.25))
    raise RuntimeError(f"Muse Spark failed after {retries + 1} attempts: {describe(last_error)}")


def describe(error: BaseException | None) -> str:
    """A plain-string error summary. SDK exception objects don't survive pickling between
    Modal containers, so stages re-raise API failures as RuntimeError(describe(e))."""
    if isinstance(error, openai.APIStatusError):
        return f"HTTP {error.status_code}: {error.message}"
    return f"{type(error).__name__}: {error}"


# ---------- SAM 3.1 ----------


async def finish_text_lanes(stream: AsyncIterable[Any]) -> AsyncIterator[dict[str, Any]]:
    """Pass SDK events to the SAM parser as dicts, closing each text lane.

    The API sends `response.content_part.done` but not `response.output_text.done`,
    which the parser needs to close a lane, so this synthesizes it. Adapted from
    Meta's cookbook (07_sam/01_api_basics/python/sam_api.py).
    """
    finished: set[tuple[Any, Any, Any]] = set()
    async for event in stream:
        data = event.model_dump() if hasattr(event, "model_dump") else dict(event)
        lane = (data.get("item_id"), data.get("output_index"), data.get("content_index"))
        if data.get("type") == "response.output_text.done":
            if lane not in finished:
                finished.add(lane)
                yield data
            continue
        yield data
        part = data.get("part")
        if (
            data.get("type") == "response.content_part.done"
            and lane not in finished
            and isinstance(part, dict)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
        ):
            finished.add(lane)
            yield {
                "type": "response.output_text.done",
                "item_id": data.get("item_id"),
                "output_index": data.get("output_index"),
                "content_index": data.get("content_index"),
                "text": part["text"],
            }


@dataclass(frozen=True)
class FrameMask:
    """One object's mask on one frame, as a full-frame boolean raster."""

    frame: int
    object_id: int
    mask: np.ndarray
    confidence: float | None


def raster_from_record(record: Any, frame_size: tuple[int, int]) -> np.ndarray:
    """Paste a box-local SAM mask into a full-frame (height, width) boolean array."""
    width, height = frame_size
    full = np.zeros((height, width), dtype=bool)
    pixels = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8)
    pixels = pixels.reshape(record.mask.height, record.mask.width).astype(bool)
    left, top = int(record.bounds.left), int(record.bounds.top)
    right, bottom = int(record.bounds.right), int(record.bounds.bottom)
    if pixels.shape != (bottom - top, right - left):
        resized = Image.fromarray(pixels.astype(np.uint8) * 255).resize(
            (right - left, bottom - top), Image.Resampling.NEAREST
        )
        pixels = np.asarray(resized) > 127
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(right, width), min(bottom, height)
    full[y0:y1, x0:x1] = pixels[y0 - top : y1 - top, x0 - left : x1 - left]
    return full


def latest_masks(records: Sequence[Any]) -> list[Any]:
    """Keep only the highest revision of each (frame, object) mask."""
    best: dict[tuple[int | None, str], Any] = {}
    for record in records:
        if record.kind != "mask":
            continue
        frame = record.frame.frame_index if record.frame is not None else 0
        key = (frame, record.object_id)
        if key not in best or record.revision > best[key].revision:
            best[key] = record
    return list(best.values())


async def _stream_sam(content: list[dict[str, Any]], metadata: dict[str, str], video: bool) -> Any:
    async with openai.AsyncOpenAI(base_url=BASE_URL, api_key=api_key(), max_retries=3) as client:
        stream = await client.responses.create(
            model=SAM_MODEL,
            stream=True,
            input=[{"type": "message", "role": "user", "content": content}],
            metadata=metadata,
        )
        response_format = video_segmentation_format() if video else image_segmentation_format()
        async with stream, aclosing(finish_text_lanes(stream)) as events:
            result = await parse_responses_stream(events, response_format).final_result()
    if result.outcome.status != "completed":
        raise RuntimeError(f"SAM 3.1 stopped early: {result.outcome}")
    errors = [d for d in result.diagnostics if d.severity == "error"]
    if errors:
        raise RuntimeError("unreadable SAM 3.1 output: " + "; ".join(d.message for d in errors))
    return result


def upload_video(path: Path) -> str:
    client = openai.OpenAI(base_url=BASE_URL, api_key=api_key(), max_retries=3)
    with path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="user_data")
    return uploaded.id


def delete_file(file_id: str) -> None:
    try:
        openai.OpenAI(base_url=BASE_URL, api_key=api_key()).files.delete(file_id)
    except openai.APIError:
        pass  # best effort; uploads are housekeeping, not correctness


async def track_boxes_in_video(
    video_file_id: str,
    frame_size: tuple[int, int],
    keyframe: int,
    boxes: Sequence[tuple[int, int, int, int]],
) -> list[FrameMask]:
    """Track each box (left, top, right, bottom; half-open, video pixels) through the video.

    Object i in `boxes` comes back as object_id i + 1. Box prompts make SAM
    independent of how an object is named, which matters because SAM's concept
    prompts are very sensitive to wording (see docs/api-notes.md).
    """
    if not boxes:
        return []
    if len(boxes) > SAM_MAX_OBJECTS:
        raise ValueError(f"SAM 3.1 tracks at most {SAM_MAX_OBJECTS} objects per video")
    prompt = build_object_prompt(
        size=frame_size,
        objects=[
            PromptObject(id=i + 1, prompts=[FramePrompt(frame=keyframe, box=PromptBox(*box))])
            for i, box in enumerate(boxes)
        ],
    )
    result = await _stream_sam(
        [{"type": "input_text", "text": prompt}, {"type": "input_video", "file_id": video_file_id}],
        {"mask_encoding": "one_bit", "include_confidence": "true", "propagation_direction": "both"},
        video=True,
    )
    masks = []
    for record in latest_masks(result.records):
        frame = record.frame.frame_index if record.frame is not None else 0
        masks.append(FrameMask(frame, int(record.object_id), raster_from_record(record, frame_size), record.confidence))
    return masks


async def segment_concept_in_image(image: Image.Image, concept: str) -> list[FrameMask]:
    """All instances of a short noun phrase in one image."""
    result = await _stream_sam(
        [{"type": "input_text", "text": concept}, {"type": "input_image", "image_url": image_data_url(image, 4096)}],
        {"mask_encoding": "one_bit", "include_confidence": "true"},
        video=False,
    )
    return [
        FrameMask(0, int(r.object_id), raster_from_record(r, image.size), r.confidence)
        for r in latest_masks(result.records)
    ]
