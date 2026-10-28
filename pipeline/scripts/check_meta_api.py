"""Smoke check: call Muse Spark once and SAM 3.1 once on a test photo.

Run from the pipeline folder:
    uv run python scripts/check_meta_api.py                   # bundled test photo
    uv run python scripts/check_meta_api.py my_desk.jpg       # your own photo
    uv run python scripts/check_meta_api.py --concept orange  # choose what SAM looks for

By default SAM looks for the label Muse Spark listed most often: SAM needs a specific
noun phrase (a generic "object" finds nothing).

Needs MODEL_API_KEY in the project's .env file (see .env.example).
Saves what comes back to pipeline/outputs/api_check/.
Request formats come from Meta's docs; see docs/api-notes.md.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import time
import traceback
from contextlib import aclosing
from pathlib import Path
from typing import Any

import numpy as np
import openai
from dotenv import load_dotenv
from meta_sam_parser import (
    ImageSegmentationResult,
    decode_mask_to_raster,
    image_segmentation_format,
    parse_responses_stream,
)
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pydantic import BaseModel, Field

from darius.models.meta_api import finish_text_lanes

PIPELINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PIPELINE_DIR.parent
DEFAULT_PHOTO = PIPELINE_DIR / "test_assets" / "tabletop.jpg"
OUTPUT_DIR = PIPELINE_DIR / "outputs" / "api_check"

BASE_URL = "https://api.meta.ai/v1"
MUSE_SPARK_MODEL = "muse-spark-1.3"
SAM_MODEL = "sam-3.1"
MAX_PHOTO_SIDE = 1280  # shrink photos before sending: fewer tokens, faster uploads

# Standard-mode prices from docs/api-notes.md, used only to print a cost estimate.
MUSE_SPARK_USD_PER_INPUT_TOKEN = 1.25 / 1_000_000
MUSE_SPARK_USD_PER_OUTPUT_TOKEN = 4.25 / 1_000_000
SAM_USD_PER_IMAGE = 2.50 / 1000

MUSE_SPARK_PROMPT = (
    "List each separate physical object resting on the table in this photo. "
    "Skip the table itself, walls, windows and furniture. For each object give a short "
    "everyday name, its main material, your best estimate of its real-world mass in "
    "kilograms, and whether it would likely break if dropped on a hard floor."
)

OVERLAY_COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
]


class TableObject(BaseModel):
    label: str = Field(description="Short everyday name, e.g. 'coffee mug'")
    material: str = Field(description="Main material, e.g. 'ceramic', 'paper', 'wood'")
    mass_kg: float = Field(description="Best estimate of the real object's mass in kilograms")
    fragile: bool = Field(description="True if it would likely break when dropped")


class TableContents(BaseModel):
    objects: list[TableObject]


def load_photo(path: Path) -> tuple[Image.Image, str]:
    """Open the photo upright, shrink it, and encode it as a base64 data URL."""
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image.thumbnail((MAX_PHOTO_SIDE, MAX_PHOTO_SIDE))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return image, f"data:image/jpeg;base64,{encoded}"


# ---------- Muse Spark: what is on the table? ----------


def ask_muse_spark(api_key: str, image_url: str) -> tuple[TableContents, dict[str, Any]]:
    """One Chat Completions call: the photo plus a JSON schema the answer must follow."""
    client = openai.OpenAI(base_url=BASE_URL, api_key=api_key)
    response = client.chat.completions.create(
        model=MUSE_SPARK_MODEL,
        reasoning_effort="low",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": MUSE_SPARK_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "TableContents", "schema": TableContents.model_json_schema()},
        },
    )
    contents = TableContents.model_validate_json(response.choices[0].message.content or "")
    return contents, response.model_dump(mode="json")


# ---------- SAM 3.1: exactly which pixels belong to each object? ----------


async def segment_with_sam(api_key: str, image_url: str, concept: str) -> ImageSegmentationResult:
    """One streamed Responses call: a short noun phrase plus the photo."""
    async with openai.AsyncOpenAI(base_url=BASE_URL, api_key=api_key) as client:
        stream = await client.responses.create(
            model=SAM_MODEL,
            stream=True,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": concept},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
            metadata={"mask_encoding": "one_bit", "include_confidence": "true"},
        )
        # Close the event generator before the stream, as Meta's cookbook does.
        async with stream, aclosing(finish_text_lanes(stream)) as events:
            result = await parse_responses_stream(events, image_segmentation_format()).final_result()
    if result.outcome.status != "completed":
        raise RuntimeError(f"SAM 3.1 stopped early: {result.outcome}")
    errors = [d for d in result.diagnostics if d.severity == "error"]
    if errors:
        raise RuntimeError("could not read SAM 3.1's answer: " + "; ".join(d.message for d in errors))
    return result


def tint_masks(image: Image.Image, result: ImageSegmentationResult) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Tint each mask SAM returned in its own color; also summarize each object."""
    tint = Image.new("RGBA", image.size, (0, 0, 0, 0))
    masks = [record for record in result.records if record.kind == "mask"]
    found = []
    for index, record in enumerate(masks):
        color = OVERLAY_COLORS[index % len(OVERLAY_COLORS)]
        pixels = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8)
        pixels = pixels.reshape(record.mask.height, record.mask.width)
        # Masks are box-local: the raster covers the object's box, from its top-left corner.
        left, top = int(record.bounds.left), int(record.bounds.top)
        size = (int(record.bounds.right) - left, int(record.bounds.bottom) - top)
        alpha = Image.fromarray(pixels * 140)
        if alpha.size != size:
            alpha = alpha.resize(size, Image.Resampling.NEAREST)
        layer = Image.new("RGBA", size, color + (0,))
        layer.putalpha(alpha)
        tint.alpha_composite(layer, dest=(left, top))
        found.append(
            {
                "object_id": record.object_id,
                "box": [left, top, left + size[0], top + size[1]],
                "confidence": record.confidence,
                "mask_pixels": int(pixels.sum()),
                "color": color,
            }
        )

    return Image.alpha_composite(image.convert("RGBA"), tint).convert("RGB"), found


def outline_objects(
    picture: Image.Image, found: list[dict[str, Any]], origin: tuple[int, int] = (0, 0), scale: float = 1.0
) -> None:
    """Draw each object's box and id; origin and scale map photo pixels onto the picture."""
    draw = ImageDraw.Draw(picture)
    font = ImageFont.load_default(size=max(14, picture.width // 45))
    for item in found:
        box = item["box"]
        left, right = ((x - origin[0]) * scale for x in (box[0], box[2]))
        top, bottom = ((y - origin[1]) * scale for y in (box[1], box[3]))
        draw.rectangle([left, top, right - 1, bottom - 1], outline=item["color"], width=3)
        draw.text(
            (left + 4, top + 2),
            f"#{item['object_id']}",
            fill=item["color"],
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )


def zoom_on_objects(tinted: Image.Image, found: list[dict[str, Any]], width: int = 900) -> Image.Image:
    """Crop around everything SAM found and enlarge it, so small objects are easy to see."""
    left = min(item["box"][0] for item in found)
    top = min(item["box"][1] for item in found)
    right = max(item["box"][2] for item in found)
    bottom = max(item["box"][3] for item in found)
    pad = max(right - left, bottom - top) // 3
    region = (max(0, left - pad), max(0, top - pad), min(tinted.width, right + pad), min(tinted.height, bottom + pad))
    scale = width / (region[2] - region[0])
    zoomed = tinted.crop(region).resize((width, round((region[3] - region[1]) * scale)), Image.Resampling.LANCZOS)
    outline_objects(zoomed, found, origin=(region[0], region[1]), scale=scale)
    return zoomed


# ---------- Running both checks ----------


def format_mass(kg: float) -> str:
    return f"{kg * 1000:.0f} g" if kg < 1 else f"{kg:.1f} kg"


def most_common_label(contents: TableContents | None) -> str | None:
    """The label Muse Spark used most often (the first one listed wins a tie)."""
    labels = [item.label for item in contents.objects] if contents else []
    return max(labels, key=labels.count) if labels else None


def explain(error: Exception) -> str:
    """Turn an API error into a plain-language hint."""
    if isinstance(error, openai.AuthenticationError):
        return "the API key was rejected (401). Check MODEL_API_KEY in .env."
    if isinstance(error, openai.PermissionDeniedError):
        return "this key has no access to that model (403). Check the dashboard at dev.meta.ai."
    if isinstance(error, openai.NotFoundError):
        return "model not found, or this account has no access to it yet (404)."
    if isinstance(error, openai.RateLimitError):
        return "too many requests (429). Wait a minute and try again."
    if isinstance(error, openai.APIStatusError):
        if error.status_code == 402:
            return "billing problem (402): no credits or payment method on the account."
        return f"HTTP {error.status_code}: {error.message}"
    if isinstance(error, openai.APIConnectionError):
        return "could not reach api.meta.ai. Check the internet connection."
    traceback.print_exc()
    return f"{type(error).__name__}: {error}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Call Muse Spark once and SAM 3.1 once on a photo.")
    parser.add_argument("photo", nargs="?", type=Path, default=DEFAULT_PHOTO, help="photo to test with")
    parser.add_argument("--concept", help="what SAM 3.1 should find (default: Muse Spark's most common label)")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("MODEL_API_KEY", "").strip()
    if not api_key:
        print("MODEL_API_KEY is not set. Copy .env.example to .env and paste your Meta Model API key into it.")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image, image_url = load_photo(args.photo)
    print(f"Photo: {args.photo.name} ({image.width}x{image.height} after resizing)")
    succeeded = 0
    contents = None

    print(f"\n1) Muse Spark ({MUSE_SPARK_MODEL}): what is on the table?")
    started = time.monotonic()
    try:
        contents, raw = ask_muse_spark(api_key, image_url)
    except Exception as error:
        print(f"   FAILED: {explain(error)}")
    else:
        succeeded += 1
        reply = {"model": MUSE_SPARK_MODEL, "answer": contents.model_dump(), "raw_response": raw}
        (OUTPUT_DIR / "muse_spark_reply.json").write_text(json.dumps(reply, indent=2) + "\n")
        for item in contents.objects:
            fragile = ", fragile" if item.fragile else ""
            print(f"   - {item.label}: {item.material}, about {format_mass(item.mass_kg)}{fragile}")
        usage = raw.get("usage") or {}
        cost = (
            usage.get("prompt_tokens", 0) * MUSE_SPARK_USD_PER_INPUT_TOKEN
            + usage.get("completion_tokens", 0) * MUSE_SPARK_USD_PER_OUTPUT_TOKEN
        )
        print(f"   {len(contents.objects)} objects in {time.monotonic() - started:.1f} s, cost about ${cost:.4f}")

    concept = args.concept or most_common_label(contents) or "object"
    print(f'\n2) SAM 3.1 ({SAM_MODEL}): find every "{concept}"')
    started = time.monotonic()
    try:
        result = asyncio.run(segment_with_sam(api_key, image_url, concept))
    except Exception as error:
        print(f"   FAILED: {explain(error)}")
    else:
        succeeded += 1
        tinted, found = tint_masks(image, result)
        picture = tinted.copy()
        outline_objects(picture, found)
        picture.save(OUTPUT_DIR / "sam_overlay.jpg", quality=90)
        if found:
            zoom_on_objects(tinted, found).save(OUTPUT_DIR / "sam_overlay_zoom.jpg", quality=90)
        (OUTPUT_DIR / "sam_raw_output.txt").write_text(result.raw_output)
        summary = {"model": SAM_MODEL, "concept": concept, "image_size": list(image.size), "objects": found}
        (OUTPUT_DIR / "sam_result.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"   Found {len(found)} in {time.monotonic() - started:.1f} s, cost about ${SAM_USD_PER_IMAGE:.4f}")
        for item in found:
            confidence = "" if item["confidence"] is None else f", confidence {item['confidence']:.0%}"
            print(f"   - #{item['object_id']}: box {item['box']}{confidence}")
        if not found:
            print("   No matches. Try a more specific word, e.g. --concept orange")
        print(
            f"   Pictures with the masks drawn on: {OUTPUT_DIR / 'sam_overlay.jpg'}"
            + (f" and {OUTPUT_DIR / 'sam_overlay_zoom.jpg'}" if found else "")
        )

    print(f"\n{succeeded} of 2 API calls succeeded. Outputs: {OUTPUT_DIR}")
    return 0 if succeeded == 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
