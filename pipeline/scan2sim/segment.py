"""Stage 3, segment: find each object once, then track it through every keyframe.

1. Muse Spark looks at one sharp keyframe and returns each object on the table
   with a short label and a bounding box on its 0-1000 grid.
2. SAM 3.1 receives those boxes as *box prompts* on that keyframe and propagates
   masks forward and backward through the keyframe clip.

Concept (text) prompts would be simpler, but live testing showed SAM 3.1 is very
sensitive to wording ("mandarin orange" finds nothing where "orange" finds six),
and Muse Spark's labels vary run to run. Boxes are wording-independent.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from . import meta_api
from .frames import FrameSet
from .materials import MATERIAL_NAMES

DETECT_PROMPT = """You are building a physics simulation of the tabletop in this photo.
List every separate, movable physical object resting on the main table or surface.
Do not list the table or surface itself, walls, floor, furniture, shadows, reflections,
printed patterns, or anything cut off by the image border so that less than half of it is visible.
Do not list parts of an object separately (a mug's handle belongs to the mug).
For each object give:
- label: a short, common noun phrase of one to three words, e.g. "coffee mug", "book"
- box_2d: [x_min, y_min, x_max, y_max] tightly around the object, on a 0-1000 grid where
  x grows to the right and y grows downward, relative to this image
- material: its main material
List at most 12 objects, largest first."""


class Detection(BaseModel):
    label: str = Field(description="Short common noun phrase, 1-3 words")
    box_2d: list[int] = Field(min_length=4, max_length=4, description="[x_min, y_min, x_max, y_max] on a 0-1000 grid")
    material: str = Field(description="One of: " + ", ".join(MATERIAL_NAMES))


class Detections(BaseModel):
    objects: list[Detection]


def choose_keyframe(frameset: FrameSet) -> int:
    """Sharpest frame in the middle half: centered so propagation runs both ways."""
    n = frameset.count
    lo, hi = n // 4, max(n // 4 + 1, 3 * n // 4)
    window = frameset.sharpness[lo:hi]
    return lo + int(np.argmax(window))


def grid_to_pixels(box: list[int], size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """0-1000 grid box -> half-open pixel box; None if degenerate. Values are clamped
    because the model occasionally emits 1001."""
    width, height = size
    x0, y0, x1, y1 = (min(max(v, 0), 1000) / 1000 for v in box)
    left, right = sorted((round(x0 * width), round(x1 * width)))
    top, bottom = sorted((round(y0 * height), round(y1 * height)))
    right, bottom = min(right, width), min(bottom, height)
    if right - left < 4 or bottom - top < 4:
        return None
    return left, top, right, bottom


def filter_detections(detections: list[Detection], size: tuple[int, int]) -> list[tuple[Detection, tuple]]:
    """Drop degenerate, tiny (<0.2% of frame) and near-whole-frame boxes."""
    area = size[0] * size[1]
    kept = []
    for detection in detections:
        box = grid_to_pixels(detection.box_2d, size)
        if box is None:
            continue
        share = (box[2] - box[0]) * (box[3] - box[1]) / area
        if 0.002 <= share <= 0.8:
            kept.append((detection, box))
    return kept[: meta_api.SAM_MAX_OBJECTS]


def label_maps(
    masks: list[meta_api.FrameMask], frames: int, frame_size: tuple[int, int], out_size: tuple[int, int]
) -> np.ndarray:
    """(frames, h, w) uint8 label maps at reconstruction resolution; 0 = background.

    Where masks overlap, the more confident one wins (painted last).
    """
    width, height = out_size
    labels = np.zeros((frames, height, width), dtype=np.uint8)
    order = sorted(masks, key=lambda m: m.confidence if m.confidence is not None else 0.5)
    for mask in order:
        if not 0 <= mask.frame < frames:
            continue
        if mask.mask.shape != (height, width):
            small = Image.fromarray(mask.mask.astype(np.uint8) * 255).resize(out_size, Image.Resampling.NEAREST)
            raster = np.asarray(small) > 127
        else:
            raster = mask.mask
        labels[mask.frame][raster] = mask.object_id
    return labels


def draw_overlay(image: Image.Image, masks: list[meta_api.FrameMask], labels: dict[int, str]) -> Image.Image:
    """Keyframe with each object's mask tinted and labeled (used as the scene thumbnail)."""
    from .materials import PALETTE

    base = image.convert("RGBA")
    tint = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(base)
    for mask in masks:
        color = PALETTE[(mask.object_id - 1) % len(PALETTE)]
        alpha = Image.fromarray((mask.mask * 120).astype(np.uint8))
        layer = Image.new("RGBA", base.size, color + (0,))
        layer.putalpha(alpha)
        tint.alpha_composite(layer)
    base.alpha_composite(tint)
    draw = ImageDraw.Draw(base)
    for mask in masks:
        ys, xs = np.nonzero(mask.mask)
        if len(xs):
            color = PALETTE[(mask.object_id - 1) % len(PALETTE)]
            draw.text(
                (int(xs.min()), max(0, int(ys.min()) - 18)),
                labels.get(mask.object_id, "?"),
                fill=color,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
                font_size=max(14, base.width // 40),
            )
    return base.convert("RGB")


def run(job_dir: Path, frameset: FrameSet, recon_size: tuple[int, int]) -> dict:
    frames_dir = job_dir / "frames"
    keyframe = choose_keyframe(frameset)
    key_image = Image.open(frames_dir / f"{keyframe:03d}.jpg").convert("RGB")

    detections, usage = meta_api.ask_structured(DETECT_PROMPT, [key_image], Detections, reasoning_effort="medium")
    kept = filter_detections(detections.objects, frameset.size)
    if not kept:
        raise RuntimeError("no objects were found on the table")

    file_id = meta_api.upload_video(job_dir / "clip.mp4")
    try:
        masks = asyncio.run(meta_api.track_boxes_in_video(file_id, frameset.size, keyframe, [box for _, box in kept]))
    finally:
        meta_api.delete_file(file_id)

    labels = label_maps(masks, frameset.count, frameset.size, recon_size)
    np.savez_compressed(job_dir / "masks.npz", labels=labels)

    names = {i + 1: d.label for i, (d, _) in enumerate(kept)}
    key_masks = [m for m in masks if m.frame == keyframe]
    draw_overlay(key_image, key_masks, names).save(job_dir / "segmentation.jpg", quality=88)

    summary = {
        "keyframe": keyframe,
        "muse_spark_usage": usage,
        "objects": [
            {
                "index": i + 1,
                "label": detection.label,
                "material_hint": detection.material,
                "box": list(box),
                "frames_tracked": len({m.frame for m in masks if m.object_id == i + 1}),
                "mean_confidence": _mean([m.confidence for m in masks if m.object_id == i + 1]),
            }
            for i, (detection, box) in enumerate(kept)
        ],
    }
    (job_dir / "detections.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _mean(values: list[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return round(float(np.mean(known)), 3) if known else None
