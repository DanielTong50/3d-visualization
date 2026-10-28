"""Track every part through a build video: Muse Spark finds and names the parts, SAM 3.1 follows them.

1. Ground. On an anchor frame at the start of the build, where every part lies apart on the floor,
   Muse Spark sees the frame and the manual's parts page and returns, per piece, its letter, a box
   and a point on the piece, on its 0-1000 grid. It looks twice: at the whole frame, then at a close
   crop around everything it found, where a thin leg spans several times the pixels.
2. Track. SAM 3.1 receives each box and point as one object's prompt on the anchor frame and
   propagates a mask for every piece through the whole video, forward and backward, in one request
   (up to 16 objects). Box and point prompts make the tracks independent of how a part is named.
3. Find again. SAM drops a part that leaves the frame (the camera comes in close to fasten) and
   does not pick it up when it comes back. Every few seconds, Muse Spark looks for the lost parts
   on the frame, given the anchor frame with every part outlined and numbered (set-of-mark
   prompting); each part it finds becomes a prompt for that part on that frame. SAM tracks the
   parts found again once more, and their new masks fill only the frames the first pass lost.
4. Read contacts. A joint of the plan is two parts held together; once it is made, the two parts'
   masks touch. Pairs of objects whose masks have touched for a second without a break are counted
   by pair of letters against the joints the plan says each step adds, which turns the tracks into
   plan-conditioned evidence of which steps are done: a second, independent source next to Muse
   Spark's step readings, fused in the tracker.

Tracks are stored as polygons with each mask's box, area and confidence. Everything downstream
(contacts, scores) is computed from the stored polygons, so a stored run re-scores exactly.
"""

from __future__ import annotations

import asyncio
import math
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from ..assembly.schema import AssemblyPlan, Observation
from ..core.errors import InputError

SAM_FPS = 5.0
SAM_SIZE = (960, 540)
CONTACT_SIZE = (320, 180)
CONTACT_GAP_PX = 1

ZOOM_PAD = 0.2  # the close look's margin around everything the first look found, as a share of its size
GROUND_VOTES = 3  # close looks at the anchor frame; each piece takes the letter most of them give it
REFIND_EVERY_S = 4.0  # how often Muse Spark looks again for the parts SAM 3.1 has lost
MIN_AREA_PX = 30  # a mask smaller than this (at SAM_SIZE) counts as lost

GROUND_PROMPT = """The first image is a frame of a video (or a close crop of one) of someone about to
assemble a piece of flat-pack furniture: the parts lie apart on the floor. The second image is the
manual's parts page, with each part's letter, count and size in millimetres.

Find every part piece in the first image and return, for each, its letter, a tight box_2d around it
and a point_2d on the piece itself, near its middle (for a thin piece lying diagonally, a point on
the piece, not on the floor beside it). Coordinates are on a 0-1000 grid where x grows to the right
and y grows downward, relative to the first image: box_2d is [x_min, y_min, x_max, y_max] and
point_2d is [x, y]. One entry per piece: if there are two pieces with the same letter, return two
entries. Use the parts' sizes and shapes to tell them apart. Leave out the hardware tray, tools, the
floor and walls, and pieces you cannot see.

Parts:
{parts}"""

REFIND_PROMPT = """The first image is a frame from early in a video of someone assembling flat-pack
furniture, with each part outlined and numbered. The second image is a later frame of the same video:
the camera has moved, and parts may have been picked up, moved or joined to others.

Find these numbered parts in the second image: {numbers}. For each one you can see, return its
number, a tight box_2d around it and a point_2d on the part itself, near its middle, on a 0-1000 grid
relative to the second image ([x_min, y_min, x_max, y_max] and [x, y]; x grows to the right, y
downward). Tell the parts apart by shape and size and by where they lie relative to the others.
Leave out any part you cannot see or are unsure of."""


class PartBox(BaseModel):
    label: str = Field(description="The part's letter from the manual")
    box_2d: list[int] = Field(description="[x_min, y_min, x_max, y_max] on a 0-1000 grid")
    point_2d: list[int] = Field(description="[x, y] on the piece, near its middle, on a 0-1000 grid")


class PartBoxes(BaseModel):
    parts: list[PartBox]


class PartPrompt(BaseModel):
    """What SAM 3.1 was asked to track: Muse Spark's box and point for one piece on one frame, in
    clip pixels (the anchor frame, or a later one where Muse Spark found a lost piece again)."""

    object: int
    label: str
    frame: int
    box: list[int]
    point: list[int] | None


class FoundPart(BaseModel):
    number: int = Field(description="The part's number in the first image")
    box_2d: list[int] = Field(description="[x_min, y_min, x_max, y_max] on a 0-1000 grid of the second image")
    point_2d: list[int] = Field(description="[x, y] on the part, near its middle, on a 0-1000 grid")


class FoundParts(BaseModel):
    parts: list[FoundPart]


class TrackFrame(BaseModel):
    """One part's mask on one frame, in the tracking clip's pixels."""

    frame: int
    box: list[int]
    area: int
    confidence: float | None
    polygon: list[int] = Field(description="Outline as x0, y0, x1, y1, ...")


class PartTrack(BaseModel):
    object: int
    label: str
    frames: list[TrackFrame]


class PartTracks(BaseModel):
    plan: str
    model: str
    grounding_model: str
    fps: float
    size: list[int]
    anchor_frame: int
    frames: int
    prompts: list[PartPrompt]
    tracks: list[PartTrack]
    seconds: float


def describe_parts(plan: AssemblyPlan) -> str:
    counts = Counter(p.label for p in plan.parts)
    first = {}
    for part in plan.parts:
        first.setdefault(part.label, part)
    lines = []
    for label, n in sorted(counts.items()):
        part = first[label]
        size = " x ".join(str(round(v)) for v in part.size_mm)
        lines.append(f"- {label}: {part.name}, {n} piece{'s' if n > 1 else ''}, {size} mm")
    return "\n".join(lines)


def grid_to_pixels(box: Sequence[int], size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """A 0-1000 grid box as a half-open pixel box (clamped: the model sometimes emits 1001)."""
    width, height = size
    x0, y0, x1, y1 = (min(max(v, 0), 1000) / 1000 for v in box)
    left, right = sorted((round(x0 * width), round(x1 * width)))
    top, bottom = sorted((round(y0 * height), round(y1 * height)))
    right, bottom = min(right, width), min(bottom, height)
    if right - left < 3 or bottom - top < 3:
        return None
    return left, top, right, bottom


def grid_point(point: Sequence[int], size: tuple[int, int]) -> tuple[int, int] | None:
    """A 0-1000 grid point as a pixel, or None if it is malformed."""
    if len(point) != 2:
        return None
    width, height = size
    x, y = (min(max(v, 0), 1000) / 1000 for v in point)
    return min(round(x * width), width - 1), min(round(y * height), height - 1)


def tracking_clip(video: Path, out: Path, fps: float = SAM_FPS, size: tuple[int, int] = SAM_SIZE) -> Path:
    """The video re-encoded for SAM 3.1: fixed frame rate and size, H.264, fast start."""
    width, height = size
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(video),
        "-vf", f"fps={fps},scale={width}:{height}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "20", "-movflags", "+faststart", "-an", str(out),
    ]  # fmt: skip
    if subprocess.run(command, capture_output=True).returncode != 0:
        raise InputError("That file doesn't look like a video we can read.")
    return out


def read_frame(video: Path, index: int | None = None, time_s: float | None = None) -> Image.Image:
    """One frame of a video, by index or by time."""
    capture = cv2.VideoCapture(str(video))
    if time_s is not None:
        capture.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
    else:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index or 0)
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise InputError("The video is too short to track parts in.")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def count_frames(clip: Path) -> int:
    capture = cv2.VideoCapture(str(clip))
    n = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return n


def zoom_region(boxes: Sequence[Sequence[int]], size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """A crop around every box (0-1000 grid) with a margin, in pixels; None if it would not zoom in."""
    if not boxes:
        return None
    width, height = size
    x0, y0 = min(b[0] for b in boxes) / 1000 * width, min(b[1] for b in boxes) / 1000 * height
    x1, y1 = max(b[2] for b in boxes) / 1000 * width, max(b[3] for b in boxes) / 1000 * height
    pad_x, pad_y = max(32, (x1 - x0) * ZOOM_PAD), max(32, (y1 - y0) * ZOOM_PAD)
    left, top = max(0, round(x0 - pad_x)), max(0, round(y0 - pad_y))
    right, bottom = min(width, round(x1 + pad_x)), min(height, round(y1 + pad_y))
    if (right - left) > 0.8 * width and (bottom - top) > 0.8 * height:
        return None
    return left, top, right, bottom


def from_crop(part: PartBox, region: tuple[int, int, int, int], size: tuple[int, int]) -> PartBox:
    """A box and point on a crop's 0-1000 grid, back on the whole frame's grid."""
    left, top, right, bottom = region
    width, height = size

    def x(v: int) -> int:
        return round((left + min(max(v, 0), 1000) / 1000 * (right - left)) / width * 1000)

    def y(v: int) -> int:
        return round((top + min(max(v, 0), 1000) / 1000 * (bottom - top)) / height * 1000)

    box = [x(part.box_2d[0]), y(part.box_2d[1]), x(part.box_2d[2]), y(part.box_2d[3])] if len(part.box_2d) == 4 else []
    point = [x(part.point_2d[0]), y(part.point_2d[1])] if len(part.point_2d) == 2 else []
    return PartBox(label=part.label, box_2d=box, point_2d=point)


def box_iou(a: Sequence[int], b: Sequence[int]) -> float:
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def vote(looks: Sequence[Sequence[PartBox]]) -> list[PartBox]:
    """The first look's pieces, each with the letter most looks gave the piece in the same place
    (box IoU 0.3 or more; ties keep the first look's letter). Pieces that look alike get their
    letters from size, and one look at it is a noisy reading."""
    out = []
    for piece in looks[0]:
        letters = [piece.label]
        for other in looks[1:]:
            best = max(other, key=lambda o: box_iou(o.box_2d, piece.box_2d), default=None)
            if best is not None and box_iou(best.box_2d, piece.box_2d) >= 0.3:
                letters.append(best.label)
        out.append(piece.model_copy(update={"label": Counter(letters).most_common(1)[0][0]}))
    return out


def ground_parts(plan: AssemblyPlan, frame: Image.Image, parts_page: Image.Image | None) -> list[PartBox]:
    """Muse Spark: per visible piece on the anchor frame, its letter, a box and a point. The first
    look sees the whole frame; then `GROUND_VOTES` close looks at a crop around everything it found,
    and each piece takes the letter most of them give it."""
    from ..models import meta_api

    labels = {p.label for p in plan.parts}

    def look(image: Image.Image) -> list[PartBox]:
        images = [image] + ([parts_page] if parts_page is not None else [])
        reading, _ = meta_api.ask_structured(
            GROUND_PROMPT.format(parts=describe_parts(plan)), images, PartBoxes, reasoning_effort="medium"
        )
        return [p for p in reading.parts if p.label in labels and len(p.box_2d) == 4]

    first = look(frame)
    region = zoom_region([p.box_2d for p in first], frame.size)
    if region is None:
        return first
    crop = frame.crop(region)
    scale = 1280 / max(crop.size)
    if scale > 1:
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
    with ThreadPoolExecutor(max_workers=GROUND_VOTES) as pool:
        looks = [[from_crop(p, region, frame.size) for p in found] for found in pool.map(look, [crop] * GROUND_VOTES)]
    looks.sort(key=len, reverse=True)  # the look that found the most pieces sets them
    return vote(looks) if len(looks[0]) >= len(first) else first


def polygon(mask: np.ndarray, epsilon: float = 1.2) -> list[int]:
    """The outline of a mask's largest region, simplified, as a flat list of points."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    points = cv2.approxPolyDP(largest, epsilon, True).reshape(-1, 2)
    return [int(v) for point in points for v in point]


def shrink(mask: np.ndarray, size: tuple[int, int] = CONTACT_SIZE) -> np.ndarray:
    """A mask at the contact resolution: a cell is on when at least half of it is."""
    small = cv2.resize(mask.astype(np.float32), size, interpolation=cv2.INTER_AREA)
    return small >= 0.5


def anchor_prompts(grounded: Sequence[PartBox], anchor: int) -> list[PartPrompt]:
    """Muse Spark's pieces as SAM 3.1 prompts on the anchor frame, numbered from 1."""
    prompts: list[PartPrompt] = []
    for part in grounded:
        box = grid_to_pixels(part.box_2d, SAM_SIZE)
        point = grid_point(part.point_2d, SAM_SIZE)
        if box is None:
            continue
        inside = point is not None and box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3]
        prompts.append(
            PartPrompt(
                object=len(prompts) + 1,
                label=part.label,
                frame=anchor,
                box=list(box),
                point=list(point) if inside else None,
            )
        )
    return prompts


def sam_tracks(file_id: str, prompts: Sequence[PartPrompt]) -> list[PartTrack]:
    """One SAM 3.1 request: every object's prompts in, its masks out as outlines."""
    from ..models import meta_api

    objects = sorted({p.object for p in prompts})
    requests = [
        [
            meta_api.BoxPrompt(
                p.frame, (p.box[0], p.box[1], p.box[2], p.box[3]), (p.point[0], p.point[1]) if p.point else None
            )
            for p in prompts
            if p.object == obj
        ]
        for obj in objects
    ]
    masks = asyncio.run(meta_api.track_objects_in_video(file_id, SAM_SIZE, requests))
    by_object: dict[int, list[meta_api.FrameMask]] = {}
    for mask in masks:
        by_object.setdefault(mask.object_id, []).append(mask)
    label = {p.object: p.label for p in prompts}
    tracks = []
    for i, obj in enumerate(objects):
        frames = []
        for record in sorted(by_object.get(i + 1, []), key=lambda m: m.frame):
            if not record.mask.any():
                continue
            ys, xs = np.nonzero(record.mask)
            outline = polygon(record.mask)
            frames.append(
                TrackFrame(
                    frame=record.frame,
                    box=[
                        record.left + int(xs.min()),
                        record.top + int(ys.min()),
                        record.left + int(xs.max()) + 1,
                        record.top + int(ys.max()) + 1,
                    ],
                    area=int(record.mask.sum()),
                    confidence=None if record.confidence is None else round(float(record.confidence), 3),
                    polygon=[v + (record.left if k % 2 == 0 else record.top) for k, v in enumerate(outline)],
                )
            )
        tracks.append(PartTrack(object=obj, label=label[obj], frames=frames))
    return tracks


def fill_gaps(first: Sequence[PartTrack], second: Sequence[PartTrack]) -> list[PartTrack]:
    """The first pass's tracks, with the second pass's masks only on the frames the first lost: a
    second pass can pick a part up again, but never overrides where the first pass held it (SAM
    propagates every prompt both ways, so a wrong later prompt could otherwise rewrite the past)."""
    again = {t.object: t for t in second}
    out = []
    for track in first:
        held = {f.frame: f for f in track.frames if f.area >= MIN_AREA_PX}
        extra = [f for f in again[track.object].frames if f.frame not in held] if track.object in again else []
        frames = sorted([*held.values(), *extra], key=lambda f: f.frame)
        out.append(PartTrack(object=track.object, label=track.label, frames=frames))
    return out


def mark(image: Image.Image, tracks: Sequence[PartTrack], frame: int) -> Image.Image:
    """The frame with each tracked part outlined and numbered, for Muse Spark to find them by."""
    canvas = np.array(image.convert("RGB"))
    scale = image.width / SAM_SIZE[0]
    for track in tracks:
        record = next((f for f in track.frames if f.frame == frame), None)
        if record is None or len(record.polygon) < 6:
            continue
        points = (np.array(record.polygon, np.float32).reshape(-1, 2) * scale).astype(np.int32)
        cv2.polylines(canvas, [points], True, (230, 40, 40), max(2, round(scale * 1.5)))
        x, y = points.mean(axis=0).astype(int)
        text = str(track.object)
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(canvas, (x - w // 2 - 5, y - h // 2 - 5), (x + w // 2 + 5, y + h // 2 + 5), (255, 255, 255), -1)
        cv2.putText(canvas, text, (x - w // 2, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2, cv2.LINE_AA)
    return Image.fromarray(canvas)


def lost_at(tracks: Sequence[PartTrack], frame: int) -> list[int]:
    """The objects with no mask (or a speck) on a frame."""
    seen = {t.object for t in tracks for f in t.frames if f.frame == frame and f.area >= MIN_AREA_PX}
    return [t.object for t in tracks if t.object not in seen]


def covered(tracks: Sequence[PartTrack], frame: int, point: tuple[int, int]) -> bool:
    """Does a point lie on some tracked part's mask on that frame?"""
    for track in tracks:
        for record in track.frames:
            if record.frame == frame and len(record.polygon) >= 6:
                outline = np.array(record.polygon, np.float32).reshape(-1, 1, 2)
                if cv2.pointPolygonTest(outline, (float(point[0]), float(point[1])), False) >= 0:
                    return True
    return False


def refind(
    video: Path,
    tracks: Sequence[PartTrack],
    marked: Image.Image,
    anchor: int,
    frames: int,
    every_s: float = REFIND_EVERY_S,
) -> list[PartPrompt]:
    """Muse Spark looks again for the parts SAM 3.1 lost: every `every_s` seconds, on the frame and
    the anchor frame with the parts outlined and numbered (set-of-mark prompting). Each part it
    finds there, off every mask SAM still follows, becomes a new prompt for that part."""
    from ..models import meta_api

    label = {t.object: t.label for t in tracks}
    step = max(1, round(every_s * SAM_FPS))
    jobs = [(f, lost) for f in range(anchor + step, frames, step) if (lost := lost_at(tracks, f))]

    def look(job: tuple[int, list[int]]) -> list[PartPrompt]:
        frame, lost = job
        later = read_frame(video, time_s=frame / SAM_FPS)
        numbers = ", ".join(str(o) for o in lost)
        reading, _ = meta_api.ask_structured(
            REFIND_PROMPT.format(numbers=numbers), [marked, later], FoundParts, reasoning_effort="medium"
        )
        found: dict[int, PartPrompt] = {}
        for part in reading.parts:
            box = grid_to_pixels(part.box_2d, SAM_SIZE)
            point = grid_point(part.point_2d, SAM_SIZE)
            if part.number not in lost or part.number in found or box is None or point is None:
                continue
            if not (box[0] <= point[0] < box[2] and box[1] <= point[1] < box[3]) or covered(tracks, frame, point):
                continue
            found[part.number] = PartPrompt(
                object=part.number, label=label[part.number], frame=frame, box=list(box), point=list(point)
            )
        return list(found.values())

    with ThreadPoolExecutor(max_workers=4) as pool:
        return [prompt for found in pool.map(look, jobs) for prompt in found]


def track_parts(
    video: Path,
    plan: AssemblyPlan,
    parts_page: Image.Image | None = None,
    anchor_s: float = 1.0,
    refind_every_s: float | None = REFIND_EVERY_S,
) -> PartTracks:
    """Ground the parts on the anchor frame with Muse Spark and track them with SAM 3.1; then let
    Muse Spark find the parts SAM lost and track again with those prompts added."""
    from ..models import meta_api

    started = time.monotonic()
    with tempfile.TemporaryDirectory() as folder:
        clip = tracking_clip(video, Path(folder) / "clip.mp4")
        frames = count_frames(clip)
        anchor = min(frames - 1, round(anchor_s * SAM_FPS))
        # Ground on the source frame (thin parts need the pixels), at the clip frame's time.
        first = read_frame(video, time_s=anchor / SAM_FPS)
        prompts = anchor_prompts(ground_parts(plan, first, parts_page), anchor)[: meta_api.SAM_MAX_OBJECTS]
        if not prompts:
            raise RuntimeError("Muse Spark found no parts to track on the anchor frame")
        file_id = meta_api.upload_video(clip)
        try:
            tracks = sam_tracks(file_id, prompts)
            if refind_every_s:
                found = refind(video, tracks, mark(first, tracks, anchor), anchor, frames, refind_every_s)
                if found:
                    prompts = sorted(prompts + found, key=lambda p: (p.object, p.frame))
                    again = {p.object for p in found}
                    tracks = fill_gaps(tracks, sam_tracks(file_id, [p for p in prompts if p.object in again]))
        finally:
            meta_api.delete_file(file_id)
    return PartTracks(
        plan=plan.id,
        model=meta_api.SAM_MODEL,
        grounding_model=meta_api.MUSE_SPARK_MODEL,
        fps=SAM_FPS,
        size=list(SAM_SIZE),
        anchor_frame=anchor,
        frames=frames,
        prompts=prompts,
        tracks=tracks,
        seconds=round(time.monotonic() - started, 2),
    )


def rasterize(frame: TrackFrame, size: tuple[int, int] = SAM_SIZE, out: tuple[int, int] = CONTACT_SIZE) -> np.ndarray:
    """A stored mask outline back as a mask, at the contact resolution."""
    canvas = np.zeros((size[1], size[0]), np.uint8)
    if len(frame.polygon) >= 6:
        cv2.fillPoly(canvas, [np.array(frame.polygon, np.int32).reshape(-1, 2)], 1)
    return shrink(canvas, out)


Masks = dict[int, tuple[str, np.ndarray]]
"""One frame's masks: object -> (letter, mask), at the contact resolution."""


def track_masks(tracks: PartTracks) -> list[Masks]:
    """Every frame's masks, rasterized from the stored outlines."""
    size = (tracks.size[0], tracks.size[1])
    frames: list[Masks] = [{} for _ in range(tracks.frames)]
    for track in tracks.tracks:
        for record in track.frames:
            if 0 <= record.frame < tracks.frames:
                frames[record.frame][track.object] = (track.label, rasterize(record, size))
    return frames


# ---------- Contacts: the plan's joints, seen in the masks ----------

HOLD_S = 1.0  # two masks must touch this long without a break before it counts as a contact


def contact_series(frames: Sequence[Masks], gap: int = CONTACT_GAP_PX) -> dict[tuple[int, int], np.ndarray]:
    """For every pair of objects, whether their masks touch (within `gap` cells) on each frame."""
    kernel = np.ones((2 * gap + 1, 2 * gap + 1), np.uint8)
    objects = sorted({o for masks in frames for o in masks})
    series = {(a, b): np.zeros(len(frames), bool) for i, a in enumerate(objects) for b in objects[i + 1 :]}
    for t, masks in enumerate(frames):
        solid = {o: m.astype(np.uint8) for o, (_, m) in masks.items() if m.any()}
        grown = {o: cv2.dilate(m, kernel) for o, m in solid.items()}
        for a, b in series:
            if a in solid and b in solid and (grown[a] & solid[b]).any():
                series[a, b][t] = True
    return series


def held_contacts(
    series: dict[tuple[int, int], np.ndarray], fps: float, hold_s: float = HOLD_S
) -> dict[tuple[int, int], float]:
    """When each pair of objects had touched for `hold_s` without a break (causal: the end of the hold)."""
    need = max(1, round(hold_s * fps))
    out = {}
    for pair, on in series.items():
        run = 0
        for t, touching in enumerate(on):
            run = run + 1 if touching else 0
            if run >= need:
                out[pair] = round(t / fps, 3)
                break
    return out


def joint_pairs(plan: AssemblyPlan) -> list[Counter[tuple[str, str]]]:
    """Each step's joints as a multiset of letter pairs."""
    label = {p.id: p.label for p in plan.parts}
    return [Counter(tuple(sorted(label[x] for x in plan.joint(j).parts)) for j in s.joints) for s in plan.steps]  # type: ignore[misc]


def contact_observation(plan: AssemblyPlan, seen: Counter[tuple[str, str]], time_s: float) -> Observation:
    """Tracker evidence from contacts. Twins share a letter, so contacts are counted by pair of letters:
    the plan says how many of each kind every step adds, contacts are credited to the steps in the
    manual's order, and a step reads as done in proportion to its contacts seen, never with certainty
    (masks touching in the image is not a fastened joint)."""
    before: Counter[tuple[str, str]] = Counter()
    step_done = {}
    for step, needed in zip(plan.steps, joint_pairs(plan), strict=True):
        if needed:
            got = sum(min(n, max(0, seen[pair] - before[pair])) for pair, n in needed.items())
            step_done[step.index] = round(0.2 + 0.6 * got / sum(needed.values()), 4)
        before += needed
    return Observation(time_s=time_s, step_done=step_done, source="sam-3.1")


def contact_observations(
    plan: AssemblyPlan, frames: Sequence[Masks], fps: float, times: Sequence[float]
) -> list[Observation]:
    """Contact evidence at each time, from the pairs of objects that have held contact by then."""
    held = held_contacts(contact_series(frames), fps)
    letter = {o: label for masks in frames for o, (label, _) in masks.items()}
    out = []
    for t in times:
        seen = Counter(tuple(sorted((letter[a], letter[b]))) for (a, b), at in held.items() if at <= t + 1e-6)
        out.append(contact_observation(plan, seen, t))  # type: ignore[arg-type]
    return out


def fuse(a: Observation, b: Observation, weight: float = 0.5) -> Observation:
    """A product of experts per step: P(done) from two independent readings, the second weighted."""
    step_done = dict(a.step_done)
    for index, q in b.step_done.items():
        p = step_done.get(index)
        if p is None:
            step_done[index] = q
            continue
        logit = math.log(p / (1 - p)) + weight * math.log(q / (1 - q))
        step_done[index] = round(1 / (1 + math.exp(-logit)), 4)
    return Observation(time_s=a.time_s, step_done=step_done, source=f"{a.source}+{b.source}")
