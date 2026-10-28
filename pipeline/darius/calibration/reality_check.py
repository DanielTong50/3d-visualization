"""Reality check: measure an object's friction from a video of it being flicked across the table.

Model. After the hand lets go, a sliding (not rolling) object decelerates at a
constant a = mu * g (Coulomb friction), so its distance along the path is

    s(t) = v0 * t - a * t^2 / 2        until it stops at t = v0 / a,

which is linear in the unknowns (v0, a). Least squares on the sliding segment
gives both; mu = a / g, and the slide distance is v0^2 / (2 mu g).

Correction. Re-simulating the same push (same v0) with the estimated and the
measured mu gives the slide-distance error before and after correction. The
browser does that re-simulation in Rapier; `predicted_distance` is the analytic
version used for the report and tests.

Video to meters. The push is filmed from a fixed phone roughly above the table;
pixels are converted to meters with the object's known footprint (from the
scanned scene) against its mask's extent in the first frame. That assumes a
near-top-down view; the README documents the limitation.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

G = 9.81


@dataclass
class SlideFit:
    v0: float  # m/s at release
    deceleration: float  # m/s^2
    friction: float
    start_s: float  # time of release in the video
    duration_s: float
    distance_m: float  # observed slide distance
    rms_residual_m: float


def predicted_distance(v0: float, friction: float) -> float:
    return v0**2 / (2 * friction * G)


def smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def sliding_segment(t: np.ndarray, s: np.ndarray, still_speed: float = 0.03) -> tuple[int, int]:
    """Indices [start, end) of the free slide: from peak speed (the hand has let go)
    until the object comes to rest."""
    speed = np.gradient(smooth(s), t)
    start = int(np.argmax(speed))
    after = np.flatnonzero(speed[start:] < still_speed)
    end = start + (int(after[0]) if len(after) else len(s) - start)
    return start, max(end, start + 3)


def fit_slide(t: np.ndarray, s: np.ndarray) -> SlideFit:
    """Fit s = v0 t - a t^2 / 2 on the sliding segment of a 1D path-distance series."""
    start, end = sliding_segment(t, s)
    tt = t[start:end] - t[start]
    ss = s[start:end] - s[start]
    # Free intercept: the detected release point is only accurate to about a frame.
    design = np.stack([np.ones_like(tt), tt, -0.5 * tt**2], axis=1)
    (s0, v0, a), *_ = np.linalg.lstsq(design, ss, rcond=None)
    a = max(float(a), 1e-3)
    residual = ss - design @ np.array([s0, v0, a])
    return SlideFit(
        v0=float(v0),
        deceleration=a,
        friction=a / G,
        start_s=float(t[start]),
        duration_s=float(tt[-1]),
        distance_m=float(ss[-1] - s0),
        rms_residual_m=float(np.sqrt(np.mean(residual**2))),
    )


def path_distance(xy: np.ndarray) -> np.ndarray:
    """Cumulative distance along a 2D track (meters)."""
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def probe_video(video: Path) -> tuple[float, tuple[int, int]]:
    """Frame rate and displayed (width, height), honoring rotation metadata."""
    import subprocess
    from fractions import Fraction

    info = json.loads(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate:stream_side_data=rotation",
                "-of",
                "json",
                str(video),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )["streams"][0]
    width, height = info["width"], info["height"]
    rotation = next((int(d.get("rotation", 0)) for d in info.get("side_data_list", [])), 0)
    if abs(rotation) % 180 == 90:
        width, height = height, width
    return float(Fraction(info["r_frame_rate"])), (width, height)


def track_centroids(video: Path, concept: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Track the object with SAM 3.1; return times, pixel centroids of the track that moves
    most, and the object's extent in pixels on its first tracked frame."""
    from . import meta_api

    fps, size = probe_video(video)
    file_id = meta_api.upload_video(video)
    try:
        result = asyncio.run(
            meta_api._stream_sam(
                [{"type": "input_text", "text": concept}, {"type": "input_video", "file_id": file_id}],
                {"mask_encoding": "one_bit"},
                video=True,
            )
        )
    finally:
        meta_api.delete_file(file_id)

    tracks: dict[str, list[tuple[int, np.ndarray, float]]] = {}
    for record in meta_api.latest_masks(result.records):
        ys, xs = np.nonzero(meta_api.raster_from_record(record, size))
        if len(xs):
            frame = record.frame.frame_index if record.frame else 0
            extent = float(max(np.ptp(xs), np.ptp(ys)))
            tracks.setdefault(record.object_id, []).append((frame, np.array([xs.mean(), ys.mean()]), extent))
    if not tracks:
        raise RuntimeError(f'SAM 3.1 found no "{concept}" in the push video')
    best = max(tracks.values(), key=lambda tr: np.ptp(np.array([c for _, c, _ in tr]), axis=0).sum())
    best.sort(key=lambda item: item[0])
    times = np.array([f / fps for f, _, _ in best])
    return times, np.array([c for _, c, _ in best]), best[0][2]


def run(video: Path, scene_dir: Path, object_id: str) -> dict:
    """Measure friction for `object_id` and write the correction into the scene package."""
    manifest_path = scene_dir / "scene.json"
    scene = json.loads(manifest_path.read_text())
    obj = next(o for o in scene["objects"] if o["id"] == object_id)
    times, centroids, extent_px = track_centroids(video, obj["label"].split()[-1])
    meters_per_px = max(obj["dimensions_m"][0], obj["dimensions_m"][2]) / extent_px
    fit = fit_slide(times, path_distance(centroids * meters_per_px))

    estimated = obj["friction"]
    before = abs(predicted_distance(fit.v0, estimated) - fit.distance_m)
    after = abs(predicted_distance(fit.v0, fit.friction) - fit.distance_m)
    report = {
        "object_id": object_id,
        "video": video.name,
        "fit": asdict(fit),
        "friction_estimated": estimated,
        "friction_measured": fit.friction,
        "real_distance_m": fit.distance_m,
        "predicted_distance_before_m": predicted_distance(fit.v0, estimated),
        "predicted_distance_after_m": predicted_distance(fit.v0, fit.friction),
        "error_before_m": before,
        "error_after_m": after,
        "track": {"t": times.round(4).tolist(), "s_m": path_distance(centroids * meters_per_px).round(4).tolist()},
    }
    reports_path = scene_dir / "reality_check.json"
    reports = json.loads(reports_path.read_text()) if reports_path.exists() else []
    reports.append(report)
    reports_path.write_text(json.dumps(reports, indent=2) + "\n")

    obj["friction"] = round(fit.friction, 3)
    obj["physics_source"] = "measured"
    manifest_path.write_text(json.dumps(scene, indent=2) + "\n")
    return report
