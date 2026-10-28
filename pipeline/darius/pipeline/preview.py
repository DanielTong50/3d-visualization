"""Small artifacts that show the pipeline at work.

While a job runs, the processing screen polls these (through the API's artifacts
endpoint) to draw each stage as it finishes: the keyframes it picked, the point
cloud VGGT reconstructed with the camera path, the cloud split into objects. The
package stage bundles the same artifacts into the scene (`scene/pipeline/`), so
the web app can replay how any shipped scene was built.

`points.bin` is little-endian: uint32 count, then count x float32 xyz, count x
uint8 rgb, count x uint8 label (0 = background, k = object k). `cameras.json`
holds the camera path in the same frame: three.js convention (+y up, a camera
looks down its -z), as [x, y, z, qx, qy, qz, qw] per keyframe.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from .jobs import load_arrays

PREVIEW = "preview"
SHEET_COLUMNS = 8
SHEET_THUMB_HEIGHT = 120
MAX_PREVIEW_POINTS = 80_000
# OpenCV cameras look down +z with +y down; three.js cameras look down -z with +y up.
CV_TO_GL = np.diag([1.0, -1.0, -1.0])


def keyframe_sheet(job_dir: Path) -> Path:
    """A contact sheet of the selected keyframes, in order."""
    frames = sorted((job_dir / "frames").glob("*.jpg"))
    thumbs = []
    for path in frames:
        with Image.open(path) as image:
            image = image.convert("RGB")
            width = round(image.width * SHEET_THUMB_HEIGHT / image.height)
            thumbs.append(image.resize((width, SHEET_THUMB_HEIGHT), Image.Resampling.LANCZOS))
    columns = min(SHEET_COLUMNS, max(1, len(thumbs)))
    rows = math.ceil(len(thumbs) / columns)
    cell = max(t.width for t in thumbs)
    sheet = Image.new("RGB", (columns * cell, rows * SHEET_THUMB_HEIGHT), (10, 10, 10))
    for i, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((i % columns) * cell, (i // columns) * SHEET_THUMB_HEIGHT))
    out = job_dir / PREVIEW / "keyframes.jpg"
    out.parent.mkdir(exist_ok=True)
    sheet.save(out, quality=82)
    return out


def write_points(path: Path, points: np.ndarray, colors: np.ndarray, labels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(np.uint32(len(points)).tobytes())
        handle.write(points.astype("<f4").tobytes())
        handle.write(colors.astype(np.uint8).tobytes())
        handle.write(labels.astype(np.uint8).tobytes())


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = path.read_bytes()
    count = int(np.frombuffer(data, "<u4", 1)[0])
    points = np.frombuffer(data, "<f4", count * 3, 4).reshape(-1, 3)
    colors = np.frombuffer(data, np.uint8, count * 3, 4 + 12 * count).reshape(-1, 3)
    labels = np.frombuffer(data, np.uint8, count, 4 + 15 * count)
    return points, colors, labels


def quaternions(matrices: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotation matrices -> (N, 4) unit quaternions [x, y, z, w].

    NumPy only, because the reconstruction image has no SciPy. Shepperd's method:
    build from the largest of w, x, y, z for numerical stability.
    """
    m = np.asarray(matrices, dtype=np.float64)
    trace = np.trace(m, axis1=1, axis2=2)
    candidates = np.stack(
        [
            trace,
            m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
            m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2],
            m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1],
        ],
        axis=1,
    )
    which = np.argmax(candidates, axis=1)
    quats = np.empty((len(m), 4))
    for i, (r, k) in enumerate(zip(m, which, strict=True)):
        s = 2 * np.sqrt(max(1 + candidates[i, k], 1e-12))
        if k == 0:
            quats[i] = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, s / 4]
        elif k == 1:
            quats[i] = [s / 4, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
        elif k == 2:
            quats[i] = [(r[0, 1] + r[1, 0]) / s, s / 4, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
        else:
            quats[i] = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, s / 4, (r[1, 0] - r[0, 1]) / s]
    return quats / np.linalg.norm(quats, axis=1, keepdims=True)


def camera_poses(world_from_camera: np.ndarray, centers: np.ndarray) -> list[list[float]]:
    """[x, y, z, qx, qy, qz, qw] per camera, from OpenCV camera-to-world rotations."""
    quats = quaternions(world_from_camera @ CV_TO_GL)
    return [[round(float(v), 5) for v in (*c, *q)] for c, q in zip(centers, quats, strict=True)]


def _sample(count: int, limit: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(count, limit, replace=False)) if count > limit else np.arange(count)


def raw_points(job_dir: Path) -> None:
    """After reconstruction: VGGT's confident points and cameras, before gravity is known.

    VGGT's world is the first camera's OpenCV frame (+y down, looking down +z), so it is
    turned to +y up for viewing; gravity comes later, from the table plane.
    """
    recon = load_arrays(job_dir / "recon.npz", ["points", "colors", "confidence", "extrinsic"])
    points = recon["points"].reshape(-1, 3)
    colors = recon["colors"].reshape(-1, 3)
    confidence = recon["confidence"].reshape(-1)
    keep = np.flatnonzero(np.isfinite(points).all(axis=1) & (confidence >= np.percentile(confidence, 50)))
    keep = keep[_sample(len(keep), MAX_PREVIEW_POINTS, np.random.default_rng(0))]
    rotation, translation = recon["extrinsic"][:, :, :3], recon["extrinsic"][:, :, 3]
    centers = -np.einsum("nji,nj->ni", rotation, translation)
    world_from_camera = np.transpose(rotation, (0, 2, 1))
    write_points(
        job_dir / PREVIEW / "points.bin",
        points[keep] @ CV_TO_GL.T,
        colors[keep],
        np.zeros(len(keep), np.uint8),
    )
    cameras = camera_poses(CV_TO_GL @ world_from_camera, centers @ CV_TO_GL.T)
    (job_dir / PREVIEW / "cameras.json").write_text(json.dumps({"frame": "vggt", "cameras": cameras}))


def lifted_points(job_dir: Path, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The gravity-aligned scene from the lift stage: background and labeled objects."""
    lift = load_arrays(job_dir / "lift.npz")
    info = json.loads((job_dir / "lift.json").read_text())
    rng = np.random.default_rng(0)
    parts_p, parts_c, parts_l = [], [], []
    object_budget = MAX_PREVIEW_POINTS // 2
    per_object = object_budget // max(1, len(info["objects"]))
    for obj in info["objects"]:
        pts, cols = lift[f"obj{obj['index']}_points"], lift[f"obj{obj['index']}_colors"]
        pick = _sample(len(pts), per_object, rng)
        parts_p.append(pts[pick])
        parts_c.append(cols[pick])
        parts_l.append(np.full(len(pick), obj["index"], np.uint8))
    background, colors = lift["background_points"], lift["background_colors"]
    pick = _sample(len(background), MAX_PREVIEW_POINTS - sum(len(p) for p in parts_p), rng)
    parts_p.insert(0, background[pick])
    parts_c.insert(0, colors[pick])
    parts_l.insert(0, np.zeros(len(pick), np.uint8))
    return np.concatenate(parts_p) * scale, np.concatenate(parts_c), np.concatenate(parts_l)


def gravity_cameras(job_dir: Path, scale: float = 1.0) -> list[list[float]]:
    """Camera path in the gravity frame: refined by the splat stage if it has run."""
    if (job_dir / "cameras.npz").exists():
        viewmats = load_arrays(job_dir / "cameras.npz", ["viewmats"])["viewmats"]
    else:
        recon = load_arrays(job_dir / "recon.npz", ["extrinsic"])
        transform = load_arrays(job_dir / "lift.npz", ["transform"])["transform"]
        viewmats = np.tile(np.eye(4), (len(recon["extrinsic"]), 1, 1))
        viewmats[:, :3, :] = recon["extrinsic"]
        viewmats = viewmats @ np.linalg.inv(transform)
    rotation, translation = viewmats[:, :3, :3], viewmats[:, :3, 3]
    centers = -np.einsum("nji,nj->ni", rotation, translation) * scale
    return camera_poses(np.transpose(rotation, (0, 2, 1)), centers)


def lift_preview(job_dir: Path) -> None:
    """After lifting: the same cloud, now gravity-aligned and split into objects."""
    points, colors, labels = lifted_points(job_dir)
    write_points(job_dir / PREVIEW / "points.bin", points, colors, labels)
    cameras = gravity_cameras(job_dir)
    (job_dir / PREVIEW / "cameras.json").write_text(json.dumps({"frame": "gravity", "cameras": cameras}))


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def bundle(job_dir: Path, out: Path, scale: float, timings: list[tuple[str, float]]) -> None:
    """scene/pipeline/: everything the web app's replay of this scene needs, in meters."""
    target = out / "pipeline"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    keyframe_sheet(job_dir)
    shutil.copy(job_dir / PREVIEW / "keyframes.jpg", target / "keyframes.jpg")
    for name in ("segmentation.jpg", "splat_test_view.jpg"):
        if (job_dir / name).exists():
            with Image.open(job_dir / name) as image:
                image = image.convert("RGB")
                image.thumbnail((1280, 1280))
                image.save(target / name, quality=84)
    points, colors, labels = lifted_points(job_dir, scale)
    write_points(target / "points.bin", points, colors, labels)
    frames = _read(job_dir / "frames.json")
    detections = _read(job_dir / "detections.json")
    physics = _read(job_dir / "physics.json")
    summary = {
        "units": "meters",
        "cameras": gravity_cameras(job_dir, scale),
        "timings": [{"stage": name, "seconds": round(seconds, 1)} for name, seconds in timings],
        "frames": {k: frames.get(k) for k in ("count", "size", "duration_s", "sharpness")},
        "reconstruct": _read(job_dir / "recon.json"),
        "segment": {
            "keyframe": detections.get("keyframe"),
            "frame_size": frames.get("size"),
            "objects": detections.get("objects", []),
        },
        "lift": {
            k: v for k, v in _read(job_dir / "lift.json").items() if k not in ("unit", "voxel", "plane_threshold")
        },
        "physics": {
            "scale_m_per_unit": physics.get("scale_m_per_unit"),
            "scale_votes": physics.get("scale_votes"),
            "scale_source": physics.get("scale_source"),
            "objects": [
                {
                    k: o.get(k)
                    for k in (
                        "index",
                        "label",
                        "category",
                        "confidence",
                        "material",
                        "mass_kg",
                        "mass_computed_kg",
                        "typical_mass_kg",
                        "measured_dimensions_cm",
                        "typical_dimensions_cm",
                        "friction",
                        "restitution",
                        "fragile",
                    )
                }
                for o in physics.get("objects", [])
            ],
        },
        "splat": _read(job_dir / "splats.json"),
        "package": _read(job_dir / "package.json"),
    }
    (target / "pipeline.json").write_text(json.dumps(summary, indent=1) + "\n")
