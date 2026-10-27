"""Stage 4, lift: turn per-frame 2D masks into separated, gravity-aligned 3D objects.

Every VGGT point comes from exactly one pixel of one keyframe, and SAM's label
map for that keyframe says which object (if any) that pixel shows. So each point
arrives already labeled, and the hard part is the noise, not the association:

1. Drop low-confidence depth.
2. Fit the table plane with RANSAC, using only background points near the
   objects (so a wall or the floor can't win), and rotate so the table is y = 0
   with +y toward the cameras. This defines gravity.
3. Voxelize. Each voxel takes a label by majority vote over every point that fell
   in it, from every frame. A mask that bleeds onto the table in one view is
   outvoted by the views where it didn't.
4. Per object: discard points on or below the table, remove statistical
   outliers, and keep the largest connected component. Objects with too few
   points stay in the background rather than becoming junk rigid bodies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import geometry
from .jobs import load_arrays
from .reconstruct import camera_centers

CONFIDENCE_PERCENTILE = 30
# The background is only scenery, so it can afford a stricter cut: low-confidence depth
# far from the table shows up as a haze of floating points in the viewer.
BACKGROUND_CONFIDENCE_PERCENTILE = 60
MIN_OBJECT_POINTS = 300
MAX_BACKGROUND_POINTS = 200_000


@dataclass
class LiftedObject:
    index: int
    label: str
    points: np.ndarray  # (P, 3) gravity-aligned, VGGT units
    colors: np.ndarray  # (P, 3) uint8


def scene_unit(points: np.ndarray, centers: np.ndarray, frame_of_point: np.ndarray) -> float:
    """Median camera-to-point distance: a natural length unit for VGGT's unscaled output."""
    return float(np.median(np.linalg.norm(points - centers[frame_of_point], axis=1)))


def table_candidates(points: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Background points within ~2.5 object radii of the objects' centroid."""
    background = labels == 0
    objects = ~background
    if objects.sum() < 50:
        return background
    centroid = points[objects].mean(axis=0)
    radius = np.percentile(np.linalg.norm(points[objects] - centroid, axis=1), 90)
    near = np.linalg.norm(points - centroid, axis=1) < 2.5 * radius
    return background & near


def run(job_dir: Path, object_labels: dict[int, str]) -> dict:
    recon = load_arrays(job_dir / "recon.npz")
    labels_map = load_arrays(job_dir / "masks.npz")["labels"]
    points = recon["points"].reshape(-1, 3)
    colors = recon["colors"].reshape(-1, 3)
    confidence = recon["confidence"].reshape(-1)
    labels = labels_map.reshape(-1).astype(np.int64)
    frames, height, width = labels_map.shape
    frame_of_point = np.repeat(np.arange(frames), height * width)
    centers = camera_centers(recon["extrinsic"])

    keep = np.isfinite(points).all(axis=1) & (confidence >= np.percentile(confidence, CONFIDENCE_PERCENTILE))
    background_confident = confidence >= np.percentile(confidence, BACKGROUND_CONFIDENCE_PERCENTILE)
    points, colors, labels, frame_of_point = points[keep], colors[keep], labels[keep], frame_of_point[keep]
    background_confident = background_confident[keep]
    unit = scene_unit(points, centers, frame_of_point)

    # Gravity: the supporting plane, oriented toward the cameras.
    candidates = table_candidates(points, labels)
    rng = np.random.default_rng(0)
    sample = np.flatnonzero(candidates)
    sample = rng.choice(sample, min(len(sample), 60_000), replace=False)
    plane_threshold = 0.006 * unit
    plane, _ = geometry.fit_plane_ransac(points[sample], plane_threshold, iterations=800)
    table_mask = candidates & (np.abs(plane.signed_distance(points)) < plane_threshold)
    transform = geometry.gravity_frame(plane, centers, points[table_mask])
    points = geometry.apply_transform(transform, points)
    cams = geometry.apply_transform(transform, centers)

    # Majority vote per voxel across all views.
    voxel = 0.004 * unit
    num_labels = int(labels.max()) + 1
    keys = geometry.voxel_keys(points, voxel)
    _, voxel_label, inverse = geometry.majority_vote(keys, labels, num_labels, min_share=0.5)
    voted = voxel_label[inverse]

    objects: list[LiftedObject] = []
    dropped: list[dict] = []
    assigned = np.zeros(len(points), dtype=bool)
    lift_margin = 2.0 * plane_threshold
    for index, label in sorted(object_labels.items()):
        mask = (voted == index) & (points[:, 1] > lift_margin)
        selected = np.flatnonzero(mask)
        if len(selected) >= MIN_OBJECT_POINTS:
            selected = selected[geometry.remove_statistical_outliers(points[selected])]
            selected = selected[geometry.largest_component(points[selected], 3 * voxel)]
        if len(selected) < MIN_OBJECT_POINTS:
            dropped.append({"index": index, "label": label, "points": int(len(selected))})
            continue
        assigned[selected] = True
        objects.append(LiftedObject(index, label, points[selected], colors[selected]))

    # Scenery: confident points, from just below the table up to twice the tallest object.
    top = max((o.points[:, 1].max() for o in objects), default=0.1 * unit)
    scenery = ~assigned & background_confident & (points[:, 1] > -0.1 * unit) & (points[:, 1] < 2 * top + 0.05 * unit)
    background = np.flatnonzero(scenery)
    background = rng.choice(background, min(len(background), MAX_BACKGROUND_POINTS), replace=False)
    table_inliers = np.abs(points[:, 1]) < plane_threshold
    table_xz = points[table_inliers & ~assigned][:, [0, 2]]
    lo, hi = np.percentile(table_xz, [2, 98], axis=0)

    arrays: dict[str, np.ndarray] = {
        "transform": transform,
        "background_points": points[background].astype(np.float32),
        "background_colors": colors[background],
        "camera_centers": cams.astype(np.float32),
        "table_bounds": np.array([lo, hi], dtype=np.float32),
        "table_color": np.median(colors[table_inliers & ~assigned], axis=0).astype(np.uint8),
    }
    for obj in objects:
        arrays[f"obj{obj.index}_points"] = obj.points.astype(np.float32)
        arrays[f"obj{obj.index}_colors"] = obj.colors
    np.savez_compressed(job_dir / "lift.npz", **arrays)

    residual = np.abs(points[table_inliers][:, 1])
    summary = {
        "unit": unit,
        "voxel": voxel,
        "plane_threshold": plane_threshold,
        "table_inlier_rms": float(np.sqrt(np.mean(residual**2))),
        "cameras_above_table": float(np.mean(cams[:, 1] > 0)),
        "points_after_confidence": int(len(points)),
        "objects": [
            {
                "index": o.index,
                "label": o.label,
                "points": int(len(o.points)),
                "dims_raw": geometry.footprint_extents(o.points)[0].round(5).tolist(),
            }
            for o in objects
        ],
        "dropped": dropped,
    }
    (job_dir / "lift.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
