"""Stage 6, package: write the scene package (scene.json + PLY visuals + OBJ hulls).

Everything is converted to meters here, once, using the scale from the physics
stage. Each object's points and hull are stored in object-local coordinates
(origin at its bounding-box center) so the simulator can attach them to a
rigid body directly.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from . import export, geometry, meta_api
from .jobs import load_arrays
from .schema import (
    Background,
    HullCollider,
    PointsVisual,
    Pose,
    Provenance,
    Scene,
    SceneObject,
    StageTiming,
    TablePlane,
)

FLOOR_DEPTH_M = 0.75


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    """Binary little-endian PLY: float32 x y z, uint8 red green blue."""
    vertex = np.empty(
        len(points), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    vertex["x"], vertex["y"], vertex["z"] = points.T.astype(np.float32)
    vertex["red"], vertex["green"], vertex["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertex.tobytes())


def write_obj(path: Path, hull: geometry.Hull, comment: str) -> None:
    lines = [f"# {comment}", "# Object-local coordinates in meters."]
    lines += [f"v {x:.5f} {y:.5f} {z:.5f}" for x, y, z in hull.vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in hull.faces]
    path.write_text("\n".join(lines) + "\n")


def hex_color(rgb: np.ndarray) -> str:
    return "#" + "".join(f"{int(c):02x}" for c in rgb)


def thin(points: np.ndarray, colors: np.ndarray, limit: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= limit:
        return points, colors
    pick = np.random.default_rng(seed).choice(len(points), limit, replace=False)
    return points[pick], colors[pick]


def run(job_dir: Path, timings: list[tuple[str, float]], meta: dict, recon_info: dict) -> Scene:
    arrays = load_arrays(job_dir / "lift.npz")
    lift = json.loads((job_dir / "lift.json").read_text())
    physics = json.loads((job_dir / "physics.json").read_text())
    scale = physics["scale_m_per_unit"]
    voxel_m = lift["voxel"] * scale

    out = job_dir / "scene"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()

    objects: list[SceneObject] = []
    for n, props in enumerate(physics["objects"], start=1):
        index = props["index"]
        points = arrays[f"obj{index}_points"] * scale
        colors = arrays[f"obj{index}_colors"]
        dims, _ = geometry.footprint_extents(points)
        support = geometry.complete_bottom(points, geometry.SUPPORT_BAND * dims[1])
        hull_world = geometry.convex_hull(support)
        lo, hi = hull_world.vertices.min(axis=0), hull_world.vertices.max(axis=0)
        center = (lo + hi) / 2
        hull_local = geometry.Hull(hull_world.vertices - center, hull_world.faces, hull_world.volume)
        object_id = f"obj_{n:02d}"
        write_obj(out / f"{object_id}_hull.obj", hull_local, f"{props['label']}: convex hull of the lifted points")
        visual_points, visual_colors = thin(points - center, colors, 60_000)
        write_ply(out / f"{object_id}.ply", visual_points, visual_colors)
        objects.append(
            SceneObject(
                id=object_id,
                label=props["label"],
                confidence=props["confidence"],
                material=props["material"],
                dimensions_m=[round(float(d), 4) for d in dims],
                volume_m3=round(props["volume_m3"], 7),
                hollow=props["hollow"],
                fill_fraction=round(props["fill_fraction"], 3),
                density_kg_m3=props["density_kg_m3"],
                mass_kg=round(props["mass_kg"], 4),
                friction=round(props["friction"], 3),
                restitution=round(props["restitution"], 3),
                rigidity=props["rigidity"],
                fragile=props["fragile"],
                visual=PointsVisual(
                    file=f"{object_id}.ply", count=len(visual_points), point_size_m=round(voxel_m * 1.6, 5)
                ),
                collider=HullCollider(file=f"{object_id}_hull.obj"),
                # Lift slightly so the reconstructed base never starts inside the table.
                pose=Pose(
                    position=[
                        round(float(center[0]), 4),
                        round(float(center[1]) + 0.002, 4),
                        round(float(center[2]), 4),
                    ]
                ),
                physics_source="estimated",
                physics_notes=props["physics_notes"],
            )
        )

    bg_points, bg_colors = thin(arrays["background_points"] * scale, arrays["background_colors"], 150_000)
    write_ply(out / "background.ply", bg_points, bg_colors)
    (lo_xz, hi_xz) = arrays["table_bounds"] * scale
    size = (hi_xz - lo_xz).round(3).tolist()
    middle = (lo_xz + hi_xz) / 2

    if (job_dir / "segmentation.jpg").exists():
        shutil.copy(job_dir / "segmentation.jpg", out / "thumbnail.jpg")

    scene = Scene(
        scene_id=job_dir.name,
        title=meta.get("title", "Scanned tabletop"),
        scale_source=physics["scale_source"],
        background=Background(
            visual=PointsVisual(file="background.ply", count=len(bg_points), point_size_m=round(voxel_m * 1.8, 5)),
            table_plane=TablePlane(
                point=[float(middle[0]), 0.0, float(middle[1])], normal=[0.0, 1.0, 0.0], size_m=size
            ),
            table_color=hex_color(arrays["table_color"]),
            floor_depth_m=FLOOR_DEPTH_M,
        ),
        objects=objects,
        provenance=Provenance(
            source=meta.get("source", "uploaded video"),
            license=meta.get("license"),
            frames_used=recon_info["frames"],
            models={
                "reconstruction": recon_info["weights"],
                "detection_and_physics": meta_api.MUSE_SPARK_MODEL,
                "segmentation": meta_api.SAM_MODEL,
            },
            timings=[StageTiming(stage=name, seconds=round(s, 1)) for name, s in timings],
            scale_estimates=physics["scale_votes"],
        ),
    )
    (out / "scene.json").write_text(scene.model_dump_json(indent=2, exclude_none=True) + "\n")
    write_export_bundle(out)
    return scene


def write_export_bundle(scene_dir: Path) -> Path:
    """scene.xml (MuJoCo) plus export.zip: the manifest, MJCF and hull meshes it references."""
    export.export(scene_dir)
    bundle = scene_dir / "export.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(scene_dir.iterdir()):
            if path.name in ("scene.json", "scene.xml") or path.suffix == ".obj":
                # Fixed timestamps keep the archive byte-identical across rebuilds.
                info = zipfile.ZipInfo(f"{scene_dir.name}/{path.name}", date_time=(1980, 1, 1, 0, 0, 0))
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
    return bundle
