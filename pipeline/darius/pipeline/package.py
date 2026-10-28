"""Stage 7, package: write the scene package (scene.json, splats, points, colliders) and its
robotics bundle.

Everything is converted to meters here, once, using the scale from the physics
stage. Each object's points, splats and hull are stored in its body frame (origin
at its bounding-box center) so a simulator can attach them to a rigid body as
they are. The robotics layer then annotates every object with its mass
distribution and grasp candidates and writes the MJCF / URDF / USD bundle
(`darius.robotics`).
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np

from ..core import geometry
from ..models import meta_api
from ..robotics.annotate import annotate
from ..robotics.export import bundle
from ..scene.io import ScenePackage, write_obj, write_ply
from ..scene.schema import (
    AppearanceQuality,
    Background,
    Capture,
    CaptureCamera,
    CaptureView,
    ConvexPartsCollider,
    HullCollider,
    PointsVisual,
    Pose,
    Provenance,
    Scene,
    SceneObject,
    SplatsVisual,
    StageTiming,
    TablePlane,
)
from . import collision, preview, splats
from .jobs import load_arrays

FLOOR_DEPTH_M = 0.75


def hex_color(rgb: np.ndarray) -> str:
    return "#" + "".join(f"{int(c):02x}" for c in rgb)


def thin(points: np.ndarray, colors: np.ndarray, limit: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if len(points) <= limit:
        return points, colors
    pick = np.random.default_rng(seed).choice(len(points), limit, replace=False)
    return points[pick], colors[pick]


def write_thumbnail(frame: Path, out: Path, max_side: int = 720) -> None:
    """The clean keyframe (no overlays), for gallery cards."""
    from PIL import Image

    with Image.open(frame) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side))
        image.save(out, quality=86)


def capture_view(job_dir: Path, keyframe: int, transform: np.ndarray, scale: float, targets: list) -> CaptureView:
    """The keyframe's camera in scene coordinates, looking at the objects' center.

    Splats are only reliable near filmed viewpoints, so viewers should open here. The
    splat stage's refined cameras are used when they exist.
    """
    target = np.mean(np.array(targets), axis=0) if targets else np.zeros(3)
    if (job_dir / "cameras.npz").exists():
        cameras = load_arrays(job_dir / "cameras.npz")
        rotation, translation = cameras["viewmats"][keyframe, :3, :3], cameras["viewmats"][keyframe, :3, 3]
        center = -rotation.T @ translation
        height = float(cameras["Ks"][keyframe, 1, 2]) * 2
        focal = float(cameras["Ks"][keyframe, 1, 1])
    else:
        from .reconstruct import camera_centers

        recon = load_arrays(job_dir / "recon.npz", ["extrinsic", "intrinsic", "confidence"])
        center = geometry.apply_transform(transform, camera_centers(recon["extrinsic"][keyframe : keyframe + 1]))[0]
        height = recon["confidence"].shape[1]
        focal = float(recon["intrinsic"][keyframe, 1, 1])
    fov = 2 * math.degrees(math.atan(height / 2 / focal))
    return CaptureView(
        position=[round(float(v), 4) for v in center * scale],
        target=[round(float(v), 4) for v in target],
        fov_deg=round(fov, 2),
    )


def capture_path(job_dir: Path, scale: float, frame_size: tuple[int, int]) -> Capture | None:
    """Every keyframe's refined camera, for drawing the filmed path and fading the backdrop."""
    if not (job_dir / "cameras.npz").exists():
        return None
    Ks = load_arrays(job_dir / "cameras.npz", ["Ks"])["Ks"]
    width, height = frame_size
    cameras = [
        CaptureCamera(position=pose[:3], rotation_quat=pose[3:]) for pose in preview.gravity_cameras(job_dir, scale)
    ]
    fov = 2 * math.degrees(math.atan(height / 2 / float(np.median(Ks[:, 1, 1]))))
    return Capture(cameras=cameras, fov_deg=round(fov, 2), aspect=round(width / height, 4), image_size=[width, height])


#: The table reaches at least this far past each object's footprint.
TABLE_MARGIN_M = 0.03


def table_polygon(points: np.ndarray, band: float, cell: float, footprints: list[np.ndarray] = ()) -> np.ndarray:
    """Outline of the table top in meters: the dense region of background points within `band`
    of y = 0, grown to cover every object's footprint (plus TABLE_MARGIN_M).

    The support surface must reach under everything resting on it. Seen at a grazing angle or
    in poor light, part of a table can be too sparse to count as dense; on a kitchen counter the
    dense region covered 42 x 28 cm and three of five objects fell off it in simulation.
    """
    from scipy.spatial import ConvexHull

    surface = points[np.abs(points[:, 1]) < band]
    outline = geometry.table_outline(surface[:, [0, 2]], cell=cell)
    if not len(footprints):
        return outline
    grown = []
    for footprint in footprints:
        center = footprint.mean(axis=0)
        radius = max(float(np.linalg.norm(footprint - center, axis=1).max()), 1e-6)
        grown.append(center + (footprint - center) * (1 + TABLE_MARGIN_M / radius))
    everything = np.concatenate([outline, *grown])
    hull = everything[ConvexHull(everything).vertices]
    return hull[::-1]  # counter-clockwise seen from +y, like geometry.table_outline


def object_footprint(points: np.ndarray) -> np.ndarray:
    """Convex outline (x, z) of the lower half of an object: where it touches the table."""
    from scipy.spatial import ConvexHull

    low = points[points[:, 1] <= points[:, 1].min() + geometry.SUPPORT_BAND * np.ptp(points[:, 1])]
    xz = low[:, [0, 2]]
    return xz[ConvexHull(xz).vertices]


def run(job_dir: Path, timings: list[tuple[str, float]], meta: dict, recon_info: dict) -> Scene:
    arrays = load_arrays(job_dir / "lift.npz")
    lift = json.loads((job_dir / "lift.json").read_text())
    physics = json.loads((job_dir / "physics.json").read_text())
    scale = physics["scale_m_per_unit"]
    voxel_m = lift["voxel"] * scale
    has_splats = (job_dir / "splats.json").exists()
    if has_splats:
        trained = load_arrays(job_dir / "splats.npz")
        boxes = {
            o["index"]: (arrays[f"obj{o['index']}_points"].min(axis=0), arrays[f"obj{o['index']}_points"].max(axis=0))
            for o in physics["objects"]
        }
        owner = splats.assign(trained, boxes, **splats.cleanup_options(lift["voxel"], scale))
        splat_info = json.loads((job_dir / "splats.json").read_text())
        antialiased = bool(splat_info.get("antialiased", False))

    out = job_dir / "scene"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()

    objects: list[SceneObject] = []
    collider_report = {}
    footprints = []
    for n, props in enumerate(physics["objects"], start=1):
        index = props["index"]
        raw = arrays[f"obj{index}_points"]
        points = raw * scale
        colors = arrays[f"obj{index}_colors"]
        dims, _ = geometry.footprint_extents(points)
        support = geometry.complete_bottom(points, geometry.SUPPORT_BAND * dims[1])
        hull_world = geometry.convex_hull(support)
        lo, hi = hull_world.vertices.min(axis=0), hull_world.vertices.max(axis=0)
        center = (lo + hi) / 2
        object_id = f"obj_{n:02d}"
        write_obj(
            out / f"{object_id}_hull.obj",
            geometry.Hull(hull_world.vertices - center, hull_world.faces, hull_world.volume),
            f"{props['label']}: convex hull of the lifted points",
        )
        parts, collider_report[object_id] = collision.best_collider(support - center, voxel=max(voxel_m, 0.004))
        if len(parts) > 1:
            files = [f"{object_id}_part{k}.obj" for k in range(len(parts))]
            for file, part in zip(files, parts, strict=True):
                write_obj(out / file, part, f"{props['label']}: convex part")
            collider = ConvexPartsCollider(files=files)
        else:
            collider = HullCollider(file=f"{object_id}_hull.obj")
        footprints.append(object_footprint(raw))

        points_local, points_colors = thin(points - center, colors, 60_000)
        write_ply(out / f"{object_id}.ply", points_local, points_colors)
        points_visual = PointsVisual(
            file=f"{object_id}.ply", count=len(points_local), point_size_m=round(voxel_m * 1.6, 5)
        )
        visual = points_visual
        if has_splats:
            count = splats.write_spz(
                out / f"{object_id}.spz", splats.subset(trained, owner == index), center, scale, antialiased
            )
            visual = SplatsVisual(
                file=f"{object_id}.spz", count=count, sh_degree=splat_info["sh_degree"], antialiased=antialiased
            )
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
                color=hex_color(np.median(colors, axis=0)),
                visual=visual,
                fallback_visual=points_visual if has_splats else None,
                collider=collider,
                pick_hull=f"{object_id}_hull.obj",
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
    background_points = PointsVisual(file="background.ply", count=len(bg_points), point_size_m=round(voxel_m * 1.8, 5))
    background_visual = background_points
    backdrop_visual = None

    polygon = table_polygon(
        arrays["background_points"] * scale,
        band=lift["plane_threshold"] * scale,
        cell=0.02,
        footprints=[footprint * scale for footprint in footprints],
    )
    lo_xz, hi_xz = polygon.min(axis=0), polygon.max(axis=0)
    middle = (lo_xz + hi_xz) / 2

    if has_splats:
        scenery = splats.fill_under(
            splats.subset(trained, owner == 0),
            footprints,
            spacing=lift["voxel"],
            table_band=2 * lift["plane_threshold"],
            grow=splats.CUT_MARGIN_M / scale,
        )
        tallest = max((o.dimensions_m[1] for o in objects), default=0.1)
        on_table = splats.table_layer(scenery, polygon, scale, max(splats.TABLE_LAYER_MIN_HEIGHT_M, 1.5 * tallest))
        count = splats.write_spz(
            out / "background.spz", splats.subset(scenery, on_table), np.zeros(3), scale, antialiased
        )
        background_visual = SplatsVisual(
            file="background.spz", count=count, sh_degree=splat_info["sh_degree"], antialiased=antialiased
        )
        if (~on_table).any():
            count = splats.write_spz(
                out / "backdrop.spz", splats.subset(scenery, ~on_table), np.zeros(3), scale, antialiased
            )
            backdrop_visual = SplatsVisual(
                file="backdrop.spz", count=count, sh_degree=splat_info["sh_degree"], antialiased=antialiased
            )

    keyframe = json.loads((job_dir / "detections.json").read_text())["keyframe"]
    write_thumbnail(job_dir / "frames" / f"{keyframe:03d}.jpg", out / "thumbnail.jpg")
    view = capture_view(job_dir, keyframe, arrays["transform"], scale, [o.pose.position for o in objects])
    frame_size = tuple(json.loads((job_dir / "frames.json").read_text())["size"])

    scene = Scene(
        scene_id=job_dir.name,
        title=meta.get("title", "Scanned tabletop"),
        scale_source=physics["scale_source"],
        background=Background(
            visual=background_visual,
            backdrop=backdrop_visual,
            fallback_visual=background_points if has_splats else None,
            table_plane=TablePlane(
                point=[round(float(middle[0]), 4), 0.0, round(float(middle[1]), 4)],
                normal=[0.0, 1.0, 0.0],
                size_m=(hi_xz - lo_xz).round(3).tolist(),
                polygon_m=polygon.round(4).tolist(),
            ),
            table_color=hex_color(arrays["table_color"]),
            floor_depth_m=FLOOR_DEPTH_M,
        ),
        objects=objects,
        view=view,
        capture=capture_path(job_dir, scale, frame_size),
        provenance=Provenance(
            source=meta.get("source", "uploaded video"),
            license=meta.get("license"),
            frames_used=recon_info["frames"],
            models={
                "reconstruction": recon_info["weights"],
                "detection_and_physics": meta_api.MUSE_SPARK_MODEL,
                "segmentation": meta_api.SAM_MODEL,
                **({"appearance": "gsplat 3D Gaussian splatting"} if has_splats else {}),
            },
            timings=[StageTiming(stage=name, seconds=round(s, 1)) for name, s in timings],
            scale_estimates=physics["scale_votes"],
            appearance=AppearanceQuality(
                test_psnr_db=splat_info.get("test_psnr"),
                train_psnr_db=splat_info.get("train_psnr"),
                gaussians=splat_info["gaussians"],
                train_seconds=splat_info["train_seconds"],
            )
            if has_splats
            else None,
        ),
    )
    package = ScenePackage(out, scene)
    package.manifest = annotate(package)
    package.save()
    (job_dir / "package.json").write_text(json.dumps({"colliders": collider_report}, indent=2) + "\n")
    preview.bundle(job_dir, out, scale, timings)
    bundle.export(out)
    return package.manifest
