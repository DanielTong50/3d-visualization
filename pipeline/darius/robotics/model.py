"""The robot's-eye view of a scene: everything the exporters write, in the world frame.

`build` reads a scene package once and converts it to the robotics conventions
(`darius.core.frames`): +z up, the table top at z = 0, body frames with world axes,
collision meshes re-expressed in them. Every writer (MJCF, URDF, USD, the JSON
manifest) works from this one model, so they cannot disagree about a pose, a mass
or a grasp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .. import __version__
from ..core import frames
from ..scene.io import Mesh, ScenePackage
from . import grasps as grasping
from . import mass

TABLE_THICKNESS_M = 0.04
DEFAULT_RGBA = (0.62, 0.65, 0.69, 1.0)


@dataclass(frozen=True)
class Shape:
    """One collision shape in its body's frame (world axes).

    kind is "mesh" (a convex mesh, `mesh` set), "box" (`size` = half extents),
    "sphere" (`size` = [radius]) or "cylinder" (`size` = [radius, half height], along +z).
    """

    name: str
    kind: str
    size: tuple[float, ...] = ()
    mesh: Mesh | None = None


@dataclass(frozen=True)
class Body:
    id: str
    label: str
    material: str
    position: np.ndarray  # world
    quat: np.ndarray  # [x, y, z, w], body to world
    mass: mass.MassProperties  # body frame, world axes
    friction: float
    restitution: float
    shapes: list[Shape]
    visual: Mesh | None  # the convex hull of a scanned object, for display; None for primitives
    grasps: list[grasping.Grasp]  # body frame, world axes
    rgba: tuple[float, float, float, float]
    physics_source: str
    dimensions_m: tuple[float, float, float]  # [length, width, height], height along +z


@dataclass(frozen=True)
class Camera:
    """A capture camera: OpenGL axes (x right, y up, looking along -z) in `quat`, world frame."""

    name: str
    position: np.ndarray
    quat: np.ndarray
    fovy_deg: float
    width: int
    height: int

    def intrinsics(self) -> np.ndarray:
        """OpenCV K for the camera's image size (principal point at the center)."""
        fy = self.height / 2 / math.tan(math.radians(self.fovy_deg) / 2)
        return np.array([[fy, 0, self.width / 2], [0, fy, self.height / 2], [0, 0, 1.0]])

    def cv_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """World-from-camera rotation and position with OpenCV camera axes."""
        return frames.quat_to_matrix(self.quat) @ frames.CV_FROM_GL, self.position


@dataclass
class RobotScene:
    scene_id: str
    title: str
    gravity: np.ndarray
    table_outline: np.ndarray  # (N, 2) world [x, y], counter-clockwise seen from above
    table_rgba: tuple[float, float, float, float]
    floor_depth_m: float | None
    bodies: list[Body]
    cameras: list[Camera] = field(default_factory=list)
    source: str = "hand-authored"
    generator: str = f"Darius {__version__}"

    def table_mesh(self) -> Mesh:
        """The table top as a closed convex prism, top face at z = 0."""
        outline = self.table_outline
        n = len(outline)
        top = np.column_stack([outline, np.zeros(n)])
        bottom = np.column_stack([outline, np.full(n, -TABLE_THICKNESS_M)])
        faces = [(0, k, k + 1) for k in range(1, n - 1)]  # top fan, counter-clockwise from above
        faces += [(n, n + k + 1, n + k) for k in range(1, n - 1)]  # bottom fan, reversed
        for k in range(n):
            j = (k + 1) % n
            faces += [(k, n + k, n + j), (k, n + j, j)]
        return Mesh(np.concatenate([top, bottom]), np.array(faces, dtype=np.int64))


def hex_rgba(color: str | None) -> tuple[float, float, float, float]:
    if not color or not color.startswith("#") or len(color) != 7:
        return DEFAULT_RGBA
    return tuple(round(int(color[i : i + 2], 16) / 255, 3) for i in (1, 3, 5)) + (1.0,)  # type: ignore[return-value]


def counter_clockwise(points: np.ndarray) -> np.ndarray:
    area = 0.5 * float(np.sum(points[:, 0] * np.roll(points[:, 1], -1) - np.roll(points[:, 0], -1) * points[:, 1]))
    return points if area >= 0 else points[::-1]


def build(package: ScenePackage, max_cameras: int = 8) -> RobotScene:
    """Convert a scene package into the robotics model (computing mass properties and grasps
    where scene.json doesn't carry them)."""
    manifest = package.manifest
    w = frames.WORLD_FROM_SCENE
    bodies = []
    for obj in manifest.objects:
        props = mass.from_inertial(obj.mass_kg, obj.inertial) if obj.inertial else mass.for_object(package, obj)
        if obj.grasps is not None:
            found = [grasping.Grasp.from_spec(g) for g in obj.grasps]
        else:
            found = grasping.for_object(package, obj, props.com_m)
        collider = obj.collider
        if collider.type == "box":
            hx, hy, hz = collider.half_extents
            shapes = [Shape(f"{obj.id}", "box", (hx, hz, hy))]
        elif collider.type == "sphere":
            shapes = [Shape(f"{obj.id}", "sphere", (collider.radius,))]
        elif collider.type == "cylinder":  # scene.json's cylinders run along +y, which is world +z
            shapes = [Shape(f"{obj.id}", "cylinder", (collider.radius, collider.half_height))]
        else:
            meshes = package.collision_meshes(obj)
            names = [obj.id] if len(meshes) == 1 else [f"{obj.id}_part{k}" for k in range(len(meshes))]
            shapes = [Shape(name, "mesh", mesh=m.transformed(w)) for name, m in zip(names, meshes, strict=True)]
        dims = obj.dimensions_m
        bodies.append(
            Body(
                id=obj.id,
                label=obj.label,
                material=obj.material,
                position=frames.to_world(obj.pose.position),
                quat=frames.rotation_to_world(obj.pose.rotation_quat),
                mass=props.to_world(),
                friction=obj.friction,
                restitution=obj.restitution,
                shapes=shapes,
                visual=package.hull(obj).transformed(w) if collider.type in ("convex_hull", "convex_parts") else None,
                grasps=[
                    grasping.Grasp(w @ g.position, w @ g.approach, w @ g.closing, g.width_m, g.score) for g in found
                ],
                rgba=hex_rgba(obj.color),
                physics_source=obj.physics_source,
                dimensions_m=(dims[0], dims[2], dims[1]),
            )
        )
    outline = package.table_outline()  # scene [x, z]; world y = -z
    table_xy = counter_clockwise(np.column_stack([outline[:, 0], -outline[:, 1]]))
    table_y = manifest.background.table_plane.point[1]
    if abs(table_y) > 1e-9:
        raise ValueError("the table top must be the plane y = 0 in scene.json")
    return RobotScene(
        scene_id=manifest.scene_id,
        title=manifest.title,
        gravity=frames.to_world(manifest.gravity),
        table_outline=table_xy,
        table_rgba=hex_rgba(manifest.background.table_color),
        floor_depth_m=manifest.background.floor_depth_m,
        bodies=bodies,
        cameras=capture_cameras(package, max_cameras),
        source=manifest.provenance.source if manifest.provenance else "hand-authored",
    )


def capture_cameras(package: ScenePackage, limit: int) -> list[Camera]:
    """The opening view, plus up to `limit` - 1 cameras spread along the filmed path."""
    manifest = package.manifest
    cameras: list[Camera] = []
    capture = manifest.capture
    width, height = capture.image_size if capture and capture.image_size else (768, 1024)
    if manifest.view:
        view = manifest.view
        position = np.array(view.position, float)
        rotation = frames.look_rotation(np.array(view.target, float) - position, np.array([0.0, 1.0, 0.0]))
        quat = frames.matrix_to_quat(frames.WORLD_FROM_SCENE @ rotation)
        cameras.append(Camera("view", frames.to_world(position), quat, view.fov_deg, width, height))
    if capture:
        picks = np.unique(np.linspace(0, len(capture.cameras) - 1, max(1, limit - len(cameras))).round().astype(int))
        for k in picks:
            camera = capture.cameras[k]
            rotation = frames.WORLD_FROM_SCENE @ frames.quat_to_matrix(np.array(camera.rotation_quat))
            cameras.append(
                Camera(
                    f"capture_{k:02d}",
                    frames.to_world(camera.position),
                    frames.matrix_to_quat(rotation),
                    capture.fov_deg,
                    width,
                    height,
                )
            )
    return cameras
