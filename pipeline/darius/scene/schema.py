"""The scene package manifest (`scene.json`): the contract between pipeline and simulator.

The TypeScript mirror lives in frontend/src/sim/scene.ts; docs/scene-format.md is
the human-readable spec, and `darius schema` regenerates
docs/scene.schema.json from these models.
"""

from __future__ import annotations

import json
import sys
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Quat = Annotated[list[float], Field(min_length=4, max_length=4, description="x, y, z, w")]

Rigidity = Literal["rigid", "soft"]
PhysicsSource = Literal["estimated", "measured", "hand-authored"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoxCollider(Strict):
    type: Literal["box"] = "box"
    half_extents: Vec3


class CylinderCollider(Strict):
    type: Literal["cylinder"] = "cylinder"
    radius: float = Field(gt=0)
    half_height: float = Field(gt=0)


class SphereCollider(Strict):
    type: Literal["sphere"] = "sphere"
    radius: float = Field(gt=0)


class HullCollider(Strict):
    type: Literal["convex_hull"] = "convex_hull"
    file: str = Field(description="Wavefront OBJ in object-local meters")


class ConvexPartsCollider(Strict):
    """A compound of convex pieces (approximate convex decomposition), for concave objects."""

    type: Literal["convex_parts"] = "convex_parts"
    files: list[str] = Field(min_length=1, description="Wavefront OBJs in object-local meters, one convex piece each")


Collider = Annotated[
    BoxCollider | CylinderCollider | SphereCollider | HullCollider | ConvexPartsCollider,
    Field(discriminator="type"),
]


class PointsVisual(Strict):
    type: Literal["points"] = "points"
    file: str = Field(description="Binary little-endian PLY with float xyz and uchar rgb")
    count: int = Field(ge=0)
    point_size_m: float = Field(gt=0)


class SplatsVisual(Strict):
    """3D Gaussian splats, stored as Niantic SPZ (v3, gzip) in the scene's own frame (+y up, right-handed)."""

    type: Literal["splats"] = "splats"
    file: str = Field(description=".spz file")
    count: int = Field(ge=0)
    sh_degree: int = Field(ge=0, le=3)
    antialiased: bool = Field(
        default=False,
        description="Trained with anti-aliased rasterization: render with a 0.3 px² blur and opacity compensation. "
        "Otherwise the 0.3 px² dilation is part of each splat (classic 3DGS) and must be added without compensation.",
    )


Visual = Annotated[PointsVisual | SplatsVisual, Field(discriminator="type")]


class Pose(Strict):
    position: Vec3
    rotation_quat: Quat = [0.0, 0.0, 0.0, 1.0]


class Inertial(Strict):
    """How an object's mass is distributed, in its body frame (`darius.robotics.mass`)."""

    com_m: Vec3 = Field(description="Center of mass")
    inertia_kg_m2: Annotated[list[float], Field(min_length=6, max_length=6)] = Field(
        description="Inertia tensor about the center of mass, body axes: [ixx, iyy, izz, ixy, ixz, iyz]"
    )
    model: Literal["solid", "shell"] = Field(
        description="solid: uniform density over the collider. shell: the mass sits in a wall that holds "
        "`fill_fraction` of the hull's volume (hollow objects)"
    )


class Grasp(Strict):
    """A parallel-jaw grasp candidate in the object's body frame (`darius.robotics.grasps`)."""

    position: Vec3 = Field(description="Midpoint between the two fingertip contacts")
    rotation_quat: Quat = Field(description="Gripper frame: +z is the approach direction, +y the closing axis")
    width_m: float = Field(gt=0, description="Distance between the contacts; the gripper must open wider")
    score: float = Field(
        ge=0, le=1, description="Friction-cone margin, closeness to the center of mass, approach from above"
    )


class SceneObject(Strict):
    id: str
    label: str
    confidence: float = Field(ge=0, le=1)
    material: str
    dimensions_m: Vec3
    volume_m3: float = Field(ge=0)
    hollow: bool
    fill_fraction: float = Field(gt=0, le=1)
    density_kg_m3: float = Field(gt=0)
    mass_kg: float = Field(gt=0)
    friction: float = Field(ge=0, le=2)
    restitution: float = Field(ge=0, le=1)
    rigidity: Rigidity
    fragile: bool
    visual: Visual | None = None
    fallback_visual: PointsVisual | None = Field(default=None, description="Cheaper visual for low-end devices")
    color: str | None = Field(default=None, description="Hex color used when there is no visual")
    collider: Collider
    pick_hull: str | None = Field(default=None, description="Single convex hull OBJ for picking and raycasts")
    pose: Pose
    physics_source: PhysicsSource
    physics_notes: str | None = None
    inertial: Inertial | None = Field(default=None, description="Center of mass and inertia tensor")
    grasps: list[Grasp] | None = Field(default=None, description="Parallel-jaw grasp candidates, best first")


class TablePlane(Strict):
    point: Vec3
    normal: Vec3
    size_m: Annotated[list[float], Field(min_length=2, max_length=2)]
    polygon_m: list[Annotated[list[float], Field(min_length=2, max_length=2)]] | None = Field(
        default=None,
        min_length=3,
        description="Convex outline of the table top as [x, z] pairs, counter-clockwise seen from +y",
    )


class Background(Strict):
    visual: Visual | None = None
    backdrop: SplatsVisual | None = Field(
        default=None,
        description="Splats beyond the table (walls, floor, surroundings), world coordinates. Only reliable near "
        "the filmed viewpoints, so viewers should fade them out as the camera moves away from `capture.cameras`.",
    )
    fallback_visual: PointsVisual | None = None
    collider: Literal["table_plane"] = "table_plane"
    table_plane: TablePlane
    table_color: str | None = None
    floor_depth_m: float | None = Field(default=None, description="Floor this far below the table; null = none")


class StageTiming(Strict):
    stage: str
    seconds: float


class AppearanceQuality(Strict):
    """How well the splats reproduce the video: PSNR on held-out keyframes the optimizer never saw."""

    test_psnr_db: float | None = None
    train_psnr_db: float | None = None
    gaussians: int = Field(ge=0)
    train_seconds: float = Field(ge=0)


class Provenance(Strict):
    source: str = Field(description="Where the input video came from")
    license: str | None = None
    frames_used: int
    models: dict[str, str]
    timings: list[StageTiming] = []
    scale_estimates: list[float] = []
    appearance: AppearanceQuality | None = None


class CaptureView(Strict):
    """A camera the scene was filmed from: splats look best from viewpoints like this."""

    position: Vec3
    target: Vec3
    fov_deg: float = Field(gt=0, lt=180, description="Vertical field of view")


class CaptureCamera(Strict):
    """One keyframe's camera: position, and rotation from the camera's frame (three.js convention:
    looking down -z with +y up) to the scene frame."""

    position: Vec3
    rotation_quat: Quat


class Capture(Strict):
    """The path the phone took while filming, after camera refinement."""

    cameras: list[CaptureCamera] = Field(min_length=1)
    fov_deg: float = Field(gt=0, lt=180, description="Vertical field of view")
    aspect: float = Field(gt=0, description="Frame width / height")
    image_size: Annotated[list[int], Field(min_length=2, max_length=2)] | None = Field(
        default=None, description="Keyframe [width, height] in pixels, for camera intrinsics"
    )


class Scene(Strict):
    version: Literal[1] = 1
    scene_id: str
    title: str
    units: Literal["meters"] = "meters"
    gravity: Vec3 = [0.0, -9.81, 0.0]
    scale_source: str
    background: Background
    objects: list[SceneObject]
    view: CaptureView | None = None
    capture: Capture | None = None
    provenance: Provenance | None = None


if __name__ == "__main__":
    json.dump(Scene.model_json_schema(), sys.stdout, indent=2)
    sys.stdout.write("\n")
