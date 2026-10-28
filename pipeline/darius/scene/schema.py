"""The scene package manifest (`scene.json`): the contract between pipeline and simulator.

The TypeScript mirror lives in frontend/src/sim/scene.ts; docs/scene-format.md is
the human-readable spec, and `python -m scan2sim.schema` regenerates
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


Collider = Annotated[BoxCollider | CylinderCollider | SphereCollider | HullCollider, Field(discriminator="type")]


class PointsVisual(Strict):
    type: Literal["points"] = "points"
    file: str = Field(description="Binary little-endian PLY with float xyz and uchar rgb")
    count: int = Field(ge=0)
    point_size_m: float = Field(gt=0)


class Pose(Strict):
    position: Vec3
    rotation_quat: Quat = [0.0, 0.0, 0.0, 1.0]


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
    visual: PointsVisual | None = None
    color: str | None = Field(default=None, description="Hex color used when there is no visual")
    collider: Collider
    pose: Pose
    physics_source: PhysicsSource
    physics_notes: str | None = None


class TablePlane(Strict):
    point: Vec3
    normal: Vec3
    size_m: Annotated[list[float], Field(min_length=2, max_length=2)]


class Background(Strict):
    visual: PointsVisual | None = None
    collider: Literal["table_plane"] = "table_plane"
    table_plane: TablePlane
    table_color: str | None = None
    floor_depth_m: float | None = Field(default=None, description="Floor this far below the table; null = none")


class StageTiming(Strict):
    stage: str
    seconds: float


class Provenance(Strict):
    source: str = Field(description="Where the input video came from")
    license: str | None = None
    frames_used: int
    models: dict[str, str]
    timings: list[StageTiming] = []
    scale_estimates: list[float] = []


class Scene(Strict):
    version: Literal[1] = 1
    scene_id: str
    title: str
    units: Literal["meters"] = "meters"
    gravity: Vec3 = [0.0, -9.81, 0.0]
    scale_source: str
    background: Background
    objects: list[SceneObject]
    provenance: Provenance | None = None


if __name__ == "__main__":
    json.dump(Scene.model_json_schema(), sys.stdout, indent=2)
    sys.stdout.write("\n")
