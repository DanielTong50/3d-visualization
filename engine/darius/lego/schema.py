"""Typed LEGO-compatible task IR. Positions are integer stud/plate grid coordinates.

The frame is right handed, +Y up. A brick's position is its footprint's minimum
X/Z corner and its body-bottom Y. Rotations are clockwise quarter turns viewed
from above. The compiler uses nominal pitch, not manufacturing tolerances.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Grid3 = Annotated[list[Annotated[int, Field(ge=0, le=96, strict=True)]], Field(min_length=3, max_length=3)]
Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Studs2 = Annotated[list[Annotated[int, Field(ge=1, le=32, strict=True)]], Field(min_length=2, max_length=2)]
Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Source(Strict):
    kind: Literal["original", "reviewed-import"] = "original"
    description: str = "Original Darius curriculum build; not reconstructed from an official LEGO set."
    url: str | None = None
    training_rights: Literal["project-original", "permission-required", "permission-documented"] = "project-original"
    permission_reference: str | None = None
    review_status: Literal["original", "pending", "reviewed"] = "original"

    @model_validator(mode="after")
    def imported_rights(self) -> Source:
        if self.kind == "reviewed-import" and self.training_rights == "project-original":
            raise ValueError("imported instructions require a separate permission review for training")
        if self.kind == "reviewed-import" and self.review_status == "original":
            raise ValueError("imported instructions must declare pending or reviewed status")
        if self.training_rights == "permission-documented" and not (self.permission_reference or "").strip():
            raise ValueError("documented training permission requires a nonempty permission_reference")
        return self


class Baseplate(Strict):
    studs: Studs2 = [16, 16]
    color_hex: str = Field(default="#d7dccf", pattern=r"^#[0-9a-fA-F]{6}$")
    height_mm: float = Field(default=3.2, gt=0, le=20)


class Brick(Strict):
    id: Identifier
    part_id: str = Field(pattern=r"^[0-9]{3,7}$")
    color: str = Field(min_length=1, max_length=40)
    color_hex: str = Field(pattern=r"^#[0-9a-fA-F]{6}$")
    studs: Studs2
    height_plates: int = Field(default=3, ge=1, le=12, strict=True)
    position: Grid3
    rotation: Literal[0, 90, 180, 270] = 0

    @property
    def footprint(self) -> tuple[int, int]:
        return (self.studs[1], self.studs[0]) if self.rotation in (90, 270) else (self.studs[0], self.studs[1])

    @property
    def cells(self) -> set[tuple[int, int]]:
        x, _, z = self.position
        width, depth = self.footprint
        return {(x + i, z + j) for i in range(width) for j in range(depth)}


class BuildStep(Strict):
    index: int = Field(ge=1, le=256, strict=True)
    title: str = Field(min_length=1, max_length=160)
    brick_ids: list[Identifier] = Field(min_length=1, max_length=32)


class LegoAssembly(Strict):
    schema_version: Literal["1.0"] = "1.0"
    id: Identifier
    title: str = Field(min_length=1, max_length=120)
    description: str = Field(max_length=1000)
    source: Source = Field(default_factory=Source)
    baseplate: Baseplate = Field(default_factory=Baseplate)
    bricks: list[Brick] = Field(min_length=1, max_length=256)
    steps: list[BuildStep] = Field(min_length=1, max_length=256)
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_schedule(self) -> LegoAssembly:
        ids = [b.id for b in self.bricks]
        if len(set(ids)) != len(ids):
            raise ValueError("brick ids must be unique")
        if [s.index for s in self.steps] != list(range(1, len(self.steps) + 1)):
            raise ValueError("step indices must be consecutive, starting at 1")
        scheduled = [brick for step in self.steps for brick in step.brick_ids]
        if len(scheduled) != len(set(scheduled)) or set(scheduled) != set(ids):
            raise ValueError("every brick must be scheduled exactly once, with no unknown ids")
        return self


class Connection(Strict):
    id: str
    lower: str
    upper: str
    studs: list[list[int]]
    step: int


class Primitive(Strict):
    kind: Literal["move", "grasp", "release", "press", "verify"]
    position_mm: Vec3
    yaw_deg: float = 0.0
    duration_s: float = Field(gt=0)
    gripper: Literal["open", "closed", "hold"] = "hold"
    force_limit_n: float | None = Field(default=None, gt=0)
    description: str = ""


class CompiledStep(Strict):
    index: int
    source_step: int
    title: str
    brick_id: str
    dependencies: list[str]
    connection_ids: list[str]
    support_ratio: float = Field(ge=0, le=1)
    source_center_mm: Vec3
    target_center_mm: Vec3
    source_yaw_deg: float = Field(default=0.0, description="Clockwise plan-frame yaw of the brick in the indexed tray")
    target_yaw_deg: float = Field(description="Clockwise plan-frame yaw at the assembled pose")
    primitives: list[Primitive]


class Validation(Strict):
    valid: bool
    errors: list[str]
    warnings: list[str]


class Metrics(Strict):
    bricks: int
    connections: int
    stud_connections: int
    steps: int
    primitives: int
    nominal_duration_s: float


class Execution(Strict):
    status: Literal["simulation_only"] = "simulation_only"
    blockers: list[str]


class CompiledPlan(Strict):
    schema_version: Literal["1.0"] = "1.0"
    assembly: LegoAssembly
    connections: list[Connection]
    steps: list[CompiledStep]
    validation: Validation
    metrics: Metrics
    execution: Execution
    coordinate_system: str = "+Y up; x/z = 8 mm stud pitch; y = 3.2 mm plate pitch; TCP positions in mm"
