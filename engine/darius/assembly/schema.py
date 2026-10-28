"""The assembly plan: one furniture product as a graph of parts, hardware, joints and steps.

A plan is what a manual means. Parts (the lettered panels, legs and shelves) are
the nodes; joints (two parts held together by dowels, cam locks or screws) are the
edges; steps order the edges into the sequence a person follows. The same model is
produced three ways and compared on equal terms:

- procedurally, with exact geometry (`darius.furniture`), as ground truth;
- by reading a manual's pages (`darius.perception.manual`);
- by watching a video of the build (`darius.perception.video`), as a timeline of
  which steps were done when.

Geometry is optional: a plan read from a manual has no poses, but every graph
operation works without them. When present, lengths are millimeters in the
furniture frame: +y up, the floor is y = 0, the furniture centered on the origin,
its front facing +z. Rotations are quaternions [x, y, z, w].

`darius schema` writes the JSON Schema; the web app's TypeScript mirror is
frontend/src/engine/plan.ts.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Vec3 = Annotated[list[float], Field(min_length=3, max_length=3)]
Quat = Annotated[list[float], Field(min_length=4, max_length=4, description="x, y, z, w")]

PartKind = Literal[
    "side", "top", "bottom", "shelf", "back", "door", "drawer_front", "drawer_side", "drawer_back",
    "drawer_bottom", "leg", "rail", "plinth", "panel", "misc",
]  # fmt: skip
HardwareKind = Literal[
    "dowel", "cam_lock", "cam_bolt", "screw", "bolt", "nail", "shelf_pin", "hinge", "bracket", "glide", "allen_key",
]  # fmt: skip
JointKind = Literal["dowel", "cam_lock", "screw", "bolt", "nail", "slot", "pin", "glue"]
ActionKind = Literal["place", "insert", "rotate", "flip", "stand_up", "lay_down", "tighten", "align", "slide"]
Orientation = Literal["upright", "on_back", "on_front", "on_side", "upside_down"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Pose(Strict):
    position: Vec3 = Field(description="Millimeters, furniture frame")
    rotation: Quat = [0.0, 0.0, 0.0, 1.0]


class Part(Strict):
    """One physical piece. Identical pieces (two side panels) are separate parts sharing a label."""

    id: str = Field(description="Unique instance id, e.g. 'A1'")
    label: str = Field(description="The letter the manual prints, e.g. 'A'")
    name: str = Field(description="What it is, e.g. 'side panel'")
    kind: PartKind
    size_mm: Vec3 = Field(description="Extents along the part's own x, y, z")
    material: str = "particleboard, white foil"
    color: str = "#efece6"
    assembled: Pose | None = Field(default=None, description="Pose in the finished furniture")
    layout: Pose | None = Field(default=None, description="Where it lies on the floor before it is used")


class Hardware(Strict):
    id: str = Field(description="The part number the manual prints, e.g. '101350'")
    name: str
    kind: HardwareKind
    count: int = Field(ge=1)
    size_mm: list[float] = Field(default_factory=list, description="e.g. [8, 30] for an 8 x 30 mm dowel")


class HardwareUse(Strict):
    id: str
    count: int = Field(ge=1)


class Fastener(Strict):
    """One placed piece of hardware: where the AR view draws its insertion marker."""

    hardware: str
    position: Vec3
    axis: Vec3 = Field(description="Insertion direction (unit vector)")


class Joint(Strict):
    """Two parts held together: an edge of the assembly graph."""

    id: str
    parts: Annotated[list[str], Field(min_length=2, max_length=2)]
    kind: JointKind
    hardware: list[HardwareUse] = []
    fasteners: list[Fastener] = []


class Action(Strict):
    kind: ActionKind
    target: str = Field(description="A part id, or 'assembly' for everything built so far")
    axis: Literal["x", "y", "z"] | None = None
    degrees: float | None = None
    text: str


class Step(Strict):
    index: int = Field(ge=1)
    title: str
    instruction: str
    parts: list[str] = Field(default_factory=list, description="Parts this step brings in")
    joints: list[str] = Field(default_factory=list, description="Joints this step makes")
    hardware: list[HardwareUse] = []
    actions: list[Action] = []
    orientation: Orientation = Field(default="upright", description="How the assembly lies during this step")
    tools: list[str] = []


class AssemblyPlan(Strict):
    version: Literal[1] = 1
    id: str
    title: str
    category: str
    source: Literal["procedural", "manual", "hand-authored"]
    units: Literal["mm"] = "mm"
    dimensions_mm: Vec3 | None = None
    parts: list[Part]
    hardware: list[Hardware] = []
    joints: list[Joint]
    steps: list[Step]
    notes: list[str] = []

    def part(self, part_id: str) -> Part:
        for part in self.parts:
            if part.id == part_id:
                return part
        raise KeyError(f"no part {part_id!r} in {self.id}")

    def joint(self, joint_id: str) -> Joint:
        for joint in self.joints:
            if joint.id == joint_id:
                return joint
        raise KeyError(f"no joint {joint_id!r} in {self.id}")


class Observation(Strict):
    """What one look at the build (a camera frame) says about it: for each step, the probability
    that it is done. Steps the frame cannot speak to are left out (no evidence either way)."""

    time_s: float = 0.0
    step_done: dict[int, float] = Field(default_factory=dict)
    visible_parts: list[str] = []
    source: str = "vision"


class Belief(Strict):
    """The tracker's posterior over how many steps are complete."""

    probabilities: list[float] = Field(description="P(exactly k steps complete), k = 0..N")
    completed: int = Field(ge=0, description="The most probable number of completed steps")
    confidence: float = Field(ge=0, le=1)
