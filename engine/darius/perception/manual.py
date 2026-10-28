"""Read a flat-pack manual into an assembly plan.

All pages go to Muse Spark in one request, with a schema that asks for exactly what a plan
needs: the parts by letter with counts, the hardware by article number with counts, and for
every step the parts it brings in, the hardware it uses, the build orientation shown, and the
pairs of parts it joins. `to_plan` turns that reading into a plan (one instance per piece of a
repeated letter, joints in step order) that `validate` can check and `compare` can score against
the truth. A manual reading is a graph, not geometry: parts have no poses.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field

from ..assembly.graph import validate
from ..assembly.schema import (
    AssemblyPlan,
    Hardware,
    HardwareUse,
    Joint,
    JointKind,
    Orientation,
    Part,
    Step,
)

MANUAL_PROMPT = """You are reading the assembly manual of one piece of flat-pack furniture.
The images are its pages, in order.

Report what the manual says, exactly as printed:

1. title: the product name on the cover. category: what it is (bookcase, nightstand, side table, desk, ...).
2. parts: every part on the parts page, by the letter printed with it. name: what the part is (side panel,
   top, bottom, shelf, back panel, plinth, leg, rail, drawer side, ...). count: how many pieces carry that
   letter. size_mm: the three numbers printed under it, if any.
3. hardware: every item on the hardware page: the article number, its kind, a short name, and the total
   count printed.
4. steps: one entry per numbered step page, in order:
   - index: the step number; title: the caption under the number.
   - adds: the letters of the parts this step brings in (white parts with letter callouts and arrows). One
     entry per piece: if two pieces with the same letter go on, list the letter twice.
   - hardware: each item in the step's hardware box, as article number and count.
   - orientation: how the assembly lies in the drawing: upright, on_back, on_front (face down), on_side,
     upside_down.
   - tools: the tools pictured (screwdriver, hammer, allen key).
   - connections: every joint the step makes, as the two letters of the parts joined and the joint kind. A
     part that goes on between or onto two parts makes two connections. Joints between a new part and parts
     already in place count. A subassembly that goes on as a whole (a drawer, a frame) connects through its
     parts that touch the rest.
     Joint kinds: cam_lock (dowels with cam bolts and cam locks), dowel (dowels only), screw, bolt, nail,
     slot (slides into a groove or onto runners, no hardware), pin (rests on shelf pins), glue.

Parts drawn in gray are already in place; white parts with letters and arrows are the ones going on now.
Read counts and article numbers from the page; do not guess what is not shown."""


class PartReading(BaseModel):
    label: str = Field(description="The letter printed with the part")
    name: str = Field(description="What the part is, e.g. side panel")
    count: int = Field(ge=1, le=24)
    size_mm: list[float] = Field(description="The size printed under the part (three numbers), or empty")


class HardwareReading(BaseModel):
    id: str = Field(description="The article number")
    kind: Literal["dowel", "cam_bolt", "cam_lock", "nail", "shelf_pin", "screw", "bolt", "other"]
    name: str
    count: int = Field(ge=1, le=400)


class HardwareCount(BaseModel):
    id: str
    count: int = Field(ge=1, le=400)


class ConnectionReading(BaseModel):
    a: str = Field(description="Letter of one part")
    b: str = Field(description="Letter of the other part")
    kind: JointKind


class StepReading(BaseModel):
    index: int
    title: str
    adds: list[str]
    hardware: list[HardwareCount]
    orientation: Orientation
    tools: list[str]
    connections: list[ConnectionReading]


class ManualReading(BaseModel):
    title: str
    category: str
    parts: list[PartReading]
    hardware: list[HardwareReading]
    steps: list[StepReading]


# Which joint kinds a piece of hardware belongs to, to share a step's hardware among its joints.
HARDWARE_JOINTS: dict[str, tuple[str, ...]] = {
    "dowel": ("dowel", "cam_lock"),
    "cam_bolt": ("cam_lock",),
    "cam_lock": ("cam_lock",),
    "nail": ("nail",),
    "screw": ("screw",),
    "bolt": ("bolt",),
    "shelf_pin": ("pin",),
}

PART_KINDS = {
    "side": "side", "top": "top", "bottom": "bottom", "shelf": "shelf", "back": "back", "door": "door",
    "drawer front": "drawer_front", "drawer side": "drawer_side", "drawer back": "drawer_back",
    "drawer bottom": "drawer_bottom", "leg": "leg", "rail": "rail", "apron": "rail", "plinth": "plinth",
}  # fmt: skip


def part_kind(name: str) -> str:
    name = name.lower()
    for words, kind in sorted(PART_KINDS.items(), key=lambda item: -len(item[0])):
        if words in name:
            return kind
    return "panel"


def instance_ids(parts: Sequence[PartReading]) -> dict[str, list[str]]:
    """Part ids per letter, the way the library names them: 'A' alone, or 'A1', 'A2' for repeats."""
    return {p.label: [p.label] if p.count == 1 else [f"{p.label}{i}" for i in range(1, p.count + 1)] for p in parts}


@dataclass
class _Pick:
    """What instance resolution looks at within one step."""

    available: dict[str, list[str]]
    brought: list[str]
    pairs: set[frozenset[str]]
    in_step: Counter[str]
    joint_count: Counter[str]


def _pick(label: str, other: str | None, state: _Pick) -> str | None:
    """The instance of `label` a connection most likely means (see `to_plan`)."""
    candidates = state.available.get(label) or []
    if not candidates:
        return None

    def rank(candidate: str) -> tuple[bool, bool, int, int, str]:
        paired = other is not None and frozenset((candidate, other)) in state.pairs
        return (
            candidate not in state.brought,
            paired,
            state.in_step[candidate],
            state.joint_count[candidate],
            candidate,
        )

    return min(candidates, key=rank)


def to_plan(reading: ManualReading, plan_id: str = "manual") -> tuple[AssemblyPlan, list[str]]:
    """Turn a manual reading into a plan, and list what did not add up.

    Repeated letters become instances; a step's `adds` takes the next unused instance of each
    letter, and a part a connection needs before any step has brought it in comes in with that
    step. Each connection is resolved to instances: a piece brought in by this step first,
    then one not yet joined to the other side, then the one with the fewest joints so far
    (scores compare by letter, so this only has to give a sensible graph).
    """
    notes: list[str] = []
    ids = instance_ids(reading.parts)
    printed = {p.label: p for p in reading.parts}
    parts = [
        Part(
            id=pid,
            label=label,
            name=printed[label].name,
            kind=part_kind(printed[label].name),  # type: ignore[arg-type]
            size_mm=printed[label].size_mm if len(printed[label].size_mm) == 3 else [0.0, 0.0, 0.0],
        )
        for label, pids in ids.items()
        for pid in pids
    ]
    hardware_kind = {h.id: h.kind for h in reading.hardware}
    hardware = [
        Hardware(id=h.id, name=h.name, kind=h.kind if h.kind != "other" else "bracket", count=h.count)  # type: ignore[arg-type]
        for h in reading.hardware
    ]

    used: dict[str, int] = defaultdict(int)  # instances of each letter brought in so far
    joint_count: Counter[str] = Counter()
    pairs: set[frozenset[str]] = set()
    joints: list[Joint] = []
    steps: list[Step] = []
    for k, step in enumerate(reading.steps, start=1):
        brought: list[str] = []
        for label in step.adds:
            pool = ids.get(label)
            if pool is None:
                notes.append(f"step {k} adds part {label}, which is not on the parts page")
                continue
            if used[label] >= len(pool):
                notes.append(f"step {k} adds more {label} than the parts page lists")
                continue
            brought.append(pool[used[label]])
            used[label] += 1
        available = {label: pool[: used[label]] for label, pool in ids.items()}
        step_joints: list[str] = []
        in_step: Counter[str] = Counter()

        state = _Pick(available, brought, pairs, in_step, joint_count)
        for connection in step.connections:
            # The part a step builds on is often drawn without an arrow (the first side panel lies
            # there), so a part joined before any step brings it in comes in with this step.
            for label in (connection.a, connection.b):
                pool = ids.get(label)
                if pool and not available.get(label) and used[label] < len(pool):
                    brought.append(pool[used[label]])
                    used[label] += 1
                    available[label] = pool[: used[label]]
            a = _pick(connection.a, None, state)
            b = _pick(connection.b, a, state)
            if a is not None and b == a:
                b = None
            if a is None or b is None:
                notes.append(f"step {k} joins {connection.a}-{connection.b}, but one of them is not in the build yet")
                continue
            # Re-pick a now that b is known, so a repeated letter does not pair twice with b.
            a = _pick(connection.a, b, state) or a
            if a == b:
                continue
            joint_id = f"J{len(joints) + 1}"
            joints.append(Joint(id=joint_id, parts=[a, b], kind=connection.kind, hardware=[], fasteners=[]))
            step_joints.append(joint_id)
            pairs.add(frozenset((a, b)))
            for pid in (a, b):
                joint_count[pid] += 1
                in_step[pid] += 1

        # Share the step's hardware among its joints of the matching kinds.
        for use in step.hardware:
            kinds = HARDWARE_JOINTS.get(hardware_kind.get(use.id, ""), ())
            takers = [j for j in joints if j.id in step_joints and j.kind in kinds]
            if not takers:
                continue
            share, extra = divmod(use.count, len(takers))
            for i, joint in enumerate(takers):
                count = share + (1 if i < extra else 0)
                if count:
                    joint.hardware.append(HardwareUse(id=use.id, count=count))

        steps.append(
            Step(
                index=k,
                title=step.title,
                instruction=step.title,
                parts=brought,
                joints=step_joints,
                hardware=[HardwareUse(id=h.id, count=h.count) for h in step.hardware],
                orientation=step.orientation,
                tools=step.tools,
            )
        )
    plan = AssemblyPlan(
        id=plan_id,
        title=reading.title,
        category=reading.category,
        source="manual",
        parts=parts,
        hardware=hardware,
        joints=joints,
        steps=steps,
    )
    notes += validate(plan)
    return plan, notes


class ManualResult(BaseModel):
    """A manual read by the model: the raw reading, the plan made from it, and the bookkeeping."""

    reading: ManualReading
    plan: AssemblyPlan
    notes: list[str]
    model: str
    pages: int
    seconds: float
    usage: dict[str, int | float | None | dict] = {}


def read_manual(pages: Sequence[Image.Image], plan_id: str = "manual", effort: str = "high") -> ManualResult:
    """Read a manual's pages (in order) into a plan with Muse Spark."""
    from ..models import meta_api

    if not pages:
        raise ValueError("a manual needs at least one page")
    if len(pages) > 50:
        raise ValueError("Muse Spark takes up to 50 images per request; split the manual")
    started = time.monotonic()
    reading, usage = meta_api.ask_structured(MANUAL_PROMPT, pages, ManualReading, reasoning_effort=effort)
    plan, notes = to_plan(reading, plan_id)
    return ManualResult(
        reading=reading,
        plan=plan,
        notes=notes,
        model=meta_api.MUSE_SPARK_MODEL,
        pages=len(pages),
        seconds=round(time.monotonic() - started, 2),
        usage=usage,
    )
