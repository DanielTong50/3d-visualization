"""Guidance: from the plan and the tracked progress, what the person should do next.

The output is display-ready and geometry-aware: which parts to pick up (with their
letters and where they lie), which hardware and how many, which way to turn the
assembly first, where each fastener goes, and what the finished step must look
like. The AR view draws it; the voice and text views read it.
"""

from __future__ import annotations

from pydantic import Field

from .graph import subassemblies
from .schema import Action, AssemblyPlan, HardwareUse, Strict

ORIENTATION_TEXT = {
    "upright": "standing upright",
    "on_back": "lying on its back",
    "on_front": "lying on its front",
    "on_side": "lying on its side",
    "upside_down": "upside down",
}


class PartCue(Strict):
    id: str
    label: str
    name: str
    target: list[float] | None = Field(default=None, description="Where it goes (mm, furniture frame)")


class Guidance(Strict):
    completed: int
    total: int
    done: bool
    step: int | None = None
    title: str = ""
    instruction: str = ""
    reorient: str | None = Field(default=None, description="Turn the assembly first, e.g. 'lay it on its back'")
    parts: list[PartCue] = []
    hardware: list[HardwareUse] = []
    hardware_names: dict[str, str] = {}
    actions: list[Action] = []
    joints: list[str] = []
    checks: list[str] = Field(default_factory=list, description="What the finished step must look like")
    subassemblies: int = Field(default=1, description="Separate pieces on the floor after this step")


def guide(plan: AssemblyPlan, completed: int) -> Guidance:
    total = len(plan.steps)
    completed = max(0, min(completed, total))
    if completed == total:
        return Guidance(
            completed=completed, total=total, done=True, title="Done", instruction=f"{plan.title} is assembled."
        )
    step = plan.steps[completed]
    previous = plan.steps[completed - 1].orientation if completed else "upright"
    reorient = None
    if step.orientation != previous and completed:
        reorient = f"Turn the assembly so it is {ORIENTATION_TEXT[step.orientation]}."
    parts = []
    for part_id in step.parts:
        part = plan.part(part_id)
        parts.append(
            PartCue(
                id=part.id,
                label=part.label,
                name=part.name,
                target=part.assembled.position if part.assembled else None,
            )
        )
    checks = []
    names = {p.id: f"{p.name} ({p.label})" for p in plan.parts}
    for joint_id in step.joints:
        joint = plan.joint(joint_id)
        a, b = (names[p] for p in joint.parts)
        hardware = ", ".join(f"{u.count} x {plan_hardware_name(plan, u.id)}" for u in joint.hardware)
        checks.append(f"{a} is joined to {b}" + (f" with {hardware}" if hardware else ""))
    return Guidance(
        completed=completed,
        total=total,
        done=False,
        step=step.index,
        title=step.title,
        instruction=step.instruction,
        reorient=reorient,
        parts=parts,
        hardware=step.hardware,
        hardware_names={u.id: plan_hardware_name(plan, u.id) for u in step.hardware},
        actions=step.actions,
        joints=step.joints,
        checks=checks,
        subassemblies=len(subassemblies(plan, completed + 1)),
    )


def plan_hardware_name(plan: AssemblyPlan, hardware_id: str) -> str:
    for item in plan.hardware:
        if item.id == hardware_id:
            return item.name
    return hardware_id
