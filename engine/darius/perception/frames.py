"""Verify build progress on camera frames: which steps of the plan are done in each one.

Muse Spark sees a few consecutive frames at a time, with the plan's parts and a description of
the build after every step, and reports for each frame: which steps are visibly complete (with a
confidence), which part letters are in the assembly and which lie loose, and how the assembly
is oriented. One frame is weak evidence (a panel hides a joint, the camera looks away); the
tracker in `assembly.tracking` filters the readings over the plan's order.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field

from ..assembly.graph import subassemblies
from ..assembly.schema import AssemblyPlan, Observation, Orientation

ORIENTATION_TEXT = {
    "upright": "standing upright",
    "on_back": "lying on its back",
    "on_front": "lying face down",
    "on_side": "lying on its side",
    "upside_down": "upside down",
}


class StepCheck(BaseModel):
    index: int
    done: bool
    confidence: float = Field(ge=0, le=1, description="How sure the frame makes you, 0.5 = cannot tell")


class FrameReading(BaseModel):
    frame: int = Field(description="Which image of this request, from 1")
    orientation: Orientation | Literal["none"] = Field(description="How the main assembly lies; none before any joint")
    attached: list[str] = Field(description="Letters of the parts joined into an assembly (one per piece)")
    loose: list[str] = Field(description="Letters of the parts still lying loose (one per piece)")
    steps: list[StepCheck]


class BatchReading(BaseModel):
    frames: list[FrameReading]


def _counted(labels: Counter[str]) -> list[str]:
    return [f"{label} x{n}" if n > 1 else label for label, n in sorted(labels.items())]


def describe_plan(plan: AssemblyPlan) -> str:
    """The plan as the frame verifier needs it: the parts, and what the build looks like after each step."""
    labels = Counter(p.label for p in plan.parts)
    names = {p.label: p.name for p in plan.parts}
    lines = ["Parts:"]
    lines += [f"- {label}: {names[label]} x{count}" for label, count in sorted(labels.items())]
    lines.append("")
    lines.append("Steps, and what the build looks like once each is done:")
    by_id = {p.id: p for p in plan.parts}
    for step in plan.steps:
        groups = [g for g in subassemblies(plan, step.index) if len(g) > 1]
        joined = "; ".join(" + ".join(_counted(Counter(by_id[p].label for p in g))) for g in groups)
        lies = ORIENTATION_TEXT[step.orientation]
        lines.append(f"{step.index}. {step.title}. {step.instruction}")
        lines.append(f"   After step {step.index}: joined = {joined}; the assembly is {lies}.")
    return "\n".join(lines)


FRAMES_PROMPT = """These {count} images are consecutive frames, {every:.1f} s apart, of someone assembling
the furniture below by its manual. The furniture is built step by step in this order; a step is done
when all of its parts are in place and joined as described (fastening is the last thing; a part only
held in position is not done).

{plan}

For each frame, report:
- steps: for every step 1..{steps}, whether it is done in this frame and how sure you are (0.5 = cannot tell
  from this frame; use the build's order: if a later step is clearly done, earlier ones are too).
- attached: the letters of the parts joined into an assembly (one entry per piece); loose: the letters of
  parts still lying apart.
- orientation: how the main assembly lies (none if nothing is joined yet)."""


@dataclass(frozen=True)
class Frame:
    time_s: float
    image: Image.Image


def to_observation(reading: FrameReading, time_s: float, steps: int) -> Observation:
    """A frame reading as tracker evidence: P(step done) per step."""
    step_done: dict[int, float] = {}
    for check in reading.steps:
        if 1 <= check.index <= steps:
            p = check.confidence if check.done else 1 - check.confidence
            step_done[check.index] = round(min(max(p, 0.02), 0.98), 4)
    seen = reading.attached + reading.loose
    return Observation(time_s=time_s, step_done=step_done, visible_parts=seen, source="muse-spark")


def verify_frames(
    plan: AssemblyPlan,
    frames: Sequence[Frame],
    *,
    batch: int = 4,
    workers: int = 6,
    effort: str = "low",
    on_batch: Callable[[int, int], None] | None = None,
) -> list[FrameReading]:
    """Read every frame, `batch` consecutive frames per request, requests in parallel."""
    from ..models import meta_api

    if not frames:
        return []
    every = (frames[-1].time_s - frames[0].time_s) / max(1, len(frames) - 1)
    description = describe_plan(plan)
    chunks = [list(frames[i : i + batch]) for i in range(0, len(frames), batch)]
    done = 0

    def read(chunk: list[Frame]) -> list[FrameReading]:
        nonlocal done
        prompt = FRAMES_PROMPT.format(count=len(chunk), every=every, plan=description, steps=len(plan.steps))
        result, _ = meta_api.ask_structured(prompt, [f.image for f in chunk], BatchReading, reasoning_effort=effort)
        by_frame = {r.frame: r for r in result.frames}
        # A missing frame reads as "cannot tell": no evidence either way.
        blank = [
            FrameReading(frame=i + 1, orientation="none", attached=[], loose=[], steps=[]) for i in range(len(chunk))
        ]
        out = [by_frame.get(i + 1) or blank[i] for i in range(len(chunk))]
        done += 1
        if on_batch:
            on_batch(done, len(chunks))
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(read, chunks))
    return [reading for chunk in results for reading in chunk]
