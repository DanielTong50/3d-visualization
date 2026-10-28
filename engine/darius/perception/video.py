"""Watch a video of a build and recover the assembly graph as it was built.

Frames are sampled at a fixed interval, read by the frame verifier, and decoded over the whole
sequence by the tracker (Viterbi for the path, forward-backward for its confidence). Given the
plan, the decoded path is the graph's history: the joints of step k were made when step k
finished. The video tells when each joint was made, in what order, and how sure the model was.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import cv2
from PIL import Image
from pydantic import BaseModel

from ..assembly.schema import AssemblyPlan
from ..assembly.tracking import decode
from ..core.errors import InputError
from .frames import Frame, FrameReading, to_observation, verify_frames


class FrameRecord(BaseModel):
    """One sampled frame and what the model read on it."""

    time_s: float
    step_done: dict[int, float]
    attached: list[str]
    loose: list[str]
    orientation: str


class JointEvent(BaseModel):
    """An edge of the assembly graph and when the video shows it made."""

    joint: str
    parts: list[str]
    labels: list[str]
    kind: str
    step: int
    made_at_s: float | None


class VideoAnalysis(BaseModel):
    plan: str
    model: str
    every_s: float
    duration_s: float
    frames: list[FrameRecord]
    path: list[int]
    confidence: list[float]
    posteriors: list[list[float]]
    finished_at: dict[int, float]
    joints: list[JointEvent]
    seconds: float


def sample_frames(video: Path, every_s: float = 1.0, max_side: int = 960) -> tuple[list[Frame], float]:
    """Frames at every_s, 2·every_s, … (decoded in order, which is exact and fast), and the duration."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise InputError("That file doesn't look like a video we can read.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = count / fps
    wanted = {round(k * every_s * fps): k * every_s for k in range(1, int(duration / every_s) + 1)}
    frames: list[Frame] = []
    for index in range(count):
        if not capture.grab():
            break
        if index not in wanted:
            continue
        ok, bgr = capture.retrieve()
        if not ok:
            continue
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        image.thumbnail((max_side, max_side))
        frames.append(Frame(time_s=round(wanted[index], 3), image=image))
    capture.release()
    if len(frames) < 2:
        raise InputError("The video is too short to follow a build.")
    return frames, duration


def analyze_readings(
    plan: AssemblyPlan, frames: list[Frame], readings: list[FrameReading], model: str, every_s: float, duration: float
) -> VideoAnalysis:
    """Decode frame readings into the build's timeline and the graph's history."""
    steps = len(plan.steps)
    observations = [to_observation(r, f.time_s, steps) for r, f in zip(readings, frames, strict=True)]
    timeline = decode(observations, steps)
    labels = {p.id: p.label for p in plan.parts}
    joints = [
        JointEvent(
            joint=joint_id,
            parts=list(plan.joint(joint_id).parts),
            labels=[labels[p] for p in plan.joint(joint_id).parts],
            kind=plan.joint(joint_id).kind,
            step=step.index,
            made_at_s=timeline.finished_at.get(step.index),
        )
        for step in plan.steps
        for joint_id in step.joints
    ]
    return VideoAnalysis(
        plan=plan.id,
        model=model,
        every_s=every_s,
        duration_s=round(duration, 3),
        frames=[
            FrameRecord(
                time_s=o.time_s, step_done=o.step_done, attached=r.attached, loose=r.loose, orientation=r.orientation
            )
            for o, r in zip(observations, readings, strict=True)
        ],
        path=timeline.path,
        confidence=timeline.confidence,
        posteriors=timeline.posteriors,
        finished_at=timeline.finished_at,
        joints=joints,
        seconds=0.0,
    )


def analyze(
    video: Path,
    plan: AssemblyPlan,
    *,
    every_s: float = 1.0,
    batch: int = 4,
    workers: int = 6,
    on_batch: Callable[[int, int], None] | None = None,
) -> VideoAnalysis:
    """Sample, read and decode a build video against its plan."""
    from ..models.meta_api import MUSE_SPARK_MODEL

    started = time.monotonic()
    frames, duration = sample_frames(video, every_s)
    readings = verify_frames(plan, frames, batch=batch, workers=workers, on_batch=on_batch)
    analysis = analyze_readings(plan, frames, readings, MUSE_SPARK_MODEL, every_s, duration)
    analysis.seconds = round(time.monotonic() - started, 2)
    return analysis
