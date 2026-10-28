"""The deterministic half of perception: model readings in, plans and timelines out (no API calls)."""

from __future__ import annotations

from collections import Counter

import pytest

from darius.assembly.graph import compare
from darius.assembly.schema import AssemblyPlan
from darius.eval.benchmark import score_session
from darius.furniture.catalog import CATALOG
from darius.perception.frames import Frame, FrameReading, StepCheck, describe_plan, to_observation
from darius.perception.manual import (
    ConnectionReading,
    HardwareCount,
    HardwareReading,
    ManualReading,
    PartReading,
    StepReading,
    to_plan,
)
from darius.perception.video import analyze_readings

HARDWARE_KINDS = {"dowel", "cam_bolt", "cam_lock", "nail", "shelf_pin", "screw", "bolt"}


def perfect_reading(truth: AssemblyPlan) -> ManualReading:
    """What a flawless reader would report for a library plan's manual."""
    labels = {p.id: p.label for p in truth.parts}
    first = {}
    for part in truth.parts:
        first.setdefault(part.label, part)
    counts = Counter(p.label for p in truth.parts)
    return ManualReading(
        title=truth.title,
        category=truth.category,
        parts=[
            PartReading(label=label, name=first[label].name, count=n, size_mm=list(first[label].size_mm))
            for label, n in sorted(counts.items())
        ],
        hardware=[
            HardwareReading(id=h.id, kind=h.kind if h.kind in HARDWARE_KINDS else "other", name=h.name, count=h.count)
            for h in truth.hardware
        ],
        steps=[
            StepReading(
                index=s.index,
                title=s.title,
                adds=[labels[p] for p in s.parts],
                hardware=[HardwareCount(id=h.id, count=h.count) for h in s.hardware],
                orientation=s.orientation,
                tools=s.tools,
                connections=[
                    ConnectionReading(
                        a=labels[truth.joint(j).parts[0]], b=labels[truth.joint(j).parts[1]], kind=truth.joint(j).kind
                    )
                    for j in s.joints
                ],
            )
            for s in truth.steps
        ],
    )


@pytest.mark.parametrize("plan_id", list(CATALOG))
def test_a_perfect_reading_gives_back_the_graph(plan_id: str) -> None:
    truth = CATALOG[plan_id]()
    plan, notes = to_plan(perfect_reading(truth), plan_id)
    score = compare(plan, truth)
    assert notes == []
    assert (score.parts_f1, score.edges_f1, score.steps_exact, score.hardware_accuracy) == (1.0, 1.0, 1.0, 1.0)


def test_the_part_a_step_builds_on_comes_in_without_being_listed() -> None:
    # Manuals draw the first side panel lying there, with no arrow: readers leave it out of `adds`.
    truth = CATALOG["halden-bookcase"]()
    reading = perfect_reading(truth)
    reading.steps[0].adds = [a for a in reading.steps[0].adds if a != "A"]
    plan, notes = to_plan(reading, truth.id)
    assert notes == []
    assert compare(plan, truth).edges_f1 == 1.0
    assert "A1" in plan.steps[0].parts


def test_repeated_letters_join_different_instances() -> None:
    truth = CATALOG["varde-nightstand"]()
    plan, _ = to_plan(perfect_reading(truth), truth.id)
    first = [plan.joint(j).parts for j in plan.steps[0].joints]
    # Bottom C between two side panels A: one joint to each panel, not two to the same one.
    assert sorted(p for pair in first for p in pair if p.startswith("A")) == ["A1", "A2"]


def test_unknown_parts_are_reported_not_invented() -> None:
    truth = CATALOG["ostra-desk"]()
    reading = perfect_reading(truth)
    reading.steps[0].adds.append("Z")
    _, notes = to_plan(reading, truth.id)
    assert any("Z" in note for note in notes)


def test_frame_readings_become_tracker_evidence() -> None:
    reading = FrameReading(
        frame=1,
        orientation="on_side",
        attached=["A", "B"],
        loose=["C"],
        steps=[StepCheck(index=1, done=True, confidence=0.9), StepCheck(index=2, done=False, confidence=0.8)],
    )
    observation = to_observation(reading, 3.0, steps=2)
    assert observation.step_done == {1: 0.9, 2: pytest.approx(0.2)}
    assert observation.visible_parts == ["A", "B", "C"]


def test_the_plan_description_names_every_step() -> None:
    plan = CATALOG["lund-side-table"]()
    text = describe_plan(plan)
    for step in plan.steps:
        assert f"After step {step.index}:" in text


def test_decoding_readings_dates_every_joint() -> None:
    plan = CATALOG["ostra-desk"]()
    steps = len(plan.steps)
    # Step k done from t = 10 k on, read with 0.9 confidence; one misread frame at t = 12.
    frames = [Frame(time_s=float(t), image=None) for t in range(1, 41)]  # type: ignore[arg-type]
    readings = []
    for frame in frames:
        done = min(steps, int(frame.time_s // 10))
        checks = [StepCheck(index=k, done=k <= done, confidence=0.9) for k in range(1, steps + 1)]
        if frame.time_s == 12:
            checks = [StepCheck(index=k, done=True, confidence=0.9) for k in range(1, steps + 1)]
        readings.append(FrameReading(frame=1, orientation="upright", attached=[], loose=[], steps=checks))
    analysis = analyze_readings(plan, frames, readings, "test", 1.0, 40.0)
    assert analysis.path[-1] == steps
    assert [round(analysis.finished_at[k]) for k in range(1, steps + 1)] == [10, 20, 30]
    assert all(j.made_at_s is not None for j in analysis.joints)

    truth = {
        "fps": 1,
        "completed": [min(steps, t // 10) for t in range(0, 42)],
        "steps": [{"index": k, "done_at": 10.0 * k} for k in range(1, steps + 1)],
    }
    score = score_session(analysis, plan, truth)
    assert score.frame_accuracy == 1.0
    assert score.raw_frame_accuracy < 1.0  # the misread frame fools a frame on its own
    assert score.steps_found == steps and score.step_error_s == 0.0
