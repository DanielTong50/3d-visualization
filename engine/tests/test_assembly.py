import numpy as np
import pytest

from darius.assembly import graph, guidance, tracking
from darius.assembly.schema import AssemblyPlan, Observation
from darius.furniture import catalog
from darius.furniture.builder import Box, contact, lying_footprint, lying_rotation

PLANS = {name: build() for name, build in catalog.CATALOG.items()}


@pytest.mark.parametrize("plan_id", sorted(PLANS))
def test_every_catalog_plan_is_valid_and_ends_in_one_piece(plan_id):
    plan = PLANS[plan_id]
    assert graph.validate(plan) == []
    assert len(graph.subassemblies(plan, len(plan.steps))) == 1
    assert graph.subassemblies(plan, 0) == [[p.id] for p in sorted(plan.parts, key=lambda p: p.id)]
    # The plan survives JSON.
    assert AssemblyPlan.model_validate_json(plan.model_dump_json()) == plan


def test_the_bookcase_is_built_the_way_the_manual_says():
    plan = PLANS["halden-bookcase"]
    assert [s.orientation for s in plan.steps] == ["on_side", "on_side", "on_side", "on_front", "upright"]
    counts = {h.name: h.count for h in plan.hardware}
    assert counts["cam lock 15 mm"] == counts["cam bolt 34 mm"] == 12  # six cam-lock joints, two each
    assert counts["shelf pin 5 mm"] == 8
    # After step 3 the carcass is one piece, with the back and both shelves still loose.
    assert graph.subassemblies(plan, 3)[0] == sorted(["A1", "A2", "B", "C", "D", "E"])
    assert len(graph.subassemblies(plan, 3)) == 4


def test_fasteners_sit_on_the_shared_face():
    plan = PLANS["halden-bookcase"]
    joint = plan.joint("J1")  # top B to left side A1
    side = plan.part("A1")
    face_x = side.assembled.position[0] + side.size_mm[0] / 2
    assert len(joint.fasteners) == 4
    assert all(f.position[0] == pytest.approx(face_x) for f in joint.fasteners)
    assert {f.hardware for f in joint.fasteners} == {"101350", "118331"}


def test_contacts_and_floor_layout():
    a = Box(np.array([0.0, 0, 0]), np.array([10.0, 10, 10]))
    assert contact(a, Box(np.array([10.0, 0, 0]), np.array([10.0, 10, 10]))).axis == 0
    assert contact(a, Box(np.array([20.0, 0, 0]), np.array([10.0, 10, 10]))) is None
    assert lying_footprint(np.array([18.0, 1060, 280])) == (1060, 280)  # a side panel lies on its face
    assert lying_rotation(np.array([800.0, 18, 280])) == [0.0, 0.0, 0.0, 1.0]
    for plan in PLANS.values():
        assert all(p.layout.position[1] == pytest.approx(min(p.size_mm) / 2) for p in plan.parts)


def test_validation_catches_broken_plans():
    plan = PLANS["lund-side-table"]
    early = plan.model_copy(deep=True)
    early.steps[0].parts.remove("B2")  # rail C1 is bolted to a leg that isn't in hand yet
    assert any("before part B2" in p for p in graph.validate(early))
    twice = plan.model_copy(deep=True)
    twice.steps[1].joints.append(twice.steps[0].joints[0])
    assert any("made by 2 steps" in p for p in graph.validate(twice))


def test_comparison_is_by_label_and_perfect_on_itself():
    plan = PLANS["varde-nightstand"]
    score = graph.compare(plan, plan)
    assert score.edges_f1 == score.parts_f1 == score.steps_exact == score.hardware_accuracy == 1.0
    fewer = plan.model_copy(deep=True)
    fewer.joints = fewer.joints[:-2]
    fewer.steps[-1].joints = []
    assert graph.compare(fewer, plan).edges_recall < 1.0


def test_guidance_for_each_step_and_when_done():
    plan = PLANS["halden-bookcase"]
    first = guidance.guide(plan, 0)
    assert first.step == 1 and {c.label for c in first.parts} == {"A", "B", "C"}
    assert first.reorient is None
    nailing = guidance.guide(plan, 3)
    assert nailing.reorient == "Turn the assembly so it is lying on its front."
    assert any("nail" in name for name in nailing.hardware_names.values())
    assert guidance.guide(plan, 5).done


def test_transitions_are_stochastic_and_mostly_forward():
    t = tracking.transition_matrix(5)
    assert np.allclose(t.sum(axis=1), 1)
    assert t[2, 3] > t[2, 1] and t[0, 0] > 0.8


def test_the_tracker_follows_evidence_and_ignores_a_single_bad_frame():
    tracker = tracking.Tracker(steps=4)
    for _ in range(3):
        belief = tracker.update(Observation(step_done={1: 0.95, 2: 0.9, 3: 0.1, 4: 0.05}))
    assert belief.completed == 2 and belief.confidence > 0.8
    belief = tracker.update(Observation(step_done={1: 0.9, 2: 0.9, 3: 0.95, 4: 0.95}))  # one wild frame
    assert belief.completed == 2
    assert tracker.update(Observation()).completed == 2  # no evidence: the belief only drifts slowly


def test_decoding_a_noisy_video_recovers_when_each_step_finished():
    rng = np.random.default_rng(0)
    truth = [0] * 5 + [1] * 6 + [2] * 6 + [3] * 8  # steps finished at frames 5, 11 and 17
    observations = []
    for t, k in enumerate(truth):
        p = {i: float(np.clip((0.88 if i <= k else 0.12) + rng.normal(0, 0.12), 0.01, 0.99)) for i in range(1, 4)}
        if t == 13:
            p = {1: 0.9, 2: 0.2, 3: 0.95}  # an occluded, contradictory frame
        observations.append(Observation(time_s=t * 0.5, step_done=p))
    timeline = tracking.decode(observations, steps=3)
    assert timeline.path == truth
    assert timeline.finished_at == {1: 2.5, 2: 5.5, 3: 8.5}
    assert np.median(timeline.confidence) > 0.9  # confidence dips only around the moments a step changes
