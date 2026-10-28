"""Part tracking without the API: prompt geometry, stored outlines, contacts and their evidence."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from darius.assembly.schema import Observation
from darius.eval import benchmark
from darius.furniture.catalog import CATALOG
from darius.perception.parts import (
    CONTACT_SIZE,
    PartBox,
    PartTrack,
    PartTracks,
    TrackFrame,
    contact_observation,
    contact_series,
    fill_gaps,
    from_crop,
    fuse,
    grid_point,
    grid_to_pixels,
    held_contacts,
    joint_pairs,
    polygon,
    rasterize,
    track_masks,
    vote,
    zoom_region,
)
from darius.perception.video import VideoAnalysis


def square(x: int, y: int, side: int, size: tuple[int, int] = CONTACT_SIZE) -> np.ndarray:
    mask = np.zeros((size[1], size[0]), bool)
    mask[y : y + side, x : x + side] = True
    return mask


def test_grid_boxes_and_points_become_pixels():
    assert grid_to_pixels([100, 200, 300, 400], (1000, 500)) == (100, 100, 300, 200)
    assert grid_to_pixels([300, 400, 100, 200], (1000, 500)) == (100, 100, 300, 200)  # corners swapped
    assert grid_to_pixels([0, 0, 1001, 1001], (960, 540)) == (0, 0, 960, 540)  # the model sometimes says 1001
    assert grid_to_pixels([500, 500, 501, 501], (960, 540)) is None  # too small to prompt with
    assert grid_point([500, 1001], (960, 540)) == (480, 539)
    assert grid_point([500], (960, 540)) is None


def test_the_close_look_maps_back_onto_the_frame():
    region = zoom_region([[400, 400, 500, 500], [450, 450, 600, 550]], (1000, 1000))
    assert region == (360, 368, 640, 582)  # the union, with 20% of its size or 32 px around it
    assert zoom_region([[0, 0, 1000, 1000]], (1000, 1000)) is None  # nothing to zoom into
    whole = from_crop(PartBox(label="A", box_2d=[0, 0, 1000, 1000], point_2d=[500, 500]), region, (1000, 1000))
    assert whole.box_2d == [360, 368, 640, 582]
    assert whole.point_2d == [500, 475]


def test_stored_outlines_rasterize_back_to_the_mask():
    mask = np.zeros((540, 960), bool)
    mask[100:220, 300:540] = True
    frame = TrackFrame(frame=0, box=[300, 100, 540, 220], area=int(mask.sum()), confidence=0.9, polygon=polygon(mask))
    back = rasterize(frame, (960, 540))
    exact = mask.reshape(180, 3, 320, 3).mean(axis=(1, 3)) >= 0.5
    assert back.shape == (180, 320)
    assert benchmark.iou(back, exact) > 0.97


def test_contacts_count_once_they_have_held():
    fps = 5.0
    frames = []
    for t in range(20):
        apart = t < 8 or t == 12  # touching from frame 8, with one dropped frame at 12
        b = square(60 if apart else 40, 10, 10)
        frames.append({1: ("A", square(30, 10, 10)), 2: ("B", b), 3: ("C", square(200, 100, 10))})
    series = contact_series(frames)
    assert series[1, 2].tolist() == [8 <= t and t != 12 for t in range(20)]
    assert not series[1, 3].any() and not series[2, 3].any()
    # Five unbroken frames are needed: 13..17 is the first run that long (8..11 is cut short).
    assert held_contacts(series, fps) == {(1, 2): 17 / fps}
    assert held_contacts(series, fps, hold_s=0.6) == {(1, 2): 10 / fps}


def test_contacts_are_credited_to_steps_in_the_manuals_order():
    plan = CATALOG["lund-side-table"]()
    assert joint_pairs(plan) == [
        Counter({("B", "C"): 2}),
        Counter({("B", "C"): 2}),
        Counter({("B", "C"): 4}),
        Counter({("A", "C"): 4}),
    ]
    # Three leg-rail contacts: the first step's two, and one of the second step's.
    reading = contact_observation(plan, Counter({("B", "C"): 3}), 12.0)
    assert reading.step_done == {1: 0.8, 2: 0.5, 3: 0.2, 4: 0.2}
    # The top's contacts speak for the last step whatever the frames show.
    reading = contact_observation(plan, Counter({("A", "C"): 4}), 40.0)
    assert reading.step_done[4] == 0.8 and reading.step_done[1] == 0.2


def test_fusion_is_a_product_of_experts():
    a = Observation(time_s=1.0, step_done={1: 0.8, 2: 0.3}, source="muse-spark")
    b = Observation(time_s=1.0, step_done={1: 0.5, 2: 0.8, 3: 0.2}, source="sam-3.1")
    fused = fuse(a, b)
    assert fused.step_done[1] == pytest.approx(0.8, abs=1e-4)  # an undecided second reading changes nothing
    assert fused.step_done[2] > 0.3  # disagreement pulls toward the second, at half weight
    assert fused.step_done[3] == 0.2  # a step only the second reading speaks to
    assert fused.source == "muse-spark+sam-3.1"


def perfect_tracks(plan_id: str) -> PartTracks:
    """Tracks that follow every part exactly: the session's label maps as outlines."""
    labels, part_ids, fps = benchmark.load_ids(plan_id)
    plan = CATALOG[plan_id]()
    letter = {p.id: p.label for p in plan.parts}
    tracks = []
    for i, pid in enumerate(part_ids):
        frames = []
        for f, frame in enumerate(labels):
            mask = frame == i + 1
            if mask.any():
                ys, xs = np.nonzero(mask)
                box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                frames.append(TrackFrame(frame=f, box=box, area=int(mask.sum()), confidence=1.0, polygon=polygon(mask)))
        tracks.append(PartTrack(object=i + 1, label=letter[pid], frames=frames))
    return PartTracks(
        plan=plan_id,
        model="exact",
        grounding_model="exact",
        fps=fps,
        size=list(CONTACT_SIZE),
        anchor_frame=5,
        frames=len(labels),
        prompts=[],
        tracks=tracks,
        seconds=0.0,
    )


@pytest.mark.parametrize("plan_id", ["lund-side-table", "ostra-desk"])
def test_perfect_tracks_score_like_the_exact_masks(plan_id):
    plan = CATALOG[plan_id]()
    folder = benchmark.LIBRARY / plan_id / "session"
    truth = json.loads((folder / "truth.json").read_text())
    analysis = VideoAnalysis.model_validate(
        {k: v for k, v in json.loads((folder / "analysis.json").read_text()).items() if k != "score"}
    )
    tracks = perfect_tracks(plan_id)
    assert len(track_masks(tracks)) == tracks.frames
    score, match, _ = benchmark.score_tracking(tracks, plan, truth, analysis)
    assert len(match) == len(plan.parts) and score.letters_right == len(plan.parts)
    assert score.mean_iou > 0.85 and score.frames_tracked > 0.9  # outlines keep the largest region only
    assert score.contacts_recall > 0.9
    assert score.frames_right_contacts == pytest.approx(score.frames_right_exact_contacts, abs=0.08)


def test_a_second_pass_only_fills_the_frames_the_first_lost():
    def frame(f: int, area: int = 100) -> TrackFrame:
        return TrackFrame(frame=f, box=[0, 0, 10, 10], area=area, confidence=0.9, polygon=[0, 0, 10, 0, 10, 10])

    first = [
        PartTrack(object=1, label="A", frames=[frame(0), frame(1), frame(2, area=3)]),
        PartTrack(object=2, label="B", frames=[frame(0)]),
    ]
    second = [PartTrack(object=1, label="A", frames=[frame(1, area=50), frame(2), frame(5)])]
    merged = fill_gaps(first, second)
    assert [(f.frame, f.area) for f in merged[0].frames] == [(0, 100), (1, 100), (2, 100), (5, 100)]
    assert merged[1] == first[1]  # not tracked again: unchanged


def test_each_piece_takes_the_letter_most_looks_give_it():
    def piece(label: str, x: int) -> PartBox:
        return PartBox(label=label, box_2d=[x, 100, x + 100, 200], point_2d=[x + 50, 150])

    looks = [
        [piece("B", 0), piece("C", 300), piece("A", 600)],
        [piece("C", 2), piece("C", 305)],  # misses a piece
        [piece("C", 0), piece("B", 300), piece("A", 900)],  # the last piece somewhere else
    ]
    voted = vote(looks)
    assert [p.label for p in voted] == ["C", "C", "A"]
    assert [p.box_2d for p in voted] == [p.box_2d for p in looks[0]]  # the first look's places
