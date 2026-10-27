import numpy as np
import pytest

from scan2sim import materials, physics, reality_check
from scan2sim.physics import Estimate, ObjectProperties


def props(**overrides):
    base = dict(
        label="coffee mug",
        material="ceramic",
        typical_dimensions_cm=[12, 10, 8.5],
        typical_mass_kg=0.33,
        hollow=True,
        fill_fraction=0.25,
        friction=0.55,
        restitution=0.15,
        fragile=True,
        rigidity="rigid",
        confidence=0.9,
    )
    return ObjectProperties(**(base | overrides))


def test_scale_vote_matches_dimensions_by_rank():
    # Measured in VGGT units at exactly 1 unit = 0.5 m, listed in a different order.
    measured = np.array([0.17, 0.24, 0.2])
    assert physics.scale_vote(np.array([0.085, 0.12, 0.10]), measured) == pytest.approx(0.5)


def test_weighted_median_ignores_one_outlier():
    assert physics.weighted_median(np.array([0.5, 0.52, 3.0]), np.array([1, 1, 1])) == 0.52
    assert physics.weighted_median(np.array([0.5, 3.0]), np.array([0.2, 0.9])) == 3.0


def test_scene_scale_ignores_low_confidence_objects():
    good = [Estimate(i, "mug", props(), "ceramic", np.array([0.24, 0.2, 0.17]), 1.0) for i in range(2)]
    bad = Estimate(
        9,
        "blob",
        props(confidence=0.2, typical_dimensions_cm=[300, 300, 300]),
        "ceramic",
        np.array([0.2, 0.2, 0.2]),
        1.0,
    )
    scale, votes, source = physics.estimate_scale(good + [bad])
    assert scale == pytest.approx(0.5)
    assert len(votes) == 2 and source.startswith("known_objects:")


def test_mass_falls_back_to_typical_when_geometry_disagrees():
    assert physics.resolve_mass(0.30, 0.33) == (0.30, None)
    mass, note = physics.resolve_mass(3.0, 0.33)
    assert mass == 0.33 and "typical" in note


def test_sanitize_clamps_and_normalizes():
    fixed, material = physics.sanitize(props(material="Porcelain", friction=1.9, rigidity="bendy"))
    assert material == "ceramic"
    assert fixed.friction == 1.5 and fixed.rigidity == "rigid"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Ceramic", "ceramic"),
        ("stainless steel", "metal"),
        ("plush fabric", "fabric"),
        ("mystery goo", "unknown"),
        ("aluminium", "aluminum"),
    ],
)
def test_material_normalize(text, expected):
    assert materials.normalize(text) == expected


def test_every_material_is_physically_sane():
    for name in materials.MATERIAL_NAMES:
        m = materials.lookup(name)
        assert 10 < m["density_kg_m3"] < 20000
        assert 0 < m["friction"] < 1.5 and 0 <= m["restitution"] < 1


@pytest.mark.parametrize("mu,v0", [(0.3, 1.2), (0.55, 2.0), (0.9, 1.5)])
def test_slide_fit_recovers_friction_from_noisy_track(mu, v0):
    fps = 60
    rng = np.random.default_rng(0)
    push = np.arange(0, 0.15, 1 / fps)  # hand accelerating the object
    s_push = 0.5 * (v0 / 0.15) * push**2
    slide_t = np.arange(1 / fps, v0 / (mu * 9.81) + 0.5, 1 / fps)
    slide_t_stop = np.minimum(slide_t, v0 / (mu * 9.81))
    s_slide = s_push[-1] + v0 * slide_t_stop - 0.5 * mu * 9.81 * slide_t_stop**2
    t = np.concatenate([push, push[-1] + slide_t])
    s = np.concatenate([s_push, s_slide]) + rng.normal(0, 0.001, len(push) + len(slide_t))
    fit = reality_check.fit_slide(t, s)
    assert fit.friction == pytest.approx(mu, rel=0.12)
    assert fit.v0 == pytest.approx(v0, rel=0.1)
    # After correction, re-simulating the same push lands where the real object stopped.
    assert reality_check.predicted_distance(fit.v0, fit.friction) == pytest.approx(fit.distance_m, rel=0.08)
