import math

import numpy as np
import pytest

from darius.core import geometry
from darius.pipeline import collision, splats


def random_splats(n, rng, means=None):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return {
        "means": (rng.normal(0, 0.2, (n, 3)) if means is None else means).astype(np.float32),
        "quats": q.astype(np.float32),
        "log_scales": rng.uniform(-7, -4, (n, 3)).astype(np.float32),
        "opacity_logits": rng.uniform(-3, 3, n).astype(np.float32),
        "sh0": rng.uniform(-1, 1, (n, 3)).astype(np.float32),
        "shN": rng.uniform(-0.5, 0.5, (n, 3, 3)).astype(np.float32),
        "labels": np.zeros(n, np.int16),
    }


def test_spz_round_trip_through_the_reference_reader(tmp_path):
    spz = pytest.importorskip("spz")
    rng = np.random.default_rng(0)
    s = random_splats(400, rng)
    offset, scale = np.array([0.1, 0.0, -0.2]), 0.5
    splats.write_spz(tmp_path / "x.spz", s, offset, scale)
    options = spz.UnpackOptions()
    options.to_coord = spz.RUB
    back = spz.load_spz(str(tmp_path / "x.spz"), options)
    assert back.sh_degree == 1
    np.testing.assert_allclose(np.asarray(back.positions).reshape(-1, 3), s["means"] * scale - offset, atol=3e-4)
    np.testing.assert_allclose(np.asarray(back.scales).reshape(-1, 3), s["log_scales"] + math.log(scale), atol=0.04)
    q = np.asarray(back.rotations).reshape(-1, 4)  # x, y, z, w
    assert np.abs((q * s["quats"][:, [1, 2, 3, 0]]).sum(axis=1)).min() > 0.9999
    np.testing.assert_allclose(np.asarray(back.sh).reshape(-1, 3, 3), s["shN"], atol=0.04)


def test_assign_keeps_objects_whole_and_drops_ghosts():
    rng = np.random.default_rng(1)
    cases = [  # (position, label, sigma, opacity logit, expected owner)
        ([0.00, 0.05, 0.00], 1, 0.001, 3, 1),  # object splat inside its box: stays with the object
        ([0.50, 0.05, 0.00], 1, 0.001, 3, -1),  # object splat that drifted away: dropped, it would ghost
        ([0.01, 0.05, 0.00], 0, 0.001, 3, 1),  # background inside the box: it drew the object, so it joins it
        ([0.30, 0.00, 0.30], 0, 0.001, 3, 0),  # table splat far from the object: kept
        ([0.08, 0.05, 0.00], 0, 0.02, 3, -1),  # large background splat reaching into the object: dropped
        ([0.09, 0.05, 0.00], 0, 0.001, 3, 0),  # small background splat beside the object: kept
        ([0.00, -0.02, 0.00], 0, 0.001, 3, -1),  # background under the table inside the footprint: dropped
        ([0.01, 0.06, 0.01], 1, [0.003, 0.0002, 0.0002], 3, -1),  # needle-shaped object splat: dropped
        ([0.02, 0.06, 0.02], 1, [0.003, 0.003, 0.0001], 3, 1),  # flat disc on the object's surface: kept
        ([0.02, 0.06, 0.02], 1, 0.01, 3, -1),  # oversized object splat: dropped
        ([0.00, 0.04, 0.00], 1, 0.001, -6, -1),  # practically transparent: dropped
    ]
    s = random_splats(len(cases), rng, means=np.array([c[0] for c in cases]))
    s["labels"] = np.array([c[1] for c in cases], np.int16)
    s["opacity_logits"] = np.array([c[3] for c in cases], np.float32)
    s["log_scales"] = np.log(np.array([np.broadcast_to(c[2], 3) for c in cases])).astype(np.float32)
    box = (np.array([-0.05, 0, -0.05]), np.array([0.05, 0.1, 0.05]))
    owner = splats.assign(s, {1: box}, margin=0.005, max_object_sigma=0.005)
    assert owner.tolist() == [c[4] for c in cases]


def test_table_layer_separates_the_table_top_from_the_room():
    rng = np.random.default_rng(6)
    means = np.array(
        [
            [0.0, 0.0, 0.0],  # on the table
            [0.1, 0.05, 0.1],  # something left on the table
            [0.32, 0.0, 0.0],  # just past the edge: still the table layer
            [0.6, 0.0, 0.0],  # the floor beyond the table
            [0.0, 0.4, 0.0],  # high above: the room
            [0.0, -0.3, 0.0],  # under the table
        ]
    )
    s = random_splats(len(means), rng, means=means)
    square = np.array([[-0.3, -0.3], [0.3, -0.3], [0.3, 0.3], [-0.3, 0.3]])
    assert splats.table_layer(s, square, scale=1.0, top_m=0.12).tolist() == [True, True, True, False, False, False]


def test_fill_under_covers_the_footprint_with_table_color():
    rng = np.random.default_rng(2)
    table_xz = rng.uniform(-0.3, 0.3, (4000, 2))
    table_xz = table_xz[np.linalg.norm(table_xz, axis=1) > 0.06]  # hole where the object stood
    means = np.stack([table_xz[:, 0], np.zeros(len(table_xz)), table_xz[:, 1]], axis=1)
    background = random_splats(len(means), rng, means=means)
    background["sh0"][:] = [0.5, -0.2, 0.1]
    square = np.array([[-0.05, -0.05], [0.05, -0.05], [0.05, 0.05], [-0.05, 0.05]])
    filled = splats.fill_under(background, [square], spacing=0.01, table_band=0.005)
    added = filled["means"][len(means) :]
    assert len(added) >= 100 and np.abs(added[:, 1]).max() == 0
    np.testing.assert_allclose(filled["sh0"][len(means) :], [[0.5, -0.2, 0.1]] * len(added), atol=1e-5)


def test_table_outline_ignores_stray_coplanar_points():
    rng = np.random.default_rng(3)
    table = rng.uniform([-0.4, -0.3], [0.4, 0.3], (20_000, 2))
    strays = rng.uniform([2.0, 2.0], [2.2, 2.2], (40, 2))
    polygon = geometry.table_outline(np.vstack([table, strays]), cell=0.02)
    assert np.all(polygon.max(axis=0) < 0.45) and np.all(polygon.min(axis=0) > -0.45)
    x, z = polygon[:, 0], polygon[:, 1]
    signed_area_xz = 0.5 * np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z)
    assert signed_area_xz < 0  # counter-clockwise seen from +y means clockwise in (x, z)
    assert abs(abs(signed_area_xz) - 0.8 * 0.6) < 0.05


def test_voxel_surface_mesh_is_closed_and_outward():
    rng = np.random.default_rng(4)
    vertices, faces = geometry.voxel_surface_mesh(rng.uniform(0, 0.1, (5000, 3)), voxel=0.01, dilate=0)
    volume = (
        np.einsum("ij,ij->i", vertices[faces[:, 0]], np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]])).sum() / 6
    )
    assert volume == pytest.approx(0.1**3, rel=0.25)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts % 2 == 0)  # every edge shared: watertight


def test_concave_objects_get_parts_and_convex_ones_do_not():
    rng = np.random.default_rng(5)
    arm = rng.uniform([0, 0, 0], [0.2, 0.04, 0.04], (6000, 3))
    post = rng.uniform([0, 0, 0], [0.04, 0.2, 0.04], (6000, 3))
    parts, info = collision.best_collider(np.vstack([arm, post]), voxel=0.01)
    assert info["used_parts"] and len(parts) >= 2
    parts, info = collision.best_collider(rng.uniform(0, 0.1, (8000, 3)), voxel=0.02)
    assert not info["used_parts"] and len(parts) == 1
