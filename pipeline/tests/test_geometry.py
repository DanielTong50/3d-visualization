import numpy as np
import pytest

from darius.core import geometry


def noisy_plane(normal, offset, n=4000, noise=0.002, seed=0):
    rng = np.random.default_rng(seed)
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    basis = np.linalg.svd(normal[None])[2][1:]
    uv = rng.uniform(-1, 1, (n, 2))
    points = uv @ basis - offset * normal
    return points + rng.normal(0, noise, (n, 3))


def test_ransac_recovers_plane_despite_outliers():
    rng = np.random.default_rng(1)
    plane_points = noisy_plane([0.2, 1.0, -0.1], 0.3)
    clutter = rng.uniform(-1, 1, (1500, 3))
    plane, inliers = geometry.fit_plane_ransac(np.vstack([plane_points, clutter]), threshold=0.01)
    expected = np.array([0.2, 1.0, -0.1]) / np.linalg.norm([0.2, 1.0, -0.1])
    assert abs(abs(plane.normal @ expected) - 1) < 1e-4
    assert inliers[: len(plane_points)].mean() > 0.98


@pytest.mark.parametrize("source", [[0, 0, 1], [1, 1, 0], [0, -1, 0], [0, 1, 0], [0.3, -0.2, 0.9]])
def test_rotation_aligning_maps_source_to_target(source):
    source = np.array(source, float) / np.linalg.norm(source)
    rotation = geometry.rotation_aligning(source, np.array([0.0, 1.0, 0.0]))
    assert np.allclose(rotation @ source, [0, 1, 0], atol=1e-9)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(rotation), 1.0)


def test_gravity_frame_puts_table_at_y0_and_cameras_above():
    table = noisy_plane([1, 0, 0], -2.0, noise=0.0)  # plane x = 2
    cameras = np.array([[3.0, 0.5, 0.0], [3.2, -0.4, 0.3]])  # on the +x side
    plane = geometry.refine_plane(table).flipped()  # deliberately wrong sign
    transform = geometry.gravity_frame(plane, cameras, table)
    aligned = geometry.apply_transform(transform, table)
    assert np.abs(aligned[:, 1]).max() < 1e-9
    assert np.all(geometry.apply_transform(transform, cameras)[:, 1] > 0)
    assert np.allclose(aligned.mean(axis=0), 0, atol=1e-9)


def test_majority_vote_needs_a_clear_winner():
    keys = np.array([[0, 0, 0]] * 5 + [[1, 0, 0]] * 4 + [[2, 0, 0]])
    labels = np.array([1, 1, 1, 0, 2] + [1, 1, 2, 2] + [1])
    unique, winner, inverse = geometry.majority_vote(keys, labels, num_labels=3, min_share=0.55)
    by_key = {tuple(k): w for k, w in zip(unique, winner, strict=True)}
    assert by_key[(0, 0, 0)] == 1  # 3 of 5
    assert by_key[(1, 0, 0)] == 0  # tie: no clear winner
    assert by_key[(2, 0, 0)] == 0  # a single vote is not enough
    assert len(inverse) == len(keys)


def test_largest_component_drops_floaters():
    rng = np.random.default_rng(2)
    blob = rng.uniform(0, 0.1, (2000, 3))
    floater = rng.uniform(0, 0.01, (50, 3)) + 1.0
    keep = geometry.largest_component(np.vstack([blob, floater]), voxel=0.02)
    assert keep[:2000].all() and not keep[2000:].any()


def test_convex_hull_of_cube():
    corners = np.array([[x, y, z] for x in (0, 2) for y in (0, 2) for z in (0, 2)], float)
    rng = np.random.default_rng(3)
    hull = geometry.convex_hull(np.vstack([corners, rng.uniform(0.1, 1.9, (500, 3))]))
    assert np.isclose(hull.volume, 8.0)
    assert len(hull.vertices) == 8
    center = hull.vertices.mean(axis=0)
    for a, b, c in hull.faces:  # every face winds outward
        normal = np.cross(hull.vertices[b] - hull.vertices[a], hull.vertices[c] - hull.vertices[a])
        assert normal @ (hull.vertices[a] - center) > 0


def test_convex_hull_respects_vertex_budget():
    rng = np.random.default_rng(4)
    sphere = rng.normal(size=(5000, 3))
    sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
    hull = geometry.convex_hull(sphere, max_vertices=64)
    assert len(hull.vertices) <= 64
    assert 0.7 * (4 / 3 * np.pi) < hull.volume < 4 / 3 * np.pi


def test_footprint_extents_of_rotated_box():
    rng = np.random.default_rng(5)
    box = rng.uniform([-0.12, 0, -0.04], [0.12, 0.05, 0.04], (6000, 3))
    yaw = 0.6
    rotation = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    dims, _ = geometry.footprint_extents(box @ rotation.T)
    assert np.allclose(dims, [0.24, 0.05, 0.08], atol=0.006)


def test_complete_bottom_adds_a_flat_base():
    rng = np.random.default_rng(6)
    shell = rng.uniform([-0.05, 0.01, -0.05], [0.05, 0.1, 0.05], (1000, 3))
    completed = geometry.complete_bottom(shell, band=0.01)
    assert completed[:, 1].min() == 0.0
    assert len(completed) > len(shell)
