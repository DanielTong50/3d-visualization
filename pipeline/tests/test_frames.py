import math

import numpy as np
import pytest

from darius.core import frames


def random_rotations(n, seed=0):
    q = np.random.default_rng(seed).normal(size=(n, 4))
    return [frames.quat_to_matrix(v / np.linalg.norm(v)) for v in q]


def test_world_frame_is_z_up_and_right_handed():
    w = frames.WORLD_FROM_SCENE
    assert np.allclose(w @ [0, 1, 0], [0, 0, 1])  # scene up is world up
    assert np.allclose(w @ [1, 0, 0], [1, 0, 0])
    assert np.allclose(w @ [0, 0, 1], [0, -1, 0])  # toward the filming side is world -y
    assert np.allclose(w @ w.T, np.eye(3)) and np.linalg.det(w) == pytest.approx(1)
    assert np.allclose(frames.quat_to_matrix(frames.WORLD_FROM_SCENE_QUAT), w)
    point = np.array([0.1, 0.2, 0.3])
    assert np.allclose(frames.to_scene(frames.to_world(point)), point)


def test_quaternion_round_trip_and_hemisphere():
    for r in random_rotations(50):
        q = frames.matrix_to_quat(r)
        assert q[3] >= 0 and np.linalg.norm(q) == pytest.approx(1)
        assert np.allclose(frames.quat_to_matrix(q), r, atol=1e-9)
    batch = frames.matrix_to_quat(np.stack(random_rotations(5, seed=1)))
    assert batch.shape == (5, 4)


def test_quaternion_product_composes_rotations():
    a, b = random_rotations(2, seed=2)
    qa, qb = frames.matrix_to_quat(a), frames.matrix_to_quat(b)
    assert np.allclose(frames.quat_to_matrix(frames.quat_multiply(qa, qb)), a @ b, atol=1e-9)


def test_a_yaw_in_the_scene_is_a_yaw_about_world_z():
    angle = 0.7
    yaw_scene = np.array([0, math.sin(angle / 2), 0, math.cos(angle / 2)])  # about scene +y
    yaw_world = frames.rotation_to_world(yaw_scene)
    assert np.allclose(yaw_world, [0, 0, math.sin(angle / 2), math.cos(angle / 2)], atol=1e-12)
    assert np.allclose(frames.rotation_to_world([0, 0, 0, 1]), [0, 0, 0, 1])


def test_roll_pitch_yaw_round_trip():
    for roll, pitch, yaw in [(0.1, -0.4, 2.0), (-2.5, 1.2, -0.3), (0.0, 0.0, 0.0)]:
        r = frames.rpy_to_matrix(roll, pitch, yaw)
        assert np.allclose(frames.rpy_to_matrix(*frames.matrix_to_rpy(r)), r, atol=1e-9)
    locked = frames.rpy_to_matrix(0.3, math.pi / 2, 0.2)  # gimbal lock still reproduces the matrix
    assert np.allclose(frames.rpy_to_matrix(*frames.matrix_to_rpy(locked)), locked, atol=1e-9)


def test_look_rotation_points_the_camera():
    r = frames.look_rotation(np.array([0.0, -1.0, -1.0]), np.array([0.0, 1.0, 0.0]))
    assert np.allclose(-r[:, 2], np.array([0, -1, -1]) / math.sqrt(2))  # OpenGL cameras look along -z
    assert r[1, 1] > 0 and np.linalg.det(r) == pytest.approx(1)
