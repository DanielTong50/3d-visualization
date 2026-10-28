"""Coordinate frames, and the fixed transforms between them.

Every frame is right-handed and measured in meters.

    scene   scene.json and every asset in a scene package (splats, points, hulls).
            +y is up, the table top is the plane y = 0, and +z points toward the side
            the video was mostly filmed from. This is glTF's and three.js's convention,
            so a browser draws a package as it is.
    world   every robotics output: MJCF, URDF, USD and `darius.robotics`. +z is up
            (ROS REP 103) and the table top is z = 0. It is the scene frame turned
            +90 degrees about x, so the scene point (x, y, z) is the world point
            (x, -z, y).
    body    each object's own frame. Its origin is the center of the object's bounding
            box at capture time. In scene.json its axes are the scene's; in robotics
            outputs they are the world's.
    camera  OpenCV in calibration outputs: x right, y down, looking along +z.
            OpenGL in scene.json's `capture` and in MJCF: x right, y up, looking
            along -z.

Quaternions are [x, y, z, w] throughout, except where a name says `wxyz` (the
order MuJoCo and USD write).

NumPy only: the reconstruction image imports this module and has no SciPy.
"""

from __future__ import annotations

import math

import numpy as np

#: Scene to world: +90 degrees about x. world = WORLD_FROM_SCENE @ scene.
WORLD_FROM_SCENE = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
#: The same rotation as a quaternion [x, y, z, w].
WORLD_FROM_SCENE_QUAT = np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
#: Camera axes between the OpenGL and OpenCV conventions; the flip is its own inverse.
CV_FROM_GL = np.diag([1.0, -1.0, -1.0])


def to_world(points: np.ndarray) -> np.ndarray:
    """Scene-frame points or directions, shape (3,) or (N, 3), in the world frame."""
    return np.asarray(points, dtype=np.float64) @ WORLD_FROM_SCENE.T


def to_scene(points: np.ndarray) -> np.ndarray:
    """World-frame points or directions in the scene frame (the inverse of `to_world`)."""
    return np.asarray(points, dtype=np.float64) @ WORLD_FROM_SCENE


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b of [x, y, z, w] quaternions: rotate by b, then by a."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([-x, -y, -z, w])


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of an [x, y, z, w] quaternion (normalized first)."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(matrices: np.ndarray) -> np.ndarray:
    """Rotation matrices, (3, 3) or (N, 3, 3), as unit [x, y, z, w] quaternions with w >= 0.

    Shepperd's method: build the quaternion from the largest of |w|, |x|, |y|, |z|,
    which keeps the division well conditioned for every rotation.
    """
    m = np.asarray(matrices, dtype=np.float64)
    single = m.ndim == 2
    m = m.reshape(-1, 3, 3)
    trace = np.trace(m, axis1=1, axis2=2)
    candidates = np.stack(
        [
            trace,
            m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2],
            m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2],
            m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1],
        ],
        axis=1,
    )
    which = np.argmax(candidates, axis=1)
    quats = np.empty((len(m), 4))
    for i, (r, k) in enumerate(zip(m, which, strict=True)):
        s = 2 * math.sqrt(max(1 + candidates[i, k], 1e-12))
        if k == 0:
            quats[i] = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, s / 4]
        elif k == 1:
            quats[i] = [s / 4, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
        elif k == 2:
            quats[i] = [(r[0, 1] + r[1, 0]) / s, s / 4, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
        else:
            quats[i] = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, s / 4, (r[1, 0] - r[0, 1]) / s]
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    quats[quats[:, 3] < 0] *= -1
    return quats[0] if single else quats


def rotation_to_world(quat: np.ndarray) -> np.ndarray:
    """A body orientation in the scene frame, re-expressed with world axes on both sides.

    A body's frame in scene.json has scene axes; in robotics outputs it has world
    axes. Its orientation therefore changes by conjugation: W q W^-1.
    """
    return quat_multiply(
        quat_multiply(WORLD_FROM_SCENE_QUAT, np.asarray(quat, float)), quat_conjugate(WORLD_FROM_SCENE_QUAT)
    )


def wxyz(quat: np.ndarray) -> np.ndarray:
    """An [x, y, z, w] quaternion in MuJoCo's and USD's [w, x, y, z] order."""
    x, y, z, w = quat
    return np.array([w, x, y, z])


def matrix_to_rpy(matrix: np.ndarray) -> tuple[float, float, float]:
    """URDF's roll, pitch, yaw: the matrix is Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r = np.asarray(matrix, dtype=np.float64)
    pitch = math.asin(float(np.clip(-r[2, 0], -1.0, 1.0)))
    if abs(r[2, 0]) < 1 - 1e-9:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:  # gimbal lock: only roll - yaw (or roll + yaw) is defined, so put it all in yaw
        roll = 0.0
        yaw = math.atan2(-r[0, 1], r[1, 1])
    return roll, pitch, yaw


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll),
        math.sin(roll),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
    )
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def look_rotation(forward: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Rotation whose columns are an OpenGL camera's axes (x right, y up, looking along -z)
    for a camera looking along `forward` with `up` roughly up."""
    back = -np.asarray(forward, float) / np.linalg.norm(forward)
    right = np.cross(up, back)
    if np.linalg.norm(right) < 1e-9:  # looking straight along `up`: any right will do
        right = np.cross([1.0, 0.0, 0.0] if abs(back[0]) < 0.9 else [0.0, 1.0, 0.0], back)
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(back, right), back], axis=1)
