"""Grasp candidates for a parallel-jaw gripper.

A parallel-jaw grasp squeezes an object between two fingertips along a closing
axis, and holds it by friction. It holds when each contact's surface normal lies
within the friction cone around the closing axis, i.e. at most atan(mu) from it:
an antipodal grasp (Nguyen, 1988).

Candidates are synthesized on the object's convex hull, in its body frame:

1. Sample contact points p on the hull, area-weighted, with outward normals n.
2. Close along -n: the ray from p into the object leaves the hull at q, where it
   crosses the first face plane it is heading out of (exact for a convex hull).
3. Keep the pair if the width |q - p| fits the gripper, q's normal is inside the
   friction cone, and both contacts clear the table, so the fingers do not hit it.
4. Approach perpendicular to the closing axis, as close to straight down as the
   axis allows: top-down grasps where they exist, side grasps otherwise.
5. Score each grasp (0 to 1) by its friction-cone margin, how close the closing
   line passes to the center of mass (gravity then exerts little torque about the
   fingertips, which only torsional friction resists), and a preference for top-down
   approaches; then suppress near-duplicates and keep the best `limit`.

The hull overestimates concave objects (a mug's handle loop is filled in), so a
candidate can reach into a concavity the real object has. Treat candidates as
proposals for a planner, not as verified grasps.

Grasp frame, as written to scene.json and every export: origin midway between the
fingertips; +z is the approach direction (palm toward fingertips); +y is the
closing axis; +x = y cross z.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core import frames
from ..scene.io import Mesh, ScenePackage
from ..scene.schema import Grasp as GraspSpec
from ..scene.schema import SceneObject


@dataclass(frozen=True)
class Gripper:
    """A parallel-jaw gripper. The default opens to 140 mm, like a Robotiq 2F-140: tabletop
    objects such as mugs, shoes and cameras are 9 to 14 cm across, beyond a Franka Hand's 80 mm
    (`Gripper(max_width_m=0.08)`)."""

    max_width_m: float = 0.14
    min_width_m: float = 0.004
    #: The fingertips must stay this far above the table ...
    clearance_m: float = 0.008
    #: ... plus half a finger's width when approaching from the side.
    finger_width_m: float = 0.02


@dataclass(frozen=True)
class Grasp:
    position: np.ndarray  # (3,) midpoint between the contacts, body frame
    approach: np.ndarray  # (3,) unit
    closing: np.ndarray  # (3,) unit, from one contact to the other
    width_m: float
    score: float

    def rotation(self) -> np.ndarray:
        """Gripper frame as a rotation matrix: columns x, y (closing), z (approach)."""
        x = np.cross(self.closing, self.approach)
        return np.stack([x, self.closing, self.approach], axis=1)

    def quat(self) -> np.ndarray:
        return frames.matrix_to_quat(self.rotation())

    def spec(self) -> GraspSpec:
        return GraspSpec(
            position=[round(float(v), 5) + 0.0 for v in self.position],
            rotation_quat=[round(float(v), 6) + 0.0 for v in self.quat()],
            width_m=round(self.width_m, 4),
            score=round(self.score, 3),
        )

    @classmethod
    def from_spec(cls, spec: GraspSpec) -> Grasp:
        rotation = frames.quat_to_matrix(np.array(spec.rotation_quat))
        return cls(np.array(spec.position, float), rotation[:, 2], rotation[:, 1], spec.width_m, spec.score)


def face_planes(hull: Mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit outward normals, plane offsets (n . x = h) and areas of the hull's faces."""
    a, b, c = (hull.vertices[hull.faces[:, k]] for k in range(3))
    cross = np.cross(b - a, c - a)
    area = np.linalg.norm(cross, axis=1) / 2
    keep = area > 1e-12
    normals = cross[keep] / (2 * area[keep, None])
    return normals, np.einsum("ij,ij->i", normals, a[keep]), area[keep]


def sample_surface(hull: Mesh, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted random points on the hull, with their faces' outward normals."""
    a, b, c = (hull.vertices[hull.faces[:, k]] for k in range(3))
    cross = np.cross(b - a, c - a)
    area = np.linalg.norm(cross, axis=1) / 2
    keep = area > 1e-12
    a, b, c, cross, area = a[keep], b[keep], c[keep], cross[keep], area[keep]
    face = rng.choice(len(area), count, p=area / area.sum())
    u, v = rng.random(count), rng.random(count)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    points = a[face] + u[:, None] * (b[face] - a[face]) + v[:, None] * (c[face] - a[face])
    return points, cross[face] / (2 * area[face, None])


def exit_points(points: np.ndarray, directions: np.ndarray, normals: np.ndarray, offsets: np.ndarray):
    """Where rays starting on the hull leave it, and the normals of the faces they leave through."""
    toward = directions @ normals.T  # (S, F): > 0 for faces the ray is heading out of
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (offsets[None, :] - points @ normals.T) / toward
    t = np.where(toward > 1e-9, t, np.inf)
    t = np.where(t > 1e-7, t, np.inf)  # skip the face the ray starts on
    face = np.argmin(t, axis=1)
    distance = t[np.arange(len(points)), face]
    return points + distance[:, None] * directions, normals[face], distance


def antipodal_grasps(
    hull: Mesh,
    com: np.ndarray,
    up: np.ndarray,
    table_height: float,
    friction: float,
    gripper: Gripper = Gripper(),
    samples: int = 1500,
    limit: int = 8,
    seed: int = 0,
) -> list[Grasp]:
    """Grasp candidates on a convex hull, best first.

    `up` is the world's up direction and `table_height` the table's height along it
    (up . x for points on the table), both in the hull's frame. `friction` is the
    object's Coulomb coefficient.
    """
    rng = np.random.default_rng(seed)
    up = np.asarray(up, float) / np.linalg.norm(up)
    # Grasp the object as it rests: its lowest point is on the table (or above it, if stacked).
    table_height = max(table_height, float((hull.vertices @ up).min()))
    normals, offsets, _ = face_planes(hull)
    p, n_p = sample_surface(hull, samples, rng)
    q, n_q, width = exit_points(p, -n_p, normals, offsets)
    axis = -n_p  # closing direction, from contact p to contact q
    cone = math.atan(max(friction, 1e-3))
    angle_q = np.arccos(np.clip(np.einsum("ij,ij->i", n_q, axis), -1, 1))
    lowest = np.minimum(p @ up, q @ up) - table_height  # the lower fingertip's height above the table
    # Closing roughly vertically means approaching from the side, with a finger's width below the contact.
    side_extra = np.where(np.abs(axis @ up) > 0.7, gripper.finger_width_m / 2, 0.0)
    ok = (
        np.isfinite(width)
        & (width >= gripper.min_width_m)
        & (width <= gripper.max_width_m)
        & (angle_q <= cone)
        & (lowest >= gripper.clearance_m + side_extra)
    )
    if not ok.any():
        return []
    p, q, axis, width, angle_q = p[ok], q[ok], axis[ok], width[ok], angle_q[ok]
    center = (p + q) / 2

    # Approach: straight down, minus its component along the closing axis.
    down = -up
    approach = down[None, :] - (axis @ down)[:, None] * axis
    length = np.linalg.norm(approach, axis=1)
    sideways = length < 1e-6  # closing axis is vertical: approach horizontally
    if sideways.any():
        helper = np.cross(axis[sideways], up)
        helper_length = np.linalg.norm(helper, axis=1)
        fallback = np.where(helper_length[:, None] > 1e-6, helper, np.cross(axis[sideways], [1.0, 0, 0]))
        approach[sideways] = fallback
        length[sideways] = np.linalg.norm(fallback, axis=1)
    approach /= length[:, None]

    size = float(np.ptp(hull.vertices, axis=0).max())
    rel = com[None, :] - center
    off_axis = np.linalg.norm(rel - np.einsum("ij,ij->i", rel, axis)[:, None] * axis, axis=1)
    margin = 1 - angle_q / cone
    centered = np.exp(-((off_axis / (0.25 * size)) ** 2))
    from_above = 0.6 + 0.4 * np.clip(approach @ down, 0, 1)
    roomy = np.where(width <= 0.85 * gripper.max_width_m, 1.0, 1 - 0.5 * (width / gripper.max_width_m - 0.85) / 0.15)
    score = np.sqrt(np.clip(margin, 0, 1)) * centered * from_above * roomy

    chosen: list[Grasp] = []
    for i in np.argsort(-score):
        duplicate = any(
            np.linalg.norm(center[i] - g.position) < max(0.01, 0.1 * size) and abs(float(axis[i] @ g.closing)) > 0.94
            for g in chosen
        )
        if duplicate:
            continue
        chosen.append(Grasp(center[i], approach[i], axis[i], float(width[i]), float(score[i])))
        if len(chosen) == limit:
            break
    return chosen


def for_object(package: ScenePackage, obj: SceneObject, com: np.ndarray, gripper: Gripper = Gripper()) -> list[Grasp]:
    """Grasp candidates for an object as it rests at its pose in the package, in its body frame."""
    plane = package.manifest.background.table_plane
    rotation = frames.quat_to_matrix(np.array(obj.pose.rotation_quat))
    up_scene = np.array(plane.normal, float) / np.linalg.norm(plane.normal)
    up_body = rotation.T @ up_scene
    # A point x on the table, in body coordinates: up_body . x = up_scene . (table point - body origin).
    table_height = float(up_scene @ (np.array(plane.point, float) - np.array(obj.pose.position, float)))
    return antipodal_grasps(package.hull(obj), com, up_body, table_height, obj.friction, gripper)
