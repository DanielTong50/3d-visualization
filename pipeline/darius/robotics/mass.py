"""Mass properties: how an object's mass is distributed, which is what a robot feels.

A rigid body in a simulator needs its mass, its center of mass, and its inertia
tensor about that center. The pipeline estimates the mass (hull volume x density x
fill fraction, `darius.pipeline.physics`); this module distributes it over the
object's collision geometry.

Solid objects get uniform density. The volume integrals use the divergence
theorem on the closed triangle meshes: each face spans a signed tetrahedron with
the origin, and a tetrahedron's moments have closed forms, so the result is exact
for the meshes as given (and correct for convex parts that only touch).

Hollow objects (a mug, a shoe, a box) carry their mass in a wall, not in the volume
the hull encloses, and a solid model would make them too easy to spin. Their wall is
modeled as the hull minus a copy of itself scaled by s about the centroid, with s
chosen so the wall holds the object's fill fraction f of the hull's volume:
1 - s^3 = f. Both solids have the same shape, and a solid's inertia per unit mass
scales with the square of its size, so the wall's inertia per unit mass is the solid
hull's times

    k = (1 - s^5) / (1 - s^3),

which runs from 1 (f = 1, solid) to 5/3 (f -> 0, a thin shell): a solid sphere's
2/5 m r^2 becomes the thin spherical shell's 2/3 m r^2. The center of mass is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import frames
from ..scene.io import Mesh, ScenePackage
from ..scene.schema import Inertial, SceneObject


@dataclass(frozen=True)
class MassProperties:
    mass_kg: float
    com_m: np.ndarray  # (3,) in the body frame
    inertia_kg_m2: np.ndarray  # (3, 3) about the center of mass, body axes

    @property
    def principal(self) -> tuple[np.ndarray, np.ndarray]:
        """Principal moments and a proper rotation whose columns are the principal axes.

        Canonical, so exports are reproducible on any machine: a tensor that is already
        diagonal keeps the body axes (symmetric shapes have no unique principal axes),
        and otherwise each axis points along its largest component's positive direction.
        """
        tensor = self.inertia_kg_m2
        off_diagonal = abs(tensor[0, 1]) + abs(tensor[0, 2]) + abs(tensor[1, 2])
        if off_diagonal <= 1e-9 * max(float(np.trace(tensor)), 1e-12):
            return np.diag(tensor).copy(), np.eye(3)
        moments, axes = np.linalg.eigh(tensor)
        for k in range(2):
            if axes[np.argmax(np.abs(axes[:, k])), k] < 0:
                axes[:, k] *= -1
        axes[:, 2] = np.cross(axes[:, 0], axes[:, 1])
        return moments, axes

    def rotated(self, rotation: np.ndarray) -> MassProperties:
        """The same distribution with the body's axes rotated (e.g. scene to world axes)."""
        r = np.asarray(rotation, float)
        return MassProperties(self.mass_kg, r @ self.com_m, r @ self.inertia_kg_m2 @ r.T)

    def to_world(self) -> MassProperties:
        return self.rotated(frames.WORLD_FROM_SCENE)

    def six(self) -> list[float]:
        """[ixx, iyy, izz, ixy, ixz, iyz], the order of scene.json, MJCF's fullinertia and URDF."""
        i = self.inertia_kg_m2
        return [float(v) for v in (i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2])]


def volume_moments(meshes: list[Mesh]) -> tuple[float, np.ndarray, np.ndarray]:
    """Volume, first moment (integral of x) and second moment (integral of x x^T) of closed meshes.

    A tetrahedron (0, a, b, c) has signed volume V = a . (b x c) / 6, first moment
    V (a + b + c) / 4, and second moment V / 20 (a a^T + b b^T + c c^T + s s^T) with
    s = a + b + c. Summed over every face, the parts outside the solid cancel.
    """
    volume, first, second = 0.0, np.zeros(3), np.zeros((3, 3))
    for mesh in meshes:
        a, b, c = (mesh.vertices[mesh.faces[:, k]] for k in range(3))
        v = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
        s = a + b + c
        volume += float(v.sum())
        first += (v[:, None] * s).sum(axis=0) / 4
        outer = sum(np.einsum("ni,nj->nij", x, x) for x in (a, b, c, s))
        second += np.einsum("n,nij->ij", v, outer) / 20
    return volume, first, second


def shell_factor(fill_fraction: float) -> float:
    """k = (1 - s^5) / (1 - s^3) with 1 - s^3 = fill_fraction (see the module docstring)."""
    f = float(np.clip(fill_fraction, 1e-6, 1.0))
    s = (1 - f) ** (1 / 3)
    return (1 - s**5) / f


def mass_properties(
    meshes: list[Mesh], mass_kg: float, fill_fraction: float = 1.0, hollow: bool = False
) -> MassProperties:
    """Mass properties of `mass_kg` spread over the meshes: solid, or as a wall if `hollow`."""
    volume, first, second = volume_moments(meshes)
    if volume <= 0:
        raise ValueError("collision meshes enclose no volume (are they closed and wound outward?)")
    com = first / volume
    covariance = second / volume - np.outer(com, com)  # per unit volume, about the center of mass
    inertia = mass_kg * (np.trace(covariance) * np.eye(3) - covariance)
    if hollow and fill_fraction < 1:
        inertia *= shell_factor(fill_fraction)
    inertia = (inertia + inertia.T) / 2
    return MassProperties(float(mass_kg), com, inertia)


def for_object(package: ScenePackage, obj: SceneObject) -> MassProperties:
    """The object's mass properties in its body frame (scene axes)."""
    return mass_properties(package.collision_meshes(obj), obj.mass_kg, obj.fill_fraction, obj.hollow)


def inertial(props: MassProperties, hollow: bool, fill_fraction: float) -> Inertial:
    """The scene.json form of `props`."""
    six = props.six()
    noise = 1e-9 * max(abs(v) for v in six)  # products of inertia of symmetric shapes come out as ~1e-21
    return Inertial(
        com_m=[round(float(v), 5) + 0.0 for v in props.com_m],  # + 0.0 turns -0.0 into 0.0
        inertia_kg_m2=[0.0 if abs(v) < noise else float(f"{v:.6g}") for v in six],
        model="shell" if hollow and fill_fraction < 1 else "solid",
    )


def from_inertial(mass_kg: float, value: Inertial) -> MassProperties:
    ixx, iyy, izz, ixy, ixz, iyz = value.inertia_kg_m2
    tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64)
    return MassProperties(mass_kg, np.array(value.com_m, dtype=np.float64), tensor)
