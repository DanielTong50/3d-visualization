import numpy as np
import pytest

from darius.robotics import mass
from darius.scene.io import box_mesh, cylinder_mesh, sphere_mesh


def test_box_matches_the_closed_form():
    a, b, c, m = 0.24, 0.04, 0.17, 1.2
    props = mass.mass_properties([box_mesh([a / 2, b / 2, c / 2])], m)
    expected = m / 12 * np.diag([b**2 + c**2, a**2 + c**2, a**2 + b**2])
    assert np.allclose(props.inertia_kg_m2, expected, rtol=1e-9, atol=1e-15)
    assert np.allclose(props.com_m, 0, atol=1e-12)


def test_volume_and_moments_of_a_unit_cube():
    volume, first, second = mass.volume_moments([box_mesh([0.5, 0.5, 0.5])])
    assert volume == pytest.approx(1)
    assert np.allclose(first, 0, atol=1e-12)
    assert np.allclose(second, np.eye(3) / 12)


def test_cylinder_and_sphere_match_the_closed_forms():
    r, h, m = 0.04, 0.1, 0.3
    cylinder = mass.mass_properties([cylinder_mesh(r, h / 2)], m).inertia_kg_m2  # along +y
    assert cylinder[1, 1] == pytest.approx(m * r**2 / 2, rel=0.01)
    assert cylinder[0, 0] == pytest.approx(m * (3 * r**2 + h**2) / 12, rel=0.01)
    sphere = mass.mass_properties([sphere_mesh(0.03)], 0.06).inertia_kg_m2
    assert np.allclose(np.diag(sphere), 0.4 * 0.06 * 0.03**2, rtol=0.01)


def test_a_thin_shell_approaches_two_thirds_m_r_squared():
    solid = mass.mass_properties([sphere_mesh(0.05)], 1.0)
    shell = mass.mass_properties([sphere_mesh(0.05)], 1.0, fill_fraction=0.005, hollow=True)
    assert shell.inertia_kg_m2[0, 0] == pytest.approx(2 / 3 * 0.05**2, rel=0.02)
    assert shell.inertia_kg_m2[0, 0] > solid.inertia_kg_m2[0, 0]
    assert np.allclose(shell.com_m, solid.com_m)
    assert mass.shell_factor(1.0) == pytest.approx(1)
    factors = [mass.shell_factor(f) for f in (0.9, 0.5, 0.2, 0.05)]
    assert factors == sorted(factors) and factors[-1] < 5 / 3


def test_center_of_mass_follows_the_geometry_and_inertia_does_not():
    box = box_mesh([0.05, 0.02, 0.03])
    centered = mass.mass_properties([box], 0.5)
    moved = mass.mass_properties([box.transformed(np.eye(3), [0.1, -0.2, 0.05])], 0.5)
    assert np.allclose(moved.com_m, [0.1, -0.2, 0.05])
    assert np.allclose(moved.inertia_kg_m2, centered.inertia_kg_m2, atol=1e-12)


def test_two_touching_parts_equal_one_box():
    left = box_mesh([0.025, 0.02, 0.03]).transformed(np.eye(3), [-0.025, 0, 0])
    right = box_mesh([0.025, 0.02, 0.03]).transformed(np.eye(3), [0.025, 0, 0])
    whole = mass.mass_properties([box_mesh([0.05, 0.02, 0.03])], 0.5)
    parts = mass.mass_properties([left, right], 0.5)
    assert np.allclose(parts.inertia_kg_m2, whole.inertia_kg_m2) and np.allclose(parts.com_m, 0, atol=1e-12)


def test_rotating_to_world_axes_and_back_through_scene_json():
    props = mass.mass_properties([box_mesh([0.05, 0.01, 0.02]).transformed(np.eye(3), [0, 0.01, 0])], 0.2)
    world = props.to_world()
    assert world.com_m == pytest.approx([0, 0, 0.01])
    assert world.inertia_kg_m2[2, 2] == pytest.approx(props.inertia_kg_m2[1, 1])  # scene y is world z
    stored = mass.inertial(props, hollow=False, fill_fraction=1.0)
    assert stored.model == "solid" and stored.inertia_kg_m2[3:] == [0.0, 0.0, 0.0]  # no -1e-21 noise
    back = mass.from_inertial(0.2, stored)
    assert np.allclose(back.inertia_kg_m2, props.inertia_kg_m2, rtol=1e-5, atol=1e-12)


def test_principal_axes_are_canonical():
    moments, axes = mass.mass_properties([box_mesh([0.05, 0.01, 0.02])], 1.0).principal
    assert np.allclose(axes, np.eye(3))  # already diagonal: keep the body axes
    tilted = mass.mass_properties([box_mesh([0.05, 0.01, 0.02])], 1.0).rotated(
        np.array([[0.8, -0.6, 0], [0.6, 0.8, 0], [0, 0, 1.0]])
    )
    moments, axes = tilted.principal
    assert np.linalg.det(axes) == pytest.approx(1)
    assert np.allclose(axes @ np.diag(moments) @ axes.T, tilted.inertia_kg_m2, atol=1e-12)


def test_an_open_mesh_is_rejected():
    box = box_mesh([0.05, 0.05, 0.05])
    inside_out = type(box)(box.vertices, box.faces[:, ::-1])
    with pytest.raises(ValueError):
        mass.mass_properties([inside_out], 1.0)
