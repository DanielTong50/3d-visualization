import numpy as np
import pytest

from darius.robotics import grasps
from darius.scene.io import box_mesh, sphere_mesh

UP = np.array([0.0, 1.0, 0.0])


def resting(mesh):
    """A mesh in its body frame with the table under its lowest point."""
    return mesh, float(mesh.vertices[:, 1].min())


def test_a_narrow_box_is_grasped_across_its_thin_sides_from_above():
    hull, table = resting(box_mesh([0.02, 0.05, 0.03]))  # 4 x 10 x 6 cm, standing
    found = grasps.antipodal_grasps(hull, np.zeros(3), UP, table, friction=0.5)
    assert found, "expected grasp candidates"
    assert all(g.width_m <= 0.14 for g in found)
    best = found[0]
    assert best.width_m == pytest.approx(0.04, abs=1e-6)  # across the thinnest side
    assert abs(best.closing[0]) == pytest.approx(1, abs=1e-6)
    assert best.approach @ -UP == pytest.approx(1, abs=1e-6)  # straight down
    assert {round(g.width_m, 3) for g in found} <= {0.04, 0.06}  # never pinched top to bottom
    assert all(0 <= g.score <= 1 for g in found) and [g.score for g in found] == sorted(
        (g.score for g in found), reverse=True
    )


def test_objects_wider_than_the_gripper_get_no_grasps():
    hull, table = resting(box_mesh([0.08, 0.08, 0.08]))
    assert grasps.antipodal_grasps(hull, np.zeros(3), UP, table, friction=0.5) == []
    hull, table = resting(box_mesh([0.06, 0.06, 0.06]))
    franka = grasps.Gripper(max_width_m=0.08)
    assert grasps.antipodal_grasps(hull, np.zeros(3), UP, table, friction=0.5, gripper=franka) == []
    assert grasps.antipodal_grasps(hull, np.zeros(3), UP, table, friction=0.5)  # 12 cm fits the default


def test_a_book_lying_flat_cannot_be_pinched_through_the_table():
    hull, table = resting(box_mesh([0.12, 0.02, 0.085]))  # 24 x 4 x 17 cm, lying down
    assert grasps.antipodal_grasps(hull, np.zeros(3), UP, table, friction=0.5) == []  # 17 cm across at best
    hull, table = resting(box_mesh([0.06, 0.015, 0.05]))  # a 12 x 3 x 10 cm paperback: grasped by its edges
    found = grasps.antipodal_grasps(hull, np.zeros(3), UP, table, friction=0.5)
    assert found and all(abs(g.closing[1]) < 1e-6 for g in found)  # never closing vertically


def test_a_ball_is_grasped_across_its_diameter():
    hull, table = resting(sphere_mesh(0.03, subdivisions=2))
    found = grasps.antipodal_grasps(hull, np.zeros(3), UP, table, friction=0.8)
    assert found and all(g.width_m == pytest.approx(0.06, rel=0.05) for g in found)
    assert all((g.position @ UP) - table >= 0.008 for g in found)


def test_grasp_frames_are_orthonormal_and_survive_scene_json():
    hull, table = resting(box_mesh([0.02, 0.05, 0.03]))
    for g in grasps.antipodal_grasps(hull, np.zeros(3), UP, table, friction=0.5):
        r = g.rotation()
        assert np.allclose(r.T @ r, np.eye(3), atol=1e-9) and np.linalg.det(r) == pytest.approx(1)
        back = grasps.Grasp.from_spec(g.spec())
        assert np.allclose(back.closing, g.closing, atol=1e-5) and np.allclose(back.approach, g.approach, atol=1e-5)
        assert back.position == pytest.approx(g.position, abs=1e-5)


def test_low_friction_narrows_the_cone():
    hull = sphere_mesh(0.03, subdivisions=1)  # coarse: opposite faces are not quite parallel
    table = float(hull.vertices[:, 1].min())
    grippy = grasps.antipodal_grasps(hull, np.zeros(3), UP, table, friction=1.0, limit=100)
    slippery = grasps.antipodal_grasps(hull, np.zeros(3), UP, table, friction=0.02, limit=100)
    assert len(slippery) < len(grippy)
