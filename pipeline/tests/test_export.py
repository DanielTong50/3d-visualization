import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from darius.core import frames
from darius.robotics import mass
from darius.robotics.annotate import annotate
from darius.robotics.export import bundle, mjcf
from darius.robotics.model import build
from darius.scene.io import Mesh, dumps, load_scene
from darius.scene.schema import Scene

REPO = Path(__file__).resolve().parents[2]
SANDBOX = REPO / "scenes" / "sandbox"
SCANNED = sorted(p.parent for p in (REPO / "scenes").glob("*/scene.json") if p.parent.name != "sandbox")


@pytest.fixture(scope="module")
def sandbox_bundle(tmp_path_factory):
    package = load_scene(SANDBOX)
    package.manifest = annotate(package)
    return package, bundle.write(package, tmp_path_factory.mktemp("bundle") / "robotics")


def test_restitution_maps_to_a_damping_ratio():
    assert mjcf.damping_ratio(0.5) == pytest.approx(0.2155, abs=1e-4)
    assert mjcf.damping_ratio(0.99) < 0.01 and mjcf.damping_ratio(0.0) > 0.9


def test_the_bundle_has_every_format(sandbox_bundle):
    package, folder = sandbox_bundle
    names = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    assert {"scene.xml", "scene.usda", "world.json", "README.md", "urdf/table.urdf", "meshes/table.obj"} <= names
    assert {f"urdf/{o.id}.urdf" for o in package.objects} <= names


def test_world_json_is_the_scene_in_z_up(sandbox_bundle):
    package, folder = sandbox_bundle
    world = json.loads((folder / "world.json").read_text())
    assert world["frame"]["up"] == "+z" and world["gravity"] == pytest.approx([0, 0, -9.81])
    for entry, obj in zip(world["objects"], package.objects, strict=True):
        assert entry["pose"]["position"] == pytest.approx(frames.to_world(obj.pose.position).tolist(), abs=1e-6)
        assert entry["mass_kg"] == pytest.approx(obj.mass_kg)
        rotation = frames.quat_to_matrix(np.array(entry["pose"]["rotation_quat"]))
        for grasp in entry["grasps"]:  # world grasp = object pose composed with the body-frame grasp
            expected = np.array(entry["pose"]["position"]) + rotation @ np.array(grasp["position"])
            assert grasp["world"]["position"] == pytest.approx(expected.tolist(), abs=1e-5)


def test_the_table_mesh_is_a_closed_slab():
    scene = build(load_scene(SANDBOX))
    table = scene.table_mesh()
    volume, _, _ = mass.volume_moments([table])
    assert volume == pytest.approx(1.2 * 0.75 * 0.04, rel=1e-6)
    assert table.vertices[:, 2].max() == pytest.approx(0)


def test_urdfs_carry_mass_inertia_and_contact(sandbox_bundle):
    package, folder = sandbox_bundle
    for obj in package.objects:
        root = ET.parse(folder / "urdf" / f"{obj.id}.urdf").getroot()
        assert float(root.find("link/inertial/mass").attrib["value"]) == pytest.approx(obj.mass_kg)
        inertia = root.find("link/inertial/inertia").attrib
        assert all(float(inertia[k]) > 0 for k in ("ixx", "iyy", "izz"))
        assert float(root.find("link/contact/lateral_friction").attrib["value"]) == pytest.approx(obj.friction)


def test_mjcf_loads_in_mujoco_with_the_scene_physics(sandbox_bundle):
    mujoco = pytest.importorskip("mujoco")
    package, folder = sandbox_bundle
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    assert model.nbody == len(package.objects) + 1
    assert model.nkey == 1 and model.nsite == sum(len(o.grasps) for o in package.objects)
    for obj in package.objects:
        body = model.body(obj.id)
        assert body.mass[0] == pytest.approx(obj.mass_kg, rel=1e-4)
        assert np.allclose(model.body_ipos[body.id], frames.to_world(obj.inertial.com_m), atol=1e-5)
        moments = np.linalg.eigvalsh(mass.from_inertial(obj.mass_kg, obj.inertial).inertia_kg_m2)
        assert np.sort(model.body_inertia[body.id]) == pytest.approx(np.sort(moments), rel=1e-3)
        geom = model.geom(model.body_geomadr[body.id])
        assert geom.friction[0] == pytest.approx(obj.friction) and geom.priority[0] == 1


def test_urdfs_compile_in_mujoco(sandbox_bundle):
    mujoco = pytest.importorskip("mujoco")
    package, folder = sandbox_bundle
    for obj in package.objects:
        model = mujoco.MjModel.from_xml_path(str(folder / "urdf" / f"{obj.id}.urdf"))
        assert model.body_mass.sum() == pytest.approx(obj.mass_kg, rel=1e-3)


def test_usd_opens_with_rigid_bodies_and_materials(sandbox_bundle):
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    package, folder = sandbox_bundle
    stage = Usd.Stage.Open(str(folder / "scene.usda"))
    assert UsdGeom.GetStageUpAxis(stage) == "Z" and UsdGeom.GetStageMetersPerUnit(stage) == 1
    assert pxr is not None and stage.GetDefaultPrim().GetPath() == "/World"
    scene = UsdPhysics.Scene(stage.GetPrimAtPath("/World/PhysicsScene"))
    assert scene.GetGravityMagnitudeAttr().Get() == pytest.approx(9.81)
    for obj in package.objects:
        prim = stage.GetPrimAtPath(f"/World/Objects/{obj.id}")
        assert prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.HasAPI(UsdPhysics.MassAPI)
        assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() == pytest.approx(obj.mass_kg, rel=1e-5)
        position = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
        assert list(position) == pytest.approx(frames.to_world(obj.pose.position).tolist(), abs=1e-5)
        colliders = [c for c in prim.GetChildren() if c.HasAPI(UsdPhysics.CollisionAPI)]
        assert colliders
        material, _ = UsdShade.MaterialBindingAPI(colliders[0]).ComputeBoundMaterial("physics")
        friction = UsdPhysics.MaterialAPI(material.GetPrim()).GetDynamicFrictionAttr().Get()
        assert friction == pytest.approx(obj.friction)


@pytest.mark.parametrize("scene_dir", [SANDBOX, *SCANNED], ids=lambda p: p.name)
def test_exported_scenes_come_to_rest_in_mujoco(scene_dir, tmp_path):
    mujoco = pytest.importorskip("mujoco")
    folder = bundle.write(load_scene(scene_dir), tmp_path / "robotics")
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    start = data.xpos.copy()
    for _ in range(int(3.5 / model.opt.timestep)):  # the sandbox ball is dropped in and bounces ~2.6 s
        mujoco.mj_step(model, data)
    before = data.xpos.copy()
    for _ in range(int(0.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    drift = np.linalg.norm((data.xpos - start)[1:, :2], axis=1).max()
    moving = np.linalg.norm((data.xpos - before)[1:], axis=1).max()  # over the last half second
    assert drift < 0.03 and moving < 0.003


def test_a_dropped_ball_rebounds_to_e_squared_of_its_height(sandbox_bundle):
    mujoco = pytest.importorskip("mujoco")
    package, folder = sandbox_bundle
    ball = package.object("obj_04")  # rubber, restitution 0.85, dropped from 25 cm in the scene
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    body = model.body("obj_04").id
    radius, drop = ball.collider.radius, ball.pose.position[1] - ball.collider.radius
    heights, falling = [], True
    for _ in range(int(1.2 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        vz = data.cvel[body][5]
        if falling and vz > 0:
            falling = False
        elif not falling and vz <= 0:
            heights.append(data.xpos[body, 2] - radius)
            break
    assert heights and heights[0] == pytest.approx(ball.restitution**2 * drop, rel=0.15)


def test_compact_scene_json_round_trips():
    package = load_scene(SANDBOX)
    text = dumps(package.manifest)
    assert Scene.model_validate_json(text) == package.manifest
    assert '"gravity": [0, -9.81, 0]' in text  # short arrays stay on one line, whole numbers stay whole


def test_mesh_transform_is_rigid():
    mesh = Mesh(np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([[0, 1, 2]]))
    turned = mesh.transformed(frames.WORLD_FROM_SCENE, [0, 0, 1])
    assert np.allclose(turned.vertices[1], [0, 0, 2])
    assert math.isclose(np.linalg.norm(turned.vertices[0] - turned.vertices[1]), math.sqrt(2))


def test_the_scene_graph_relates_every_object():
    from darius.robotics import graph

    scene = build(load_scene(SANDBOX))
    g = graph.build(scene)
    ids = {n["id"] for n in g.nodes}
    assert {"world", "table"} <= ids and {o.id for o in scene.bodies} <= ids
    supported = {target for source, target in g.relations("supports")}
    # Everything rests on the table except the rubber ball, which the sandbox drops in from 25 cm.
    assert supported == {"obj_01", "obj_02", "obj_03", "obj_05"}
    parts = [n for n in g.nodes if n["type"] == "part"]
    grasps = [n for n in g.nodes if n["type"] == "grasp"]
    assert not parts  # the sandbox has no compound colliders
    assert len(grasps) == sum(len(b.grasps) for b in scene.bodies)
    assert all(source.startswith(target) for source, target in g.relations("grasp_of"))
    near = {(e["source"], e["target"]): e["distance_m"] for e in g.edges if e["relation"] == "near"}
    assert all(0 <= d < graph.NEAR_M for d in near.values())
    data = g.as_dict(scene)
    assert data["format"] == graph.FORMAT and data["counts"]["edges"] == len(g.edges)


def test_scanned_scene_graphs_include_collision_parts():
    from darius.robotics import graph

    scene = build(load_scene(REPO / "scenes" / "kitchen-counter"))
    g = graph.build(scene)
    assert len({target for _, target in g.relations("supports")}) == len(scene.bodies)  # all rest on something
    assert any(n["type"] == "part" for n in g.nodes)
    assert g.relations("observes")
