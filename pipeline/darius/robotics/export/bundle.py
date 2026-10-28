"""The robotics bundle: one folder a robot stack can load as it is, and a zip of it.

    <scene>/robotics/
    ├── README.md        what is here, the conventions, and how to load it in each simulator
    ├── world.json       the scene in the world frame: poses, mass properties, grasps, cameras
    ├── scene_graph.json 3D scene graph: surfaces, objects, parts, grasps, cameras and their relations
    ├── scene.xml        MuJoCo MJCF
    ├── scene.usda       OpenUSD with UsdPhysics (Isaac Sim / Isaac Lab)
    ├── urdf/<id>.urdf   one URDF per object, plus table.urdf (PyBullet, Drake, Genesis, ROS)
    └── meshes/*.obj     collision meshes and hull visuals in each body's frame, +z up

    <scene>/export.zip   scene.json + robotics/, what the web app's download button serves

Every file is a pure function of the scene package, so `darius gallery` can rebuild
and check them.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from ...core import frames
from ...scene.io import ScenePackage, write_obj
from .. import graph as scene_graph
from ..model import RobotScene, build
from . import mjcf, urdf, usd

FOLDER = "robotics"
FORMAT = "darius.robotics/1"


def _r(values, digits: int = 6) -> list[float]:
    return [float(f"{float(v):.{digits}g}") for v in np.ravel(values)]


def world_manifest(scene: RobotScene) -> dict:
    """world.json: everything a robot stack needs to place, weigh and grasp each object."""
    objects = []
    for body in scene.bodies:
        rotation = frames.quat_to_matrix(body.quat)
        grasps = []
        for g in body.grasps:
            grasps.append(
                {
                    "position": _r(g.position),
                    "rotation_quat": _r(g.quat()),
                    "width_m": round(g.width_m, 4),
                    "score": round(g.score, 3),
                    "world": {
                        "position": _r(body.position + rotation @ g.position),
                        "rotation_quat": _r(frames.matrix_to_quat(rotation @ g.rotation())),
                    },
                }
            )
        objects.append(
            {
                "id": body.id,
                "label": body.label,
                "material": body.material,
                "urdf": f"urdf/{body.id}.urdf",
                "pose": {"position": _r(body.position), "rotation_quat": _r(body.quat)},
                "dimensions_m": _r(body.dimensions_m),
                "mass_kg": round(body.mass.mass_kg, 5),
                "com_m": _r(body.mass.com_m),
                "inertia_kg_m2": _r(body.mass.six()),
                "friction": body.friction,
                "restitution": body.restitution,
                "physics_source": body.physics_source,
                "grasps": grasps,
            }
        )
    cameras = []
    for camera in scene.cameras:
        rotation, position = camera.cv_pose()
        cameras.append(
            {
                "name": camera.name,
                "position": _r(position),
                "rotation_quat": _r(frames.matrix_to_quat(rotation)),
                "image_size": [camera.width, camera.height],
                "K": [_r(row) for row in camera.intrinsics()],
                "fovy_deg": round(camera.fovy_deg, 3),
            }
        )
    return {
        "format": FORMAT,
        "generator": scene.generator,
        "scene_id": scene.scene_id,
        "title": scene.title,
        "source": scene.source,
        "frame": {
            "up": "+z",
            "units": "meters, kilograms, seconds",
            "origin": "on the table top, which is the plane z = 0",
            "rotation_quats": "[x, y, z, w]",
            "world_from_scene_quat": _r(frames.WORLD_FROM_SCENE_QUAT),
            "grasp_frame": "+z approach direction, +y closing axis, origin midway between the fingertips",
            "camera_frame": "OpenCV: x right, y down, looking along +z",
        },
        "gravity": _r(scene.gravity),
        "table": {
            "urdf": "urdf/table.urdf",
            "outline_xy": [_r(p) for p in scene.table_outline],
            "thickness_m": 0.04,
        },
        "floor_z": None if scene.floor_depth_m is None else -scene.floor_depth_m,
        "objects": objects,
        "cameras": cameras,
    }


README = """# {title}: robotics bundle

Written by {generator} from `{scene_id}` ({source}). Every file here is generated from
`../scene.json`; rebuild with `darius export ..`.

**Conventions.** World frame: +z up, meters, kilograms, seconds; the table top is the plane
z = 0. Quaternions are `[x, y, z, w]` in `world.json` and `[w, x, y, z]` in MJCF and USD.
Each object's frame has its origin at its capture-time bounding-box center; mass properties
and grasps are given in that frame. Grasp frame: +z is the approach direction, +y the closing
axis, origin midway between the fingertips. Cameras in `world.json` use OpenCV axes.

| Object | Material | Mass | Friction | Restitution | Grasps (best width) |
|---|---|---|---|---|---|
{rows}

## MuJoCo

```python
import mujoco
model = mujoco.MjModel.from_xml_path("scene.xml")
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("capture").id)
```

Grasp candidates are sites named `<object>/grasp_<k>`, best first.

## Scene graph

`scene_graph.json` lists nodes (the table, each object at its center of mass, its convex parts,
its grasps, the cameras) and edges (`supports`, `near` with the gap in meters, `part_of`,
`grasp_of`, `observes`), ready for a planner or for `networkx.node_link_graph`-style loading.

## PyBullet

```python
import json, pybullet as p
world = json.load(open("world.json"))
p.connect(p.DIRECT)
p.setGravity(*world["gravity"])
p.loadURDF(world["table"]["urdf"], useFixedBase=True)
for o in world["objects"]:
    p.loadURDF(o["urdf"], o["pose"]["position"], o["pose"]["rotation_quat"])
```

## Isaac Sim / Isaac Lab

Open `scene.usda` (or reference it into a stage). Objects are rigid bodies under
`/World/Objects` with explicit mass properties and convex-hull colliders, bound to
physics materials in `/World/PhysicsMaterials`.

## Python

```python
import darius
scene = darius.load_scene("..")
```

See `docs/robotics.md` in the Darius repository for how every number here is computed.
"""


def readme(scene: RobotScene) -> str:
    rows = []
    for body in scene.bodies:
        best = f"{len(body.grasps)} ({body.grasps[0].width_m * 100:.1f} cm)" if body.grasps else "none fit"
        rows.append(
            f"| `{body.id}` {body.label} | {body.material} | {body.mass.mass_kg * 1000:.0f} g | "
            f"{body.friction:.2f} | {body.restitution:.2f} | {best} |"
        )
    return README.format(
        title=scene.title, generator=scene.generator, scene_id=scene.scene_id, source=scene.source, rows="\n".join(rows)
    )


def write(package: ScenePackage, out: Path | None = None) -> Path:
    """Write the bundle folder (default: <scene>/robotics/) and return it."""
    scene = build(package)
    out = Path(out) if out else package.root / FOLDER
    if out.exists():
        shutil.rmtree(out)
    meshes = out / "meshes"
    meshes.mkdir(parents=True)
    write_obj(
        meshes / "table.obj", scene.table_mesh(), f"{scene.title}: table top, a convex slab with its top at z = 0"
    )
    for body in scene.bodies:
        for shape in body.shapes:
            if shape.kind == "mesh":
                write_obj(meshes / f"{shape.name}.obj", shape.mesh, f"{body.label}: convex collision part")
        if body.visual is not None:
            write_obj(meshes / f"{body.id}_visual.obj", body.visual, f"{body.label}: convex hull, for display")
    mjcf.write(scene, out / "scene.xml")
    usd.write(scene, out / "scene.usda")
    urdf.write(scene, out / "urdf")
    (out / "world.json").write_text(json.dumps(world_manifest(scene), indent=2) + "\n")
    graph = scene_graph.build(scene).as_dict(scene)
    (out / "scene_graph.json").write_text(json.dumps(graph, indent=1) + "\n")
    (out / "README.md").write_text(readme(scene))
    return out


def write_zip(scene_dir: Path) -> Path:
    """export.zip: scene.json and the robotics folder, with fixed timestamps so rebuilds are
    byte-identical."""
    scene_dir = Path(scene_dir)
    bundle = scene_dir / "export.zip"
    files = [scene_dir / "scene.json"] + sorted(p for p in (scene_dir / FOLDER).rglob("*") if p.is_file())
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            info = zipfile.ZipInfo(
                f"{scene_dir.name}/{path.relative_to(scene_dir).as_posix()}", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
    return bundle


def export(scene_dir: Path, out: Path | None = None) -> Path:
    """Write the robotics bundle for a scene folder; with no `out`, also refresh its export.zip."""
    package = ScenePackage.load(scene_dir)
    folder = write(package, out)
    if out is None:
        write_zip(package.root)
    return folder


__all__ = ["FOLDER", "export", "world_manifest", "write", "write_zip"]
