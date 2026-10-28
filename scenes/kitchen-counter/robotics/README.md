# Kitchen counter: robotics bundle

Written by Darius 0.2.0 from `3bb8174ebe2e` (Objectron bottle/batch-3/14, Google). Every file here is generated from
`../scene.json`; rebuild with `darius export ..`.

**Conventions.** World frame: +z up, meters, kilograms, seconds; the table top is the plane
z = 0. Quaternions are `[x, y, z, w]` in `world.json` and `[w, x, y, z]` in MJCF and USD.
Each object's frame has its origin at its capture-time bounding-box center; mass properties
and grasps are given in that frame. Grasp frame: +z is the approach direction, +y the closing
axis, origin midway between the fingertips. Cameras in `world.json` use OpenCV axes.

| Object | Material | Mass | Friction | Restitution | Grasps (best width) |
|---|---|---|---|---|---|
| `obj_01` toaster oven | metal | 23006 g | 0.50 | 0.10 | 8 (12.1 cm) |
| `obj_02` woven basket | plastic | 677 g | 0.50 | 0.15 | 8 (8.8 cm) |
| `obj_03` liquor bottle | glass | 1343 g | 0.35 | 0.15 | 8 (9.5 cm) |
| `obj_04` scissors | metal | 60 g | 0.40 | 0.20 | 1 (8.7 cm) |
| `obj_05` folded towel | fabric | 272 g | 0.70 | 0.05 | 8 (4.7 cm) |

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
