"""OpenUSD with UsdPhysics: the scene as a stage that NVIDIA Isaac Sim / Isaac Lab, Omniverse
and any other UsdPhysics consumer opens directly.

Written as plain USDA text (no USD library needed to export; the tests read it back
with `pxr`). Stage metadata declares +z up, meters and kilograms.

    /World
        PhysicsScene                gravity
        PhysicsMaterials/<id>       friction and restitution (PhysicsMaterialAPI)
        Table                       static convex collider
        Floor                       static box, if the scene has a floor
        Objects/<id>                rigid body: PhysicsRigidBodyAPI + PhysicsMassAPI
            collision[_k]           convex collider(s) (PhysicsMeshCollisionAPI: convexHull)
        Cameras/<name>              capture cameras (UsdGeomCamera)

Mass properties are explicit: mass, center of mass, principal moments
(diagonalInertia) and the rotation to the principal axes (principalAxes).
PhysX averages two materials' friction by default; each material also sets
PhysX's combine mode to "max" with the table at zero, so a table contact uses the
object's coefficient (the scene format's meaning). Consumers other than PhysX
ignore those two attributes.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ...core import frames
from ..model import Body, RobotScene, Shape

FLOOR_HALF_THICKNESS_M = 0.05
#: Camera film: USD focal length and aperture are in tenths of a scene unit (mm-style numbers).
VERTICAL_APERTURE = 20.955


def _f(value: float) -> str:
    return f"{float(value):.6g}"


def _tuple(values) -> str:
    return "(" + ", ".join(_f(v) for v in values) + ")"


def _array(rows) -> str:
    return "[" + ", ".join(_tuple(r) for r in rows) + "]"


def _ints(values) -> str:
    return "[" + ", ".join(str(int(v)) for v in values) + "]"


def _mesh_lines(mesh, indent: str) -> list[str]:
    return [
        f"{indent}int[] faceVertexCounts = {_ints([3] * len(mesh.faces))}",
        f"{indent}int[] faceVertexIndices = {_ints(mesh.faces.ravel())}",
        f"{indent}point3f[] points = {_array(mesh.vertices)}",
        f'{indent}uniform token subdivisionScheme = "none"',
    ]


def _collider(shape: Shape, prim: str, material: str, color, indent: str, translate=None) -> list[str]:
    schemas = '"PhysicsCollisionAPI", "MaterialBindingAPI"'
    inner = indent + "    "
    ops = [f"{inner}double3 xformOp:translate = {_tuple(translate)}"] if translate is not None else []
    order = ['"xformOp:translate"'] if translate is not None else []
    if shape.kind == "mesh":
        lines = [
            f'{indent}def Mesh "{prim}" (',
            f'{indent}    prepend apiSchemas = [{schemas}, "PhysicsMeshCollisionAPI"]',
            f"{indent})",
            f"{indent}{{",
            f'{inner}uniform token physics:approximation = "convexHull"',
            *_mesh_lines(shape.mesh, inner),
        ]
    elif shape.kind == "box":
        lines = [
            f'{indent}def Cube "{prim}" (',
            f"{indent}    prepend apiSchemas = [{schemas}]",
            f"{indent})",
            f"{indent}{{",
            f"{inner}double size = 1",
            *ops,
            f"{inner}float3 xformOp:scale = {_tuple(2 * v for v in shape.size)}",
        ]
        order.append('"xformOp:scale"')
    elif shape.kind == "sphere":
        lines = [
            f'{indent}def Sphere "{prim}" (',
            f"{indent}    prepend apiSchemas = [{schemas}]",
            f"{indent})",
            f"{indent}{{",
            f"{inner}double radius = {_f(shape.size[0])}",
        ]
    else:
        lines = [
            f'{indent}def Cylinder "{prim}" (',
            f"{indent}    prepend apiSchemas = [{schemas}]",
            f"{indent})",
            f"{indent}{{",
            f'{inner}uniform token axis = "Z"',
            f"{inner}double radius = {_f(shape.size[0])}",
            f"{inner}double height = {_f(2 * shape.size[1])}",
        ]
    if shape.kind != "box":
        lines += ops
    if order:
        lines.append(f"{inner}uniform token[] xformOpOrder = [{', '.join(order)}]")
    lines += [
        f"{inner}color3f[] primvars:displayColor = [{_tuple(color[:3])}]",
        f"{inner}rel material:binding:physics = <{material}>",
        f"{indent}}}",
    ]
    return lines


def _material(name: str, friction: float, restitution: float, indent: str) -> list[str]:
    inner = indent + "    "
    return [
        f'{indent}def Material "{name}" (',
        f'{indent}    prepend apiSchemas = ["PhysicsMaterialAPI", "PhysxMaterialAPI"]',
        f"{indent})",
        f"{indent}{{",
        f"{inner}float physics:dynamicFriction = {_f(friction)}",
        f"{inner}float physics:staticFriction = {_f(friction)}",
        f"{inner}float physics:restitution = {_f(restitution)}",
        f'{inner}uniform token physxMaterial:frictionCombineMode = "max"',
        f'{inner}uniform token physxMaterial:restitutionCombineMode = "max"',
        f"{indent}}}",
    ]


def _body(body: Body, indent: str) -> list[str]:
    inner = indent + "    "
    moments, axes = body.mass.principal
    material = f"/World/PhysicsMaterials/{body.id}"
    lines = [
        f'{indent}def Xform "{body.id}" (',
        f'{indent}    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]',
        f'{indent}    doc = "{body.label}: {body.material}, {body.mass.mass_kg * 1000:.0f} g, {body.physics_source}"',
        f"{indent})",
        f"{indent}{{",
        f"{inner}double3 xformOp:translate = {_tuple(body.position)}",
        f"{inner}quatf xformOp:orient = {_tuple(frames.wxyz(body.quat))}",
        f'{inner}uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]',
        f"{inner}float physics:mass = {_f(body.mass.mass_kg)}",
        f"{inner}point3f physics:centerOfMass = {_tuple(body.mass.com_m)}",
        f"{inner}float3 physics:diagonalInertia = {_tuple(moments)}",
        f"{inner}quatf physics:principalAxes = {_tuple(frames.wxyz(frames.matrix_to_quat(axes)))}",
        f'{inner}string darius:label = "{body.label}"',
        f'{inner}string darius:material = "{body.material}"',
    ]
    for k, shape in enumerate(body.shapes):
        prim = "collision" if len(body.shapes) == 1 else f"collision_{k}"
        lines += [""] + _collider(shape, prim, material, body.rgba, inner)
    lines.append(f"{indent}}}")
    return lines


def _camera(camera, indent: str) -> list[str]:
    inner = indent + "    "
    focal = VERTICAL_APERTURE / 2 / math.tan(math.radians(camera.fovy_deg) / 2)
    return [
        f'{indent}def Camera "{camera.name}"',
        f"{indent}{{",
        f"{inner}double3 xformOp:translate = {_tuple(camera.position)}",
        f"{inner}quatf xformOp:orient = {_tuple(frames.wxyz(camera.quat))}",
        f'{inner}uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]',
        f"{inner}float focalLength = {_f(focal)}",
        f"{inner}float verticalAperture = {_f(VERTICAL_APERTURE)}",
        f"{inner}float horizontalAperture = {_f(VERTICAL_APERTURE * camera.width / camera.height)}",
        f"{inner}float2 clippingRange = (0.01, 100)",
        f"{indent}}}",
    ]


def to_usda(scene: RobotScene) -> str:
    gravity = np.asarray(scene.gravity, float)
    magnitude = float(np.linalg.norm(gravity))
    direction = gravity / magnitude if magnitude > 0 else np.array([0.0, 0.0, -1.0])
    i1, i2 = "    ", "        "
    lines = [
        "#usda 1.0",
        "(",
        f'    doc = "{scene.title}. Written by {scene.generator} from {scene.source}. '
        '+z up, meters, table top at z = 0."',
        '    defaultPrim = "World"',
        "    metersPerUnit = 1",
        "    kilogramsPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        'def Xform "World"',
        "{",
        f'{i1}def PhysicsScene "PhysicsScene"',
        f"{i1}{{",
        f"{i2}vector3f physics:gravityDirection = {_tuple(direction)}",
        f"{i2}float physics:gravityMagnitude = {_f(magnitude)}",
        f"{i1}}}",
        "",
        f'{i1}def Scope "PhysicsMaterials"',
        f"{i1}{{",
    ]
    lines += _material("table", 0.0, 0.0, i2)
    for body in scene.bodies:
        lines += _material(body.id, body.friction, body.restitution, i2)
    lines += [f"{i1}}}", ""]
    table = Shape("table", "mesh", mesh=scene.table_mesh())
    lines += _collider(table, "Table", "/World/PhysicsMaterials/table", scene.table_rgba, i1)
    if scene.floor_depth_m is not None:
        floor = Shape("floor", "box", (5.0, 5.0, FLOOR_HALF_THICKNESS_M))
        below = (0.0, 0.0, -scene.floor_depth_m - FLOOR_HALF_THICKNESS_M)
        lines += [""] + _collider(floor, "Floor", "/World/PhysicsMaterials/table", (0.2, 0.21, 0.23), i1, below)
    lines += ["", f'{i1}def Scope "Objects"', f"{i1}{{"]
    for k, body in enumerate(scene.bodies):
        lines += ([""] if k else []) + _body(body, i2)
    lines += [f"{i1}}}"]
    if scene.cameras:
        lines += ["", f'{i1}def Scope "Cameras"', f"{i1}{{"]
        for k, camera in enumerate(scene.cameras):
            lines += ([""] if k else []) + _camera(camera, i2)
        lines += [f"{i1}}}"]
    lines += ["}", ""]
    return "\n".join(lines)


def write(scene: RobotScene, path: Path) -> Path:
    Path(path).write_text(to_usda(scene))
    return Path(path)
