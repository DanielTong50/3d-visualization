"""URDF: one file per object (and one for the table), for PyBullet, Drake, Genesis, ROS
tooling and anything else that reads the Unified Robot Description Format.

A URDF describes one kinematic tree, so a scene of free objects is a set of files:
load each at its world pose from `scene.json`'s robotics manifest (`world.json`)
with a floating base, and the table with a fixed base.

URDF has no standard place for contact parameters. Each link carries PyBullet's
<contact> extension (lateral friction, restitution), which other loaders ignore,
and a <mujoco> block so MuJoCo can compile the file directly (it needs the mesh
directory, since it drops paths from mesh file names). MuJoCo welds a URDF's root
link to the world; for free objects in MuJoCo, use scene.xml.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from ..model import Body, RobotScene
from .mjcf import vec

MESHES = "../meshes"


def _geometry(parent: ET.Element, shape) -> None:
    geometry = ET.SubElement(parent, "geometry")
    if shape.kind == "mesh":
        ET.SubElement(geometry, "mesh", filename=f"{MESHES}/{shape.name}.obj")
    elif shape.kind == "box":
        ET.SubElement(geometry, "box", size=vec(2 * v for v in shape.size))
    elif shape.kind == "sphere":
        ET.SubElement(geometry, "sphere", radius=f"{shape.size[0]:.6g}")
    else:
        ET.SubElement(geometry, "cylinder", radius=f"{shape.size[0]:.6g}", length=f"{2 * shape.size[1]:.6g}")


def _header(robot: ET.Element, text: str) -> None:
    robot.append(ET.Comment(f" {text} "))
    mujoco = ET.SubElement(robot, "mujoco")
    ET.SubElement(
        mujoco, "compiler", meshdir=f"{MESHES}/", discardvisual="true", balanceinertia="true", fusestatic="false"
    )


def object_urdf(body: Body, generator: str) -> ET.Element:
    robot = ET.Element("robot", name=body.id)
    _header(
        robot,
        f"{body.label}: {body.material}, {body.mass.mass_kg * 1000:.0f} g. Written by {generator}. "
        "Link frame: +z up, origin at the capture-time bounding-box center; place it at the pose in world.json",
    )
    link = ET.SubElement(robot, "link", name=body.id)
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz=vec(body.mass.com_m), rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{body.mass.mass_kg:.6g}")
    ixx, iyy, izz, ixy, ixz, iyz = body.mass.six()
    ET.SubElement(
        inertial,
        "inertia",
        ixx=f"{ixx:.6g}",
        ixy=f"{ixy:.6g}",
        ixz=f"{ixz:.6g}",
        iyy=f"{iyy:.6g}",
        iyz=f"{iyz:.6g}",
        izz=f"{izz:.6g}",
    )
    visual = ET.SubElement(link, "visual")
    if body.visual is not None:
        ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename=f"{MESHES}/{body.id}_visual.obj")
    else:
        _geometry(visual, body.shapes[0])
    material = ET.SubElement(visual, "material", name=f"{body.id}_color")
    ET.SubElement(material, "color", rgba=vec(body.rgba))
    for shape in body.shapes:
        _geometry(ET.SubElement(link, "collision", name=shape.name), shape)
    contact = ET.SubElement(link, "contact")
    ET.SubElement(contact, "lateral_friction", value=f"{body.friction:.4g}")
    ET.SubElement(contact, "restitution", value=f"{body.restitution:.4g}")
    return robot


def table_urdf(scene: RobotScene) -> ET.Element:
    robot = ET.Element("robot", name="table")
    _header(robot, f"The table top of {scene.title}: a static convex slab, top face at z = 0. Load with a fixed base")
    link = ET.SubElement(robot, "link", name="table")
    inertial = ET.SubElement(link, "inertial")  # a fixed base ignores it, but some loaders insist on one
    ET.SubElement(inertial, "mass", value="20")
    ET.SubElement(inertial, "inertia", ixx="1", ixy="0", ixz="0", iyy="1", iyz="0", izz="1")
    for tag in ("visual", "collision"):
        part = ET.SubElement(link, tag)
        ET.SubElement(ET.SubElement(part, "geometry"), "mesh", filename=f"{MESHES}/table.obj")
        if tag == "visual":
            ET.SubElement(ET.SubElement(part, "material", name="table_color"), "color", rgba=vec(scene.table_rgba))
    contact = ET.SubElement(link, "contact")
    ET.SubElement(contact, "lateral_friction", value="1")
    return robot


def _write(element: ET.Element, path: Path) -> Path:
    tree = ET.ElementTree(element)
    ET.indent(tree)
    with path.open("w") as handle:
        handle.write('<?xml version="1.0"?>\n')
        tree.write(handle, encoding="unicode", xml_declaration=False)
        handle.write("\n")
    return path


def write(scene: RobotScene, folder: Path) -> list[Path]:
    """Write <id>.urdf for every object, and table.urdf, into `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    written = [_write(object_urdf(body, scene.generator), folder / f"{body.id}.urdf") for body in scene.bodies]
    written.append(_write(table_urdf(scene), folder / "table.urdf"))
    return written
