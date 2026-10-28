"""MuJoCo MJCF: the scene as free rigid bodies on a static table, ready for MuJoCo, MJX and
the learning stacks built on them.

What each part of the scene becomes:

- object  a body with a free joint, an explicit <inertial> (mass, center of mass and
          the full inertia tensor from `darius.robotics.mass`), and one geom per convex
          collision part. Grasp candidates are <site>s (frame: +z approach, +y closing).
- table   a static convex mesh geom (the scanned outline, 4 cm thick), top at z = 0.
- cameras the opening view and capture cameras along the filmed path.

Contacts. scene.json gives each object's friction and restitution against the table.
Object geoms have `priority` 1 and the table 0, so a table contact uses exactly the
object's parameters; between two objects MuJoCo takes the larger friction, as the
browser simulator does.

MuJoCo has no restitution coefficient; a contact is a damped spring set by solref =
(time constant, damping ratio). The damping ratio follows from restitution, zeta =
-ln e / sqrt(pi^2 + ln^2 e). The time constant must then be chosen with care: MuJoCo's
contact stiffness grows as 1 / (time constant * zeta)^2, so a bouncy object (rubber, zeta
~ 0.05) with the usual 0.01 s gets a contact that oscillates faster than the 2 ms step
can resolve, and gains energy (a dropped ball left the table at 24 m/s). The time
constant is therefore set so the contact's natural period is CONTACT_PERIOD_STEPS
timesteps for every object; a ball dropped in the exported scene rebounds to e^2 of its
drop height (tests/test_export.py).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from ...core import frames
from ..model import Body, RobotScene

TIMESTEP = 0.002
CONTACT_PERIOD_STEPS = 8  # the contact spring's natural period, in timesteps


def damping_ratio(restitution: float) -> float:
    """Damping ratio of a mass-spring contact whose bounce has coefficient `restitution`.

    For an underdamped oscillator, e = exp(-zeta * pi / sqrt(1 - zeta^2)).
    """
    e = min(max(restitution, 1e-3), 0.999)
    log_e = math.log(e)
    return -log_e / math.sqrt(math.pi**2 + log_e**2)


def contact_solref(restitution: float) -> tuple[float, float]:
    """MuJoCo solref (time constant, damping ratio) for a contact with this restitution."""
    zeta = damping_ratio(restitution)
    omega = 2 * math.pi / (CONTACT_PERIOD_STEPS * TIMESTEP)  # natural frequency of the contact
    return max(2 * TIMESTEP, 1 / (omega * zeta)), zeta


def vec(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def mesh_file(name: str) -> str:
    return f"{name}.obj"


def to_mjcf(scene: RobotScene, meshdir: str = "meshes") -> ET.Element:
    root = ET.Element("mujoco", model=scene.scene_id)
    root.append(
        ET.Comment(
            f" {scene.title}. Written by {scene.generator} from {scene.source}. "
            "World frame: +z up, meters, the table top is z = 0. "
        )
    )
    ET.SubElement(root, "compiler", angle="radian", meshdir=meshdir, autolimits="true")
    ET.SubElement(root, "option", timestep=f"{TIMESTEP}", gravity=vec(scene.gravity), integrator="implicitfast")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")

    default = ET.SubElement(root, "default")
    objects = ET.SubElement(default, "default", {"class": "object"})
    ET.SubElement(objects, "geom", condim="4", priority="1")
    grasp = ET.SubElement(default, "default", {"class": "grasp"})
    ET.SubElement(grasp, "site", type="box", size="0.002 0.004 0.012", rgba="0.1 0.6 1 0.5", group="4")

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", name="table", file=mesh_file("table"))
    for body in scene.bodies:
        for shape in body.shapes:
            if shape.kind == "mesh":
                ET.SubElement(asset, "mesh", name=shape.name, file=mesh_file(shape.name))

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", name="key", pos="0.4 -0.6 1.6", dir="-0.25 0.37 -0.9", directional="true")
    ET.SubElement(
        world, "geom", name="table", type="mesh", mesh="table", rgba=vec(scene.table_rgba), friction="1 0.005 0.0001"
    )
    if scene.floor_depth_m is not None:
        ET.SubElement(
            world,
            "geom",
            name="floor",
            type="plane",
            size="5 5 0.1",
            pos=vec([0, 0, -scene.floor_depth_m]),
            rgba="0.2 0.21 0.23 1",
        )
    for camera in scene.cameras:
        ET.SubElement(
            world,
            "camera",
            name=camera.name,
            pos=vec(camera.position),
            quat=vec(frames.wxyz(camera.quat)),
            fovy=f"{camera.fovy_deg:.4g}",
        )
    for body in scene.bodies:
        world.append(body_element(body))

    keyframe = ET.SubElement(root, "keyframe")
    qpos = [v for body in scene.bodies for v in (*body.position, *frames.wxyz(body.quat))]
    ET.SubElement(keyframe, "key", name="capture", qpos=vec(qpos))
    return root


def body_element(body: Body) -> ET.Element:
    element = ET.Element("body", name=body.id, pos=vec(body.position), quat=vec(frames.wxyz(body.quat)))
    element.append(
        ET.Comment(f" {body.label}: {body.material}, {body.mass.mass_kg * 1000:.0f} g, {body.physics_source} ")
    )
    ET.SubElement(element, "freejoint", name=body.id)
    ET.SubElement(
        element, "inertial", pos=vec(body.mass.com_m), mass=f"{body.mass.mass_kg:.6g}", fullinertia=vec(body.mass.six())
    )
    contact = {
        "class": "object",
        "friction": f"{body.friction:.4g} 0.005 0.0001",
        "solref": vec(contact_solref(body.restitution)),
        "rgba": vec(body.rgba),
    }
    for shape in body.shapes:
        geom = {"name": shape.name, **contact}
        if shape.kind == "mesh":
            geom |= {"type": "mesh", "mesh": shape.name}
        elif shape.kind == "box":
            geom |= {"type": "box", "size": vec(shape.size)}
        elif shape.kind == "sphere":
            geom |= {"type": "sphere", "size": vec(shape.size)}
        else:
            geom |= {"type": "cylinder", "size": vec(shape.size)}
        ET.SubElement(element, "geom", geom)
    for k, g in enumerate(body.grasps):
        ET.SubElement(
            element,
            "site",
            {
                "name": f"{body.id}/grasp_{k}",
                "class": "grasp",
                "pos": vec(g.position),
                "quat": vec(frames.wxyz(g.quat())),
            },
        )
    return element


def write(scene: RobotScene, path: Path, meshdir: str = "meshes") -> Path:
    tree = ET.ElementTree(to_mjcf(scene, meshdir))
    ET.indent(tree)
    tree.write(path, encoding="unicode", xml_declaration=False)
    with Path(path).open("a") as handle:
        handle.write("\n")
    return Path(path)
