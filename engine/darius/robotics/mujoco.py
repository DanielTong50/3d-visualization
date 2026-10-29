"""Export a loadable, deliberately simple MuJoCo fixture scene.

Box collision bodies and visual-only studs provide a starting point for contact
experiments. They do not implement LEGO clutch, a calibrated gripper, an arm,
inverse kinematics, or an assembly environment. Goal sites are reference poses.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .dataset import robot_position, validated_plan


def _values(values: list[float]) -> str:
    return " ".join(f"{v:.9g}" for v in values)


def scene_xml(compiled: Any) -> str:
    plan = validated_plan(compiled)
    if not plan.get("validation", {}).get("valid"):
        raise ValueError("A valid compiled LEGO plan is required for a scene")
    assembly = plan["assembly"]
    root = ET.Element("mujoco", model=f"darius_{assembly['id']}")
    ET.SubElement(root, "compiler", angle="radian")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81")
    custom = ET.SubElement(root, "custom")
    source = assembly["source"]
    ET.SubElement(
        custom,
        "text",
        name="provenance",
        data=f"{source['kind']}; review={source['review_status']}; rights={source['training_rights']}",
    )
    ET.SubElement(
        custom,
        "text",
        name="limitations",
        data="No clutch, robot arm, IK, calibrated force model or demonstrated assembly success",
    )
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", friction="0.6 0.005 0.0001", solref="0.01 1", solimp="0.9 0.95 0.001")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0.2 -0.1 0.8", dir="0 0 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "camera", name="overhead", pos="0.128 -0.064 0.65", quat="1 0 0 0")
    ET.SubElement(world, "geom", name="table", type="plane", size="0.6 0.6 0.02", rgba="0.93 0.92 0.88 1")
    nx, nz = assembly["baseplate"]["studs"]
    base_h = assembly["baseplate"]["height_mm"] / 1000
    ET.SubElement(
        world,
        "geom",
        name="baseplate",
        type="box",
        pos=_values([nx * 0.004, -nz * 0.004, base_h / 2]),
        size=_values([nx * 0.004, nz * 0.004, base_h / 2]),
        rgba="0.35 0.4 0.35 1",
    )
    # A noncolliding mocap TCP marker is a visualization reference, not a robot.
    tcp = ET.SubElement(world, "body", name="reference_tcp", mocap="true", pos="0.1 -0.1 0.15")
    ET.SubElement(tcp, "geom", type="sphere", size="0.004", rgba="0.15 0.3 0.9 0.6", contype="0", conaffinity="0")
    for brick in assembly["bricks"]:
        step = next(s for s in plan["steps"] if s["brick_id"] == brick["id"])
        h = brick["height_plates"] * 0.0032
        width, depth = brick["studs"]
        # Local brick dimensions are rotated by the body quaternion, so use the unrotated footprint here.
        grasp = next(p for p in step["primitives"] if p["kind"] == "grasp")
        source = step.get("source_center_mm") or [
            grasp["position_mm"][0],
            grasp["position_mm"][1] - h * 500 - 6,
            grasp["position_mm"][2],
        ]
        position = robot_position(source)
        source_yaw = -math.radians(step["source_yaw_deg"])
        target_yaw = -math.radians(step["target_yaw_deg"])
        body = ET.SubElement(
            world,
            "body",
            name=brick["id"],
            pos=_values(position),
            quat=_values([math.cos(source_yaw / 2), 0, 0, math.sin(source_yaw / 2)]),
        )
        ET.SubElement(body, "freejoint", name=f"free_{brick['id']}")
        rgb = [int(brick["color_hex"][i : i + 2], 16) / 255 for i in (1, 3, 5)]
        rgba = _values([*rgb, 1])
        mass = 0.0003 * width * depth  # Scaffold mass, not a measured part-property database.
        ET.SubElement(
            body,
            "geom",
            name=f"solid_{brick['id']}",
            type="box",
            size=_values([width * 0.004 - 0.0002, depth * 0.004 - 0.0002, h / 2]),
            mass=f"{mass:.6g}",
            rgba=rgba,
        )
        for ix in range(width):
            for iz in range(depth):
                ET.SubElement(
                    body,
                    "geom",
                    type="cylinder",
                    size="0.0024 0.0009",
                    pos=_values([(ix + 0.5 - width / 2) * 0.008, -(iz + 0.5 - depth / 2) * 0.008, h / 2 + 0.0009]),
                    rgba=rgba,
                    contype="0",
                    conaffinity="0",
                    mass="0",
                )
        press = next(p for p in step["primitives"] if p["kind"] == "press")
        target = step.get("target_center_mm") or [
            press["position_mm"][0],
            press["position_mm"][1] - h * 500 - 6,
            press["position_mm"][2],
        ]
        ET.SubElement(
            world,
            "site",
            name=f"goal_{brick['id']}",
            type="box",
            pos=_values(robot_position(target)),
            size=_values([width * 0.004, depth * 0.004, h / 2]),
            quat=_values([math.cos(target_yaw / 2), 0, 0, math.sin(target_yaw / 2)]),
            rgba=_values([*rgb, 0.15]),
        )
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def export_scene(compiled: Any, output: Path) -> Path:
    output = Path(output)
    xml = scene_xml(compiled)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(xml)
    return output
