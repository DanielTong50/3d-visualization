"""Export a scene package as MuJoCo MJCF, for robot learning and evaluation stacks.

    uv run python -m scan2sim.export ../scenes/objectron-mugs            # writes scene.xml there

Each object becomes a free body with its convex hull (or primitive) as the
collision geom and the estimated mass; the table is a static box. Friction maps
directly onto MuJoCo's sliding coefficient. MuJoCo has no restitution
coefficient; contacts get a soft `solref` whose damping ratio is derived from
restitution, which is the usual approximation and is documented here rather
than hidden.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

from .schema import Scene


def damping_ratio(restitution: float) -> float:
    """Damping ratio of a mass-spring contact whose bounce has coefficient `restitution`.

    For an underdamped oscillator, e = exp(-zeta * pi / sqrt(1 - zeta^2)).
    """
    e = min(max(restitution, 1e-3), 0.999)
    log_e = math.log(e)
    return -log_e / math.sqrt(math.pi**2 + log_e**2)


def _vec(values) -> str:
    return " ".join(f"{v:.5g}" for v in values)


# scene.json is y-up; MuJoCo is z-up. R_UP rotates +90 degrees about x: (x, y, z) -> (x, -z, y).
R_UP = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)  # w, x, y, z


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def z_up(point) -> list[float]:
    x, y, z = point
    return [x, -z, y]


def z_up_rotation(quat_xyzw) -> tuple[float, ...]:
    """Conjugate a y-up rotation into the z-up frame: R_UP * q * R_UP^-1 (wxyz)."""
    qx, qy, qz, qw = quat_xyzw
    inverse = (R_UP[0], -R_UP[1], -R_UP[2], -R_UP[3])
    return _qmul(_qmul(R_UP, (qw, qx, qy, qz)), inverse)


def to_mjcf(scene: Scene) -> ET.Element:
    root = ET.Element("mujoco", model=scene.scene_id)
    ET.SubElement(root, "compiler", angle="radian", meshdir=".")
    ET.SubElement(root, "option", timestep="0.002", gravity=_vec(z_up(scene.gravity)))
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 1.5", dir="0 0 -1", directional="true")

    plane = scene.background.table_plane
    sx, sz = plane.size_m
    center = z_up(plane.point)
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="box",
        size=_vec([sx / 2, sz / 2, 0.02]),
        pos=_vec([center[0], center[1], -0.02]),
        friction="1 0.005 0.0001",
        rgba="0.6 0.45 0.3 1",
    )
    if scene.background.floor_depth_m is not None:
        ET.SubElement(
            world, "geom", name="floor", type="plane", size="5 5 0.1", pos=_vec([0, 0, -scene.background.floor_depth_m])
        )
    for obj in scene.objects:
        body = ET.SubElement(
            world,
            "body",
            name=obj.id,
            pos=_vec(z_up(obj.pose.position)),
            quat=_vec(z_up_rotation(obj.pose.rotation_quat)),
        )
        body.append(ET.Comment(f" {obj.label}: {obj.material}, {obj.mass_kg * 1000:.0f} g "))
        ET.SubElement(body, "freejoint")
        geom = {
            "name": f"{obj.id}_geom",
            "mass": f"{obj.mass_kg:.5g}",
            "friction": f"{obj.friction:.4g} 0.005 0.0001",
            "solref": f"0.01 {damping_ratio(obj.restitution):.4g}",
        }
        collider = obj.collider
        if collider.type == "convex_hull":
            # Hull vertices are y-up object-local; the geom frame rotates them upright.
            ET.SubElement(asset, "mesh", name=f"{obj.id}_hull", file=collider.file)
            geom |= {"type": "mesh", "mesh": f"{obj.id}_hull", "quat": _vec(R_UP)}
        elif collider.type == "box":
            hx, hy, hz = collider.half_extents
            geom |= {"type": "box", "size": _vec([hx, hz, hy])}
        elif collider.type == "sphere":
            geom |= {"type": "sphere", "size": f"{collider.radius:.5g}"}
        else:  # y-aligned cylinder in scene.json is z-aligned here, MuJoCo's native axis
            geom |= {"type": "cylinder", "size": _vec([collider.radius, collider.half_height])}
        ET.SubElement(body, "geom", geom)
    return root


def export(scene_dir: Path, out: Path | None = None) -> Path:
    scene = Scene.model_validate(json.loads((scene_dir / "scene.json").read_text()))
    tree = ET.ElementTree(to_mjcf(scene))
    ET.indent(tree)
    out = out or scene_dir / "scene.xml"
    tree.write(out, encoding="unicode", xml_declaration=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a scene package to MuJoCo MJCF.")
    parser.add_argument("scene_dir", type=Path)
    parser.add_argument("--out", type=Path)
    print(export(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
