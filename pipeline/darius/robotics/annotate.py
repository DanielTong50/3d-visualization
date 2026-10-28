"""Robotics annotations in scene.json: each object's inertial and grasp candidates.

They are derived data (from the collider, the mass and the friction), stored in the
manifest so every consumer sees the same values: the browser draws them, and the
exporters write them. The pipeline's package stage annotates new scenes, and
`darius gallery` re-annotates every shipped scene.
"""

from __future__ import annotations

from ..scene.io import ScenePackage
from ..scene.schema import Scene
from . import grasps as grasping
from . import mass


def annotate(package: ScenePackage, gripper: grasping.Gripper = grasping.Gripper()) -> Scene:
    """The package's manifest with every object's `inertial` and `grasps` (re)computed."""
    objects = []
    for obj in package.objects:
        props = mass.for_object(package, obj)
        found = grasping.for_object(package, obj, props.com_m, gripper)
        objects.append(
            obj.model_copy(
                update={
                    "inertial": mass.inertial(props, obj.hollow, obj.fill_fraction),
                    "grasps": [g.spec() for g in found],
                }
            )
        )
    return package.manifest.model_copy(update={"objects": objects})
