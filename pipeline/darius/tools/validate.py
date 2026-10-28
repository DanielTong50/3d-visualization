"""Check a scene package against the contract (docs/scene-format.md) and basic physics.

Schema validation catches malformed manifests; this also catches what a schema
can't: missing or open meshes, an inertia tensor no rigid body can have, a center
of mass outside the object, grasps wider than they claim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import ValidationError

from ..robotics import mass
from ..scene.io import ScenePackage


def _files(package: ScenePackage) -> list[str]:
    manifest = package.manifest
    names: list[str] = []
    background = manifest.background
    for visual in (background.visual, background.backdrop, background.fallback_visual):
        if visual:
            names.append(visual.file)
    for obj in manifest.objects:
        for visual in (obj.visual, obj.fallback_visual):
            if visual:
                names.append(visual.file)
        if obj.collider.type == "convex_hull":
            names.append(obj.collider.file)
        elif obj.collider.type == "convex_parts":
            names += obj.collider.files
        if obj.pick_hull:
            names.append(obj.pick_hull)
    return names


def validate(scene_dir: Path) -> list[str]:
    """Problems found in the package at `scene_dir` (empty when it is valid)."""
    try:
        package = ScenePackage.load(scene_dir)
    except FileNotFoundError:
        return ["no scene.json"]
    except ValidationError as error:
        return [f"scene.json: {e['loc']}: {e['msg']}" for e in error.errors()]
    problems = [f"missing file {name}" for name in _files(package) if not (package.root / name).is_file()]
    if problems:
        return problems

    plane = package.manifest.background.table_plane
    if abs(plane.point[1]) > 1e-6 or not np.allclose(plane.normal, [0, 1, 0], atol=1e-6):
        problems.append("the table must be the plane y = 0 with normal +y (robotics exports rely on it)")
    for obj in package.objects:
        name = f"{obj.id} ({obj.label})"
        meshes = package.collision_meshes(obj)
        volume, _, _ = mass.volume_moments(meshes)
        if volume <= 0:
            problems.append(f"{name}: collision meshes enclose no volume (open, or wound inward)")
            continue
        extent = np.concatenate([m.vertices for m in meshes])
        lo, hi = extent.min(axis=0), extent.max(axis=0)
        if obj.inertial:
            props = mass.from_inertial(obj.mass_kg, obj.inertial)
            moments = np.linalg.eigvalsh(props.inertia_kg_m2)
            if moments.min() <= 0:
                problems.append(f"{name}: inertia tensor is not positive definite")
            elif moments.max() > moments.sum() - moments.max() + 1e-12:
                problems.append(f"{name}: principal moments violate the triangle inequality")
            if np.any(props.com_m < lo - 1e-3) or np.any(props.com_m > hi + 1e-3):
                problems.append(f"{name}: center of mass lies outside the collider's bounds")
        for k, grasp in enumerate(obj.grasps or []):
            if abs(np.linalg.norm(grasp.rotation_quat) - 1) > 1e-3:
                problems.append(f"{name}: grasp {k} has a non-unit rotation")
            if np.any(np.array(grasp.position) < lo - grasp.width_m) or np.any(
                np.array(grasp.position) > hi + grasp.width_m
            ):
                problems.append(f"{name}: grasp {k} is far from the object")
    return problems
