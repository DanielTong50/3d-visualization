"""Darius: compile a phone video of a tabletop into a simulation-ready digital twin.

Layers, lowest first. A layer may import from the layers listed before it, never after:

    darius.core         geometry, coordinate frames, materials, cross-process errors
    darius.scene        the scene package contract: schema, asset I/O, loading
    darius.models       clients for the foundation models the pipeline calls
    darius.robotics     mass properties, grasp synthesis, MJCF / URDF / USD export, MuJoCo env
    darius.pipeline     the video-to-scene compiler: a graph of resumable stages
    darius.calibration  real-to-sim calibration: friction measured from a filmed push
    darius.service      the HTTP API and "what if" planning
    darius.eval         benchmark readers (Objectron ground truth)
    darius.tools        repository maintenance (the demo gallery)

Importing `darius` is cheap: subpackages load on first use, because each cloud
image installs only what its own stages need (the GPU images have no Pydantic).

    >>> import darius
    >>> scene = darius.load_scene("scenes/objectron-mugs")
    >>> [(o.label, o.mass_kg) for o in scene.objects]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__version__ = "0.2.0"
__all__ = ["Scene", "ScenePackage", "__version__", "load_scene"]

if TYPE_CHECKING:
    from .scene.io import ScenePackage, load_scene
    from .scene.schema import Scene


def __getattr__(name: str):
    if name == "Scene":
        from .scene.schema import Scene

        return Scene
    if name in ("ScenePackage", "load_scene"):
        from .scene import io

        return getattr(io, name)
    raise AttributeError(f"module 'darius' has no attribute {name!r}")
