"""Job folders and stage bookkeeping, shared by the Modal app and local runs.

A job is a folder. Each stage reads files written by the stages it needs and
writes its own outputs, ending with a marker file; a stage whose marker exists is
skipped. That makes every stage individually re-runnable (`--start segment`)
without paying for VGGT again.

Stages form a small graph rather than a line: reconstruction (GPU) and
segmentation (API calls) only need the keyframes, so they run at the same time,
and so do physics and splat training once objects are lifted into 3D.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Stage:
    name: str
    label: str  # shown to the user on the processing screen
    marker: str  # file whose existence means the stage is done
    needs: tuple[str, ...] = ()  # stages whose outputs this one reads


STAGES: tuple[Stage, ...] = (
    Stage("ingest", "Picking sharp frames", "frames.json"),
    Stage("reconstruct", "Finding camera positions and building 3D", "recon.json", ("ingest",)),
    Stage("segment", "Finding the objects", "masks.npz", ("ingest",)),
    Stage("lift", "Separating objects in 3D", "lift.json", ("reconstruct", "segment")),
    Stage("physics", "Identifying objects and their physics", "physics.json", ("lift",)),
    Stage("splat", "Rendering photo-real appearance", "splats.json", ("lift",)),
    Stage("package", "Building your world", "scene/scene.json", ("physics", "splat")),
)

STAGE_NAMES = [stage.name for stage in STAGES]


def stages_from(start: str | None) -> list[Stage]:
    """`start` and every stage that depends on it, directly or not (all stages for None)."""
    if start is None:
        return list(STAGES)
    if start not in STAGE_NAMES:
        raise ValueError(f"unknown stage {start!r}; choose from {', '.join(STAGE_NAMES)}")
    affected = {start}
    for stage in STAGES:  # registry order is a topological order
        if any(need in affected for need in stage.needs):
            affected.add(stage.name)
    return [stage for stage in STAGES if stage.name in affected]


def ready_stages(done: set[str], started: set[str]) -> list[Stage]:
    """Stages not yet started whose inputs are all done."""
    return [s for s in STAGES if s.name not in done | started and all(n in done for n in s.needs)]


def invalidate(job_dir: Path, start: str) -> None:
    """Delete the markers of `start` and its dependents so those stages run again."""
    for stage in stages_from(start):
        (job_dir / stage.marker).unlink(missing_ok=True)


def load_arrays(path: Path, keys: list[str] | None = None) -> dict[str, np.ndarray]:
    """Read every array from an .npz and close the file.

    np.load on an .npz is lazy and keeps the file open. Modal reuses warm containers
    across jobs, and an open handle makes the next job's volume reload fail
    ("there are open files preventing the operation").
    """
    with np.load(path) as archive:
        return {name: archive[name] for name in (keys or archive.files)}


def find_input(job_dir: Path) -> Path:
    for path in sorted(job_dir.glob("input.*")):
        return path
    raise FileNotFoundError(f"no input video in {job_dir}")


class Status:
    """status.json: what the processing screen polls."""

    def __init__(self, job_dir: Path):
        self.path = job_dir / "status.json"
        if self.path.exists():
            self.data: dict[str, Any] = json.loads(self.path.read_text())
            # Jobs created before a stage was added get it, in registry order.
            known = {s["name"]: s for s in self.data["stages"]}
            self.data["stages"] = [
                known.get(s.name, {"name": s.name, "label": s.label, "state": "pending", "seconds": None})
                for s in STAGES
            ]
        else:
            self.data = {
                "job_id": job_dir.name,
                "state": "queued",
                "created": time.time(),
                "stages": [{"name": s.name, "label": s.label, "state": "pending", "seconds": None} for s in STAGES],
                "error": None,
            }

    def stage(self, name: str) -> dict[str, Any]:
        return next(s for s in self.data["stages"] if s["name"] == name)

    def update(self, name: str | None = None, **fields: Any) -> None:
        target = self.stage(name) if name else self.data
        target.update(fields)
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")
