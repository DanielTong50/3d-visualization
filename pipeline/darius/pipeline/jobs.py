"""Job folders and stage bookkeeping, shared by the Modal app and local runs.

A job is a folder. Each stage reads files written by earlier stages and writes
its own outputs, ending with a marker file; a stage whose marker exists is
skipped. That makes every stage individually re-runnable (`--from segment`)
without paying for VGGT again.
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


STAGES: tuple[Stage, ...] = (
    Stage("ingest", "Picking sharp frames", "frames.json"),
    Stage("reconstruct", "Finding camera positions and building 3D", "recon.json"),
    Stage("segment", "Finding the objects", "masks.npz"),
    Stage("lift", "Separating objects in 3D", "lift.json"),
    Stage("physics", "Identifying objects and their physics", "physics.json"),
    Stage("package", "Building your world", "scene/scene.json"),
)

STAGE_NAMES = [stage.name for stage in STAGES]


def stages_from(start: str | None) -> list[Stage]:
    if start is None:
        return list(STAGES)
    if start not in STAGE_NAMES:
        raise ValueError(f"unknown stage {start!r}; choose from {', '.join(STAGE_NAMES)}")
    return list(STAGES[STAGE_NAMES.index(start) :])


def invalidate(job_dir: Path, start: str) -> None:
    """Delete markers from `start` onward so those stages run again."""
    for stage in stages_from(start):
        (job_dir / stage.marker).unlink(missing_ok=True)


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    """Read every array from an .npz and close the file.

    np.load on an .npz is lazy and keeps the file open. Modal reuses warm containers
    across jobs, and an open handle makes the next job's volume reload fail
    ("there are open files preventing the operation").
    """
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


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
