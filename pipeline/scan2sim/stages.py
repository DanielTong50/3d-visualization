"""The six pipeline stages as functions of a job folder (see jobs.py for the layout).

Modal wraps each in its own function (modal_app.py); tests and local runs can
call them directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import frames, jobs


def _read(job_dir: Path, name: str) -> dict:
    return json.loads((job_dir / name).read_text())


def ingest(job_dir: Path) -> None:
    frames.extract_keyframes(jobs.find_input(job_dir), job_dir)


def reconstruct(job_dir: Path, model) -> None:
    """`model` is a loaded VGGT (reconstruct.load_model), kept warm by the caller."""
    from . import reconstruct as vggt

    info = vggt.run_vggt(model, job_dir / "frames", job_dir / "recon.npz")
    (job_dir / "recon.json").write_text(json.dumps(info, indent=2) + "\n")


def segment(job_dir: Path) -> None:
    from . import segment as seg

    recon = _read(job_dir, "recon.json")
    seg.run(job_dir, frames.load_frameset(job_dir), tuple(recon["resolution"]))


def lift(job_dir: Path) -> None:
    from . import lift as lifting

    detections = _read(job_dir, "detections.json")
    lifting.run(job_dir, {o["index"]: o["label"] for o in detections["objects"]})


def physics(job_dir: Path) -> None:
    from . import physics as phys

    phys.run(job_dir)


def package(job_dir: Path) -> None:
    from . import package as pack

    status = _read(job_dir, "status.json") if (job_dir / "status.json").exists() else {"stages": []}
    timings = [(s["name"], s["seconds"]) for s in status["stages"] if s.get("seconds")]
    meta = _read(job_dir, "meta.json") if (job_dir / "meta.json").exists() else {}
    pack.run(job_dir, timings, meta, _read(job_dir, "recon.json"))
