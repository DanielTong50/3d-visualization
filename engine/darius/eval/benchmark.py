"""Run and score the perception models on the plan library."""

from __future__ import annotations

import json
import statistics
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel

from ..assembly.graph import compare
from ..assembly.schema import AssemblyPlan, Observation
from ..assembly.tracking import log_emission
from ..furniture.catalog import CATALOG
from ..perception.frames import Frame, FrameReading, StepCheck
from ..perception.manual import ManualReading, ManualResult, read_manual, to_plan
from ..perception.video import VideoAnalysis, analyze, analyze_readings

REPO = Path(__file__).resolve().parents[3]
LIBRARY = REPO / "assemblies"


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


# ---------- Manuals ----------


def manual_pages(plan_id: str) -> list[Image.Image]:
    paths = sorted((LIBRARY / plan_id / "manual").glob("page-*.png"))
    if not paths:
        raise FileNotFoundError(f"no manual pages for {plan_id}: run `node scripts/capture.mjs --manuals` in frontend/")
    return [Image.open(p).convert("RGB") for p in paths]


def stored(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"nothing to replay: {path} does not exist yet")
    return json.loads(path.read_text())


def eval_manual(plan_id: str, replay: bool = False) -> dict[str, Any]:
    """Read a library manual with the model (or, replaying, re-use the stored reading) and score it."""
    truth = CATALOG[plan_id]()
    path = LIBRARY / plan_id / "manual" / "reading.json"
    if replay:
        previous = stored(path)
        reading = ManualReading.model_validate(previous["reading"])
        plan, notes = to_plan(reading, plan_id)
        result = ManualResult(
            reading=reading,
            plan=plan,
            notes=notes,
            model=previous["model"],
            pages=previous["pages"],
            seconds=previous["seconds"],
            usage=previous["usage"],
        )
    else:
        result = read_manual(manual_pages(plan_id), plan_id)
    score = compare(result.plan, truth)
    record = {
        "plan": plan_id,
        "model": result.model,
        "pages": result.pages,
        "seconds": result.seconds,
        "usage": result.usage,
        "score": {
            "parts_f1": round(score.parts_f1, 4),
            "edges_precision": round(score.edges_precision, 4),
            "edges_recall": round(score.edges_recall, 4),
            "edges_f1": round(score.edges_f1, 4),
            "steps_exact": round(score.steps_exact, 4),
            "hardware_accuracy": round(score.hardware_accuracy, 4),
        },
        "notes": result.notes,
        "reading": result.reading.model_dump(),
        "recovered": result.plan.model_dump(exclude_none=True),
    }
    write_json(path, record)
    return record


# ---------- Sessions ----------


class SessionScore(BaseModel):
    """How a decoded session compares with the truth. In a rendered session nothing visible happens
    while a step is fastened, so a step counts as done once its parts are in place."""

    frames: int
    frame_accuracy: float
    raw_frame_accuracy: float
    steps_found: int
    steps: int
    step_error_s: float | None
    step_offset_s: float | None
    edges_recall: float


def frame_estimate(observation: Observation, steps: int) -> int:
    """What one frame alone says: the k that best explains it (no temporal model)."""
    return int(np.argmax(log_emission(observation, steps)))


def placed_at(truth: dict[str, Any]) -> dict[int, float]:
    """When each step's parts are in place: the end of its place phase."""
    return {s["step"]: s["start"] + s["duration"] for s in truth["segments"] if s["phase"] == "place"}


def score_session(analysis: VideoAnalysis, plan: AssemblyPlan, truth: dict[str, Any]) -> SessionScore:
    steps = len(plan.steps)
    placed = placed_at(truth)

    def truth_at(t: float) -> int:
        return sum(1 for k in range(1, steps + 1) if placed.get(k, float("inf")) <= t + 1e-6)

    times = [f.time_s for f in analysis.frames]
    expected = [truth_at(t) for t in times]
    raw = [frame_estimate(Observation(time_s=f.time_s, step_done=f.step_done), steps) for f in analysis.frames]
    errors = []
    offsets = []
    for k, at in placed.items():
        found = analysis.finished_at.get(k)
        if found is not None:
            errors.append(abs(found - at))
            offsets.append(found - at)
    final = analysis.path[-1] if analysis.path else 0
    made = sum(len(s.joints) for s in plan.steps[:final])
    return SessionScore(
        frames=len(times),
        frame_accuracy=round(sum(p == e for p, e in zip(analysis.path, expected, strict=True)) / max(1, len(times)), 4),
        raw_frame_accuracy=round(sum(r == e for r, e in zip(raw, expected, strict=True)) / max(1, len(times)), 4),
        steps_found=len(errors),
        steps=steps,
        step_error_s=round(statistics.mean(errors), 3) if errors else None,
        step_offset_s=round(statistics.median(offsets), 3) if offsets else None,
        edges_recall=round(made / max(1, len(plan.joints)), 4),
    )


def eval_session(plan_id: str, every_s: float = 1.0, replay: bool = False) -> dict[str, Any]:
    """Watch a library session with the model (or, replaying, re-decode the stored readings) and score it."""
    folder = LIBRARY / plan_id / "session"
    video, truth_path = folder / "session.mp4", folder / "truth.json"
    if not video.exists():
        raise FileNotFoundError(f"no session for {plan_id}: run `node scripts/capture.mjs --sessions` in frontend/")
    plan = CATALOG[plan_id]()
    truth = json.loads(truth_path.read_text())
    if replay:
        previous = VideoAnalysis.model_validate(
            {k: v for k, v in stored(folder / "analysis.json").items() if k != "score"}
        )
        frames = [Frame(time_s=f.time_s, image=None) for f in previous.frames]  # type: ignore[arg-type]
        readings = [
            FrameReading(
                frame=i + 1,
                orientation=f.orientation,  # type: ignore[arg-type]
                attached=f.attached,
                loose=f.loose,
                steps=[
                    StepCheck(index=k, done=p >= 0.5, confidence=p if p >= 0.5 else 1 - p)
                    for k, p in f.step_done.items()
                ],
            )
            for i, f in enumerate(previous.frames)
        ]
        analysis = analyze_readings(plan, frames, readings, previous.model, previous.every_s, previous.duration_s)
        analysis.seconds = previous.seconds
    else:
        step_pages = manual_pages(plan_id)[3:]  # cover, parts and hardware come first
        analysis = analyze(video, plan, references=step_pages, every_s=every_s)
    score = score_session(analysis, plan, truth)
    record = {"score": score.model_dump(), **analysis.model_dump()}
    write_json(folder / "analysis.json", record)
    return record


# ---------- Summary ----------


def summarize() -> dict[str, Any]:
    """Collect every stored result into assemblies/benchmarks.json."""
    manuals, sessions = {}, {}
    model = None
    for plan_id in CATALOG:
        reading = LIBRARY / plan_id / "manual" / "reading.json"
        if reading.exists():
            data = json.loads(reading.read_text())
            manuals[plan_id] = {**data["score"], "seconds": data["seconds"], "pages": data["pages"]}
            model = data["model"]
        analysis = LIBRARY / plan_id / "session" / "analysis.json"
        if analysis.exists():
            data = json.loads(analysis.read_text())
            sessions[plan_id] = {**data["score"], "seconds": data["seconds"], "duration_s": data["duration_s"]}
            model = data["model"]

    def mean(rows: dict[str, dict[str, Any]], key: str) -> float | None:
        values = [r[key] for r in rows.values() if r.get(key) is not None]
        return round(statistics.mean(values), 4) if values else None

    summary = {
        "model": model,
        "date": date.today().isoformat(),
        "manuals": {
            "items": manuals,
            "mean": {k: mean(manuals, k) for k in ("parts_f1", "edges_f1", "steps_exact", "hardware_accuracy")},
        },
        "sessions": {
            "items": sessions,
            "mean": {
                k: mean(sessions, k) for k in ("frame_accuracy", "raw_frame_accuracy", "step_error_s", "edges_recall")
            },
        },
    }
    write_json(LIBRARY / "benchmarks.json", summary)
    return summary
