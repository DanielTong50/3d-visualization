"""Run and score the perception models on the plan library."""

from __future__ import annotations

import gzip
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
from ..assembly.tracking import decode, log_emission
from ..furniture.catalog import CATALOG
from ..perception.frames import Frame, FrameReading, StepCheck
from ..perception.manual import ManualReading, ManualResult, read_manual, to_plan
from ..perception.parts import (
    Masks,
    PartTracks,
    contact_observations,
    contact_series,
    fuse,
    held_contacts,
    track_masks,
    track_parts,
)
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


def eval_manual(plan_id: str, replay: bool = False, write: bool = True) -> dict[str, Any]:
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
    if write:
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


def eval_session(plan_id: str, every_s: float = 1.0, replay: bool = False, write: bool = True) -> dict[str, Any]:
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
    if write:
        write_json(folder / "analysis.json", record)
    return record


# ---------- Part tracking (SAM 3.1) ----------

MIN_VISIBLE_PX = 12


class TrackingScore(BaseModel):
    """SAM 3.1's part tracks against the parts' exact masks, and what their contacts add to the tracker.
    The frame accuracies decode the same frames and differ only in their evidence."""

    pieces: int
    grounded: int
    letters_right: int
    mean_iou: float
    frames_tracked: float
    contacts_precision: float
    contacts_recall: float
    frames_right_spark: float
    frames_right_contacts: float
    frames_right_fused: float
    frames_right_exact_contacts: float
    step_error_fused_s: float | None


def load_ids(plan_id: str) -> tuple[np.ndarray, list[str], float]:
    """The session's exact part label maps (frames, height, width), the part ids by label, and their rate."""
    folder = LIBRARY / plan_id / "session"
    meta = json.loads((folder / "ids.json").read_text())
    data = gzip.decompress((folder / "ids.bin.gz").read_bytes())
    labels = np.frombuffer(data, np.uint8).reshape(meta["frames"], meta["height"], meta["width"])
    return labels, meta["parts"], float(meta["fps"])


def exact_masks(plan: AssemblyPlan, labels: np.ndarray, part_ids: list[str]) -> list[Masks]:
    """The label maps as masks per frame, object i + 1 being part i: what perfect tracks would give."""
    letter = {p.id: p.label for p in plan.parts}
    frames: list[Masks] = []
    for frame in labels:
        masks = {}
        for i, pid in enumerate(part_ids):
            mask = frame == i + 1
            if mask.any():
                masks[i + 1] = (letter[pid], mask)
        frames.append(masks)
    return frames


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def frame_accuracy(
    observations: list[Observation], plan: AssemblyPlan, truth: dict[str, Any]
) -> tuple[float, float | None]:
    """Decode observations and compare with when each step's parts were really in place."""
    steps = len(plan.steps)
    placed = placed_at(truth)
    timeline = decode(observations, steps)
    expected = [sum(1 for k in range(1, steps + 1) if placed.get(k, float("inf")) <= t + 1e-6) for t in timeline.times]
    right = sum(p == e for p, e in zip(timeline.path, expected, strict=True)) / max(1, len(expected))
    errors = [abs(timeline.finished_at[k] - at) for k, at in placed.items() if k in timeline.finished_at]
    return round(right, 4), (round(statistics.mean(errors), 3) if errors else None)


def match_tracks(tracks: PartTracks, frames: list[Masks], labels: np.ndarray, part_ids: list[str]) -> dict[int, str]:
    """Each track's part, matched where the tracks were prompted (the anchor frame) by mask overlap."""
    anchor = min(tracks.anchor_frame, len(frames) - 1, len(labels) - 1)
    candidates = [
        (iou(mask, labels[anchor] == i + 1), obj, pid)
        for obj, (_, mask) in frames[anchor].items()
        for i, pid in enumerate(part_ids)
    ]
    match: dict[int, str] = {}
    for overlap, obj, pid in sorted(candidates, reverse=True):
        if overlap >= 0.1 and obj not in match and pid not in match.values():
            match[obj] = pid
    return match


def score_tracking(
    tracks: PartTracks, plan: AssemblyPlan, truth: dict[str, Any], analysis: VideoAnalysis
) -> tuple[TrackingScore, dict[int, str], dict[int, float]]:
    """The score, each track's matched part, and each track's mean overlap with that part."""
    labels, part_ids, _ = load_ids(plan.id)
    letter = {p.id: p.label for p in plan.parts}
    frames = track_masks(tracks)
    match = match_tracks(tracks, frames, labels, part_ids)

    # Overlap with the part's exact mask on every frame where the part is visible.
    overlaps: dict[int, list[float]] = {obj: [] for obj in match}
    for f in range(min(len(frames), len(labels))):
        for obj, pid in match.items():
            exact = labels[f] == part_ids.index(pid) + 1
            if exact.sum() >= MIN_VISIBLE_PX:
                overlaps[obj].append(iou(frames[f][obj][1], exact) if obj in frames[f] else 0.0)
    per_track = {obj: round(statistics.mean(v), 4) for obj, v in overlaps.items() if v}
    every = [v for values in overlaps.values() for v in values]

    # The contact graph by the end of the video, against the plan's joints (through the match).
    joints = {frozenset(j.parts) for j in plan.joints}
    held = held_contacts(contact_series(frames), tracks.fps)
    named = [frozenset((match[a], match[b])) for a, b in held if a in match and b in match]
    true_held = sum(1 for pair in named if pair in joints)

    # Contacts as tracker evidence, next to Muse Spark's readings of the same frames.
    times = [f.time_s for f in analysis.frames]
    spark = [Observation(time_s=f.time_s, step_done=f.step_done, source="muse-spark") for f in analysis.frames]
    contacts = contact_observations(plan, frames, tracks.fps, times)
    exact = contact_observations(plan, exact_masks(plan, labels, part_ids), tracks.fps, times)
    fused_right, fused_error = frame_accuracy([fuse(a, b) for a, b in zip(spark, contacts, strict=True)], plan, truth)
    score = TrackingScore(
        pieces=len(plan.parts),
        grounded=len(tracks.tracks),
        letters_right=sum(1 for t in tracks.tracks if t.object in match and t.label == letter[match[t.object]]),
        mean_iou=round(statistics.mean(per_track.values()), 4) if per_track else 0.0,
        frames_tracked=round(sum(1 for v in every if v >= 0.5) / len(every), 4) if every else 0.0,
        contacts_precision=round(true_held / len(named), 4) if named else 0.0,
        contacts_recall=round(len(set(named) & joints) / len(joints), 4),
        frames_right_spark=frame_accuracy(spark, plan, truth)[0],
        frames_right_contacts=frame_accuracy(contacts, plan, truth)[0],
        frames_right_fused=fused_right,
        frames_right_exact_contacts=frame_accuracy(exact, plan, truth)[0],
        step_error_fused_s=fused_error,
    )
    return score, match, per_track


def eval_tracking(plan_id: str, replay: bool = False, write: bool = True) -> dict[str, Any]:
    """Track a library session's parts with Muse Spark + SAM 3.1 (or re-use the stored tracks) and score them."""
    folder = LIBRARY / plan_id / "session"
    path = folder / "tracks.json"
    plan = CATALOG[plan_id]()
    truth = json.loads((folder / "truth.json").read_text())
    analysis = VideoAnalysis.model_validate({k: v for k, v in stored(folder / "analysis.json").items() if k != "score"})
    if replay:
        keep = set(PartTracks.model_fields)
        tracks = PartTracks.model_validate({k: v for k, v in stored(path).items() if k in keep})
    else:
        tracks = track_parts(folder / "session.mp4", plan, parts_page=manual_pages(plan_id)[1])
    score, match, per_track = score_tracking(tracks, plan, truth, analysis)
    held = held_contacts(contact_series(track_masks(tracks)), tracks.fps)
    letter = {t.object: t.label for t in tracks.tracks}
    record = {
        "score": score.model_dump(),
        "parts": {str(obj): {"part": pid, "iou": per_track.get(obj)} for obj, pid in sorted(match.items())},
        "contacts": [
            {"objects": [a, b], "letters": [letter[a], letter[b]], "at": at}
            for (a, b), at in sorted(held.items(), key=lambda kv: kv[1])
        ],
        **tracks.model_dump(),
    }
    if write:
        write_json(path, record)
    return record


def check(plan_ids: list[str]) -> list[str]:
    """Re-score every stored result (no API calls) and list the scores that no longer match."""
    problems = []
    for plan_id in plan_ids:
        for kind, path in (
            ("manual", LIBRARY / plan_id / "manual" / "reading.json"),
            ("session", LIBRARY / plan_id / "session" / "analysis.json"),
            ("tracking", LIBRARY / plan_id / "session" / "tracks.json"),
        ):
            if not path.exists():
                continue
            stored_score = json.loads(path.read_text())["score"]
            if kind == "manual":
                fresh = eval_manual(plan_id, replay=True, write=False)["score"]
            elif kind == "session":
                fresh = eval_session(plan_id, replay=True, write=False)["score"]
            else:
                fresh = eval_tracking(plan_id, replay=True, write=False)["score"]
            if fresh != stored_score:
                problems.append(f"{plan_id} {kind}: stored {stored_score}, re-scored {fresh}")
    return problems


# ---------- Summary ----------


def summarize() -> dict[str, Any]:
    """Collect every stored result into assemblies/benchmarks.json."""
    manuals, sessions, tracking = {}, {}, {}
    model = None
    tracking_models = None
    for plan_id in CATALOG:
        reading = LIBRARY / plan_id / "manual" / "reading.json"
        if reading.exists():
            data = json.loads(reading.read_text())
            manuals[plan_id] = {**data["score"], "seconds": data["seconds"], "pages": data["pages"]}
            model = data["model"]
        tracks = LIBRARY / plan_id / "session" / "tracks.json"
        if tracks.exists():
            data = json.loads(tracks.read_text())
            refound = sum(1 for prompt in data["prompts"] if prompt["frame"] != data["anchor_frame"])
            tracking[plan_id] = {
                **data["score"],
                "seconds": data["seconds"],
                "frames": data["frames"],
                "prompts": len(data["prompts"]),
                "refound": refound,
            }
            tracking_models = {"segmentation": data["model"], "grounding": data["grounding_model"]}
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
        "tracking": {
            "models": tracking_models,
            "items": tracking,
            "mean": {
                k: mean(tracking, k)
                for k in (
                    "mean_iou",
                    "frames_tracked",
                    "contacts_recall",
                    "frames_right_spark",
                    "frames_right_contacts",
                    "frames_right_fused",
                    "frames_right_exact_contacts",
                )
            },
        },
    }
    write_json(LIBRARY / "benchmarks.json", summary)
    return summary
