"""Deterministic plan -> synthetic state/action episodes with an auditable contract.

This deliberately uses inspectable JSONL rather than claiming LeRobot compatibility.
There is no camera, physics, joint state, force sensor or physical success signal.
The primitive goal is known task conditioning, not a future measured observation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from ..lego.schema import CompiledPlan

PHASES = ("move", "grasp", "release", "press", "verify")
STATE_NAMES = ("ee.x_m", "ee.y_m", "ee.z_m", "ee.yaw_rad", "gripper.open_fraction")
LIMITATIONS = [
    "Synthetic kinematic references: neither robot demonstrations nor physics rollouts.",
    "No RGB/depth, measured joint state, tactile signal, force measurement or physical success labels.",
    "Cartesian goals require calibrated robot-frame transforms, IK and collision checks before hardware use.",
    "Stud clutch, friction, insertion compliance and grasp stability are not simulated.",
    "JSONL interchange format; convert and validate separately for LeRobotDataset v3.",
]


def _dict(value: Any) -> dict:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else dict(value)


def validated_plan(value: Any) -> dict:
    """Revalidate public dict inputs, including imported-source review/permission invariants."""
    return CompiledPlan.model_validate(_dict(value)).model_dump(mode="json")


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def robot_position(position_mm: list[float]) -> list[float]:
    """Proper rotation from Darius +Y up into a right-handed, +Z up robot frame."""
    x, y, z = position_mm
    return [x / 1000, -z / 1000, y / 1000]


def _round(values: list[float]) -> list[float]:
    return [round(v, 9) for v in values]


def _episode(plan: dict, index: int, seed: int, fps: int, split: str) -> dict:
    # A distinct local RNG per episode permits reproducible generation in parallel.
    rng = random.Random(seed + index * 104729)
    translation = [rng.uniform(-0.012, 0.012), rng.uniform(-0.012, 0.012), 0.0]
    speed = rng.uniform(0.8, 1.2)
    state = [translation[0], translation[1] - 0.15, 0.15, 0.0, 1.0]
    records: list[dict] = []
    step_count = len(plan["steps"])
    for step in plan["steps"]:
        for primitive_index, primitive in enumerate(step["primitives"]):
            kind = primitive["kind"]
            if kind not in PHASES:
                raise ValueError(f"Unknown primitive kind: {kind}")
            position = robot_position(primitive["position_mm"])
            position = [v + translation[i] for i, v in enumerate(position)]
            yaw = -math.radians(primitive.get("yaw_deg", 0))
            yaw = state[3] + (yaw - state[3] + math.pi) % (2 * math.pi) - math.pi
            grip = {"open": 1.0, "closed": 0.0, "hold": state[4]}[primitive.get("gripper", "hold")]
            goal = [*position, yaw, grip]
            start = state.copy()
            samples = max(1, math.ceil(primitive["duration_s"] * speed * fps))
            for sample in range(samples):
                t = (sample + 1) / samples
                blend = t * t * (3 - 2 * t)
                next_state = [a + (b - a) * blend for a, b in zip(start, goal, strict=True)]
                action = [b - a for a, b in zip(state, next_state, strict=True)]
                records.append(
                    {
                        "episode_index": index,
                        "frame_index": len(records),
                        "timestamp": round(len(records) / fps, 9),
                        "step_index": step["index"],
                        "primitive_index": primitive_index,
                        "brick_id": step["brick_id"],
                        "observation.state": _round(state),
                        "observation.goal": _round(goal),
                        "observation.phase": [int(kind == phase) for phase in PHASES],
                        "observation.progress": [(step["index"] - 1) / step_count],
                        "action": _round(action),
                        "phase": kind,
                        "done": False,
                        "measured_force_n": None,
                        "physical_success": None,
                    }
                )
                state = next_state
    if not records:
        raise ValueError("Cannot export a plan without motion primitives")
    records[-1]["done"] = True
    episode_id = f"{plan['assembly']['id']}-{index:04d}"
    return {
        "episode_id": episode_id,
        "episode_index": index,
        "split": split,
        "seed": seed + index * 104729,
        "source": "synthetic_kinematic_reference",
        "randomization": {"scene_translation_m": _round(translation), "duration_multiplier": round(speed, 9)},
        "plan_complete": True,
        "physical_success": None,
        "records": records,
    }


def training_bundle(compiled: Any, episodes: int = 4, seed: int = 7, fps: int = 10) -> dict:
    """Return a deterministic, bounded synthetic dataset bundle for CLI/API/browser use."""
    if not isinstance(episodes, int) or isinstance(episodes, bool) or not 2 <= episodes <= 64:
        raise ValueError("episodes must be an integer between 2 and 64 (whole-episode train/validation split)")
    if not isinstance(fps, int) or isinstance(fps, bool) or not 1 <= fps <= 30:
        raise ValueError("fps must be an integer between 1 and 30")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    plan = validated_plan(compiled)
    if not plan.get("validation", {}).get("valid", False):
        raise ValueError("A valid compiled LEGO plan is required")
    if not plan.get("steps"):
        raise ValueError("A nonempty compiled LEGO plan is required")
    source = plan["assembly"].get("source", {})
    if source.get("review_status") == "pending":
        raise ValueError("Manual candidates need explicit geometric review before dataset export")
    if source.get("training_rights") not in {"project-original", "permission-documented"}:
        raise ValueError("Dataset export requires original content or a documented permission basis")
    # Bound work before materializing potentially large requests.
    reference_frames = sum(
        max(1, math.ceil(float(p["duration_s"]) * 1.2 * fps)) for s in plan["steps"] for p in s["primitives"]
    )
    if reference_frames * episodes > 250_000:
        raise ValueError("Dataset exceeds 250,000 reference frames; reduce episodes or fps")
    validation_count = max(1, math.ceil(episodes * 0.2))
    train_count = episodes - validation_count
    data = [_episode(plan, i, seed, fps, "train" if i < train_count else "validation") for i in range(episodes)]
    assembly = plan["assembly"]
    manifest = {
        "schema_version": "darius.synthetic.v1",
        "assembly_id": assembly["id"],
        "source": "synthetic_kinematic_reference",
        "execution": "simulation_only",
        "seed": seed,
        "fps": fps,
        "episodes": episodes,
        "frames": sum(len(e["records"]) for e in data),
        "assembly_sha256": _digest(assembly),
        "compiled_plan_sha256": _digest(plan),
        "provenance": assembly.get("source", {}),
        "coordinate_frame": {
            "name": "robot_reference",
            "handedness": "right",
            "up": "+Z",
            "position_units": "m",
            "angle_units": "rad",
            "from_plan_mm": "[x, -z, y] / 1000; yaw_robot = -yaw_plan",
            "calibration": "identity reference only; not a calibrated hardware frame",
        },
        "features": {
            "observation.state": {"shape": [5], "dtype": "float64", "names": list(STATE_NAMES)},
            "observation.goal": {
                "shape": [5],
                "dtype": "float64",
                "names": list(STATE_NAMES),
                "meaning": "known primitive target from the compiler, not a future measured state",
            },
            "observation.phase": {"shape": [len(PHASES)], "dtype": "int64", "names": list(PHASES)},
            "observation.progress": {"shape": [1], "dtype": "float64", "names": ["prior_steps_fraction"]},
            "action": {
                "shape": [5],
                "dtype": "float64",
                "names": [f"delta.{name}" for name in STATE_NAMES],
                "meaning": "one-frame Cartesian/yaw/gripper delta, not velocity, torque or a hardware command",
            },
        },
        "split": {
            "unit": "episode",
            "train": [e["episode_id"] for e in data if e["split"] == "train"],
            "validation": [e["episode_id"] for e in data if e["split"] == "validation"],
            "generalization": "same original assembly, held-out random scene translations and durations only",
        },
        "randomization": {
            "scene_translation_xy_m": [-0.012, 0.012],
            "duration_multiplier": [0.8, 1.2],
            "cameras_materials_friction": "not generated",
        },
        "limitations": LIMITATIONS,
        "training_draft": {
            "baseline": "python -m darius.robotics.train DATASET_DIR --out MODEL_DIR",
            "next_policy": "ACT with measured joint states and synchronized wrist/overhead RGB demonstrations",
            "acceptance": "held-out set-level physical insertion and full-build success; no claim from synthetic MSE",
        },
    }
    return {"manifest": manifest, "episodes": data}


def export_dataset(compiled: Any, output: Path, episodes: int = 8, seed: int = 7, fps: int = 10) -> dict:
    """Write reproducible JSONL episodes and SHA-256 hashes; never overwrite an existing dataset."""
    bundle = training_bundle(compiled, episodes=episodes, seed=seed, fps=fps)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Dataset directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    episode_dir = output / "episodes"
    episode_dir.mkdir()
    plan = validated_plan(compiled)
    manifest = bundle["manifest"]
    manifest["episode_files"] = []
    for episode in bundle["episodes"]:
        relative = f"episodes/{episode['episode_index']:04d}.jsonl"
        records = episode.pop("records")
        (output / relative).write_text("".join(canonical_json(row) + "\n" for row in records))
        manifest["episode_files"].append({**episode, "path": relative, "frames": len(records)})
    (output / "assembly.json").write_text(canonical_json(plan["assembly"]) + "\n")
    (output / "compiled-plan.json").write_text(canonical_json(plan) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    checksums = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    (output / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    return manifest
