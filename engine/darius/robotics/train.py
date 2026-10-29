"""A small, runnable behavior-cloning *plumbing check*, requiring only NumPy.

    python -m darius.robotics.train outputs/lego-data --out outputs/lego-baseline

This fits a standardized ridge regressor on synthetic plan-conditioned Cartesian
deltas. Evaluation holds out entire episodes. It does not train a visual policy,
run a closed-loop robot or estimate insertion success. ACT training is a separate
research draft in docs/robot-training.md and training/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

INPUT_KEYS = ("observation.state", "observation.goal", "observation.phase", "observation.progress")
INPUT_SIZES = (5, 5, 5, 1)


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"Dataset path is missing or leaves its directory: {relative}")
    return path


def _load_dataset(dataset: Path) -> tuple[dict, dict[str, tuple[np.ndarray, np.ndarray]], str]:
    dataset = Path(dataset)
    manifest_path = dataset / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema_version") != "darius.synthetic.v1":
        raise ValueError("Expected a darius.synthetic.v1 dataset")
    checksums = json.loads((dataset / "checksums.json").read_text())
    if "manifest.json" not in checksums:
        raise ValueError("Dataset checksums must cover manifest.json")
    for relative, expected in checksums.items():
        actual = hashlib.sha256(_inside(dataset, relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Dataset checksum mismatch: {relative}")
    train_ids = set(manifest["split"]["train"])
    validation_ids = set(manifest["split"]["validation"])
    if not train_ids or not validation_ids or train_ids & validation_ids:
        raise ValueError("Nonempty, disjoint whole-episode train/validation splits are required")
    features: dict[str, list] = {"train": [], "validation": []}
    targets: dict[str, list] = {"train": [], "validation": []}
    seen: set[str] = set()
    seen_indices: set[int] = set()
    seen_paths: set[Path] = set()
    for episode in manifest["episode_files"]:
        episode_id = episode["episode_id"]
        split = episode["split"]
        if split not in features or episode_id in seen:
            raise ValueError("Episode identities must be unique and assigned to train or validation")
        expected_ids = train_ids if split == "train" else validation_ids
        if episode_id not in expected_ids:
            raise ValueError("Episode metadata disagrees with the split manifest")
        episode_path = _inside(dataset, episode["path"])
        if episode["episode_index"] in seen_indices or episode_path in seen_paths:
            raise ValueError("Each episode must have a unique index and file path across splits")
        seen.add(episode_id)
        seen_indices.add(episode["episode_index"])
        seen_paths.add(episode_path)
        if episode["path"] not in checksums:
            raise ValueError("All episode files must have checksums")
        rows = [json.loads(line) for line in episode_path.read_text().splitlines()]
        if not rows or len(rows) != episode["frames"]:
            raise ValueError("Episode frame count does not match metadata")
        for i, row in enumerate(rows):
            if row["frame_index"] != i or row["episode_index"] != episode["episode_index"]:
                raise ValueError("Episode/frame indices must agree with metadata")
            if bool(row["done"]) != (i == len(rows) - 1):
                raise ValueError("Only the final frame of each episode may be terminal")
            if any(len(row[key]) != size for key, size in zip(INPUT_KEYS, INPUT_SIZES, strict=True)):
                raise ValueError("Observation feature dimensions disagree with the contract")
            if len(row["action"]) != 5:
                raise ValueError("Action must have five components")
            features[split].append([v for key in INPUT_KEYS for v in row[key]])
            targets[split].append(row["action"])
    if seen != train_ids | validation_ids:
        raise ValueError("Every split episode must have exactly one episode file")
    arrays = {
        split: (np.asarray(features[split], dtype=float), np.asarray(targets[split], dtype=float)) for split in features
    }
    if any(not np.isfinite(a).all() for pair in arrays.values() for a in pair):
        raise ValueError("Dataset contains nonfinite features or actions")
    return manifest, arrays, hashlib.sha256(manifest_bytes).hexdigest()


def _metrics(predicted: np.ndarray, targets: np.ndarray) -> dict:
    error = predicted - targets
    return {
        "samples": len(targets),
        "translation_delta_rmse_mm": float(np.sqrt(np.mean(error[:, :3] ** 2)) * 1000),
        "yaw_delta_rmse_deg": float(np.degrees(np.sqrt(np.mean(error[:, 3] ** 2)))),
        "gripper_delta_rmse": float(np.sqrt(np.mean(error[:, 4] ** 2))),
        "mse_by_component": [float(v) for v in np.mean(error**2, axis=0)],
    }


def train_baseline(dataset: Path, output: Path, regularization: float = 0.01) -> dict:
    """Fit on training episodes, normalize with training data only, report held-out prediction error."""
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("regularization must be finite and positive")
    manifest, arrays, manifest_sha = _load_dataset(Path(dataset))
    train_x, train_y = arrays["train"]
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1
    design = np.column_stack(((train_x - mean) / scale, np.ones(len(train_x))))
    penalty = np.eye(design.shape[1]) * regularization
    penalty[-1, -1] = 0  # The intercept is not regularized.
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    report = {
        "schema_version": "darius.baseline.v1",
        "method": "standardized ridge behavior-cloning sanity baseline",
        "dataset_manifest_sha256": manifest_sha,
        "assembly_id": manifest["assembly_id"],
        "regularization": regularization,
        "input_keys": list(INPUT_KEYS),
        "normalization_fit": "training episodes only",
        "split": manifest["split"],
        "evaluation": {},
        "limitations": [
            "Open-loop one-step prediction on synthetic Cartesian references only.",
            "The compiler supplies the primitive goal and phase as task conditioning.",
            "Validation contains the same build with held-out translations and durations; no held-out-set claim.",
            "No learned vision, dynamics, grasping, clutch, insertion-force control or physical robot evaluation.",
            "The model must never be used as an unsupervised physical robot controller.",
        ],
    }
    for split, (x, y) in arrays.items():
        prediction = np.column_stack(((x - mean) / scale, np.ones(len(x)))) @ weights
        report["evaluation"][split] = {
            "ridge": _metrics(prediction, y),
            "zero_delta_baseline": _metrics(np.zeros_like(y), y),
            "mean_delta_baseline": _metrics(np.broadcast_to(train_y.mean(axis=0), y.shape), y),
        }
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Model directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "model.npz", weights=weights, mean=mean, scale=scale, input_keys=np.array(INPUT_KEYS))
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def predict_delta(model_path: Path, observation: dict) -> list[float]:
    """Offline model inspection. This returns a prediction, never dispatches a command."""
    x = np.asarray([v for key in INPUT_KEYS for v in observation[key]], dtype=float)
    if x.shape != (sum(INPUT_SIZES),) or not np.isfinite(x).all():
        raise ValueError("Expected 16 finite state, goal, phase and progress features")
    with np.load(model_path, allow_pickle=False) as model:
        result = np.append((x - model["mean"]) / model["scale"], 1) @ model["weights"]
    return [float(v) for v in result]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--regularization", type=float, default=0.01)
    args = parser.parse_args(argv)
    try:
        report = train_baseline(args.dataset, args.out, args.regularization)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
