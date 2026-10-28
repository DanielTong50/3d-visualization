"""Evaluate the pipeline against Objectron's metric ground truth.

Objectron's phone videos come with ARKit-measured 3D boxes, so every clip tells
us whether we found each object, separated it in 3D, named it correctly, and
recovered its real size (which tests the whole scale-estimation chain).

    uv run python scripts/evaluate_objectron.py --per-class 4
    uv run python scripts/evaluate_objectron.py --report-only   # re-score cached jobs
    uv run python scripts/evaluate_objectron.py --start physics # re-run later stages only

Needs the app deployed (`modal deploy modal_app.py`). Results go to
docs/results/objectron.json and are summarized in docs/results.md.
Cost: roughly $0.05 per clip (GPU seconds + Meta API calls).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
import numpy as np

from darius.eval.objectron import parse_sequence

PIPELINE = Path(__file__).resolve().parents[1]
REPO = PIPELINE.parent
CACHE = PIPELINE / "outputs" / "objectron"
RESULTS = REPO / "docs" / "results" / "objectron.json"
BUCKET = "https://storage.googleapis.com/objectron"
VIDEO_FPS = 30  # Objectron clips are 30 fps

# Tabletop-sized Objectron classes, with words that count as a correct label.
CLASSES = {
    "cup": ("cup", "mug", "tumbler", "glass"),
    "book": ("book", "novel", "paperback", "hardcover", "notebook"),
    "bottle": ("bottle",),
    "camera": ("camera",),
    "cereal_box": ("box", "cereal"),
    "shoe": ("shoe", "sneaker", "boot", "sandal", "heel", "loafer", "trainer", "slipper"),
}
MATCH_RADIUS = 0.12  # normalized image distance between GT and detected box centers


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
    return path


def clip_list(per_class: int) -> list[str]:
    clips = []
    for name in CLASSES:
        index = fetch(f"{BUCKET}/v1/index/{name}_annotations_test", CACHE / f"index_{name}.txt")
        clips += index.read_text().split()[:per_class]
    return clips


def job_id_for(clip: str) -> str:
    """Stable 12-hex id per clip, so re-runs reuse cached stages on the volume."""
    import hashlib

    return hashlib.sha1(f"objectron:{clip}".encode()).hexdigest()[:12]


def submit(clips: list[str], volume: modal.Volume) -> dict[str, str]:
    jobs = {}
    with volume.batch_upload(force=True) as batch:
        for clip in clips:
            job = job_id_for(clip)
            jobs[clip] = job
            local = fetch(f"{BUCKET}/videos/{clip}/video.MOV", CACHE / clip.replace("/", "_") / "video.MOV")
            fetch(f"{BUCKET}/annotations/{clip}.pbdata", local.parent / "annotation.pbdata")
            batch.put_file(local, f"/{job}/input.mov")
            meta = {"title": clip, "source": f"Objectron {clip}", "license": "C-UDA-1.0"}
            meta_path = local.parent / "meta.json"
            meta_path.write_text(json.dumps(meta))
            batch.put_file(meta_path, f"/{job}/meta.json")
    return jobs


def read_json(volume: modal.Volume, path: str) -> dict | None:
    try:
        return json.loads(b"".join(volume.read_file(path)))
    except Exception:
        return None


def label_correct(clip_class: str, *labels: str) -> bool:
    text = " ".join(labels).lower()
    return any(word in text for word in CLASSES[clip_class])


def score_clip(clip: str, job: str, volume: modal.Volume, variants: list[str]) -> dict:
    annotation = parse_sequence((CACHE / clip.replace("/", "_") / "annotation.pbdata").read_bytes())
    clip_class = clip.split("/")[0]
    status = read_json(volume, f"/{job}/status.json") or {}
    record = {
        "clip": clip,
        "job": job,
        "class": clip_class,
        "state": status.get("state"),
        "error": status.get("error"),
        "gt_objects": len(annotation.objects),
    }
    detections = read_json(volume, f"/{job}/detections.json")
    frames = read_json(volume, f"/{job}/frames.json")
    if not detections or not frames:
        return record

    # Match each GT object to the nearest detection at the keyframe, in image space.
    t = frames["source_seconds"][detections["keyframe"]]
    frame_id = min(annotation.centers, key=lambda f: abs(f - t * VIDEO_FPS))
    width, height = frames["size"]
    matches = {}
    for gt in annotation.objects:
        center = annotation.centers[frame_id].get(gt.id)
        if center is None:
            continue
        best, best_d = None, MATCH_RADIUS
        for det in detections["objects"]:
            x0, y0, x1, y1 = det["box"]
            d = np.hypot((x0 + x1) / 2 / width - center[0], (y0 + y1) / 2 / height - center[1])
            if d < best_d and det["index"] not in matches.values():
                best, best_d = det["index"], d
        if best is not None:
            matches[gt.id] = best
    record["detected"] = len(matches)
    record["timings"] = {s["name"]: s["seconds"] for s in status.get("stages", [])}
    record["splats"] = read_json(volume, f"/{job}/splats.json")
    record["colliders"] = (read_json(volume, f"/{job}/package.json") or {}).get("colliders")

    for variant in variants:
        physics = read_json(volume, f"/{job}/{variant}")
        if not physics:
            continue
        by_index = {o["index"]: o for o in physics["objects"]}
        objects = []
        for gt in annotation.objects:
            index = matches.get(gt.id)
            if index is None or index not in by_index:
                continue
            pred = by_index[index]
            gt_dims = np.array(gt.scale_m)  # x, y (up), z
            pred_dims = np.array(pred["measured_dimensions_cm"]) / 100  # length, height, width
            objects.append(
                {
                    "label": pred["label"],
                    "category": pred.get("category"),
                    "label_correct": label_correct(clip_class, pred["label"], pred.get("category", "")),
                    "gt_dims_m": gt_dims.round(4).tolist(),
                    "pred_dims_m": pred_dims.round(4).tolist(),
                    # Objectron boxes live in the object's canonical frame (a cereal box's y is
                    # always its tall side, even lying down), so sizes are compared as sorted
                    # extents, independent of orientation. size_ratio is their geometric mean.
                    "size_ratio": float(np.exp(np.mean(np.log(np.sort(pred_dims) / np.sort(gt_dims))))),
                    "sorted_dims_rel_error": float(
                        np.mean(np.abs(np.sort(pred_dims) - np.sort(gt_dims)) / np.sort(gt_dims))
                    ),
                    "mass_kg": pred["mass_kg"],
                }
            )
        record[variant] = {"separated": len(objects), "objects": objects, "scale": physics["scale_m_per_unit"]}
    return record


def summarize(records: list[dict], variant: str) -> dict:
    objects = [o for r in records for o in r.get(variant, {}).get("objects", [])]
    gt = sum(r["gt_objects"] for r in records)
    ratios = np.array([o["size_ratio"] for o in objects])
    errors = np.array([o["sorted_dims_rel_error"] for o in objects])
    return {
        "clips": len(records),
        "clips_completed": sum(r.get("state") == "done" for r in records),
        "gt_objects": gt,
        "detected": sum(r.get("detected", 0) for r in records),
        "separated": len(objects),
        "label_accuracy": float(np.mean([o["label_correct"] for o in objects])) if objects else None,
        "median_size_ratio": float(np.median(ratios)) if len(ratios) else None,
        "median_abs_log2_size_ratio": float(np.median(np.abs(np.log2(ratios)))) if len(ratios) else None,
        "within_25pct_size": float(np.mean(np.abs(ratios - 1) <= 0.25)) if len(ratios) else None,
        "median_dims_rel_error": float(np.median(errors)) if len(errors) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--start", help="re-run every clip from this stage onward (e.g. physics)")
    args = parser.parse_args()

    volume = modal.Volume.from_name("darius-jobs")
    clips = clip_list(args.per_class)
    jobs = {clip: job_id_for(clip) for clip in clips}
    variants = ["physics.json", "physics_hinted.json"]

    if not args.report_only:
        jobs = submit(clips, volume)
        run_job = modal.Function.from_name("darius", "run_job")
        variant_fn = modal.Function.from_name("darius", "physics_variant")
        started = time.monotonic()
        print(f"Running {len(jobs)} clips...")

        def run(item, start=args.start):
            clip, job = item
            try:
                run_job.remote(job, start)
                variant_fn.remote(job, True, "physics_hinted.json")
                return clip, None
            except Exception as error:
                return clip, str(error)[:200]

        failed = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            for clip, error in pool.map(run, jobs.items()):
                print(f"  {clip:<22} {'ok' if error is None else 'FAILED: ' + error}")
                if error:
                    failed[clip] = jobs[clip]
        # Second pass, one at a time: stages resume where they stopped, so this only
        # repeats work lost to transient API errors (see docs/api-notes.md).
        for clip, job in failed.items():
            _, error = run((clip, job), start=None)
            print(f"  retry {clip:<16} {'ok' if error is None else 'FAILED: ' + error}")
        print(f"Done in {time.monotonic() - started:.0f} s")

    volume = modal.Volume.from_name("darius-jobs")
    records = [score_clip(clip, job, volume, variants) for clip, job in jobs.items()]
    summary = {v: summarize(records, v) for v in variants}
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({"summary": summary, "clips": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
