"""Scan2Sim on Modal: the processing pipeline as GPU/CPU functions plus a small web API.

    modal run modal_app.py --video ../test_videos/scene1.mp4 --name scene1
        Run the whole pipeline on one video and download the scene package to
        ../scenes/scene1. Add --start segment to re-run from a later stage.

    modal deploy modal_app.py
        Deploy the web API (upload, job status, scene files, "what if") used by
        the frontend.

Secrets come from the Modal secret `scan2sim` (MODEL_API_KEY, HF_TOKEN):
    modal secret create scan2sim --from-dotenv ../.env
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import modal

APP_NAME = "scan2sim"
JOBS = Path("/jobs")
VGGT_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"  # facebookresearch/vggt main, 2026-05-19

app = modal.App(APP_NAME)
jobs_volume = modal.Volume.from_name("scan2sim-jobs", create_if_missing=True)
weights_volume = modal.Volume.from_name("scan2sim-hf-cache", create_if_missing=True)
secret = modal.Secret.from_name("scan2sim")

_ignore = ["**/__pycache__", "**/*.pyc"]

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .uv_pip_install(
        "numpy==2.4.6",
        "scipy==1.17.1",
        "opencv-python-headless==5.0.0.93",
        "pillow==13.0.0",
        "pydantic==2.14.1",
        "openai==3.39.0",
        "meta-sam-parser==0.1.0",
        "fastapi==0.162.0",
        "python-multipart==0.0.32",
    )
    .add_local_dir("scan2sim", "/root/scan2sim", ignore=_ignore)
)

# VGGT pins torch 2.3.1 and numpy<2, so it gets its own image.
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "numpy==1.26.4",
        "pillow==11.3.0",
        "huggingface_hub==0.34.4",
        "einops==0.8.1",
        "safetensors==0.6.2",
        "opencv-python-headless==4.10.0.84",
        f"vggt @ git+https://github.com/facebookresearch/vggt.git@{VGGT_COMMIT}",
    )
    .env({"HF_HOME": "/hf-cache"})
    .add_local_dir("scan2sim", "/root/scan2sim", ignore=_ignore)
)


def _job(job_id: str) -> Path:
    jobs_volume.reload()
    return JOBS / job_id


def _run_stage(name: str, job_id: str, *args) -> None:
    """Run one stage, re-raising failures as plain RuntimeErrors (or InputError, which is
    ours), because third-party exception types can't be unpickled by the caller."""
    from scan2sim import stages
    from scan2sim.frames import InputError
    from scan2sim.meta_api import describe

    try:
        getattr(stages, name)(_job(job_id), *args)
    except InputError:
        raise
    except Exception as error:
        raise RuntimeError(describe(error)) from None
    finally:
        jobs_volume.commit()


# ---------- Stages ----------


@app.function(image=cpu_image, volumes={JOBS: jobs_volume}, timeout=600)
def ingest(job_id: str) -> None:
    _run_stage("ingest", job_id)


@app.cls(
    image=gpu_image,
    gpu=["L40S", "A100-40GB", "L4"],
    volumes={JOBS: jobs_volume, "/hf-cache": weights_volume},
    secrets=[secret],
    timeout=900,
    scaledown_window=300,
    max_containers=4,
)
class Reconstructor:
    """VGGT stays loaded between videos; a cold start costs ~45 s of weight loading."""

    @modal.enter()
    def load(self) -> None:
        from scan2sim.reconstruct import load_model

        self.model = load_model()
        weights_volume.commit()

    @modal.method()
    def run(self, job_id: str) -> None:
        _run_stage("reconstruct", job_id, self.model)


@app.function(image=cpu_image, volumes={JOBS: jobs_volume}, secrets=[secret], timeout=900)
def segment(job_id: str) -> None:
    _run_stage("segment", job_id)


@app.function(image=cpu_image, volumes={JOBS: jobs_volume}, cpu=4, memory=8192, timeout=600)
def lift(job_id: str) -> None:
    _run_stage("lift", job_id)


@app.function(image=cpu_image, volumes={JOBS: jobs_volume}, secrets=[secret], timeout=600)
def physics(job_id: str) -> None:
    _run_stage("physics", job_id)


@app.function(image=cpu_image, volumes={JOBS: jobs_volume}, secrets=[secret], timeout=600)
def physics_variant(job_id: str, hint_proportions: bool, output: str) -> dict:
    """Re-run only the physics stage with a prompt variant (for the scale ablation)."""
    from scan2sim import physics as phys

    result = phys.run(_job(job_id), hint_proportions=hint_proportions, output=output)
    jobs_volume.commit()
    return result


@app.function(image=cpu_image, volumes={JOBS: jobs_volume}, timeout=300)
def package(job_id: str) -> None:
    _run_stage("package", job_id)


STAGE_FUNCTIONS = {
    "ingest": ingest,
    "reconstruct": Reconstructor().run,
    "segment": segment,
    "lift": lift,
    "physics": physics,
    "package": package,
}


@app.function(image=cpu_image, volumes={JOBS: jobs_volume}, timeout=3600)
def run_job(job_id: str, start: str | None = None) -> dict:
    """Run stages in order, skipping any whose output already exists."""
    from scan2sim.frames import InputError
    from scan2sim.jobs import Status, invalidate, stages_from

    job_dir = _job(job_id)
    if start:
        invalidate(job_dir, start)
    status = Status(job_dir)
    status.update(state="running", error=None)
    jobs_volume.commit()
    for stage in stages_from(None):
        if (job_dir / stage.marker).exists():
            if status.stage(stage.name)["state"] != "done":
                status.update(stage.name, state="done")
            continue
        status.update(stage.name, state="running")
        jobs_volume.commit()
        started = time.monotonic()
        try:
            STAGE_FUNCTIONS[stage.name].remote(job_id)
        except Exception as error:  # surfaced to the user on the processing screen
            message = str(error) if isinstance(error, InputError) else f"{stage.label} failed: {error}"
            jobs_volume.reload()
            status = Status(job_dir)
            status.update(stage.name, state="failed")
            status.update(state="failed", error=message, failed_stage=stage.name)
            jobs_volume.commit()
            raise
        jobs_volume.reload()
        status = Status(job_dir)
        status.update(stage.name, state="done", seconds=round(time.monotonic() - started, 1))
        jobs_volume.commit()
    status.update(state="done")
    jobs_volume.commit()
    return status.data


# ---------- Web API ----------


@app.function(image=cpu_image, volumes={JOBS: jobs_volume}, secrets=[secret], timeout=600)
@modal.concurrent(max_inputs=20)
@modal.asgi_app(label="scan2sim-api")
def api():
    from scan2sim.api import create_app

    def start_job(job_id: str) -> None:
        jobs_volume.commit()
        run_job.spawn(job_id)

    return create_app(JOBS, start_job=start_job, reload=jobs_volume.reload)


# ---------- CLI ----------


@app.local_entrypoint()
def main(video: str = "", name: str = "", start: str = "", job: str = "", title: str = "", source: str = ""):
    """Process one video end to end and download its scene package."""
    import json

    repo = Path(__file__).resolve().parent.parent
    if not (video or job):
        raise SystemExit("pass --video PATH, or --job ID to resume an existing job")
    job_id = job or uuid.uuid4().hex[:12]
    name = name or (Path(video).stem if video else job_id)
    if not job:
        video_path = Path(video)
        with jobs_volume.batch_upload() as batch:
            batch.put_file(video_path, f"/{job_id}/input{video_path.suffix.lower()}")
            meta = {"title": title or name.replace("_", " ").title(), "source": source or video_path.name}
            batch.put_file(_bytes_file(json.dumps(meta)), f"/{job_id}/meta.json")
        print(f"Uploaded {video_path.name} as job {job_id}")
    started = time.monotonic()
    result = run_job.remote(job_id, start or None)
    for stage in result["stages"]:
        seconds = "cached" if stage["seconds"] is None else f"{stage['seconds']:.1f} s"
        print(f"  {stage['label']:<45} {seconds}")
    print(f"Pipeline finished in {time.monotonic() - started:.0f} s")

    out = repo / "scenes" / name
    out.mkdir(parents=True, exist_ok=True)
    for entry in jobs_volume.listdir(f"/{job_id}/scene"):
        target = out / Path(entry.path).name
        with target.open("wb") as handle:
            for chunk in jobs_volume.read_file(entry.path):
                handle.write(chunk)
    debug = repo / "pipeline" / "outputs" / "jobs" / job_id
    debug.mkdir(parents=True, exist_ok=True)
    for name_ in ("status.json", "frames.json", "recon.json", "detections.json", "lift.json", "physics.json"):
        try:
            (debug / name_).write_bytes(b"".join(jobs_volume.read_file(f"/{job_id}/{name_}")))
        except FileNotFoundError:
            pass
    print(f"Scene package: {out}\nJob diagnostics: {debug}")


def _bytes_file(text: str):
    import io

    return io.BytesIO(text.encode())
