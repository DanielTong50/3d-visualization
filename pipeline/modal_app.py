"""Darius on Modal: the compiler as GPU/CPU functions plus a small web API.

    modal run modal_app.py --video ../test_videos/scene1.mp4 --name scene1
        Run the whole pipeline on one video and download the scene package to
        ../scenes/scene1. Add --start segment to re-run from a later stage.

    modal deploy modal_app.py
        Deploy the web API (upload, job status, scene files, "what if") used by
        the frontend.

Secrets come from the Modal secret `darius` (MODEL_API_KEY, HF_TOKEN, optional DARIUS_ACCESS_CODE):
    modal secret create darius --from-dotenv ../.env
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import modal

APP_NAME = "darius"
JOBS = Path("/jobs")
VGGT_COMMIT = "a288dd0f14786c93483e45524328726ab7b1b4ce"  # facebookresearch/vggt main, 2026-05-19

app = modal.App(APP_NAME)
jobs_volume = modal.Volume.from_name("darius-jobs", create_if_missing=True)
weights_volume = modal.Volume.from_name("darius-hf-cache", create_if_missing=True)
secret = modal.Secret.from_name("darius")

_ignore = ["**/__pycache__", "**/*.pyc"]

SPZ_COMMIT = "affd0ecea7fbb4c265ee119475af7ee5b2997482"  # nianticlabs/spz, Python bindings v1.1.0
SPZ = f"spz @ git+https://github.com/nianticlabs/spz.git@{SPZ_COMMIT}"

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "build-essential", "zlib1g-dev")
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
        "coacd==1.0.14",
        SPZ,
    )
    .add_local_dir("darius", "/root/darius", ignore=_ignore)
)

# gsplat publishes prebuilt CUDA wheels for Python 3.10 + torch 2.4, so splat training
# gets its own image (and compiles nothing at runtime).
splat_image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install(
        "torch==2.4.1",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "pillow==11.3.0",
        "gsplat==1.5.3",
        "packaging==24.2",  # imported by torch's extension loader, undeclared by gsplat
        extra_index_url="https://docs.gsplat.studio/whl/pt24cu121",
        extra_options="--index-strategy unsafe-best-match",
    )
    .add_local_dir("darius", "/root/darius", ignore=_ignore)
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
    .add_local_dir("darius", "/root/darius", ignore=_ignore)
)


def _job(job_id: str) -> Path:
    jobs_volume.reload()
    return JOBS / job_id


def _run_stage(name: str, job_id: str, *args) -> None:
    """Run one stage, re-raising failures as plain RuntimeErrors (or InputError, which is
    ours), because third-party exception types can't be unpickled by the caller."""
    from darius.core.errors import InputError, describe
    from darius.pipeline import stages

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
        from darius.pipeline.reconstruct import load_model

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
    from darius.pipeline import physics as phys

    result = phys.run(_job(job_id), hint_proportions=hint_proportions, output=output)
    jobs_volume.commit()
    return result


@app.function(image=splat_image, gpu=["L40S", "A100-40GB"], volumes={JOBS: jobs_volume}, timeout=1200)
def splat(job_id: str) -> None:
    _run_stage("splat", job_id)


@app.function(image=splat_image, gpu=["L40S", "A100-40GB"], volumes={JOBS: jobs_volume}, timeout=300)
def background_preview(job_id: str, overrides: dict | None = None) -> bytes:
    """Keyframe next to a render of the scene with every object removed (docs figure)."""
    from darius.pipeline.splats import render_background_preview

    return render_background_preview(_job(job_id), overrides=overrides).read_bytes()


@app.function(image=cpu_image, volumes={JOBS: jobs_volume}, cpu=4, memory=8192, timeout=900)
def package(job_id: str) -> None:
    _run_stage("package", job_id)


STAGE_FUNCTIONS = {
    "ingest": ingest,
    "reconstruct": Reconstructor().run,
    "segment": segment,
    "lift": lift,
    "physics": physics,
    "splat": splat,
    "package": package,
}


@app.function(image=splat_image, gpu=["L40S", "A100-40GB"], volumes={JOBS: jobs_volume}, timeout=1200)
def splat_variant(job_id: str, strategy: str, antialiased: bool, output: str) -> dict:
    """Train splats with other settings into `<output>.*`, leaving the job's own untouched (ablations)."""
    from darius.pipeline.splats import train

    try:
        return train(_job(job_id), strategy_name=strategy, antialiased=antialiased, output=output)
    finally:
        jobs_volume.commit()


@app.function(image=cpu_image, volumes={JOBS: jobs_volume}, timeout=3600)
def run_job(job_id: str, start: str | None = None) -> dict:
    """Run the stage graph, each stage as soon as the stages it needs are done.

    Stages whose output already exists are skipped. Reconstruction and segmentation
    run side by side, and so do physics and splat training. Only this orchestrator
    writes status.json, so concurrent stages never race on it.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    from darius.core.errors import InputError
    from darius.pipeline.jobs import STAGES, Status, invalidate, ready_stages

    job_dir = _job(job_id)
    if start:
        invalidate(job_dir, start)
    status = Status(job_dir)
    status.update(state="running", error=None, failed_stage=None)
    done = {stage.name for stage in STAGES if (job_dir / stage.marker).exists()}
    for name in done:
        if status.stage(name)["state"] != "done":
            status.update(name, state="done")
    jobs_volume.commit()

    running: dict = {}  # future -> (stage, started)
    failure: tuple | None = None
    with ThreadPoolExecutor(max_workers=len(STAGES)) as pool:
        while True:
            if failure is None:
                for stage in ready_stages(done, {s.name for s, _ in running.values()}):
                    status.update(stage.name, state="running", started=round(time.time(), 1))
                    running[pool.submit(STAGE_FUNCTIONS[stage.name].remote, job_id)] = (stage, time.monotonic())
                jobs_volume.commit()
            if not running:
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            jobs_volume.reload()
            status = Status(job_dir)
            for future in finished:
                stage, started = running.pop(future)
                try:
                    future.result()
                except Exception as error:  # surfaced to the user on the processing screen
                    message = str(error) if isinstance(error, InputError) else f"{stage.label} failed: {error}"
                    status.update(stage.name, state="failed")
                    failure = failure or (stage, message, error)
                    continue
                done.add(stage.name)
                status.update(stage.name, state="done", seconds=round(time.monotonic() - started, 1))
            jobs_volume.commit()
    if failure is not None:
        stage, message, error = failure
        status.update(state="failed", error=message, failed_stage=stage.name)
        jobs_volume.commit()
        raise error
    status.update(state="done", finished=round(time.time(), 1))
    jobs_volume.commit()
    return status.data


# ---------- Web API ----------


@app.function(image=cpu_image, volumes={JOBS: jobs_volume}, secrets=[secret], timeout=600)
@modal.concurrent(max_inputs=20)
@modal.asgi_app(label="darius-api")
def api():
    from darius.service.api import create_app

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
    download_scene(job_id, out)
    debug = repo / "pipeline" / "outputs" / "jobs" / job_id
    debug.mkdir(parents=True, exist_ok=True)
    for name_ in (
        "status.json",
        "frames.json",
        "recon.json",
        "detections.json",
        "lift.json",
        "physics.json",
        "splats.json",
        "package.json",
    ):
        try:
            (debug / name_).write_bytes(b"".join(jobs_volume.read_file(f"/{job_id}/{name_}")))
        except FileNotFoundError:
            pass
    print(f"Scene package: {out}\nJob diagnostics: {debug}")


def download_scene(job_id: str, out: Path) -> None:
    """Copy a job's scene package (including its pipeline/ replay folder) into `out`."""
    root = f"{job_id}/scene/"
    for entry in jobs_volume.listdir(f"/{job_id}/scene", recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        target = out / entry.path.lstrip("/").removeprefix(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            for chunk in jobs_volume.read_file(entry.path):
                handle.write(chunk)


def _bytes_file(text: str):
    import io

    return io.BytesIO(text.encode())
