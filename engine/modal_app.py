"""Darius on Modal: the copilot's API as one CPU web function.

    modal deploy modal_app.py
        Deploy the API the web app's live mode calls: the library, /manuals, /frames, /videos
        (see darius/service/api.py). It ships with the plan library and the manuals' step pages,
        which /frames uses as references for library plans.

    modal run modal_app.py --video build.mp4 --plan halden-bookcase
        Decode one build video in the cloud and print when each step was finished.

Secrets come from the Modal secret `darius` (MODEL_API_KEY, optional DARIUS_ACCESS_CODE):
    modal secret create darius --from-dotenv ../.env
"""

from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "darius"
LIBRARY = Path("/root/assemblies")
REPO = Path(__file__).resolve().parent.parent

app = modal.App(APP_NAME)
secret = modal.Secret.from_name("darius")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "fastapi==0.162.0",
        "numpy==2.4.6",
        "openai==3.39.0",
        "opencv-python-headless==5.0.0.93",
        "pillow==13.0.0",
        "pydantic==2.14.1",
        "python-multipart==0.0.32",
    )
    .add_local_dir("darius", "/root/darius", ignore=["**/__pycache__", "**/*.pyc"])
    # Plans and manual pages only: the rendered sessions and model outputs stay out of the image.
    .add_local_dir(REPO / "assemblies", str(LIBRARY), ignore=["**/session/**", "**/reading.json", "benchmarks.json"])
)


@app.function(image=image, secrets=[secret], timeout=600, scaledown_window=300)
@modal.concurrent(max_inputs=16)
@modal.asgi_app(label="darius-api")
def api():
    from darius.service.api import create_app

    return create_app(LIBRARY)


@app.function(image=image, secrets=[secret], timeout=900)
def decode_video(video: bytes, suffix: str, plan_id: str, every_s: float) -> dict:
    import tempfile

    from darius.perception.video import analyze
    from darius.service.api import Library

    library = Library(LIBRARY)
    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        handle.write(video)
        handle.flush()
        result = analyze(
            Path(handle.name), library.plan(plan_id), references=library.step_pages(plan_id), every_s=every_s
        )
    return result.model_dump()


@app.local_entrypoint()
def main(video: str, plan: str = "halden-bookcase", every: float = 1.0) -> None:
    path = Path(video)
    result = decode_video.remote(path.read_bytes(), path.suffix.lower(), plan, every)
    print(f"{len(result['frames'])} frames read in {result['seconds']:.0f} s")
    for step, at in sorted(result["finished_at"].items(), key=lambda item: int(item[0])):
        print(f"  step {step} finished at {at:.1f} s")
