"""HTTP API used by the web app: upload a video, poll its job, fetch the scene, run "what if".

Served by Modal in production (modal_app.py::api). For local development of the
"what if" box only:

    uv run uvicorn scan2sim.api:local_app --port 8787
"""

from __future__ import annotations

import hmac
import json
import os
import re
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from . import whatif
from .jobs import Status

ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v"}
MAX_UPLOAD_BYTES = 500 * 2**20
JOB_ID = re.compile(r"^[0-9a-f]{12}$")
SCENE_FILE = re.compile(r"^[A-Za-z0-9_.-]+\.(json|ply|obj|jpg|xml|zip)$")


def create_app(
    jobs_root: Path,
    start_job: Callable[[str], None] | None = None,
    reload: Callable[[], None] = lambda: None,
) -> FastAPI:
    app = FastAPI(title="Scan2Sim API", version="0.1.0")
    access_code = os.environ.get("SCAN2SIM_ACCESS_CODE", "")

    def require_access(x_access_code: str = Header("")) -> None:
        """Endpoints that spend GPU time or API credits can be gated behind a shared code."""
        if access_code and not hmac.compare_digest(x_access_code, access_code):
            raise HTTPException(401, "An access code is required to process videos on this server.")

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

    def job_dir(job_id: str) -> Path:
        if not JOB_ID.match(job_id):
            raise HTTPException(404, "unknown job")
        reload()
        path = jobs_root / job_id
        if not path.is_dir():
            raise HTTPException(404, "unknown job")
        return path

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "uploads": start_job is not None, "access_code_required": bool(access_code)}

    @app.post("/jobs", dependencies=[Depends(require_access)])
    def create_job(video: UploadFile = File(...), title: str = Form("My tabletop")) -> dict:
        if start_job is None:
            raise HTTPException(503, "this server does not process videos")
        suffix = Path(video.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(415, "Please upload an MP4 or MOV video.")
        job_id = uuid.uuid4().hex[:12]
        path = jobs_root / job_id
        path.mkdir(parents=True)
        written = 0
        with (path / f"input{suffix}").open("wb") as out:
            while chunk := video.file.read(2**20):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    out.close()
                    shutil.rmtree(path)
                    raise HTTPException(413, "That video is over 500 MB. Trim it to about a minute.")
                out.write(chunk)
        (path / "meta.json").write_text(json.dumps({"title": title[:80], "source": "uploaded video"}))
        Status(path).update(state="queued")
        start_job(job_id)
        return {"job_id": job_id}

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        return Status(job_dir(job_id)).data

    @app.get("/jobs/{job_id}/scene/{name}")
    def scene_file(job_id: str, name: str) -> FileResponse:
        if not SCENE_FILE.match(name):
            raise HTTPException(404, "not found")
        path = job_dir(job_id) / "scene" / name
        if not path.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(path, headers={"Cache-Control": "public, max-age=300"})

    @app.post("/what-if", dependencies=[Depends(require_access)])
    def what_if(request: whatif.WhatIfRequest) -> whatif.WhatIfResponse:
        try:
            return whatif.plan(request)
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error

    return app


def _local_app() -> FastAPI:
    from dotenv import load_dotenv

    repo = Path(__file__).resolve().parents[2]
    load_dotenv(repo / ".env")
    jobs = repo / "pipeline" / "outputs" / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    return create_app(jobs)


def __getattr__(name: str):  # lazy so importing this module never reads .env
    if name == "local_app":
        return _local_app()
    raise AttributeError(name)
