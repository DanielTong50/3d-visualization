"""LEGO assembly intelligence API, with the historical furniture perception endpoints.

    GET  /lego/library                    original LEGO build summaries
    GET  /lego/builds/{id}                compiled graph, source/target poses and nominal primitives
    POST /lego/compile                    validate and compile a custom source assembly
    GET  /lego/instructions/{set_number}  official instruction metadata and PDF links
    POST /lego/manual-drafts              page images -> unapproved vision-language candidate
    GET  /lego/builds/{id}/training       bounded, seeded synthetic Cartesian episodes

    GET  /health               version, model, whether an access code is required
    GET  /library              the furniture library (plan summaries)
    GET  /library/{id}         one library plan
    POST /manuals              manual page images -> the plan read from them (and the raw reading)
    POST /frames               camera frames + a plan -> per-frame readings and tracker evidence
    POST /videos               a build video + a plan -> decoded timeline and the graph's history

The POST endpoints call the Meta Model API, which costs credits, so a deployment can gate them
behind a shared code (DARIUS_ACCESS_CODE, sent as the X-Access-Code header). The tracker itself
runs in the client (frontend/src/engine/tracking.ts), so /frames is stateless.

Local development:

    uv run uvicorn darius.service.api:local_app --factory --port 8787
"""

from __future__ import annotations

import hmac
import io
import json
import os
import re
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from .. import __version__
from ..assembly.graph import validate
from ..assembly.schema import AssemblyPlan, Observation
from ..core.errors import InputError
from ..lego import CompiledPlan, LegoAssembly, compile_assembly, get_assembly
from ..lego import library_index as lego_library_index
from ..lego.instructions import InstructionLookupError, InstructionResult, lookup_instructions
from ..perception.frames import Frame, FrameReading, to_observation, verify_frames
from ..perception.manual import ManualResult, read_manual
from ..perception.video import VideoAnalysis, analyze

PLAN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
MAX_IMAGE_BYTES = 15 * 2**20
MAX_VIDEO_BYTES = 300 * 2**20
MAX_FRAMES = 8
MAX_REFERENCES = 30
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm"}


def _image(upload: UploadFile) -> Image.Image:
    data = upload.file.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(413, f"{upload.filename or 'an image'} is over 15 MB")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(415, f"{upload.filename or 'a file'} is not an image we can read") from error
    return image.convert("RGB")


class Library:
    """The plan library on disk (assemblies/): summaries, plans and their manuals' step pages."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def index(self) -> list[dict]:
        path = self.root / "index.json"
        return json.loads(path.read_text()) if path.exists() else []

    def plan(self, plan_id: str) -> AssemblyPlan:
        if not PLAN_ID.match(plan_id) or not (self.root / plan_id / "plan.json").is_file():
            raise HTTPException(404, f"no plan {plan_id!r} in the library")
        return AssemblyPlan.model_validate_json((self.root / plan_id / "plan.json").read_text())

    def step_pages(self, plan_id: str) -> list[Image.Image]:
        """The manual's step pages (the cover, parts and hardware pages come first)."""
        pages = sorted((self.root / plan_id / "manual").glob("page-*.png"))[3:]
        out = []
        for path in pages:
            image = Image.open(path).convert("RGB")
            image.thumbnail((900, 900))
            out.append(image)
        return out


def create_app(library_root: Path) -> FastAPI:
    app = FastAPI(title="Darius LEGO Assembly Intelligence API", version=__version__)
    library = Library(library_root)
    access_code = os.environ.get("DARIUS_ACCESS_CODE", "")

    def require_access(x_access_code: str = Header("")) -> None:
        """Endpoints that spend model credits can be gated behind a shared code."""
        if access_code and not hmac.compare_digest(x_access_code, access_code):
            raise HTTPException(401, "This server needs an access code for model calls.")

    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

    def resolve_plan(plan_id: str, plan: str) -> AssemblyPlan:
        if plan:
            try:
                parsed = AssemblyPlan.model_validate_json(plan)
            except ValidationError as error:
                raise HTTPException(422, f"the plan is not valid: {error.errors()[0]['msg']}") from error
            if not parsed.steps:
                raise HTTPException(422, "the plan has no steps")
            return parsed
        if plan_id:
            return library.plan(plan_id)
        raise HTTPException(422, "send a plan (JSON) or a library plan_id")

    @app.get("/health")
    def health() -> dict:
        from ..models.meta_api import MUSE_SPARK_MODEL

        return {
            "ok": True,
            "version": __version__,
            "model": MUSE_SPARK_MODEL,
            "access_code_required": bool(access_code),
            "library": [p["id"] for p in library.index()],
            "lego_library": [p["id"] for p in lego_library_index()],
            "robot_execution": "simulation_only",
        }

    @app.get("/library")
    def library_index() -> list[dict]:
        return library.index()

    @app.get("/library/{plan_id}")
    def library_plan(plan_id: str) -> AssemblyPlan:
        return library.plan(plan_id)

    @app.get("/lego/library")
    def lego_library() -> list[dict]:
        """Original LEGO-compatible curriculum, compiled locally without model credits."""
        return lego_library_index()

    def lego_plan(build_id: str) -> CompiledPlan:
        try:
            return compile_assembly(get_assembly(build_id))
        except KeyError as error:
            raise HTTPException(404, str(error)) from error

    @app.get("/lego/builds/{build_id}")
    def lego_build(build_id: str) -> CompiledPlan:
        return lego_plan(build_id)

    @app.post("/lego/compile")
    def lego_compile(assembly: LegoAssembly) -> CompiledPlan:
        """Validate and compile an original/reviewed task into guarded reference primitives."""
        return compile_assembly(assembly)

    @app.get("/lego/instructions/{set_number}")
    def lego_instructions(set_number: str, refresh: bool = False) -> InstructionResult:
        """Official public instruction metadata; does not semantically reconstruct the PDF."""
        try:
            return lookup_instructions(set_number, refresh=refresh)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except InstructionLookupError as error:
            raise HTTPException(502, str(error)) from error

    @app.get("/lego/builds/{build_id}/training")
    def lego_training(
        build_id: str,
        episodes: int = Query(4, ge=2, le=64),
        seed: int = Query(7, ge=0, le=2**32 - 1),
        fps: int = Query(10, ge=1, le=30),
    ) -> dict:
        from ..robotics.dataset import training_bundle

        try:
            return training_bundle(lego_plan(build_id), episodes=episodes, seed=seed, fps=fps)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/lego/manual-drafts", dependencies=[Depends(require_access)])
    def lego_manual_drafts(
        pages: list[UploadFile] = File(...),
        build_id: str = Form("manual-draft"),
        source_url: str = Form(""),
    ) -> dict:
        """Optional model inference into an uncertain candidate; always requires manual review."""
        from ..lego.manual import read_manual_draft

        if not 1 <= len(pages) <= 30:
            raise HTTPException(422, "send between 1 and 30 ordered page images")
        if not PLAN_ID.fullmatch(build_id):
            raise HTTPException(422, "build_id: lowercase letters, digits and dashes")
        try:
            return read_manual_draft(
                [_image(page) for page in pages], build_id, source_url=source_url or None
            ).model_dump(mode="json")
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error

    @app.post("/manuals", dependencies=[Depends(require_access)])
    def manuals(pages: list[UploadFile] = File(...), plan_id: str = Form("manual")) -> ManualResult:
        """Read a manual's pages (in order) into a plan."""
        if not 1 <= len(pages) <= 50:
            raise HTTPException(422, "send between 1 and 50 pages")
        if not PLAN_ID.match(plan_id):
            raise HTTPException(422, "plan_id: lowercase letters, digits and dashes")
        images = [_image(p) for p in pages]
        try:
            return read_manual(images, plan_id)
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error

    @app.post("/frames", dependencies=[Depends(require_access)])
    def frames(
        frames: list[UploadFile] = File(...),
        times: str = Form("", description="Comma-separated seconds of each frame in the session"),
        plan_id: str = Form(""),
        plan: str = Form("", description="A plan as JSON, e.g. one read by /manuals"),
        references: list[UploadFile] = File(default=[], description="Step pages of the manual, in order"),
    ) -> dict:
        """Read a few consecutive frames against a plan. Library plans bring their own step pages."""
        if not 1 <= len(frames) <= MAX_FRAMES:
            raise HTTPException(422, f"send between 1 and {MAX_FRAMES} frames")
        if len(references) > MAX_REFERENCES:
            raise HTTPException(422, f"send at most {MAX_REFERENCES} reference pages")
        chosen = resolve_plan(plan_id, plan)
        if references:
            refs = [_image(r) for r in references]
        else:
            refs = library.step_pages(plan_id) if plan_id and not plan else []
        stamps = [float(t) for t in times.split(",") if t.strip()] if times else []
        if stamps and len(stamps) != len(frames):
            raise HTTPException(422, "times: one per frame")
        batch = [Frame(time_s=stamps[i] if stamps else float(i), image=_image(f)) for i, f in enumerate(frames)]
        try:
            readings: list[FrameReading] = verify_frames(chosen, batch, references=refs, batch=MAX_FRAMES, workers=1)
        except RuntimeError as error:
            raise HTTPException(503, str(error)) from error
        steps = len(chosen.steps)
        observations: list[Observation] = [
            to_observation(r, f.time_s, steps) for r, f in zip(readings, batch, strict=True)
        ]
        return {
            "readings": [r.model_dump() for r in readings],
            "observations": [o.model_dump() for o in observations],
        }

    @app.post("/videos", dependencies=[Depends(require_access)])
    def videos(
        video: UploadFile = File(...),
        plan_id: str = Form(""),
        plan: str = Form(""),
        every_s: float = Form(2.0),
    ) -> VideoAnalysis:
        """Decode a whole build video against a plan."""
        chosen = resolve_plan(plan_id, plan)
        if problems := validate(chosen):
            raise HTTPException(422, f"the plan does not add up: {problems[0]}")
        suffix = Path(video.filename or "").suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(415, "send an MP4, MOV or WebM video")
        if not 0.5 <= every_s <= 10:
            raise HTTPException(422, "every_s: between 0.5 and 10 seconds")
        refs = library.step_pages(plan_id) if plan_id and not plan else []
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / f"video{suffix}"
            written = 0
            with path.open("wb") as out:
                while chunk := video.file.read(2**20):
                    written += len(chunk)
                    if written > MAX_VIDEO_BYTES:
                        raise HTTPException(413, "the video is over 300 MB")
                    out.write(chunk)
            try:
                return analyze(path, chosen, references=refs, every_s=every_s)
            except InputError as error:
                raise HTTPException(422, str(error)) from error
            except RuntimeError as error:
                raise HTTPException(503, str(error)) from error

    return app


def local_app() -> FastAPI:
    """The API on this machine, reading .env and the repository's library."""
    from dotenv import load_dotenv

    repo = Path(__file__).resolve().parents[3]
    load_dotenv(repo / ".env")
    return create_app(repo / "assemblies")
