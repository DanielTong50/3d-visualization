"""Optional vision-language adapter: page images -> uncertain LEGO plan draft.

The adapter never promotes a model response to an approved hardware program.
Unknown parts/poses remain unresolved; deterministic geometry validation and an
explicit human review are separate from visual inference. No paid call at import.
"""

from __future__ import annotations

from typing import Literal

from PIL import Image
from pydantic import Field, ValidationError

from ..models.meta_api import ask_structured
from .compiler import PARTS, compile_assembly
from .schema import Baseplate, Brick, BuildStep, CompiledPlan, LegoAssembly, Source, Strict


class DraftBrick(Strict):
    id: str
    part_id: str | None = Field(description="Known rectangular LEGO design id, or null if unresolved")
    color: str
    color_hex: str
    position: list[int] | None = Field(description="[x,y,z], x/z studs and y plates, or null if ambiguous")
    rotation: Literal[0, 90, 180, 270] | None
    confidence: float = Field(ge=0, le=1, description="Model self-assessment, not calibrated probability")
    source_page: int = Field(ge=1, description="1-based supplied-image index; map through source_page_map")
    evidence: str
    uncertainties: list[str]


class ManualDraftReading(Strict):
    title: str
    baseplate_studs: list[int]
    bricks: list[DraftBrick] = Field(max_length=256)
    steps: list[BuildStep] = Field(max_length=256)
    unsupported_features: list[str]
    notes: list[str]


class ManualDraftResult(Strict):
    semantic_status: Literal["candidate_draft"] = "candidate_draft"
    requires_review: Literal[True] = True
    reading: ManualDraftReading
    candidate: LegoAssembly | None
    compiled: CompiledPlan | None
    issues: list[str]
    usage: dict
    source_page_map: list[int] = Field(
        default_factory=list, description="Original document page for each supplied image, in input order"
    )
    acquisition: dict | None = None
    notice: str = (
        "Unvalidated vision-language draft. Model confidence is not calibrated. Review every part, pose, "
        "step and instruction-source permission before any dataset or hardware use."
    )


def draft_from_reading(
    reading: ManualDraftReading, build_id: str, *, source_url: str | None = None, usage: dict | None = None
) -> ManualDraftResult:
    issues = [f"Unsupported manual feature: {feature}" for feature in reading.unsupported_features]
    bricks = []
    for item in reading.bricks:
        if item.part_id not in PARTS:
            issues.append(f"{item.id}: unresolved or unsupported rectangular part {item.part_id!r}")
            continue
        if item.position is None or item.rotation is None:
            issues.append(f"{item.id}: unresolved grid pose")
            continue
        if item.confidence < 0.9 or item.uncertainties:
            issues.append(
                f"{item.id}: review pose/identity confidence {item.confidence:.2f}; " + "; ".join(item.uncertainties)
            )
        width, depth, height = PARTS[item.part_id]
        try:
            bricks.append(
                Brick(
                    id=item.id,
                    part_id=item.part_id,
                    color=item.color,
                    color_hex=item.color_hex,
                    studs=[width, depth],
                    height_plates=height,
                    position=item.position,
                    rotation=item.rotation,
                )
            )
        except ValidationError as error:
            issues.append(f"{item.id}: invalid candidate fields: {error.errors()[0]['msg']}")
    candidate = None
    compiled = None
    if len(bricks) == len(reading.bricks) and bricks and not reading.unsupported_features:
        try:
            candidate = LegoAssembly(
                id=build_id,
                title=reading.title,
                description="Vision-language manual draft requiring human review.",
                source=Source(
                    kind="reviewed-import",
                    description="Unapproved candidate inferred from manual page images.",
                    url=source_url,
                    training_rights="permission-required",
                    review_status="pending",
                ),
                baseplate=Baseplate(studs=reading.baseplate_studs),
                bricks=bricks,
                steps=reading.steps,
                notes=["MANUAL DRAFT: requires human review; do not execute or treat as ground truth.", *reading.notes],
            )
            compiled = compile_assembly(candidate)
            compiled.execution.blockers.insert(0, "Manual draft requires human review of every inferred part and pose.")
            issues.extend(compiled.validation.errors)
            issues.extend(compiled.validation.warnings)
        except ValidationError as error:
            issues.append(f"Candidate structure is invalid: {error.errors()[0]['msg']}")
    if candidate is None:
        issues.append("No complete candidate emitted: resolve missing/unsupported parts, poses or schedule first.")
    return ManualDraftResult(reading=reading, candidate=candidate, compiled=compiled, issues=issues, usage=usage or {})


def read_manual_draft(
    pages: list[Image.Image], build_id: str = "manual-draft", *, source_url: str | None = None
) -> ManualDraftResult:
    if not 1 <= len(pages) <= 30:
        raise ValueError("send 1 to 30 ordered LEGO manual page images")
    supported = ", ".join(f"{part}:{w}x{d}x{h}plates" for part, (w, d, h) in PARTS.items())
    prompt = f"""Read these ordered LEGO instruction page images into a CANDIDATE plan, never an approved robot program.
Use only visible evidence. Record all uncertainty and do not guess an occluded connection or part identity.
Supported rectangular part ids and nominal dimensions: {supported}.
Each physical brick instance has its own id. Steps contain ids introduced exactly once. Give original numbered
pages as source_page (1-based image order), evidence, confidence (self-assessment), and uncertainties per brick.
Coordinates: +Y up, x/z are integer 8mm studs, y integer 3.2mm plates. position is the footprint minimum x/z
corner and body-bottom y; rotation 90/270 swaps width/depth. Ground bricks begin y=0. Choose a consistent
nonnegative grid origin. baseplate_studs covers the whole supported build, max32x32; this fixture is a proposed
robot jig, not an assertion that a baseplate appears in the manual. Null part_id/pose if not recoverable.
List hinges, slopes, Technic, subassembly reorientation, unsupported pieces or ambiguous coordinates in
unsupported_features rather than silently substituting boxes. Use color_hex #rrggbb and a human color name.
Do not infer training rights. The downstream compiler will validate support and occupancy independently.
"""
    reading, usage = ask_structured(prompt, pages, ManualDraftReading, reasoning_effort="high")
    result = draft_from_reading(reading, build_id, source_url=source_url, usage=usage)
    result.source_page_map = list(range(1, len(pages) + 1))
    for brick in reading.bricks:
        if brick.source_page > len(pages):
            result.issues.append(f"{brick.id}: source_page {brick.source_page} exceeds the supplied-image count")
    return result
