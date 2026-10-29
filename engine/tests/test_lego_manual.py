"""Optional manual-draft adapter tested without spending model credits."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from darius.lego import get_assembly
from darius.lego.manual import DraftBrick, ManualDraftReading, draft_from_reading, read_manual_draft
from darius.service.api import create_app


def reading() -> ManualDraftReading:
    assembly = get_assembly("bridge")
    return ManualDraftReading(
        title=assembly.title,
        baseplate_studs=[16, 16],
        bricks=[
            DraftBrick(
                id=b.id,
                part_id=b.part_id,
                color=b.color,
                color_hex=b.color_hex,
                position=b.position,
                rotation=b.rotation,
                confidence=0.95,
                source_page=i + 1,
                evidence="Test reading fixture",
                uncertainties=[],
            )
            for i, b in enumerate(assembly.bricks)
        ],
        steps=assembly.steps,
        unsupported_features=[],
        notes=[],
    )


def test_reading_remains_unapproved_even_when_geometry_is_valid() -> None:
    result = draft_from_reading(reading(), "bridge-draft", source_url="https://www.lego.com/example")
    assert result.requires_review and result.semantic_status == "candidate_draft"
    assert result.compiled and result.compiled.validation.valid
    assert result.candidate.source.review_status == "pending"
    assert result.candidate.source.training_rights == "permission-required"
    assert "Manual draft" in result.compiled.execution.blockers[0]


def test_unknown_parts_and_missing_poses_do_not_get_fabricated() -> None:
    raw = reading()
    raw.bricks[0].part_id = None
    raw.bricks[1].position = None
    result = draft_from_reading(raw, "ambiguous")
    assert result.candidate is None and result.compiled is None
    assert any("unsupported" in issue for issue in result.issues)
    assert any("unresolved grid pose" in issue for issue in result.issues)
    raw = reading()
    raw.unsupported_features = ["Technic hinge"]
    assert draft_from_reading(raw, "unsupported").candidate is None


def test_low_confidence_is_retained_and_flagged_without_becoming_calibrated() -> None:
    raw = reading()
    raw.bricks[0].confidence = 0.4
    raw.bricks[0].uncertainties = ["occluded footprint"]
    result = draft_from_reading(raw, "uncertain")
    assert result.candidate is not None
    assert any("0.40" in issue and "occluded" in issue for issue in result.issues)


def test_model_adapter_passes_ordered_pages_to_existing_structured_client(monkeypatch) -> None:
    seen = {}

    def fake_ask(prompt, pages, schema, **kwargs):
        seen.update(pages=len(pages), schema=schema, prompt=prompt)
        return reading(), {"input_tokens": 123}

    monkeypatch.setattr("darius.lego.manual.ask_structured", fake_ask)
    result = read_manual_draft([Image.new("RGB", (32, 32))], "test-draft")
    assert result.usage == {"input_tokens": 123} and seen["pages"] == 1
    assert seen["schema"] is ManualDraftReading and "do not guess" in seen["prompt"]
    with pytest.raises(ValueError):
        read_manual_draft([], "empty")


def test_api_manual_draft_uses_credit_gate_and_never_approves(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("DARIUS_ACCESS_CODE", "test-code")
    client = TestClient(create_app(root / "assemblies"))
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buffer, "PNG")
    files = [("pages", ("page.png", buffer.getvalue(), "image/png"))]
    assert client.post("/lego/manual-drafts", files=files).status_code == 401
    monkeypatch.setattr("darius.lego.manual.ask_structured", lambda *a, **k: (reading(), {}))
    response = client.post("/lego/manual-drafts", files=files, headers={"X-Access-Code": "test-code"})
    assert response.status_code == 200 and response.json()["requires_review"] is True
