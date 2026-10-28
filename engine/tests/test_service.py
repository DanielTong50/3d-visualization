"""The HTTP API, with the model calls stubbed out."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from darius.perception.frames import FrameReading, StepCheck
from darius.service import api

LIBRARY = Path(__file__).resolve().parents[2] / "assemblies"


def png(color: str = "white") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.delenv("DARIUS_ACCESS_CODE", raising=False)
    return TestClient(api.create_app(LIBRARY))


def test_health_lists_the_library(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["ok"] and "halden-bookcase" in body["library"]


def test_library_plans_are_served_and_ids_are_checked(client: TestClient) -> None:
    assert len(client.get("/library").json()) == 4
    assert client.get("/library/ostra-desk").json()["title"] == "Ostra desk"
    assert client.get("/library/..%2Fsecrets").status_code == 404
    assert client.get("/library/no-such-thing").status_code == 404


def test_frames_are_read_against_a_library_plan_with_its_step_pages(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = {}

    def fake_verify(plan, frames, *, references, **_):
        seen["references"] = len(references)
        seen["times"] = [f.time_s for f in frames]
        checks = [StepCheck(index=s.index, done=s.index == 1, confidence=0.9) for s in plan.steps]
        return [
            FrameReading(frame=i + 1, orientation="on_side", attached=["A"], loose=[], steps=checks)
            for i in range(len(frames))
        ]

    monkeypatch.setattr(api, "verify_frames", fake_verify)
    response = client.post(
        "/frames",
        data={"plan_id": "halden-bookcase", "times": "12.5,13.5"},
        files=[("frames", ("a.png", png(), "image/png")), ("frames", ("b.png", png(), "image/png"))],
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert seen == {"references": 5, "times": [12.5, 13.5]}
    assert body["observations"][0]["step_done"]["1"] == 0.9
    assert body["observations"][1]["time_s"] == 13.5


def test_bad_uploads_are_refused(client: TestClient) -> None:
    files = [("frames", ("a.txt", b"not an image", "text/plain"))]
    assert client.post("/frames", data={"plan_id": "halden-bookcase"}, files=files).status_code == 415
    files = [("frames", ("a.png", png(), "image/png"))]
    assert client.post("/frames", data={"plan": "{}"}, files=files).status_code == 422
    assert client.post("/frames", files=files).status_code == 422


def test_model_calls_need_the_access_code_when_one_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DARIUS_ACCESS_CODE", "sesame")
    client = TestClient(api.create_app(LIBRARY))
    files = [("pages", ("p.png", png(), "image/png"))]
    assert client.post("/manuals", files=files).status_code == 401
    assert client.get("/health").json()["access_code_required"] is True
