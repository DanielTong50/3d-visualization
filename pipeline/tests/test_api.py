import io

import pytest
from fastapi.testclient import TestClient

from darius.pipeline.jobs import STAGE_NAMES
from darius.service.api import create_app


@pytest.fixture
def started():
    return []


@pytest.fixture
def client(tmp_path, started, monkeypatch):
    monkeypatch.setenv("DARIUS_ACCESS_CODE", "s3cret")
    return TestClient(create_app(tmp_path, start_job=started.append))


def upload(client, name="desk.mp4", code="s3cret"):
    return client.post(
        "/jobs",
        files={"video": (name, io.BytesIO(b"\x00" * 1024), "video/mp4")},
        headers={"X-Access-Code": code},
    )


def test_upload_creates_and_starts_a_job(client, started, tmp_path):
    response = upload(client)
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert started == [job_id]
    assert (tmp_path / job_id / "input.mp4").stat().st_size == 1024
    status = client.get(f"/jobs/{job_id}").json()
    assert status["state"] == "queued" and [s["name"] for s in status["stages"]] == STAGE_NAMES


def test_upload_requires_the_access_code(client, started):
    assert upload(client, code="wrong").status_code == 401
    assert started == []


def test_rejects_non_video_uploads(client):
    assert upload(client, name="notes.txt").status_code == 415


@pytest.mark.parametrize("path", ["/jobs/../../etc", "/jobs/abc", "/jobs/0123456789ab/scene/..%2Fstatus.json"])
def test_paths_cannot_escape_the_jobs_folder(client, path):
    assert client.get(path).status_code == 404


def test_scene_files_are_served(client, tmp_path):
    job = tmp_path / "0123456789ab" / "scene"
    job.mkdir(parents=True)
    (job / "scene.json").write_text("{}")
    response = client.get("/jobs/0123456789ab/scene/scene.json")
    assert response.status_code == 200 and response.json() == {}


def test_jobs_from_before_a_new_stage_pick_it_up(tmp_path):
    import json

    from darius.pipeline.jobs import Status

    job = tmp_path / "0123456789ab"
    job.mkdir()
    old = {
        "job_id": job.name,
        "state": "done",
        "error": None,
        "stages": [{"name": "ingest", "label": "x", "state": "done", "seconds": 1.0}],
    }
    (job / "status.json").write_text(json.dumps(old))
    status = Status(job)
    assert [s["name"] for s in status.data["stages"]] == STAGE_NAMES
    assert status.stage("ingest")["state"] == "done" and status.stage("splat")["state"] == "pending"


def test_replay_files_in_the_scene_are_served(client, tmp_path):
    folder = tmp_path / "0123456789ab" / "scene" / "pipeline"
    folder.mkdir(parents=True)
    (folder / "points.bin").write_bytes(b"\x01\x02")
    response = client.get("/jobs/0123456789ab/scene/pipeline/points.bin")
    assert response.status_code == 200 and response.content == b"\x01\x02"


def test_artifacts_are_served_only_from_the_allowed_list(client, tmp_path):
    job = tmp_path / "0123456789ab"
    (job / "preview").mkdir(parents=True)
    (job / "preview" / "keyframes.jpg").write_bytes(b"jpg")
    (job / "input.mp4").write_bytes(b"video")
    assert client.get("/jobs/0123456789ab/artifacts/preview/keyframes.jpg").content == b"jpg"
    assert client.get("/jobs/0123456789ab/artifacts/physics.json").status_code == 404  # not written yet
    for name in ("input.mp4", "status.json", "../0123456789ab/input.mp4", "preview/../input.mp4"):
        assert client.get(f"/jobs/0123456789ab/artifacts/{name}").status_code == 404
