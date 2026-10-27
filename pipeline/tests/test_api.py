import io

import pytest
from fastapi.testclient import TestClient

from scan2sim.api import create_app


@pytest.fixture
def started():
    return []


@pytest.fixture
def client(tmp_path, started, monkeypatch):
    monkeypatch.setenv("SCAN2SIM_ACCESS_CODE", "s3cret")
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
    assert status["state"] == "queued" and len(status["stages"]) == 6


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
