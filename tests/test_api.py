import pytest
from fastapi.testclient import TestClient
from backend.server import app, jobs


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def test_root_endpoint(client):
    res = client.get("/")
    assert res.status_code == 200


def test_load_sample_video(client):
    res = client.get("/api/sample-video")
    assert res.status_code == 200
    data = res.json()
    assert "job_id" in data
    assert data["filename"] == "sample_demo.mp4"
    assert data["duration"] > 0


def test_qa_endpoint(client):
    # Load sample job
    sample_res = client.get("/api/sample-video")
    job_id = sample_res.json()["job_id"]

    res = client.post(f"/api/qa/{job_id}", json={
        "question": "What is on the table?",
        "current_time": 6.0
    })
    assert res.status_code == 200
    data = res.json()
    assert "answer" in data
    assert len(data["answer"]) > 0


def test_tts_endpoint(client):
    res = client.post("/api/tts", json={
        "text": "Sightline AI test audio.",
        "speaker": "shubh",
        "pace": 1.0,
        "language_code": "en-IN"
    })
    assert res.status_code == 200
    data = res.json()
    assert "audio_base64" in data


def test_job_deletion(client):
    sample_res = client.get("/api/sample-video")
    job_id = sample_res.json()["job_id"]
    assert job_id in jobs

    del_res = client.delete(f"/api/job/{job_id}")
    assert del_res.status_code == 200
    assert job_id not in jobs
