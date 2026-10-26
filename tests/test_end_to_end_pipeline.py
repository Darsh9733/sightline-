import pytest
from pathlib import Path
from backend.server import app, jobs, process_video_pipeline
from backend.models import NarrationMode, DetailLevel
from fastapi.testclient import TestClient


def test_full_pipeline_execution():
    client = TestClient(app)
    
    # 1. Load sample video
    sample_res = client.get("/api/sample-video")
    assert sample_res.status_code == 200
    job_id = sample_res.json()["job_id"]
    assert job_id in jobs

    # 2. Run the full processing pipeline directly
    process_video_pipeline(
        job_id=job_id,
        mode=NarrationMode.GENERAL,
        detail_level=DetailLevel.BALANCED,
        voice="shubh",
        pace=1.0
    )

    # 3. Verify status
    job = jobs[job_id]
    assert job.status == "ready"
    assert job.progress == 100
    assert len(job.scenes) >= 1
    assert len(job.narrations) >= 1

    # 4. Check each narration item has text and audio
    for item in job.narrations:
        assert len(item.text) > 0
        assert item.timestamp >= 0.0
        # Check audio was generated
        assert item.audio_base64 is not None or item.audio_url is not None

    # 5. Check Q&A functionality on the completed job
    qa_res = client.post(f"/api/qa/{job_id}", json={
        "question": "What is the person doing?",
        "current_time": 4.0
    })
    assert qa_res.status_code == 200
    qa_data = qa_res.json()
    assert len(qa_data["answer"]) > 0

    # 6. Check Deletion cleans up everything
    del_res = client.delete(f"/api/job/{job_id}")
    assert del_res.status_code == 200
    assert job_id not in jobs
