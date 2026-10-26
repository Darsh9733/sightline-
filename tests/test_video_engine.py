import pytest
from pathlib import Path
from backend.video_engine import VideoEngine
from backend.config import SAMPLE_VIDEOS_DIR
from backend.create_sample_video import generate_sample_video


@pytest.fixture(scope="module")
def sample_video():
    vid_path = SAMPLE_VIDEOS_DIR / "test_sample.mp4"
    generate_sample_video(vid_path, duration_sec=5, fps=20)
    return vid_path


def test_video_metadata(sample_video):
    engine = VideoEngine()
    meta = engine.get_metadata(sample_video)
    assert meta["duration"] > 0
    assert meta["fps"] == 20.0
    assert meta["width"] == 640
    assert meta["height"] == 480
    assert meta["total_frames"] == 100


def test_sample_and_segment(sample_video):
    engine = VideoEngine(baseline_interval=1.0, fast_interval=0.3)
    frames, scenes = engine.sample_and_segment(sample_video, job_id="test_job_1")

    assert len(frames) > 0
    assert len(scenes) >= 1
    # Check that timestamps are normalized and strictly non-decreasing
    timestamps = [f.timestamp for f in frames]
    assert timestamps == sorted(timestamps)

    # Check that keyframe flags are set
    keyframes = [f for f in frames if f.is_keyframe]
    assert len(keyframes) > 0

    # Check spatial quadrants
    first_frame = frames[0]
    assert hasattr(first_frame, "spatial_layout")
