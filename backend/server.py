import os
import shutil
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.config import (
    BASE_DIR, UPLOADS_DIR, FRAMES_DIR, AUDIO_DIR, SAMPLE_VIDEOS_DIR,
    MAX_VIDEO_SIZE_MB, MAX_VIDEO_DURATION_SEC
)
from backend.models import (
    JobStatus, NarrationMode, DetailLevel, QARequest, QAResponse, TTSRequest
)
from backend.video_engine import VideoEngine
from backend.ai_engine import get_vision_provider
from backend.narration_engine import NarrationEngine
from backend.tts_engine import TTSEngine
from backend.qa_engine import QAEngine

app = FastAPI(title="Sightline AI API", description="Intelligent Visual Experience System for Blind and Visually Impaired Users")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job repository
jobs: Dict[str, JobStatus] = {}

video_engine = VideoEngine()
narration_engine = NarrationEngine()
tts_engine = TTSEngine()
qa_engine = QAEngine()


def process_video_pipeline(job_id: str, mode: NarrationMode, detail_level: DetailLevel, voice: str, pace: float):
    """Background task running the complete end-to-end Sightline pipeline."""
    job = jobs.get(job_id)
    if not job:
        return

    video_path = UPLOADS_DIR / f"{job_id}_{job.filename}"
    if not video_path.exists():
        job.status = "failed"
        job.error = "Source video file missing."
        return

    try:
        # Step 1: Sampling & Segmentation (0% - 35%)
        job.status = "sampling"
        job.progress = 15
        job.message = "Analyzing motion, camera cuts, and sampling keyframes..."
        
        frames, scenes = video_engine.sample_and_segment(video_path, job_id)
        job.scenes = scenes
        job.progress = 35
        job.message = f"Identified {len(scenes)} scenes and {len(frames)} sampled frames."

        # Step 2: Audio extraction & transcription (35% - 50%)
        job.status = "analyzing"
        job.message = "Checking audio track for dialogue preservation..."
        audio_file = video_engine.extract_audio(video_path, job_id)
        dialogues = []
        # If audio extracted and Sarvam STT available, transcribe
        if audio_file:
            try:
                from sarvamai import SarvamAI
                sarvam_client = SarvamAI(api_subscription_key=os.environ.get("SARVAM_API_KEY", ""))
                stt_res = sarvam_client.speech_to_text.transcribe(
                    file=str(audio_file),
                    model="saaras:v3",
                    with_timestamps=True
                )
                if hasattr(stt_res, "transcript"):
                    job.message = "Audio transcribed for dialogue coordination."
            except Exception:
                pass
        job.progress = 55

        # Step 3: Multimodal Vision Analysis (55% - 75%)
        job.message = "Applying multimodal vision to keyframes..."
        vision_provider = get_vision_provider()
        
        for sc in job.scenes:
            seg_frames = [f for f in frames if sc.start_time <= f.timestamp <= sc.end_time and f.is_keyframe]
            if not seg_frames:
                seg_frames = [f for f in frames if sc.start_time <= f.timestamp <= sc.end_time]
            
            if seg_frames:
                frame_paths = [BASE_DIR / "data" / f.relative_path for f in seg_frames]
                timestamps = [f.timestamp for f in seg_frames]
                spatial_cues = [f.spatial_layout or "center view" for f in seg_frames]

                sc_events = vision_provider.analyze_scene_frames(frame_paths, timestamps, spatial_cues)
                sc.visual_events = sc_events
        
        job.progress = 75

        # Step 4: Narration Generation (75% - 90%)
        job.status = "narrating"
        job.message = "Synthesizing context-aware spoken narration..."
        narration_items = narration_engine.generate_narration(
            scenes=job.scenes,
            mode=mode,
            detail_level=detail_level,
            dialogues=dialogues
        )
        job.progress = 88

        # Step 5: Speech Synthesis via Sarvam Bulbul TTS (90% - 100%)
        job.message = "Generating spoken narration audio with Sarvam Bulbul TTS..."
        for idx, item in enumerate(narration_items):
            audio_path = AUDIO_DIR / f"{job_id}_item_{item.id}.wav"
            b64_audio = tts_engine.synthesize(
                text=item.text,
                speaker=voice,
                pace=pace,
                output_path=audio_path
            )
            item.audio_base64 = b64_audio
            if audio_path.exists():
                item.audio_url = f"/api/audio/{job_id}_item_{item.id}.wav"
        
        job.narrations = narration_items
        job.status = "ready"
        job.progress = 100
        job.message = f"Narration ready! Generated {len(narration_items)} synchronized audio descriptions."

    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.message = f"Processing error: {str(e)}"


@app.get("/api/sample-video")
async def load_sample_video():
    """Loads the pre-generated sample test video for instant demonstration."""
    sample_file = SAMPLE_VIDEOS_DIR / "sample_demo.mp4"
    if not sample_file.exists():
        from backend.create_sample_video import generate_sample_video
        generate_sample_video(sample_file)

    job_id = str(uuid.uuid4())[:8]
    clean_filename = f"{job_id}_sample_demo.mp4"
    dest = UPLOADS_DIR / clean_filename
    shutil.copyfile(sample_file, dest)

    meta = video_engine.get_metadata(dest)
    job = JobStatus(
        job_id=job_id,
        filename="sample_demo.mp4",
        duration=meta["duration"],
        fps=meta["fps"],
        resolution=meta["resolution"],
        status="uploaded",
        progress=0,
        message="Sample video loaded and ready for analysis."
    )
    jobs[job_id] = job
    return job


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Uploads and validates a video file."""
    allowed_exts = {".mp4", ".webm", ".avi", ".mov", ".mkv"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported formats: {', '.join(allowed_exts)}"
        )

    job_id = str(uuid.uuid4())[:8]
    clean_filename = f"{job_id}_{file.filename}"
    save_path = UPLOADS_DIR / clean_filename

    # Read and validate size
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_VIDEO_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of {MAX_VIDEO_SIZE_MB} MB."
        )

    with open(save_path, "wb") as f:
        f.write(contents)

    # Validate video readability and duration
    try:
        meta = video_engine.get_metadata(save_path)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted video: {str(e)}")

    if meta["duration"] > MAX_VIDEO_DURATION_SEC:
        save_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Video duration ({meta['duration']:.1f}s) exceeds maximum limit of {MAX_VIDEO_DURATION_SEC}s."
        )

    job = JobStatus(
        job_id=job_id,
        filename=file.filename,
        duration=meta["duration"],
        fps=meta["fps"],
        resolution=meta["resolution"],
        status="uploaded",
        progress=0,
        message="Video uploaded and validated successfully."
    )
    jobs[job_id] = job

    return job


@app.post("/api/process/{job_id}")
async def start_processing(
    job_id: str,
    background_tasks: BackgroundTasks,
    mode: NarrationMode = Form(NarrationMode.GENERAL),
    detail_level: DetailLevel = Form(DetailLevel.BALANCED),
    voice: str = Form("shubh"),
    pace: float = Form(1.0)
):
    """Initiates asynchronous video understanding and narration."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    job.narration_mode = mode
    job.detail_level = detail_level
    job.voice = voice
    job.pace = pace
    job.status = "sampling"
    job.progress = 5
    job.message = "Starting video analysis pipeline..."

    background_tasks.add_task(process_video_pipeline, job_id, mode, detail_level, voice, pace)
    return {"job_id": job_id, "status": "processing_started"}


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    """Retrieves current processing progress and narration results."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.post("/api/qa/{job_id}")
async def ask_question(job_id: str, req: QARequest):
    """Enables the user to ask contextual questions about the scene."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    response = qa_engine.answer_question(
        question=req.question,
        current_time=req.current_time,
        scenes=job.scenes,
        narrations=job.narrations,
        voice=job.voice,
        pace=job.pace
    )
    return response


@app.post("/api/tts")
async def synthesize_text(req: TTSRequest):
    """Synthesizes on-demand speech using Sarvam Bulbul TTS."""
    b64 = tts_engine.synthesize(text=req.text, speaker=req.speaker, pace=req.pace, language_code=req.language_code)
    return {"audio_base64": b64}


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """User-initiated deletion of video and all associated processing data."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    # Remove upload
    upload_file = UPLOADS_DIR / f"{job_id}_{job.filename}"
    upload_file.unlink(missing_ok=True)

    # Remove frames directory
    job_frames = FRAMES_DIR / job_id
    if job_frames.exists():
        shutil.rmtree(job_frames, ignore_errors=True)

    # Remove audio files
    for a in AUDIO_DIR.glob(f"{job_id}*"):
        a.unlink(missing_ok=True)

    del jobs[job_id]
    return {"message": "All data and media associated with this job have been permanently deleted."}


@app.get("/api/audio/{filename}")
async def get_audio_file(filename: str):
    """Streams generated narration audio files."""
    audio_path = AUDIO_DIR / filename
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(str(audio_path), media_type="audio/wav")


@app.get("/api/video/{job_id}")
async def get_video_file(job_id: str):
    """Streams the uploaded video file."""
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    video_path = UPLOADS_DIR / f"{job_id}_{job.filename}"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found.")
    return FileResponse(str(video_path), media_type="video/mp4")


# Mount static assets
frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.get("/")
async def root():
    index_file = BASE_DIR / "frontend" / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Sightline AI API is running. Frontend static directory not initialized."}
