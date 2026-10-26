import os
from pathlib import Path

# Resolve base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
FRAMES_DIR = DATA_DIR / "frames"
AUDIO_DIR = DATA_DIR / "audio"
SAMPLE_VIDEOS_DIR = DATA_DIR / "sample_videos"

for d in [UPLOADS_DIR, FRAMES_DIR, AUDIO_DIR, SAMPLE_VIDEOS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load environment variables manually from .env if present
env_file = BASE_DIR / ".env"
if env_file.is_file():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ.setdefault(key.strip(), val.strip().strip("'\""))

SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))

# Processing constraints
MAX_VIDEO_SIZE_MB = 100
MAX_VIDEO_DURATION_SEC = 300  # 5 minutes for rapid prototype processing
DEFAULT_BASELINE_SAMPLE_RATE_SEC = 1.5  # Sample frame every 1.5s in static scenes
FAST_SAMPLE_RATE_SEC = 0.4  # Dense sampling during high motion
