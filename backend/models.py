from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class NarrationMode(str, Enum):
    MOVIE = "movie"
    REALTIME = "realtime"
    EDUCATIONAL = "educational"
    SPORTS = "sports"
    GENERAL = "general"


class DetailLevel(str, Enum):
    MINIMAL = "minimal"
    BALANCED = "balanced"
    DETAILED = "detailed"


class ProcessedFrame(BaseModel):
    frame_index: int
    timestamp: float
    is_keyframe: bool = False
    motion_score: float = 0.0
    relative_path: str = ""
    visual_description: Optional[str] = None
    detected_entities: List[str] = Field(default_factory=list)
    spatial_layout: Optional[str] = None


class DialogueItem(BaseModel):
    start_time: float
    end_time: float
    text: str
    speaker: Optional[str] = None


class SceneSegment(BaseModel):
    id: int
    start_time: float
    end_time: float
    summary: str = ""
    keyframe_indices: List[int] = Field(default_factory=list)
    visual_events: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    dialogues: List[DialogueItem] = Field(default_factory=list)
    spatial_relationships: Optional[str] = None
    candidate_narration: str = ""
    narrated: bool = False


class NarrationItem(BaseModel):
    id: int
    timestamp: float
    duration_estimate: float = 3.0
    text: str
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None
    priority: str = "medium"  # high, medium, low
    detail_level: DetailLevel = DetailLevel.BALANCED
    scene_id: Optional[int] = None


class JobStatus(BaseModel):
    job_id: str
    filename: str
    duration: float = 0.0
    fps: float = 0.0
    resolution: str = "0x0"
    status: str = "uploaded"  # uploaded, sampling, analyzing, narrating, ready, failed
    progress: int = 0  # 0 to 100
    message: str = "Video uploaded successfully"
    error: Optional[str] = None
    narration_mode: NarrationMode = NarrationMode.GENERAL
    detail_level: DetailLevel = DetailLevel.BALANCED
    voice: str = "shubh"
    pace: float = 1.0
    vision_provider: str = ""
    narrations: List[NarrationItem] = Field(default_factory=list)
    scenes: List[SceneSegment] = Field(default_factory=list)


class QARequest(BaseModel):
    question: str
    current_time: Optional[float] = None


class QAResponse(BaseModel):
    question: str
    answer: str
    timestamp: Optional[float] = None
    confidence: str = "high"
    audio_base64: Optional[str] = None


class TTSRequest(BaseModel):
    text: str
    speaker: str = "shubh"
    pace: float = 1.0
    language_code: str = "en-IN"
