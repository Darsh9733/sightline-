import os
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from backend.config import GEMINI_API_KEY, OPENAI_API_KEY, SARVAM_API_KEY
from backend.models import ProcessedFrame, SceneSegment


class BaseVisionProvider:
    def analyze_scene_frames(self, frames: List[Path], timestamps: List[float], spatial_cues: List[str]) -> List[str]:
        raise NotImplementedError


class GeminiVisionProvider(BaseVisionProvider):
    def __init__(self, api_key: str):
        from google import genai
        self.client = genai.Client(api_key=api_key)

    def analyze_scene_frames(self, frames: List[Path], timestamps: List[float], spatial_cues: List[str]) -> List[str]:
        from PIL import Image
        descriptions = []
        for frame_path, ts, cue in zip(frames, timestamps, spatial_cues):
            try:
                img = Image.open(frame_path)
                prompt = (
                    f"Timestamp: {ts}s. Spatial cue: {cue}. "
                    "You are the visual observer for Sightline AI assisting a blind person. "
                    "Describe directly visible facts: people, their exact physical actions, object movements, "
                    "and spatial arrangement (left, right, center, foreground, background). "
                    "Do NOT invent feelings, relationships, or unseen motives. If an object or action is partially visible, state what is observable."
                )
                response = self.client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=[img, prompt]
                )
                descriptions.append(response.text.strip())
            except Exception as e:
                descriptions.append(f"Observable action at {ts}s: {cue}")
        return descriptions


class OpenAIVisionProvider(BaseVisionProvider):
    def __init__(self, api_key: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def analyze_scene_frames(self, frames: List[Path], timestamps: List[float], spatial_cues: List[str]) -> List[str]:
        descriptions = []
        for frame_path, ts, cue in zip(frames, timestamps, spatial_cues):
            try:
                with open(frame_path, "rb") as f:
                    b64_img = base64.b64encode(f.read()).decode("utf-8")
                
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": f"Timestamp {ts}s. Spatial hint: {cue}. Describe physical actions, layout, and observable visual facts for a blind listener. Never invent emotions or hidden motives."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                                }
                            ]
                        }
                    ],
                    max_tokens=150
                )
                descriptions.append(resp.choices[0].message.content.strip())
            except Exception:
                descriptions.append(f"Visual event at {ts}s: {cue}")
        return descriptions


class SarvamPerceptionProvider(BaseVisionProvider):
    """
    Couples OpenCV perceptual feature extraction (motion vectors, spatial quadrants,
    color shifts) with Sarvam AI's 105B conversational model.
    """
    def __init__(self, api_key: str):
        from sarvamai import SarvamAI
        self.client = SarvamAI(api_subscription_key=api_key)

    def analyze_scene_frames(self, frames: List[Path], timestamps: List[float], spatial_cues: List[str]) -> List[str]:
        descriptions = []
        
        # Batch analysis via Sarvam 105B
        prompt = (
            "You are the perceptual scene reasoner for Sightline AI, an assistive visual system for blind users.\n"
            "Below is a sequence of computer-vision motion and spatial cues extracted across timestamps:\n"
        )
        for ts, cue in zip(timestamps, spatial_cues):
            prompt += f"- At {ts}s: {cue}\n"

        prompt += (
            "\nProvide a concise, grounded factual description of the observable sequence. "
            "Follow these rules strictly:\n"
            "1. Accuracy over imagination: Only describe observable physical actions and spatial positions.\n"
            "2. Do not invent emotions, intentions, or relationship histories.\n"
            "3. Format each timestamp as: '[TS] <description>'"
        )

        try:
            resp = self.client.chat.completions(
                model="sarvam-105b-conversations",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            raw_text = resp.choices[0].message.content.strip()
            # Parse lines or use raw text
            lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
            if len(lines) >= len(timestamps):
                descriptions = lines[:len(timestamps)]
            else:
                for ts, cue in zip(timestamps, spatial_cues):
                    descriptions.append(f"At {ts}s: {cue.capitalize()}.")
        except Exception as e:
            for ts, cue in zip(timestamps, spatial_cues):
                descriptions.append(f"At {ts}s: {cue.capitalize()}.")

        return descriptions


def get_vision_provider() -> BaseVisionProvider:
    """Factory selecting the best available vision provider."""
    if GEMINI_API_KEY:
        try:
            return GeminiVisionProvider(GEMINI_API_KEY)
        except Exception:
            pass
    if OPENAI_API_KEY:
        try:
            return OpenAIVisionProvider(OPENAI_API_KEY)
        except Exception:
            pass
    if SARVAM_API_KEY:
        return SarvamPerceptionProvider(SARVAM_API_KEY)
    
    # Fallback to local heuristic provider
    class LocalHeuristicProvider(BaseVisionProvider):
        def analyze_scene_frames(self, frames: List[Path], timestamps: List[float], spatial_cues: List[str]) -> List[str]:
            return [f"At {ts}s, observable action: {cue}." for ts, cue in zip(timestamps, spatial_cues)]

    return LocalHeuristicProvider()
