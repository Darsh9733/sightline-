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
                    f"Timestamp: {ts:.1f}s. "
                    "You are the visual observer for Sightline AI, describing video for a blind listener.\n"
                    "Describe directly visible facts in 1 to 2 clear, natural sentences:\n"
                    "1. Who or what is in the scene (people, clothing, animals, objects).\n"
                    "2. The exact physical action or movement taking place.\n"
                    "3. Spatial positions (e.g. on the left, center, foreground, background).\n"
                    "4. Any visible expressions or background changes.\n"
                    "Strict Rule: Only state observable facts. Never invent feelings, relationships, or unseen intentions."
                )
                try:
                    response = self.client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=[img, prompt]
                    )
                except Exception:
                    response = self.client.models.generate_content(
                        model="gemini-1.5-flash",
                        contents=[img, prompt]
                    )
                descriptions.append(response.text.strip())
            except Exception as e:
                descriptions.append(f"Observable action at {ts:.1f}s: {cue}")
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
                                    "text": (
                                        f"Timestamp {ts:.1f}s. "
                                        "Describe visible physical actions, people, and spatial layout for a blind listener. "
                                        "Be factual and natural. Never guess feelings or motives."
                                    )
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
                descriptions.append(f"Visual event at {ts:.1f}s: {cue}")
        return descriptions


class SarvamPerceptionProvider(BaseVisionProvider):
    """
    Couples OpenCV perceptual feature extraction with Sarvam 105B.
    Note: Used when multimodal vision API key is not configured.
    """
    def __init__(self, api_key: str):
        from sarvamai import SarvamAI
        self.client = SarvamAI(api_subscription_key=api_key)

    def analyze_scene_frames(self, frames: List[Path], timestamps: List[float], spatial_cues: List[str]) -> List[str]:
        descriptions = []
        prompt = (
            "You are the perceptual scene reasoner for Sightline AI, an assistive visual system for blind users.\n"
            "Below is a sequence of computer-vision motion and spatial cues extracted across timestamps:\n"
        )
        for ts, cue in zip(timestamps, spatial_cues):
            prompt += f"- At {ts:.1f}s: {cue}\n"

        prompt += (
            "\nProvide a concise factual description of what is observable. "
            "Describe only physical positions and movements. Output one line per timestamp: '[TS] description'."
        )

        try:
            resp = self.client.chat.completions(
                model="sarvam-105b-conversations",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300
            )
            raw_text = resp.choices[0].message.content.strip()
            lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
            if len(lines) >= len(timestamps):
                descriptions = lines[:len(timestamps)]
            else:
                for ts, cue in zip(timestamps, spatial_cues):
                    descriptions.append(f"At {ts:.1f}s: {cue.capitalize()}.")
        except Exception:
            for ts, cue in zip(timestamps, spatial_cues):
                descriptions.append(f"At {ts:.1f}s: {cue.capitalize()}.")

        return descriptions


def get_vision_provider(gemini_key: Optional[str] = None, openai_key: Optional[str] = None) -> BaseVisionProvider:
    """Factory selecting the best available vision provider."""
    g_key = gemini_key or os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY
    if g_key and g_key.strip():
        try:
            return GeminiVisionProvider(g_key.strip())
        except Exception:
            pass

    o_key = openai_key or os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
    if o_key and o_key.strip():
        try:
            return OpenAIVisionProvider(o_key.strip())
        except Exception:
            pass

    s_key = os.environ.get("SARVAM_API_KEY") or SARVAM_API_KEY
    if s_key and s_key.strip():
        return SarvamPerceptionProvider(s_key.strip())

    class LocalHeuristicProvider(BaseVisionProvider):
        def analyze_scene_frames(self, frames: List[Path], timestamps: List[float], spatial_cues: List[str]) -> List[str]:
            return [f"At {ts:.1f}s: observable motion {cue}." for ts, cue in zip(timestamps, spatial_cues)]

    return LocalHeuristicProvider()
