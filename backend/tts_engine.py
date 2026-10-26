import os
import base64
from pathlib import Path
from typing import Optional, Dict, Any
from backend.config import SARVAM_API_KEY, AUDIO_DIR


class TTSEngine:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or SARVAM_API_KEY

    def synthesize(
        self,
        text: str,
        speaker: str = "shubh",
        pace: float = 1.0,
        language_code: str = "en-IN",
        output_path: Optional[Path] = None
    ) -> Optional[str]:
        """
        Synthesizes speech using Sarvam AI Bulbul TTS.
        Returns base64 audio string (and saves to output_path if provided).
        """
        if not self.api_key or not text.strip():
            return None

        # Ensure valid speaker for bulbul:v3
        valid_speakers = {
            "aditya", "ritu", "ashutosh", "priya", "neha", "rahul", "pooja",
            "rohan", "simran", "kavya", "amit", "dev", "ishita", "shreya",
            "ratan", "varun", "manan", "sumit", "roopa", "kabir", "aayan",
            "shubh", "advait", "anand", "tanya", "tarun", "sunny", "mani",
            "gokul", "vijay", "shruti", "suhani", "mohit", "kavitha", "rehan",
            "soham", "rupali"
        }
        chosen_speaker = speaker.lower() if speaker.lower() in valid_speakers else "shubh"
        clamped_pace = max(0.5, min(2.0, float(pace)))

        try:
            from sarvamai import SarvamAI
            client = SarvamAI(api_subscription_key=self.api_key)
            res = client.text_to_speech.convert(
                text=text,
                language_code=language_code,
                speaker=chosen_speaker,
                pace=clamped_pace,
                model="bulbul:v3"
            )

            if res.audios and len(res.audios) > 0:
                audio_b64 = res.audios[0]
                if output_path:
                    raw_bytes = base64.b64decode(audio_b64)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_path, "wb") as f:
                        f.write(raw_bytes)
                return audio_b64
        except Exception as e:
            # Return None to trigger browser-side Web Speech API fallback gracefully
            return None

        return None
