from typing import List, Optional, Dict, Any
from backend.models import SceneSegment, NarrationItem, QAResponse
from backend.config import SARVAM_API_KEY
from backend.tts_engine import TTSEngine


class QAEngine:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or SARVAM_API_KEY
        self.tts = TTSEngine(api_key=self.api_key)

    def answer_question(
        self,
        question: str,
        current_time: Optional[float],
        scenes: List[SceneSegment],
        narrations: List[NarrationItem],
        voice: str = "shubh",
        pace: float = 1.0
    ) -> QAResponse:
        """
        Answers a user question grounded in video scene evidence and narration history.
        """
        target_time = current_time if current_time is not None else (scenes[-1].end_time if scenes else 0.0)

        # Identify relevant scene and preceding scene
        relevant_scenes = []
        for s in scenes:
            if s.start_time <= target_time <= s.end_time or abs(s.start_time - target_time) < 3.0:
                relevant_scenes.append(s)

        if not relevant_scenes and scenes:
            relevant_scenes = [scenes[-1]]

        # Gather evidence
        evidence_lines = []
        for s in relevant_scenes:
            evidence_lines.append(f"Scene from {s.start_time}s to {s.end_time}s:")
            evidence_lines.append(f"- Spatial layout: {s.spatial_relationships}")
            if s.entities:
                evidence_lines.append(f"- Tracked entities: {', '.join(s.entities)}")
            if s.visual_events:
                for ev in s.visual_events:
                    evidence_lines.append(f"- Visual event: {ev}")

        recent_narrations = [n.text for n in narrations if n.timestamp <= target_time + 1.0]

        prompt = (
            "You are the assistive question-answering assistant for Sightline AI, helping a blind user understand a video.\n"
            f"User Question: '{question}'\n"
            f"Video Time Reference: {target_time:.1f} seconds.\n\n"
            "Extracted Video Evidence at this moment:\n" +
            ("\n".join(evidence_lines) if evidence_lines else "No specific visual changes detected at this timestamp.") +
            "\n\nRecent Spoken Narration:\n" +
            ("\n".join(recent_narrations[-3:]) if recent_narrations else "No prior narration.") +
            "\n\nRules:\n"
            "1. Answer based strictly on the observable evidence provided above.\n"
            "2. If an object, person, or action is visible, describe its observable characteristics (position, movement, visible parts).\n"
            "3. If the answer cannot be determined from the available view, explicitly state what is visible and acknowledge what cannot be seen.\n"
            "4. Never invent identity, hidden motives, or details not supported by the evidence.\n"
            "5. Keep the answer direct, conversational, and concise (1-3 sentences)."
        )

        answer_text = ""
        if self.api_key:
            try:
                from sarvamai import SarvamAI
                client = SarvamAI(api_subscription_key=self.api_key)
                resp = client.chat.completions(
                    model="sarvam-105b-conversations",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150
                )
                answer_text = resp.choices[0].message.content.strip()
            except Exception:
                answer_text = self._fallback_answer(question, relevant_scenes, target_time)
        else:
            answer_text = self._fallback_answer(question, relevant_scenes, target_time)

        # Synthesize spoken audio for the answer
        audio_b64 = self.tts.synthesize(text=answer_text, speaker=voice, pace=pace)

        return QAResponse(
            question=question,
            answer=answer_text,
            timestamp=target_time,
            confidence="high" if relevant_scenes else "moderate",
            audio_base64=audio_b64
        )

    def _fallback_answer(self, question: str, scenes: List[SceneSegment], target_time: float) -> str:
        q_lower = question.lower()
        if not scenes:
            return f"At {target_time:.1f} seconds, the visual evidence is insufficient to answer definitively."

        s = scenes[0]
        if "what is happening" in q_lower or "what happened" in q_lower:
            events_str = " ".join(s.visual_events) if s.visual_events else "the scene remains stable"
            return f"At {target_time:.1f} seconds, {events_str} with {s.spatial_relationships}."
        elif "who" in q_lower or "person" in q_lower:
            if s.entities:
                return f"At {target_time:.1f} seconds, the observed entities include {', '.join(s.entities)}."
            return f"At {target_time:.1f} seconds, no specific individuals can be clearly identified in this view."
        elif "where" in q_lower:
            return f"Based on the spatial arrangement, elements are situated in the {s.spatial_relationships}."
        else:
            return f"At {target_time:.1f} seconds, observable visual cues indicate {s.spatial_relationships}."
