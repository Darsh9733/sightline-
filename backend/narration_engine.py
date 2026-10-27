import os
from typing import List, Dict, Any, Optional
from backend.models import SceneSegment, NarrationItem, NarrationMode, DetailLevel, DialogueItem
from backend.config import SARVAM_API_KEY, GEMINI_API_KEY


class NarrationEngine:
    def __init__(self, api_key: Optional[str] = None, gemini_key: Optional[str] = None):
        self.api_key = api_key or SARVAM_API_KEY
        self.gemini_key = gemini_key or GEMINI_API_KEY
        self.history: List[str] = []

    def clear_history(self):
        self.history.clear()

    def generate_narration(
        self,
        scenes: List[SceneSegment],
        mode: NarrationMode = NarrationMode.GENERAL,
        detail_level: DetailLevel = DetailLevel.BALANCED,
        dialogues: Optional[List[DialogueItem]] = None
    ) -> List[NarrationItem]:
        """
        Transforms scene segments and visual events into a coherent, non-repetitive,
        context-aware spoken narration schedule.
        """
        self.clear_history()
        narration_items: List[NarrationItem] = []
        dialogues = dialogues or []

        for seg_idx, scene in enumerate(scenes):
            # Check if this scene has visual events
            events = scene.visual_events if scene.visual_events else [scene.summary]
            
            # Compose system prompt tailored to mode and detail level
            mode_guidance = {
                NarrationMode.MOVIE: "Focus on character movements, interactions, expressions, and environmental storytelling without spoiling.",
                NarrationMode.REALTIME: "Provide concise, immediate event descriptions; prioritize urgent changes.",
                NarrationMode.EDUCATIONAL: "Detail instructional steps, diagrams, screen changes, and demonstrations clearly.",
                NarrationMode.SPORTS: "Track movement of players, ball/equipment position, and state of play factually.",
                NarrationMode.GENERAL: "Describe the visible events naturally and informatively."
            }.get(mode, "Describe visible events naturally.")

            density_guidance = {
                DetailLevel.MINIMAL: "Only describe critical scene transitions and major actions in 1 short sentence.",
                DetailLevel.BALANCED: "Describe key actions, spatial positions, and meaningful changes in 1-2 clear sentences.",
                DetailLevel.DETAILED: "Provide complete descriptions of posture, movement, spatial arrangement, and background details in 2-3 sentences."
            }.get(detail_level, "Describe key actions in 1-2 clear sentences.")

            prompt = (
                f"You are the narration synthesizer for Sightline AI, assisting a blind listener.\n"
                f"Mode: {mode.value.title()}. Detail Level: {detail_level.value.title()}.\n"
                f"Guidance: {mode_guidance} {density_guidance}\n\n"
                f"Scene {seg_idx + 1} (Start: {scene.start_time}s, End: {scene.end_time}s):\n"
                f"Spatial Layout: {scene.spatial_relationships or 'Standard'}\n"
                f"Observable Visual Facts:\n" + "\n".join(f"- {e}" for e in events) + "\n\n"
            )

            if self.history:
                prompt += (
                    "Recent narration already spoken (DO NOT REPEAT these facts unless they changed):\n" +
                    "\n".join(f"* {h}" for h in self.history[-3:]) + "\n\n"
                )

            prompt += (
                "Rules:\n"
                "1. Speak like a knowledgeable companion describing what is visible.\n"
                "2. Establish scene context at the start, then focus only on changes.\n"
                "3. Never invent intentions, unseen objects, or internal feelings.\n"
                "4. Output ONLY the narration text to be spoken, with no meta comments or markdown."
            )

            spoken_text = ""
            # Priority 1: Google Gemini Multimodal / Text Generation
            effective_gemini_key = self.gemini_key or os.environ.get("GEMINI_API_KEY", "")
            if effective_gemini_key:
                try:
                    from google import genai
                    gemini_client = genai.Client(api_key=effective_gemini_key)
                    for model_cand in ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-flash-latest"]:
                        try:
                            resp = gemini_client.models.generate_content(
                                model=model_cand,
                                contents=prompt
                            )
                            if resp.text:
                                candidate = resp.text.strip().strip('"')
                                if candidate.lower().startswith("narration:"):
                                    candidate = candidate[10:].strip()
                                spoken_text = candidate
                                break
                        except Exception:
                            continue
                except Exception:
                    pass

            # Priority 2: Sarvam AI Text Model
            if not spoken_text and self.api_key:
                try:
                    from sarvamai import SarvamAI
                    client = SarvamAI(api_subscription_key=self.api_key)
                    resp = client.chat.completions(
                        model="sarvam-105b-conversations",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200
                    )
                    candidate = resp.choices[0].message.content.strip().strip('"')
                    # Strip any "Narration:" prefix
                    if candidate.lower().startswith("narration:"):
                        candidate = candidate[10:].strip()
                    spoken_text = candidate
                except Exception:
                    pass

            # Priority 3: Fallback to structured factual text
            if not spoken_text:
                spoken_text = self._fallback_narration(scene, detail_level)

            if spoken_text and spoken_text not in self.history:
                # Schedule narration timestamp (avoiding dialogue overlap if dialogue exists)
                narration_ts = self._find_scheduled_timestamp(scene.start_time, scene.end_time, dialogues)
                
                # Estimate duration (approx 2.5 words per second)
                word_count = len(spoken_text.split())
                est_duration = max(2.0, round(word_count / 2.5, 1))

                item = NarrationItem(
                    id=len(narration_items) + 1,
                    timestamp=narration_ts,
                    duration_estimate=est_duration,
                    text=spoken_text,
                    priority="high" if seg_idx == 0 else "medium",
                    detail_level=detail_level,
                    scene_id=scene.id
                )
                narration_items.append(item)
                self.history.append(spoken_text)

        return narration_items

    def _find_scheduled_timestamp(self, start_t: float, end_t: float, dialogues: List[DialogueItem]) -> float:
        """Finds a natural gap in dialogue or defaults to scene start."""
        for d in dialogues:
            # If dialogue occupies the start of the scene, schedule right after dialogue
            if start_t <= d.start_time <= start_t + 2.0:
                if d.end_time < end_t:
                    return round(d.end_time + 0.3, 2)
        return start_t

    def _fallback_narration(self, scene: SceneSegment, detail_level: DetailLevel) -> str:
        """Deterministic, grounded narration fallback adhering to accessibility rules."""
        if detail_level == DetailLevel.MINIMAL:
            return f"At {scene.start_time} seconds, {scene.spatial_relationships}."
        elif detail_level == DetailLevel.DETAILED:
            events_str = " ".join(scene.visual_events) if scene.visual_events else "the scene remains stable."
            return f"Scene from {scene.start_time} to {scene.end_time} seconds. In the layout, {scene.spatial_relationships}. Observable events: {events_str}."
        else:
            return f"At {scene.start_time} seconds, the view shows {scene.spatial_relationships}."
