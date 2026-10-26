import pytest
from backend.models import SceneSegment, NarrationMode, DetailLevel, DialogueItem
from backend.narration_engine import NarrationEngine


def test_narration_generation_and_repetition():
    engine = NarrationEngine()
    
    scenes = [
        SceneSegment(
            id=1,
            start_time=0.0,
            end_time=3.0,
            summary="Room interior",
            visual_events=["Table on the right, doorway on the left."],
            spatial_relationships="doorway on left side, table on right side in midground"
        ),
        SceneSegment(
            id=2,
            start_time=3.0,
            end_time=6.0,
            summary="Person enters room",
            visual_events=["A person enters through the doorway and walks toward the table."],
            spatial_relationships="subject moving from left doorway across to the right table"
        )
    ]

    narrations = engine.generate_narration(scenes, mode=NarrationMode.GENERAL, detail_level=DetailLevel.BALANCED)
    
    assert len(narrations) == 2
    # Check that timestamps match scene bounds
    assert narrations[0].timestamp == 0.0
    assert narrations[1].timestamp == 3.0

    # Ensure descriptions are distinct (no repetitive duplicates)
    assert narrations[0].text != narrations[1].text


def test_dialogue_gap_scheduling():
    engine = NarrationEngine()
    scenes = [
        SceneSegment(
            id=1,
            start_time=0.0,
            end_time=5.0,
            summary="Conversation scene",
            spatial_relationships="two people seated"
        )
    ]
    # Dialogue occurs from 0.0 to 1.8s
    dialogues = [DialogueItem(start_time=0.0, end_time=1.8, text="Hello there")]

    narrations = engine.generate_narration(scenes, dialogues=dialogues)
    # The narration should be scheduled after the dialogue to avoid speaking over it
    assert narrations[0].timestamp >= 1.8
