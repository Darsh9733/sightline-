import cv2
import numpy as np
from pathlib import Path
from backend.config import SAMPLE_VIDEOS_DIR


def generate_sample_video(output_path: Path, duration_sec: int = 8, fps: int = 24):
    """
    Generates a clear 3-scene test video with distinct motion and spatial layout:
    - Scene 1 (0-2.5s): A room with a table on the right side and a door on the left.
    - Scene 2 (2.5-5.5s): A person enters through the left door and walks across toward the table.
    - Scene 3 (5.5-8.0s): The person places a rectangular object onto the table.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = duration_sec * fps

    for i in range(total_frames):
        t = i / fps
        frame = np.ones((height, width, 3), dtype=np.uint8) * 40  # Dark room background

        # Draw room geometry: Floor line
        cv2.line(frame, (0, 380), (width, 380), (80, 80, 80), 3)

        # Draw table on the right side
        cv2.rectangle(frame, (400, 300), (580, 380), (120, 90, 60), -1)
        cv2.rectangle(frame, (420, 380), (440, 440), (90, 65, 45), -1)
        cv2.rectangle(frame, (540, 380), (560, 440), (90, 65, 45), -1)

        # Draw left doorway
        cv2.rectangle(frame, (40, 160), (140, 380), (70, 70, 70), 3)

        if t < 2.5:
            # Scene 1: Empty room with doorway and table
            cv2.putText(frame, "Scene 1: Interior with doorway and table", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        elif t < 5.5:
            # Scene 2: Person enters through the doorway and moves across to the table
            progress = (t - 2.5) / 3.0
            person_x = int(90 + progress * 320)
            person_y = 280

            # Draw person (head, torso, legs)
            cv2.circle(frame, (person_x, person_y - 60), 22, (220, 180, 140), -1)  # Head
            cv2.rectangle(frame, (person_x - 18, person_y - 35), (person_x + 18, person_y + 40), (50, 80, 160), -1)  # Blue coat
            cv2.line(frame, (person_x - 10, person_y + 40), (person_x - 12, person_y + 100), (30, 30, 30), 6)  # Left leg
            cv2.line(frame, (person_x + 10, person_y + 40), (person_x + 12, person_y + 100), (30, 30, 30), 6)  # Right leg

            cv2.putText(frame, "Scene 2: Person enters left, walks toward table", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 100), 2)
        else:
            # Scene 3: Person stands beside the table and places a small box on it
            person_x = 380
            person_y = 280
            cv2.circle(frame, (person_x, person_y - 60), 22, (220, 180, 140), -1)
            cv2.rectangle(frame, (person_x - 18, person_y - 35), (person_x + 18, person_y + 40), (50, 80, 160), -1)
            cv2.line(frame, (person_x - 10, person_y + 40), (person_x - 10, person_y + 100), (30, 30, 30), 6)
            cv2.line(frame, (person_x + 10, person_y + 40), (person_x + 10, person_y + 100), (30, 30, 30), 6)

            # Draw arm reaching down to table
            cv2.line(frame, (person_x + 18, person_y - 10), (450, 300), (50, 80, 160), 5)

            # Draw rectangular object placed on table
            cv2.rectangle(frame, (450, 275), (495, 300), (200, 50, 50), -1)  # Red box on table

            cv2.putText(frame, "Scene 3: Placing object on table", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

        out.write(frame)

    out.release()
    return output_path


if __name__ == "__main__":
    out_file = SAMPLE_VIDEOS_DIR / "sample_demo.mp4"
    generate_sample_video(out_file)
    print(f"Generated sample video at: {out_file}")
