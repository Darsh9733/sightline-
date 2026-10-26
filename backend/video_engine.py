import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from backend.models import ProcessedFrame, SceneSegment, DialogueItem
from backend.config import FRAMES_DIR, AUDIO_DIR, DEFAULT_BASELINE_SAMPLE_RATE_SEC, FAST_SAMPLE_RATE_SEC


class VideoEngine:
    def __init__(self, baseline_interval: float = DEFAULT_BASELINE_SAMPLE_RATE_SEC, fast_interval: float = FAST_SAMPLE_RATE_SEC):
        self.baseline_interval = baseline_interval
        self.fast_interval = fast_interval

    def get_metadata(self, video_path: Path) -> Dict[str, Any]:
        """Extract video duration, FPS, resolution, and frame counts."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0.0

        cap.release()
        return {
            "duration": round(duration, 2),
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "resolution": f"{width}x{height}",
            "total_frames": total_frames
        }

    def sample_and_segment(self, video_path: Path, job_id: str) -> Tuple[List[ProcessedFrame], List[SceneSegment]]:
        """
        Executes adaptive frame sampling based on motion analysis,
        detects scene boundaries using color histogram correlation,
        and constructs initial SceneSegments with spatial quadrant tracking.
        """
        metadata = self.get_metadata(video_path)
        fps = metadata["fps"]
        total_frames = metadata["total_frames"]
        duration = metadata["duration"]
        width = metadata["width"]
        height = metadata["height"]

        job_frames_dir = FRAMES_DIR / job_id
        job_frames_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        
        frames_list: List[ProcessedFrame] = []
        prev_gray = None
        prev_hsv_hist = None

        curr_frame_idx = 0
        min_step = max(1, int(fps * self.fast_interval))
        max_step = max(1, int(fps * self.baseline_interval))

        current_step = max_step
        scene_boundaries: List[float] = [0.0]

        while curr_frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            timestamp = round(curr_frame_idx / fps, 2)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (160, 120))

            # Compute motion score vs previous frame
            motion_score = 0.0
            is_cut = False
            spatial_notes = []
            detected_entities = []

            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray_small)
                motion_score = float(np.mean(diff))

                # Color histogram correlation for scene cut detection
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                if prev_hsv_hist is not None:
                    corr = cv2.compareHist(prev_hsv_hist, hist, cv2.HISTCMP_CORREL)
                    # If correlation is low, a scene cut occurred
                    if corr < 0.60:
                        is_cut = True
                        scene_boundaries.append(timestamp)
                prev_hsv_hist = hist

                # Contours & Spatial Quadrant Tracking
                _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for idx, c in enumerate(contours[:3]):  # Track top 3 motion regions
                    if cv2.contourArea(c) > 60:
                        x, y, w, h = cv2.boundingRect(c)
                        # Map to normalized coordinates (0-1)
                        norm_x = (x + w / 2) / 160.0
                        norm_y = (y + h / 2) / 120.0

                        # Horizontal spatial position
                        if norm_x < 0.35:
                            h_pos = "left side"
                        elif norm_x > 0.65:
                            h_pos = "right side"
                        else:
                            h_pos = "center"

                        # Vertical / depth spatial position
                        if norm_y > 0.65:
                            v_pos = "foreground"
                        elif norm_y < 0.35:
                            v_pos = "background"
                        else:
                            v_pos = "midground"

                        entity_label = f"Subject {chr(65 + idx)}"
                        detected_entities.append(f"{entity_label} ({h_pos}, {v_pos})")
                        spatial_notes.append(f"movement of {entity_label} on the {h_pos} in the {v_pos}")

            else:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
                cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                prev_hsv_hist = hist

            prev_gray = gray_small

            # Adaptive sampling logic:
            # If high motion or cut, sample more densely; otherwise return to baseline
            is_keyframe = (motion_score > 12.0) or is_cut or (curr_frame_idx == 0)
            if motion_score > 15.0 or is_cut:
                current_step = min_step
            else:
                current_step = max_step

            # Save keyframe image for visual analysis
            frame_filename = f"frame_{curr_frame_idx:06d}_{timestamp:.2f}s.jpg"
            frame_path = job_frames_dir / frame_filename
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            p_frame = ProcessedFrame(
                frame_index=curr_frame_idx,
                timestamp=timestamp,
                is_keyframe=is_keyframe,
                motion_score=round(motion_score, 2),
                relative_path=f"frames/{job_id}/{frame_filename}",
                detected_entities=detected_entities,
                spatial_layout="; ".join(spatial_notes) if spatial_notes else "scene stable"
            )
            frames_list.append(p_frame)

            curr_frame_idx += current_step

        cap.release()

        # Build SceneSegments from boundaries
        scene_boundaries.append(duration)
        scene_boundaries = sorted(list(set(scene_boundaries)))
        segments: List[SceneSegment] = []

        for i in range(len(scene_boundaries) - 1):
            s_start = scene_boundaries[i]
            s_end = scene_boundaries[i + 1]

            # Frames belonging to this segment
            seg_frames = [f for f in frames_list if s_start <= f.timestamp <= s_end]
            key_indices = [f.frame_index for f in seg_frames if f.is_keyframe]
            
            # Aggregate spatial and entity cues
            all_entities = []
            for f in seg_frames:
                all_entities.extend(f.detected_entities)
            unique_entities = sorted(list(set(all_entities)))

            seg = SceneSegment(
                id=i + 1,
                start_time=s_start,
                end_time=s_end,
                keyframe_indices=key_indices,
                entities=unique_entities,
                summary=f"Scene segment {i + 1} from {s_start}s to {s_end}s",
                spatial_relationships=seg_frames[0].spatial_layout if seg_frames else "general scene view"
            )
            segments.append(seg)

        return frames_list, segments

    def extract_audio(self, video_path: Path, job_id: str) -> Optional[Path]:
        """
        Attempts to extract audio track to wav/mp3 for STT.
        Returns the audio file path if audio exists.
        """
        out_audio = AUDIO_DIR / f"{job_id}.wav"
        # Check if ffmpeg is available
        import shutil
        import subprocess

        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            try:
                cmd = [
                    ffmpeg_bin, "-y", "-i", str(video_path),
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                    str(out_audio)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
                if out_audio.exists() and out_audio.stat().st_size > 1000:
                    return out_audio
            except Exception:
                pass
        return None
