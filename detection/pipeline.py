"""
WareGuard AI - Detection Pipeline Module
Orchestrates video reading, YOLOv8 detection, ByteTrack trajectory tracking,
HUD annotation rendering, and structured JSON/CSV log export.
"""
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
from tqdm import tqdm

from .detector import YOLOTracker
from .tracker import TrajectoryTracker
from .visualizer import VideoVisualizer
from utils.log_exporter import export_detections_json, export_detections_csv
from config import (
    DEFAULT_MODEL,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    DEFAULT_TRACKER,
    PROCESSED_VIDEOS_DIR,
    LOGS_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WareGuard.Pipeline")


class DetectionPipeline:
    """
    End-to-end detection and tracking pipeline for warehouse video intelligence.
    """
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        conf_threshold: float = CONF_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        tracker: str = DEFAULT_TRACKER,
        target_classes: Optional[Dict[int, str]] = None
    ):
        self.detector = YOLOTracker(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            tracker=tracker,
            target_classes=target_classes
        )
        self.trajectory_tracker = TrajectoryTracker()

    def process_video(
        self,
        input_video_path: str or Path,
        output_video_path: Optional[str or Path] = None,
        save_json: bool = True,
        save_csv: bool = True,
        render_video: bool = True
    ) -> Dict[str, Any]:
        """
        Executes the detection pipeline over an input video clip.
        """
        input_path = Path(input_video_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        # Derive default output paths if not specified
        stem = input_path.stem
        if output_video_path is None:
            output_video_path = PROCESSED_VIDEOS_DIR / f"annotated_{stem}.mp4"
        else:
            output_video_path = Path(output_video_path)

        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        json_log_path = LOGS_DIR / f"detections_{stem}.json"
        csv_log_path = LOGS_DIR / f"detections_{stem}.csv"

        # Open input video
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != fps: # Check for NaN or 0
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0.0

        logger.info(f"Processing '{input_path.name}': {width}x{height} @ {fps:.1f} FPS, {total_frames} frames ({duration_sec:.2f}s)")

        # Prepare Video Writer if rendering video
        writer = None
        if render_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        visualizer = VideoVisualizer(fps=fps, total_frames=total_frames)

        all_detections = []
        start_time = time.time()
        frame_idx = 0

        pbar = tqdm(total=total_frames, desc=f"Analyzing {stem}", unit="frame")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_idx / fps

            # 1. Run YOLO detection + tracking
            raw_detections = self.detector.detect_and_track(frame, frame_idx, timestamp)

            # 2. Update trajectory kinematics (velocity, acceleration, track records)
            frame_enriched_detections = []
            for det in raw_detections:
                enriched = self.trajectory_tracker.update_track(
                    track_id=det["track_id"],
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    bbox=det["bbox"],
                    confidence=det["confidence"],
                    class_id=det["class_id"],
                    class_name=det["class_name"]
                )
                frame_enriched_detections.append(enriched)
                all_detections.append(enriched)

            # 3. Render visual overlays
            if render_video and writer is not None:
                elapsed = max(0.001, time.time() - start_time)
                instant_fps = (frame_idx + 1) / elapsed
                annotated_frame = visualizer.annotate_frame(
                    frame=frame,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    detections=frame_enriched_detections,
                    trajectories=self.trajectory_tracker.tracks,
                    current_fps=instant_fps
                )
                writer.write(annotated_frame)

            frame_idx += 1
            pbar.update(1)

        pbar.close()
        cap.release()
        if writer is not None:
            writer.release()

        total_duration = time.time() - start_time
        avg_fps = frame_idx / total_duration if total_duration > 0 else 0.0

        logger.info(f"Video processing complete in {total_duration:.2f}s ({avg_fps:.1f} FPS)")

        # Prepare Metadata
        metadata = {
            "video_name": input_path.name,
            "video_path": str(input_path.resolve()),
            "total_frames": frame_idx,
            "fps": fps,
            "duration_seconds": round(duration_sec, 2),
            "resolution": [width, height],
            "model_path": self.detector.model_path,
            "processing_time_sec": round(total_duration, 2),
            "processing_fps": round(avg_fps, 2)
        }

        # 4. Export Logs
        track_summaries = self.trajectory_tracker.get_all_summaries()

        json_file = None
        if save_json:
            json_file = export_detections_json(
                output_path=json_log_path,
                metadata=metadata,
                detections=all_detections,
                track_summaries=track_summaries
            )
            logger.info(f"Saved detection JSON log to: {json_file}")

        csv_file = None
        if save_csv:
            csv_file = export_detections_csv(
                output_path=csv_log_path,
                detections=all_detections
            )
            logger.info(f"Saved detection CSV log to: {csv_file}")

        return {
            "metadata": metadata,
            "output_video": str(output_video_path) if render_video else None,
            "json_log": json_file,
            "csv_log": csv_file,
            "total_detections": len(all_detections),
            "unique_tracks": len(track_summaries)
        }
