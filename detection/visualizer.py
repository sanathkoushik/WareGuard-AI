"""
WareGuard AI - Video Visualizer Module
Renders industrial-grade HUD, telemetry, bounding boxes, track badges, and trajectory trails.
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from config import CLASS_COLORS


class VideoVisualizer:
    """
    Renders high-contrast detection overlays, trajectory trails, and telemetry HUD on video frames.
    """
    def __init__(self, fps: float = 30.0, total_frames: int = 0):
        self.fps = fps
        self.total_frames = total_frames

    def draw_hud(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        active_objects_count: int,
        current_fps: float = 0.0
    ):
        """
        Renders a sleek top telemetry bar with system status, timestamp, frame count, and active tracks.
        """
        h, w = frame.shape[:2]
        hud_height = 48

        # Draw translucent overlay for top HUD
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_height), (20, 24, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Bottom neon border for HUD
        cv2.line(frame, (0, hud_height), (w, hud_height), (0, 180, 255), 2)

        # Left branding
        cv2.putText(
            frame,
            "WAREGUARD AI  |  WAREHOUSE VISION",
            (20, 31),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Center: Time & Frame
        mins = int(timestamp // 60)
        secs = timestamp % 60
        time_str = f"TIME: {mins:02d}:{secs:04.1f}s | FRAME: {frame_idx}"
        if self.total_frames > 0:
            time_str += f"/{self.total_frames}"

        cv2.putText(
            frame,
            time_str,
            (w // 2 - 140, 31),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            (200, 220, 240),
            1,
            cv2.LINE_AA
        )

        # Right: Objects & FPS
        fps_val = current_fps if current_fps > 0 else self.fps
        right_str = f"TRACKS: {active_objects_count}  |  FPS: {fps_val:.1f}"
        cv2.putText(
            frame,
            right_str,
            (w - 260, 31),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            (0, 230, 120),
            1,
            cv2.LINE_AA
        )

    def draw_bounding_box(
        self,
        frame: np.ndarray,
        bbox: List[float],
        track_id: int,
        class_name: str,
        confidence: float,
        speed: float = 0.0,
        trail: Optional[List[tuple]] = None
    ):
        """
        Renders corner-bracketed bounding box, trajectory trail, and informative label badge.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = CLASS_COLORS.get(class_name, CLASS_COLORS["unknown"])

        # 1. Draw Trajectory Trail (fading dots/lines)
        if trail and len(trail) > 1:
            for i in range(1, len(trail)):
                alpha = float(i) / len(trail)
                thickness = max(1, int(3 * alpha))
                cv2.line(frame, trail[i - 1], trail[i], color, thickness, cv2.LINE_AA)
                cv2.circle(frame, trail[i], max(2, int(4 * alpha)), color, -1)

        # 2. Translucent Box Fill
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

        # 3. Main Bounding Box Outline
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # 4. Corner Highlights for Tech Look
        corner_len = min(18, max(5, int((x2 - x1) * 0.2)))
        # Top-Left
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 4)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 4)
        # Top-Right
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 4)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 4)
        # Bottom-Left
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 4)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 4)
        # Bottom-Right
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 4)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 4)

        # 5. Label Badge (ID + Class + Conf)
        label = f"#{track_id} {class_name.upper()} {confidence:.2f}"
        if speed > 1.0:
            label += f" | {speed:.1f}px/f"

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.45, 1)
        badge_y1 = max(48, y1 - th - 8)
        badge_y2 = badge_y1 + th + 8
        badge_x2 = min(frame.shape[1], x1 + tw + 12)

        # Badge Background
        cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)
        # Badge Text
        cv2.putText(
            frame,
            label,
            (x1 + 6, badge_y2 - 5),
            cv2.FONT_HERSHEY_DUPLEX,
            0.45,
            (10, 10, 10),
            1,
            cv2.LINE_AA
        )

    def annotate_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
        detections: List[Dict[str, Any]],
        trajectories: Optional[Dict[int, Any]] = None,
        current_fps: float = 0.0
    ) -> np.ndarray:
        """
        Renders all detection boxes, trajectories, and HUD over a single video frame.
        """
        out_frame = frame.copy()

        # Render each detection
        for det in detections:
            tid = det.get("track_id", -1)
            trail = None
            if trajectories and tid in trajectories:
                trail = trajectories[tid].get_trail(n_points=12)

            self.draw_bounding_box(
                out_frame,
                bbox=det["bbox"],
                track_id=tid,
                class_name=det.get("class_name", "object"),
                confidence=det.get("confidence", 0.0),
                speed=det.get("speed", 0.0),
                trail=trail
            )

        # Render top HUD telemetry
        self.draw_hud(
            out_frame,
            frame_idx=frame_idx,
            timestamp=timestamp,
            active_objects_count=len(detections),
            current_fps=current_fps
        )

        return out_frame
