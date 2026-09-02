"""
WareGuard AI - Trajectory Tracker Module
Maintains multi-frame spatial trajectories, velocities, accelerations, and geometric metrics
for tracked objects across warehouse video footage.
"""
import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class TrackRecord:
    """
    Stores state history and kinematic computations for a single tracked object.
    """
    def __init__(self, track_id: int, class_id: int, class_name: str, max_history: int = 120):
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name
        self.max_history = max_history

        # Time series history
        self.frames: List[int] = []
        self.timestamps: List[float] = []
        self.bboxes: List[List[float]] = []      # [x1, y1, x2, y2]
        self.centers: List[Tuple[float, float]] = [] # (cx, cy)
        self.confidences: List[float] = []
        self.velocities: List[Tuple[float, float]] = [] # (vx, vy) in px/frame
        self.speeds: List[float] = []
        self.vertical_accelerations: List[float] = [] # a_y in px/frame^2

    def update(
        self,
        frame_idx: int,
        timestamp: float,
        bbox: List[float],
        confidence: float,
        class_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Appends a new observation and recalculates kinematics (velocity & vertical acceleration).
        """
        if class_name:
            self.class_name = class_name

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Calculate velocity relative to last frame
        if len(self.centers) > 0:
            prev_cx, prev_cy = self.centers[-1]
            prev_frame = self.frames[-1]
            df = max(1, frame_idx - prev_frame)

            vx = (cx - prev_cx) / df
            vy = (cy - prev_cy) / df
            speed = math.sqrt(vx * vx + vy * vy)
        else:
            vx, vy, speed = 0.0, 0.0, 0.0

        # Calculate vertical acceleration relative to previous velocity
        if len(self.velocities) > 0:
            _, prev_vy = self.velocities[-1]
            prev_frame = self.frames[-1]
            df = max(1, frame_idx - prev_frame)
            ay = (vy - prev_vy) / df
        else:
            ay = 0.0

        # Store in histories
        self.frames.append(frame_idx)
        self.timestamps.append(timestamp)
        self.bboxes.append([float(x1), float(y1), float(x2), float(y2)])
        self.centers.append((float(cx), float(cy)))
        self.confidences.append(float(confidence))
        self.velocities.append((float(vx), float(vy)))
        self.speeds.append(float(speed))
        self.vertical_accelerations.append(float(ay))

        # Trim history if exceeding max_history
        if len(self.frames) > self.max_history:
            self.frames.pop(0)
            self.timestamps.pop(0)
            self.bboxes.pop(0)
            self.centers.pop(0)
            self.confidences.pop(0)
            self.velocities.pop(0)
            self.speeds.pop(0)
            self.vertical_accelerations.pop(0)

        return {
            "track_id": self.track_id,
            "frame": frame_idx,
            "timestamp": timestamp,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": confidence,
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "center": [float(cx), float(cy)],
            "velocity": [float(vx), float(vy)],
            "speed": float(speed),
            "acceleration_y": float(ay),
            "width": float(x2 - x1),
            "height": float(y2 - y1),
            "ground_y": float(y2)
        }

    def get_trail(self, n_points: int = 15) -> List[Tuple[int, int]]:
        """Returns the most recent N center coordinates as integer tuples for rendering."""
        pts = self.centers[-n_points:]
        return [(int(x), int(y)) for x, y in pts]

    def get_summary(self) -> Dict[str, Any]:
        """Returns comprehensive lifetime metrics for this track."""
        if not self.frames:
            return {}
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "first_frame": self.frames[0],
            "last_frame": self.frames[-1],
            "start_time": self.timestamps[0],
            "end_time": self.timestamps[-1],
            "duration_frames": len(self.frames),
            "max_speed": max(self.speeds) if self.speeds else 0.0,
            "max_vertical_accel": max(self.vertical_accelerations) if self.vertical_accelerations else 0.0,
            "avg_confidence": sum(self.confidences) / len(self.confidences) if self.confidences else 0.0
        }


class TrajectoryTracker:
    """
    Manages active and historical track records for all objects across a video.
    """
    def __init__(self, max_history_per_track: int = 150):
        self.max_history = max_history_per_track
        self.tracks: Dict[int, TrackRecord] = {}

    def update_track(
        self,
        track_id: int,
        frame_idx: int,
        timestamp: float,
        bbox: List[float],
        confidence: float,
        class_id: int,
        class_name: str
    ) -> Dict[str, Any]:
        """
        Updates an existing track or creates a new one.
        """
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackRecord(
                track_id=track_id,
                class_id=class_id,
                class_name=class_name,
                max_history=self.max_history
            )

        return self.tracks[track_id].update(
            frame_idx=frame_idx,
            timestamp=timestamp,
            bbox=bbox,
            confidence=confidence,
            class_name=class_name
        )

    def get_track(self, track_id: int) -> Optional[TrackRecord]:
        return self.tracks.get(track_id)

    def get_all_summaries(self) -> Dict[int, Dict[str, Any]]:
        return {tid: track.get_summary() for tid, track in self.tracks.items()}
