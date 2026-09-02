"""
Detection package for WareGuard AI.
Handles YOLOv8 object detection, ByteTrack tracking, and video annotation.
"""
from .detector import YOLOTracker
from .tracker import TrajectoryTracker
from .visualizer import VideoVisualizer
from .pipeline import DetectionPipeline

__all__ = ["YOLOTracker", "TrajectoryTracker", "VideoVisualizer", "DetectionPipeline"]
