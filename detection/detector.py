"""
WareGuard AI - YOLOv8 Detector & Tracker Module
Wraps Ultralytics YOLOv8 inference with ByteTrack multi-object tracking.
"""
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO
from config import DEFAULT_MODEL, CONF_THRESHOLD, IOU_THRESHOLD, DEFAULT_TRACKER, WAREHOUSE_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WareGuard.Detector")


class YOLOTracker:
    """
    Handles frame-by-frame object detection and multi-object tracking using YOLOv8 + ByteTrack.
    """
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        conf_threshold: float = CONF_THRESHOLD,
        iou_threshold: float = IOU_THRESHOLD,
        tracker: str = DEFAULT_TRACKER,
        target_classes: Optional[Dict[int, str]] = None
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker = tracker
        self.target_classes = target_classes or WAREHOUSE_CLASSES
        self._next_fallback_id = 9000

        logger.info(f"Loading YOLO model from: {model_path}")
        self.model = YOLO(model_path)
        logger.info("YOLO model loaded successfully.")

    def detect_and_track(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        timestamp: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Runs tracking inference on a single frame.
        Returns a list of raw detected objects with tracking IDs and bounding boxes.
        """
        # Run tracking inference
        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            tracker=self.tracker,
            verbose=False
        )

        detections: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            return detections

        r = results[0]
        boxes = r.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        # Extract coordinates, classes, confidences, and track IDs
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)

        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = None

        for idx in range(len(xyxy)):
            cls_id = int(classes[idx])

            # Map class name using warehouse alias or YOLO default name
            if cls_id in self.target_classes:
                class_name = self.target_classes[cls_id]
            elif hasattr(self.model, "names") and cls_id in self.model.names:
                class_name = self.model.names[cls_id]
            else:
                class_name = f"class_{cls_id}"

            # Track ID assignment
            if track_ids is not None and idx < len(track_ids):
                track_id = int(track_ids[idx])
            else:
                track_id = self._next_fallback_id
                self._next_fallback_id += 1

            bbox = [float(coord) for coord in xyxy[idx]]
            conf = float(confs[idx])

            detections.append({
                "frame": frame_idx,
                "timestamp": timestamp,
                "track_id": track_id,
                "class_id": cls_id,
                "class_name": class_name,
                "confidence": conf,
                "bbox": bbox
            })

        return detections
