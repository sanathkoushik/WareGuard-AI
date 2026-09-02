import unittest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from detection.tracker import TrajectoryTracker, TrackRecord
from detection.visualizer import VideoVisualizer
from utils.log_exporter import export_detections_json, export_detections_csv


class TestTrajectoryTracker(unittest.TestCase):
    def test_track_kinematics(self):
        tracker = TrajectoryTracker(max_history_per_track=50)

        # Frame 0: Initial position
        res0 = tracker.update_track(
            track_id=1,
            frame_idx=0,
            timestamp=0.0,
            bbox=[100.0, 100.0, 150.0, 150.0],
            confidence=0.9,
            class_id=0,
            class_name="person"
        )
        self.assertEqual(res0["velocity"], [0.0, 0.0])
        self.assertEqual(res0["speed"], 0.0)

        # Frame 1: Move right by 10px and down by 20px
        res1 = tracker.update_track(
            track_id=1,
            frame_idx=1,
            timestamp=0.033,
            bbox=[110.0, 120.0, 160.0, 170.0],
            confidence=0.92,
            class_id=0,
            class_name="person"
        )
        self.assertAlmostEqual(res1["velocity"][0], 10.0, places=1)
        self.assertAlmostEqual(res1["velocity"][1], 20.0, places=1)
        self.assertAlmostEqual(res1["acceleration_y"], 20.0, places=1)


class TestVisualizer(unittest.TestCase):
    def test_annotation_render(self):
        vis = VideoVisualizer(fps=30.0, total_frames=100)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detections = [
            {
                "track_id": 1,
                "class_name": "person",
                "confidence": 0.88,
                "bbox": [100.0, 100.0, 200.0, 300.0],
                "speed": 5.2
            },
            {
                "track_id": 2,
                "class_name": "box",
                "confidence": 0.75,
                "bbox": [300.0, 200.0, 380.0, 280.0],
                "speed": 0.0
            }
        ]

        annotated = vis.annotate_frame(
            frame=frame,
            frame_idx=10,
            timestamp=0.333,
            detections=detections,
            current_fps=28.5
        )

        self.assertEqual(annotated.shape, (480, 640, 3))
        self.assertTrue(np.any(annotated > 0)) # Verify annotations were painted


class TestLogExporters(unittest.TestCase):
    def test_json_and_csv_export(self):
        tmp_dir = Path("data/logs/test_tmp")
        tmp_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "video_name": "test_video.mp4",
            "total_frames": 10,
            "fps": 30.0
        }
        detections = [
            {
                "frame": 0,
                "timestamp": 0.0,
                "track_id": 1,
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.95,
                "bbox": [10, 20, 50, 100],
                "center": [30, 60],
                "velocity": [1.0, 2.0],
                "speed": 2.23,
                "acceleration_y": 0.0
            }
        ]

        json_out = export_detections_json(tmp_dir / "test.json", metadata, detections)
        csv_out = export_detections_csv(tmp_dir / "test.csv", detections)

        self.assertTrue(Path(json_out).exists())
        self.assertTrue(Path(csv_out).exists())

        # Cleanup test files
        Path(json_out).unlink(missing_ok=True)
        Path(csv_out).unlink(missing_ok=True)
        tmp_dir.rmdir()


if __name__ == "__main__":
    unittest.main()
