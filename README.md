# 🛡️ WareGuard AI — Warehouse Video Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-green.svg)](https://docs.ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**WareGuard AI** is a lightweight, edge-ready AI video intelligence prototype for warehouse loading and unloading operations. It detects risky product handling behaviors (dropping, dragging, improper stacking, throwing, and rough handling) from pre-recorded video footage, assigns risk scores, and enables interactive AI supervisor queries.

---

## 📁 Repository Architecture

```
Godrej_hackathon/
├── config.py                   # Centralized configuration & warehouse class mappings
├── requirements.txt            # Python dependencies
├── run_detection.py            # CLI entry point for object detection & tracking pipeline
├── detection/                  # Core Object Detection & Tracking (Phase 1)
│   ├── detector.py             # YOLOv8 + ByteTrack inference wrapper
│   ├── tracker.py              # Multi-frame kinematics (velocity, acceleration, trajectories)
│   ├── visualizer.py           # Industrial HUD, bounding boxes & trail overlays
│   └── pipeline.py            # End-to-end video processing pipeline
├── behavior/                   # Heuristic Behavior Detection Engine (Phase 2)
├── risk/                       # Risk Assessment & Severity Engine (Phase 3)
├── dashboard/                  # Streamlit Web App (Phase 4)
├── assistant/                  # LLM Q&A Interface over Event Logs (Phase 5)
├── utils/                      # Helper utilities
│   ├── video_generator.py      # Synthetic warehouse clip generator for offline testing
│   └── log_exporter.py         # JSON and CSV log serializers
├── data/
│   ├── raw_videos/             # Input video clips
│   ├── processed_videos/       # Annotated output videos with overlays
│   └── logs/                   # Structured detection & event logs (.json, .csv)
└── docs/                       # Architecture documentation & presentation notes
```

---

## 🚀 Quickstart & Installation

### 1. Clone / Navigate to Workspace
```bash
cd Godrej_hackathon
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate a Test Clip (Optional)
If you don't have real warehouse footage yet, generate a simulated warehouse scenario:
```bash
python utils/video_generator.py
```
This generates `data/raw_videos/sample_warehouse.mp4`.

### 4. Run Detection & Tracking Pipeline
```bash
# Basic run on sample video
python run_detection.py --input data/raw_videos/sample_warehouse.mp4

# Run with custom model or confidence
python run_detection.py --input data/raw_videos/sample_warehouse.mp4 --model yolov8n.pt --conf 0.25
```

---

## 📊 Structured Output Schema

The pipeline produces two logs in `data/logs/`:

1. **`detections_<video_name>.json`**: Detailed JSON log with metadata, track summaries, and frame-by-frame kinematics.
2. **`detections_<video_name>.csv`**: Tabular CSV log for easy pandas inspection:
   - `frame`, `timestamp`, `track_id`, `class_name`, `confidence`, `bbox_x1..y2`, `center_x..y`, `velocity_x..y`, `speed`, `acceleration_y`.

---

## 🛠️ Phases Roadmap

- [x] **Phase 1: Setup & Detection Pipeline** (Repo structure, YOLOv8 + ByteTrack, log exporter, HUD video overlay)
- [ ] **Phase 2: Behavior Detection Logic** (Heuristics for drop, drag, stacking, rough handling)
- [ ] **Phase 3: Risk Scoring Engine** (Low / Medium / High / Critical with contextual factors)
- [ ] **Phase 4: Streamlit Dashboard** (Video player, timeline, event cards)
- [ ] **Phase 5: AI Assistant** (LLM-powered supervisor query agent)
- [ ] **Phase 6: Polish & Submission** (Shift summaries, deck materials)
