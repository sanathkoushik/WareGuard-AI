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
├── run_analysis.py             # CLI entry point for behavior + risk analysis
├── behavior/                   # Heuristic Behavior Detection Engine (Phase 2)
│   ├── schema.py               # Track/event data contract — the source-agnostic seam
│   ├── thresholds.py           # All tunables, in object-heights per second
│   ├── features.py             # Smoothed, normalised kinematics
│   ├── detectors.py            # drop / throw / drag / stacking / rough handling
│   ├── engine.py               # Orchestration + overlap resolution
│   └── simulation.py           # Ground-truth track generator (no video needed)
├── risk/                       # Risk Assessment & Severity Engine (Phase 3)
│   ├── scoring.py              # Explainable per-event risk model
│   ├── engine.py               # Shift roll-up & repeat-offender escalation
│   └── export.py               # Event JSON/CSV + compact LLM context
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
> **Heads up:** `torch` + `ultralytics` are large (~1–2 GB combined) and the
> first "Run Detection Pipeline" click downloads the YOLOv8 weights file over
> the network. Do this ahead of a demo, not on conference wifi in the room.

### 3. Generate a Test Clip (Optional)
If you don't have real warehouse footage yet, generate a simulated warehouse scenario:
```bash
python utils/video_generator.py
```
This generates `data/raw_videos/sample_warehouse.mp4`.

### 4. Run Behavior & Risk Analysis (Phases 2-3)
These packages are **standard-library only** — no torch, cv2, numpy or pandas
needed, so this runs even without the vision stack installed:
```bash
# Prove the engine end-to-end on ground-truth simulated tracks
python run_analysis.py --simulate demo

# Score an existing detection log
python run_analysis.py --logs data/logs/detections_sample_warehouse.json

# Run the test suite (48 tests, ~0.3s)
python -m unittest tests.test_behavior tests.test_risk
```
See [docs/behavior-risk.md](docs/behavior-risk.md) for the dashboard and
assistant integration API.

### 5. Run Detection & Tracking Pipeline
```bash
# Basic run on sample video
python run_detection.py --input data/raw_videos/sample_warehouse.mp4

# Run with custom model or confidence
python run_detection.py --input data/raw_videos/sample_warehouse.mp4 --model yolov8n.pt --conf 0.25
```

### 6. Ask the AI Assistant About a Shift (Phase 5)
Works from a saved events log or a simulated scenario, no LLM key required —
falls back to a built-in heuristic responder when `WAREGUARD_LLM_API_KEY` /
`OPENAI_API_KEY` isn't set:
```bash
# One-shot question against a saved log
python run_assistant.py --logs data/logs/events_sim_demo.json --ask "What was the worst event?"

# Interactive session
python run_assistant.py --logs data/logs/events_sim_demo.json

# Ask about a fresh simulated shift, no log file needed
python run_assistant.py --simulate demo --ask "Any repeat offenders?"

# Run the test suite
python -m unittest tests.test_assistant
```
To use a real LLM instead of the heuristic responder, set `OPENAI_API_KEY` (or
`WAREGUARD_LLM_API_KEY` for a non-OpenAI account) and, for a self-hosted or
non-OpenAI OpenAI-compatible endpoint, `WAREGUARD_LLM_BASE_URL`. These can
also be set in a `.env` file in the repo root (loaded automatically via
`python-dotenv`) instead of exporting them in the shell. The dashboard's
"AI Assistant" tab reports the *actual* outcome of the last call — a bad key
or a down API is shown as a fallback, never reported as "using the LLM".

### 7. Launch the Interactive Dashboard (Phase 4)
Ties detection, behavior/risk scoring, and the assistant together in one
Streamlit app - pick a video, run detection, then inspect kinematics, safety
events, and chat with the assistant, all in the browser:
```bash
streamlit run dashboard/app.py
```

The dashboard also offers:
- **📷 Live Camera Capture** (sidebar) — record a short clip straight from a
  webcam attached to the machine running the dashboard and it's analyzed
  immediately, no pre-recorded footage required.
- **🚨 Proactive alerts** — a Critical or High-severity event in the current
  shift surfaces as a banner and a toast the moment analysis finishes,
  instead of waiting for someone to open the Safety Events tab.
- **🏆 Cross-Shift Leaderboard** — an always-available panel ranking every
  shift already scored in `data/logs` by risk index, for spotting the
  riskiest shift or a repeat-offender pattern across footage at a glance.

---

## 📊 Structured Output Schema

The pipeline produces two logs in `data/logs/`:

1. **`detections_<video_name>.json`**: Detailed JSON log with metadata, track summaries, and frame-by-frame kinematics.
2. **`detections_<video_name>.csv`**: Tabular CSV log for easy pandas inspection:
   - `frame`, `timestamp`, `track_id`, `class_name`, `confidence`, `bbox_x1..y2`, `center_x..y`, `velocity_x..y`, `speed`, `acceleration_y`.

---

## 🌐 Deployment (Sharing a Live Link)

By default the dashboard only runs on `localhost`. For a hackathon demo you
want a link a judge can open without cloning the repo:

**Option A — Streamlit Community Cloud (fastest, free):**
1. Push this repo to GitHub (it already is).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo,
   and set the main file path to `dashboard/app.py`.
3. `requirements.txt` is installed automatically. If you want the AI
   Assistant to use a real LLM there, add `WAREGUARD_LLM_API_KEY` (or
   `OPENAI_API_KEY`) under the app's **Secrets** — it's read the same way a
   local `.env` file is.
4. Note: Community Cloud containers have no webcam, so **Live Camera
   Capture** only works when the dashboard is run locally on a machine with
   one attached — everything else works the same.

**Option B — Docker (self-hosted, works anywhere Docker runs):**
```bash
docker build -t wareguard-ai .
docker run -p 8501:8501 wareguard-ai
```
Then open `http://localhost:8501` (or the host's address if run on a server).
Same webcam caveat as above applies unless the container is given device
access (`--device=/dev/video0` on Linux).

---

## 🛠️ Phases Roadmap

- [x] **Phase 1: Setup & Detection Pipeline** (Repo structure, YOLOv8 + ByteTrack, log exporter, HUD video overlay)
- [x] **Phase 2: Behavior Detection Logic** (drop, throw, drag, improper stacking, rough handling — see [docs/behavior-risk.md](docs/behavior-risk.md))
- [x] **Phase 3: Risk Scoring Engine** (Low / Medium / High / Critical, explainable factors, shift roll-up)
- [x] **Phase 4: Streamlit Dashboard** (Video player, kinematics, detections log, safety events & risk, AI assistant chat — see `dashboard/app.py`)
- [x] **Phase 5: AI Assistant** (LLM-powered supervisor query agent, offline heuristic fallback — see `assistant/`)
- [ ] **Phase 6: Polish & Submission** (Shift summaries, deck materials)
