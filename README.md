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
├── run_demo.py                 # ONE-COMMAND end-to-end demo (Phase 6)
├── run_analysis.py             # CLI entry point for behavior + risk analysis
├── run_assistant.py            # CLI entry point for the safety assistant
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
│   ├── app.py                  # Main app: video, kinematics, logs, events, assistant
│   ├── events_panel.py         # Safety Events & Risk tab
│   └── assistant_panel.py      # Safety Assistant tab
├── assistant/                  # Q&A Interface over Event Logs (Phase 5)
│   ├── context.py              # Events log -> compact assistant context
│   ├── heuristic.py            # Offline responder (no LLM required)
│   ├── client.py               # Optional OpenAI-compatible LLM client
│   └── qa.py                   # WarehouseAssistant + LLM grounding check
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

## ⚡ Fastest Path — One Command

The whole pipeline, end to end, with **no dependencies to install**:

```bash
python run_demo.py
```

This runs simulated tracks through behavior detection, risk scoring, export and
the AI assistant, printing each stage. It needs no video, no YOLO weights, no
GPU and no API key — `behavior/`, `risk/` and `assistant/` are standard-library
only. See [Simulated vs. real footage](#-simulated-vs-real-footage) for what
that does and does not prove.

```bash
python run_demo.py --report shift_report.md   # also write a shift report
python run_demo.py --scenario drop            # one isolated behavior
python run_demo.py --logs data/logs/detections_sample_warehouse.json   # real detector output
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

### 4. Run Behavior & Risk Analysis (Phases 2-3)
These packages are **standard-library only** — no torch, cv2, numpy or pandas
needed, so this runs even without the vision stack installed:
```bash
# Prove the engine end-to-end on ground-truth simulated tracks
python run_analysis.py --simulate demo

# Score an existing detection log
python run_analysis.py --logs data/logs/detections_sample_warehouse.json

# Run the whole test suite
python -m unittest discover -s tests
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
non-OpenAI OpenAI-compatible endpoint, `WAREGUARD_LLM_BASE_URL`.

### 7. Launch the Interactive Dashboard (Phase 4)
Ties detection, behavior/risk scoring, and the assistant together in one
Streamlit app - pick a video, run detection, then inspect kinematics, safety
events, and chat with the assistant, all in the browser:
```bash
streamlit run dashboard/app.py
```

---

## 📊 Structured Output Schema

The pipeline produces two logs in `data/logs/`:

1. **`detections_<video_name>.json`**: Detailed JSON log with metadata, track summaries, and frame-by-frame kinematics.
2. **`detections_<video_name>.csv`**: Tabular CSV log for easy pandas inspection:
   - `frame`, `timestamp`, `track_id`, `class_name`, `confidence`, `bbox_x1..y2`, `center_x..y`, `velocity_x..y`, `speed`, `acceleration_y`.

---

## 🔬 Simulated vs. real footage

This distinction matters for judging what the project demonstrates, so it is
stated plainly rather than buried.

| Layer | Demo path | Real-footage path |
| --- | --- | --- |
| Tracks | Simulated by `behavior/simulation.py`, integrating real gravity | YOLOv8 + ByteTrack on video |
| Behavior detection | **Real engine** | **Real engine** |
| Risk scoring | **Real engine** | **Real engine** |
| Dashboard / assistant | **Real** | **Real** |

Only the *tracks* are synthetic on the demo path. Everything downstream is the
same code that runs on real footage, and the simulator integrates
9.81 m/s² through the pixel scale implied by the box size, so a simulated drop
and a filmed drop produce the same normalised signature.

### Known limitation: the bundled sample video

`data/raw_videos/sample_warehouse.mp4` is drawn with OpenCV primitives, and a
COCO-trained YOLOv8 does not recognise flat-shaded rectangles as cargo. On that
clip the detector produces fragmented tracks — roughly **1 usable cargo track in
32**, with most detections classified as unrelated COCO classes.

That is not enough continuous motion to measure a fall speed, a drag distance or
an acceleration from, so the behavior engine reports the shift as **unanalysable
rather than safe**:

```
!! DATA QUALITY
   Only 1 of 32 cargo tracks were usable; results are partial.
   Zero events here means 'not measurable', NOT 'no risk'.
```

See it yourself:

```bash
python run_demo.py --logs data/logs/detections_sample_warehouse.json
```

This limitation is surfaced in the CLI, the dashboard and the assistant. It is
**not** worked around by loosening thresholds, and the project does not claim
that behavior detection is validated on real video. Closing it needs either real
warehouse footage or a generator that draws objects a COCO model recognises.

---

## ⚠️ Known limitations

- **Behavior detection is validated on simulated tracks**, not on real warehouse
  footage — see above.
- **Track IDs are objects, not people.** WareGuard does not identify or track
  individual workers, and "repeat offender" refers to a load, not a person.
- **Event rates from short clips are extrapolations.** A 16-second clip with 5
  events reports 18.7/min; the summary flags this as `rate_is_reliable: false`.
- **Stack tilt is approximated.** Bounding boxes are axis-aligned, so lean is
  inferred from aspect-ratio change rather than measured rotation.
- **The assistant answers from event data only.** It cannot see the video, and
  it will say so rather than guess.
- **`assumed_cargo_height_m` (default 0.40 m)** converts pixel motion into m/s.
  Override with `--cargo-height` if the goods in frame are a different size.

---

## 🧪 Testing

```bash
python -m unittest discover -s tests
```

`behavior/`, `risk/`, `assistant/` and the dashboard panels are standard-library
only, so most of the suite runs before `pip install -r requirements.txt`. The
Phase 1 pipeline tests need the vision stack and **skip** (rather than error)
when it is absent.

---

## 🛠️ Phases Roadmap

- [x] **Phase 1: Setup & Detection Pipeline** (Repo structure, YOLOv8 + ByteTrack, log exporter, HUD video overlay)
- [x] **Phase 2: Behavior Detection Logic** (drop, throw, drag, improper stacking, rough handling — see [docs/behavior-risk.md](docs/behavior-risk.md))
- [x] **Phase 3: Risk Scoring Engine** (Low / Medium / High / Critical, explainable factors, shift roll-up)
- [x] **Phase 4: Streamlit Dashboard** (Video player, kinematics, detections log, safety events & risk, AI assistant chat — see `dashboard/app.py`)
- [x] **Phase 5: AI Assistant** (LLM-powered supervisor query agent, offline heuristic fallback — see `assistant/`)
- [x] **Phase 6: Polish & Submission** (One-command demo, Markdown shift report, assistant dashboard tab, error handling, repo cleanup)
