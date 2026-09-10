# WareGuard AI

Edge-ready warehouse video intelligence: detects risky product handling
(drop, throw, drag, improper stacking, rough handling) from video, scores
risk, and answers supervisor questions over the results.

**READ FIRST**: before modifying code in a subdirectory, read its AGENTS.md
(if one exists) for local contracts and invariants before README.md — README
is the human quickstart, this file and its children are the ground truth for
agents.

## Commands

`behavior/` and `risk/` need nothing beyond the standard library — the
commands below run and test them even with `requirements.txt` not installed.
`detection/` and `dashboard/` need the full vision stack
(`pip install -r requirements.txt`: torch, ultralytics, opencv, streamlit).

```bash
# Behavior + risk engine, no video/vision deps needed
python run_analysis.py --simulate demo                                   # prove the engine on ground-truth tracks
python run_analysis.py --logs data/logs/detections_sample_warehouse.json # score an existing detection log
python -m unittest tests.test_behavior tests.test_risk                   # 48 tests, ~0.3s
python -m unittest tests.test_behavior.TestScenarioDetection.test_drop -v # single test

# Full suite (needs the vision stack — tests.test_pipeline imports ultralytics)
python -m unittest discover -s tests

# Detection + tracking (needs torch/ultralytics/opencv)
python utils/video_generator.py                                          # generate a synthetic sample clip
python run_detection.py --input data/raw_videos/sample_warehouse.mp4

# Assistant (works with no LLM key — falls back to a heuristic responder)
python run_assistant.py --logs data/logs/events_sim_demo.json --ask "What was the worst event?"
python -m unittest tests.test_assistant

# Dashboard (needs streamlit + the full stack)
streamlit run dashboard/app.py
```

## Architecture — a pipeline, not a monolith

```
detection/  →  behavior/  →  risk/  →  dashboard/ (Streamlit UI)
(video→tracks) (tracks→events) (events→score)  ↘  assistant/ (LLM Q&A)
```

- **`detection/`** — YOLOv8 + ByteTrack inference (`detector.py`), kinematics
  tracking (`tracker.py`), HUD overlay (`visualizer.py`), pipeline glue
  (`pipeline.py`). Entry: `run_detection.py`. Writes
  `data/logs/detections_<name>.{json,csv}`.
- **`behavior/`** — heuristic event detection from tracks. See
  `behavior/AGENTS.md`.
- **`risk/`** — scores behavior events into Low/Medium/High/Critical with an
  explainable breakdown (`scoring.py`), rolls a shift up into a summary with
  repeat-offender escalation (`engine.py`), and exports compact context for
  the assistant (`export.py`). Entry: `run_analysis.py`. Writes
  `data/logs/events_<name>.{json,csv}`.
- **`dashboard/`** — Streamlit app (`app.py`, `events_panel.py`) tying
  detection, behavior/risk, and the assistant together in one UI. Run with
  `streamlit run dashboard/app.py`.
- **`assistant/`** — LLM Q&A over a risk assessment (`qa.py`, `context.py`,
  `client.py`), with a dependency-free heuristic fallback (`heuristic.py`)
  used when no API key is set. Entry: `run_assistant.py`.
- **`utils/`** — `video_generator.py` (synthetic warehouse clips for offline
  testing), `log_exporter.py` (JSON/CSV serializers).
- **`config.py`** — single source for paths, YOLO model/thresholds, and the
  COCO→warehouse class mapping (`WAREHOUSE_CLASSES`).
- **`docs/behavior-risk.md`** — the integration contract for `behavior/` +
  `risk/` (dashboard/assistant API, units, detector signatures). Read this
  before touching either package or wiring new UI against them.

## Global Invariants

- **`behavior/` and `risk/` are source-agnostic and standard-library only.**
  No torch/ultralytics/cv2/numpy/pandas imports anywhere under these two
  packages. They consume the `Track` schema (`behavior/schema.py`), never a
  video or a YOLO result directly — this is what lets the whole analysis
  stack run and test in milliseconds with no vision deps installed. Detection
  is swappable (YOLO output, simulated tracks, hand-written fixtures) without
  either package changing.
- **All behavior thresholds are in object-heights per second**, never pixels
  or frames — normalizing by the object's own bbox height cancels pixel
  scale, dividing by elapsed time cancels frame rate. Don't add a
  pixel-space or frame-count threshold; it will silently break at other
  resolutions/framerates. Covered by `TestScaleInvariance` in
  `tests/test_behavior.py`.
- **`risk_score` (severity) and `priority_score` (severity × detector
  confidence) are different numbers** — rank lists by priority, display
  severity. Never collapse them into one number.
- **An empty event list must never render as "all clear."** Always check
  `assessment.summary.data_quality_warning` — use
  `assessment.summary.headline()` rather than composing a status string from
  raw counts; it already encodes this rule.
- **The assistant works with no LLM key** — `assistant/heuristic.py` is the
  fallback responder used when `WAREGUARD_LLM_API_KEY` / `OPENAI_API_KEY`
  isn't set. Don't make LLM configuration a hard requirement anywhere in the
  pipeline.

## Anti-patterns

- Don't import `torch`/`ultralytics`/`cv2`/`numpy`/`pandas` into `behavior/`
  or `risk/` — pull data through `behavior/schema.py`'s `Track` loaders
  instead.
- Don't add a new warehouse object class by editing detector code — add it to
  `WAREHOUSE_CLASSES` / `CLASS_COLORS` in `config.py`.
- Don't bypass `risk.export.assessment_to_assistant_context` when feeding the
  assistant — raw kinematics/threshold dumps aren't meant to reach the
  prompt.

## Intent Layer

**Before modifying code in a subdirectory, read its AGENTS.md first** to
understand local patterns and invariants.

- **Behavior detection**: `behavior/AGENTS.md` - heuristic event detection
  from object tracks (drop/throw/drag/stack/rough-handling)

## Related Context

- Behavior/risk integration contract (dashboard + assistant API): `docs/behavior-risk.md`
