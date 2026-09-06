# Graph Report - WareGuard-AI  (2026-09-04)

## Corpus Check
- Corpus is ~25,086 words - fits in a single context window. You may not need a graph.

## Summary
- 481 nodes · 999 edges · 32 communities (26 shown, 6 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 72 edges (avg confidence: 0.95)
- Token cost: 73,985 input · 0 output

## Community Hubs (Navigation)
- Behavior Report Summary
- Behavior-Risk Orchestration
- Episode & Context Detection
- Risk Scoring Tests
- Track Geometry & Stacking
- Behavior Detector Classes
- Detection & Dashboard Setup
- Event Export Pipeline
- Track Lifecycle Model
- Scene Simulation Builder
- Drop Detection & Kinematics
- Video Processing Utilities
- Behavior Engine Robustness Tests
- Track Record Management
- Simulation Config & Smoothing
- Video Visualization HUD
- Scenario Detection Tests
- Scene Context & Units
- Behavior Discrimination Tests
- Synthetic Video Generator
- Kinematic Feature Extraction
- Track Loading & Schema
- Trajectory Tracking Module
- Data Quality Honesty Tests
- Scale Invariance Tests
- Dict Serialization Helpers
- Frame Detection Inference
- Assistant Package (Phase 5)
- Dashboard Package (Phase 4)
- Centralized Configuration
- Phase 6 Roadmap

## God Nodes (most connected - your core abstractions)
1. `Track` - 44 edges
2. `BehaviorEvent` - 39 edges
3. `SceneContext` - 35 edges
4. `BehaviorEngine` - 34 edges
5. `TrackPoint` - 29 edges
6. `BehaviorReport` - 25 edges
7. `TrackFeatures` - 20 edges
8. `RiskEngine` - 20 edges
9. `SceneBuilder` - 19 edges
10. `BehaviorDetector` - 18 edges

## Surprising Connections (you probably didn't know these)
- `TestScaleInvariance` --references--> `Object-heights-per-second units (scale invariance)`  [EXTRACTED]
  tests/test_behavior.py → docs/behavior-risk.md
- `TestShiftSummary` --uses--> `BehaviorReport`  [INFERRED]
  tests/test_risk.py → behavior/engine.py
- `TestDataQualityHonesty` --uses--> `SceneContext`  [INFERRED]
  tests/test_behavior.py → behavior/schema.py
- `RiskAssessment` --uses--> `BehaviorEvent`  [INFERRED]
  risk/engine.py → behavior/schema.py
- `RiskEngine` --uses--> `BehaviorEvent`  [INFERRED]
  risk/engine.py → behavior/schema.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Behavior detector taxonomy & overlap resolution** — docs_behavior_risk_drop_behavior, docs_behavior_risk_throw_behavior, docs_behavior_risk_drag_behavior, docs_behavior_risk_improper_stack_behavior, docs_behavior_risk_rough_handling_behavior [EXTRACTED 1.00]
- **WareGuard AI phased roadmap** — readme_phase1_detection_pipeline, readme_phase2_behavior_detection, readme_phase3_risk_scoring, readme_phase4_streamlit_dashboard, readme_phase5_ai_assistant, readme_phase6_polish_submission [EXTRACTED 1.00]
- **Analysis-to-UI integration pipeline** — docs_behavior_risk_behaviorengine, docs_behavior_risk_riskengine, docs_behavior_risk_assessment_to_assistant_context, readme_dashboard_package, readme_assistant_package [EXTRACTED 1.00]

## Communities (32 total, 6 thin omitted)

### Community 0 - "Behavior Report Summary"
Cohesion: 0.07
Nodes (30): BehaviorReport, Any, Everything Phase 2 produces, ready for Phase 3 to score., A human-readable warning when the tracks cannot support analysis. This exists…, assess_report(), Any, WareGuard AI - Risk Assessment Engine (Phase 3) Scores every behavior event,…, Scored events plus the shift roll-up. (+22 more)

### Community 1 - "Behavior-Risk Orchestration"
Cohesion: 0.07
Nodes (40): BehaviorEngine, data_quality_warning / zero-events handling, drag behavior signature, drop behavior signature, improper_stack behavior signature, Known limitation: detection layer track continuity, Overlapping-claim resolution (throw > drop > stack > drag > rough), RiskEngine (+32 more)

### Community 2 - "Episode & Context Detection"
Cohesion: 0.09
Nodes (23): _attach_person_context(), blend_confidence(), find_runs(), Record the closest worker across the episode, in object-heights., Contiguous index ranges where `predicate` holds. `max_break` tolerates that…, Group frame hits into continuous episodes., Abrupt, forceful cargo movement while a worker is within reach. Keyed on…, Map `value` to 0..1: 0 at the threshold, 1 at `full` and beyond. (+15 more)

### Community 3 - "Risk Scoring Tests"
Cohesion: 0.12
Nodes (12): assess_events(), make_event(), WareGuard AI - Risk Engine Tests (Phase 3) python -m unittest tests.test_risk -v, No worker detected is not evidence that no worker was there. The model must…, Severity answers 'how bad if real'; priority answers 'look at this first'.…, One critical incident makes the shift critical. A blended index must not be…, Explainability is a hard requirement: the number must equal the reasons given…, TestPriorityVsSeverity (+4 more)

### Community 4 - "Track Geometry & Stacking"
Cohesion: 0.08
Nodes (7): Vertically adjacent and horizontally overlapping., Shortest edge-to-edge distance in px; 0 when the boxes overlap., Exact-frame lookup. None when the track was not seen that frame., One observation of one object in one frame., Bottom edge - where the object meets the floor plane., Width / height. A carton that squashes on impact widens this., TrackPoint

### Community 5 - "Behavior Detector Classes"
Cohesion: 0.18
Nodes (15): BehaviorDetector, DragDetector, ImproperStackDetector, WareGuard AI - Behavior Detectors (Phase 2) Five heuristic detectors, each…, Base class. Subclasses implement `detect`., Cargo travels ballistically: airborne, fast, and carrying sideways speed. The…, Cargo pushed or pulled along the floor instead of being carried. Defined by…, An upper carton's centre of mass sits too far past its support. Works on pairs… (+7 more)

### Community 6 - "Detection & Dashboard Setup"
Cohesion: 0.16
Nodes (12): WareGuard AI - Configuration Module Central settings for object detection,…, WareGuard AI - Interactive Intelligence Dashboard Streamlit-based inspection…, WareGuard AI - YOLOv8 Detector & Tracker Module Wraps Ultralytics YOLOv8…, Handles frame-by-frame object detection and multi-object tracking using YOLOv8…, YOLOTracker, Detection package for WareGuard AI. Handles YOLOv8 object detection, ByteTrack…, DetectionPipeline, WareGuard AI - Detection Pipeline Module Orchestrates video reading, YOLOv8… (+4 more)

### Community 7 - "Event Export Pipeline"
Cohesion: 0.14
Nodes (16): assessment_to_assistant_context(), export_events_csv(), export_events_json(), Any, Path, WareGuard AI - Event Export (Phase 3) Writes scored events to JSON and CSV for…, Full fidelity: summary, events, metrics and per-event risk factors., Flat view for pandas, Excel and quick eyeballing. (+8 more)

### Community 8 - "Track Lifecycle Model"
Cohesion: 0.11
Nodes (5): The full observed life of one object, ordered by frame., Reference scale for this object. Median resists a few bad boxes., True when the detector lost the object mid-track., Fraction of frames in the track's span where it was actually seen. A low value…, Track

### Community 9 - "Scene Simulation Builder"
Cohesion: 0.16
Nodes (10): A worker standing or walking, feet planted on the floor plane., Control case: a box carried at waist height and set down gently. This must…, Held, released, free fall under gravity, hard stop on the floor., Box pushed along the floor: horizontal travel, no lift., Ballistic arc: horizontal momentum plus an upward launch. Deliberately also…, Upper carton resting on a lower one with its centre well past support., A sharp shove: high jerk at chest height with a worker alongside. Held above…, Accumulates detection rows, then hands back tracks + context. (+2 more)

### Community 10 - "Drop Detection & Kinematics"
Cohesion: 0.13
Nodes (8): DropDetector, An object falls under gravity and stops abruptly. The abrupt stop is what makes…, Locate the frame where speed collapses after the fastest fall. Returns (index,…, Kinematics, Normalised motion state for one track at one observed frame. Sign convention…, A track plus its computed kinematics, indexed for fast lookup., Whether this track can support a kinematic claim at all. A track with 3…, TrackFeatures

### Community 11 - "Video Processing Utilities"
Cohesion: 0.17
Nodes (12): Any, Path, Executes the detection pipeline over an input video clip., TestLogExporters, Utilities package for WareGuard AI. Video generator, log exporters, and helper…, export_detections_csv(), export_detections_json(), Any (+4 more)

### Community 12 - "Behavior Engine Robustness Tests"
Cohesion: 0.26
Nodes (6): BehaviorEngine, build_demo_scene(), A full simulated unloading shift containing every behavior in sequence. Each…, Real detectors flicker. The engine must still find the major events at 15%…, TestRobustness, TestEndToEnd

### Community 13 - "Track Record Management"
Cohesion: 0.16
Nodes (7): Any, Returns the most recent N center coordinates as integer tuples for rendering., Returns comprehensive lifetime metrics for this track., Stores state history and kinematic computations for a single tracked object., Updates an existing track or creates a new one., Appends a new observation and recalculates kinematics (velocity & vertical…, TrackRecord

### Community 14 - "Simulation Config & Smoothing"
Cohesion: 0.17
Nodes (6): moving_average(), Centred moving average that shrinks the window at the edges. Edge-shrinking…, WareGuard AI - Ground-Truth Track Simulator (Phase 2) Emits `Track` objects…, Free-fall acceleration in px/s^2 for this scene's scale., SimConfig, WareGuard AI - Behavior Engine Tests (Phase 2) Runs without ultralytics, torch,…

### Community 15 - "Video Visualization HUD"
Cohesion: 0.22
Nodes (8): Any, ndarray, Renders high-contrast detection overlays, trajectory trails, and telemetry HUD…, Renders all detection boxes, trajectories, and HUD over a single video frame., Renders a sleek top telemetry bar with system status, timestamp, frame count,…, Renders corner-bracketed bounding box, trajectory trail, and informative label…, VideoVisualizer, TestVisualizer

### Community 16 - "Scenario Detection Tests"
Cohesion: 0.31
Nodes (4): Each simulated behavior must produce its own event type and no other., The control case. A detector that fires on careful handling is worse than no…, TestScenarioDetection, types_in()

### Community 17 - "Scene Context & Units"
Cohesion: 0.20
Nodes (5): Path, What the detectors need to know about the scene as a whole. Thresholds are…, Free-fall acceleration expressed in object-heights per second^2. If a box of…, Convert a normalised speed back into metres/second for reporting., SceneContext

### Community 18 - "Behavior Discrimination Tests"
Cohesion: 0.24
Nodes (6): build_scenario(), One isolated behavior, for unit tests and threshold tuning., The separations between detectors, tested directly., A carried box moves fast and high but is *supported*: its vertical acceleration…, A thrown box also falls. Only the more specific claim survives, and the…, TestDiscrimination

### Community 19 - "Synthetic Video Generator"
Cohesion: 0.27
Nodes (10): draw_cardboard_box(), draw_warehouse_background(), draw_worker(), generate_sample_warehouse_video(), ndarray, WareGuard AI - Synthetic Warehouse Video Generator Generates realistic…, Generates a full synthetic warehouse video demonstrating: - Normal carry…, Draws a warehouse floor, wall, shelf racks, and dock markings. (+2 more)

### Community 20 - "Kinematic Feature Extraction"
Cohesion: 0.27
Nodes (9): build_features(), central_difference(), compute_kinematics(), estimate_floor_y(), WareGuard AI - Kinematic Feature Extraction (Phase 2) Turns raw `Track`…, Smooth, differentiate and normalise one track., Estimate the y of the floor plane from where objects come to rest. Uses a high…, Compute kinematics for every track against a shared floor estimate. (+1 more)

### Community 21 - "Track Loading & Schema"
Cohesion: 0.27
Nodes (8): load_tracks_from_csv(), load_tracks_from_json(), Path, WareGuard AI - Behavior Schema Module (Phase 2) The single data contract…, Build tracks from detection dicts (pipeline output or CSV rows)., Read a detections_<video>.csv produced by utils/log_exporter.py., Read a detections_<video>.json and recover the scene context with it., tracks_from_rows()

### Community 22 - "Trajectory Tracking Module"
Cohesion: 0.36
Nodes (4): WareGuard AI - Trajectory Tracker Module Maintains multi-frame spatial…, Manages active and historical track records for all objects across a video., TrajectoryTracker, TestTrajectoryTracker

### Community 23 - "Data Quality Honesty Tests"
Cohesion: 0.29
Nodes (3): Zero events must never be reported as 'no risk' when the input could not…, Mirrors the real sample_warehouse log: many one- and two-frame tracks with no…, TestDataQualityHonesty

### Community 26 - "Frame Detection Inference"
Cohesion: 0.50
Nodes (3): Any, ndarray, Runs tracking inference on a single frame. Returns a list of raw detected…

## Knowledge Gaps
- **13 isolated node(s):** `Phase 1: Setup & Detection Pipeline`, `Phase 6: Polish & Submission`, `run_detection.py (CLI entry point)`, `config.py (centralized configuration)`, `SceneContext.assumed_cargo_height_m` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TestScaleInvariance` connect `Scale Invariance Tests` to `Scene Simulation Builder`, `Simulation Config & Smoothing`, `Behavior-Risk Orchestration`?**
  _High betweenness centrality (0.126) - this node is a cross-community bridge._
- **Why does `Object-heights-per-second units (scale invariance)` connect `Behavior-Risk Orchestration` to `Scale Invariance Tests`?**
  _High betweenness centrality (0.121) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `Track` (e.g. with `_attach_person_context()` and `BehaviorDetector`) actually correct?**
  _`Track` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 12 inferred relationships involving `BehaviorEvent` (e.g. with `_attach_person_context()` and `BehaviorDetector`) actually correct?**
  _`BehaviorEvent` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `SceneContext` (e.g. with `BehaviorDetector` and `analyze_tracks()`) actually correct?**
  _`SceneContext` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `BehaviorEngine` (e.g. with `BehaviorDetector` and `BehaviorEvent`) actually correct?**
  _`BehaviorEngine` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `TrackPoint` (e.g. with `ImproperStackDetector` and `Kinematics`) actually correct?**
  _`TrackPoint` has 3 INFERRED edges - model-reasoned connections that need verification._