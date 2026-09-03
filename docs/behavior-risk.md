# Phases 2 & 3 — Behavior Detection & Risk Scoring

Integration notes for the dashboard (Phase 4) and assistant (Phase 5).

## What these packages are

`behavior/` infers handling behaviors from object tracks.
`risk/` scores those behaviors into Low / Medium / High / Critical with an
explainable breakdown, and rolls them up into a shift picture.

Two properties worth knowing before you integrate:

**They are source-agnostic.** Neither package knows what a video is. They
consume a track schema, so YOLOv8 output, ground-truth simulated tracks, and
real CCTV all enter through the same door. Swapping the detector changes
nothing downstream.

**They are standard-library only.** No torch, ultralytics, cv2, numpy or
pandas. You can run and test the whole analysis stack on a machine with no
vision dependencies installed, and it runs in milliseconds rather than minutes.

## Quick use

```bash
# Analyse an existing detection log
python run_analysis.py --logs data/logs/detections_sample_warehouse.json

# Prove the engine with no video and no vision stack
python run_analysis.py --simulate demo

# Test resilience against a flickering detector
python run_analysis.py --simulate demo --noise 4 --dropout 0.2
```

Writes `data/logs/events_<name>.json` and `events_<name>.csv`.

## API for the dashboard

```python
from behavior import BehaviorEngine
from risk import RiskEngine

report = BehaviorEngine(thresholds="default").analyze_json(json_log_path)
assessment = RiskEngine().assess(report)

assessment.summary.headline()       # one-line supervisor summary
assessment.summary.shift_risk_index # 0-100
assessment.summary.timeline         # 30s buckets: {start_s, end_s, events, max_risk, types}
assessment.ranked(limit=10)         # events, most-important-first

for e in assessment.ranked():
    e.event_id, e.event_type, e.severity, e.risk_score
    e.start_time, e.end_time        # seek the video player here
    e.confidence, e.description
    e.risk_factors                  # ["drop baseline (+40)", "impact at 5.0 m/s (+24)", ...]
    e.metrics["priority_score"]
```

Threshold profiles are `"default"`, `"sensitive"`, `"strict"` — good candidates
for a sidebar selector. Every field of `behavior.Thresholds` is a plain float,
so sliders can drive it directly:

```python
from behavior import Thresholds, BehaviorEngine
t = Thresholds(drop_min_peak_fall_speed=slider_value)
report = BehaviorEngine(thresholds=t).analyze(tracks, context)
```

## API for the assistant

```python
from risk.export import assessment_to_assistant_context
context = assessment_to_assistant_context(assessment)
```

Returns a compact dict — shift summary plus per-event `what_happened` and
`why_this_score`. Raw kinematics and threshold dumps are stripped, so it fits
in a prompt without burning tokens on signal the model can't use.

## Two things to respect in the UI

**1. Severity and priority are different numbers.**
`risk_score` is "how bad if this really happened". `priority_score` is that
multiplied by detector confidence — "what to review first". Rank by priority,
display severity. Showing only one of them hides dangerous-but-uncertain
events.

**2. Zero events does not mean "no risk".**
Always surface `assessment.summary.data_quality_warning` when it is not None.
On the current `sample_warehouse` log it reads:

> Only 1 of 32 cargo tracks were usable; results are partial.

An empty event list on unanalysable footage must never render as a green
"all clear" tile. `summary.headline()` already handles this correctly — use it
rather than composing your own string from the counts.

## How the detectors separate

| Behavior | Signature |
|---|---|
| `drop` | vertical fall, acceleration a real fraction of g, **abrupt stop** |
| `throw` | a fall that also carries horizontal momentum **and is unsupported** (vertical accel ≈ g) |
| `drag` | floor contact + horizontal travel + almost no vertical motion |
| `improper_stack` | two cargo boxes vertically adjacent, upper one overhanging past support |
| `rough_handling` | high speed **and** high jerk with a worker within reach |

The non-obvious ones:

- A **drop** requires the abrupt stop. Without it, every careful set-down
  scores as a drop.
- A **throw** requires free-fall vertical acceleration. Horizontal speed at
  height is not enough — a carried box and a shoved box both move fast and
  high; the difference is that a thrown box is unsupported.
- **Rough handling** is keyed on jerk, not speed. A fast smooth carry is fine;
  a sharp yank is not. The threshold (12 h/s²) sits between a controlled
  set-down (~6) and a deliberate shove (~23).

Overlapping claims on the same object at the same moment collapse to the most
specific one (throw > drop > stack > drag > rough). The suppressed alternatives
are appended to the survivor's `description` rather than discarded.

## Units

Every threshold is in **object-heights per second**, never pixels or frames.
Normalising by the object's own median bbox height cancels the pixel scale;
dividing by elapsed time cancels the frame rate. That is why the same numbers
work on 720p and 4K, and it is covered by tests
(`TestScaleInvariance` in `tests/test_behavior.py`).

Physical quantities are reported in m/s using `SceneContext.assumed_cargo_height_m`
(default 0.40 m for a standard carton). Override with `--cargo-height` if the
goods in frame are a different size.

## Tests

```bash
python -m unittest tests.test_behavior tests.test_risk -v
```

48 tests, ~0.3s, no dependencies. The important ones:

- `test_normal_carry_produces_no_events` — the control case. Careful handling
  must produce silence.
- `TestScaleInvariance` — same events at 2× resolution and half frame rate.
- `test_zero_events_on_bad_data_never_says_no_risk`.
- `test_survives_bbox_jitter_and_dropped_frames` — 15% dropout, 3px jitter.

## Known limitation

`behavior/simulation.py` exists because the current detection output cannot
support this analysis: on `sample_warehouse.mp4`, yolov8n produces 46 tracks of
which the longest survives 4 frames, with 37 classified `sports ball`. The
behavior engine is correct and tested, but it will stay quiet on that log until
the detection layer produces continuous tracks — either from real warehouse
footage or from a generator that draws objects COCO can recognise.
