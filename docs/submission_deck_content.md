# WareGuard AI — Submission Deck Content
GEG AI Video Intelligence for Warehouse Handling — draft slide content, paste into your deck.
Numbers below are real, from this project's own test runs on 2026-09-09
(7 real Mumbai-branch videos + 63 passing unit tests). Fill in [bracketed] placeholders.

---

## Slide 1 — Solution & Team

**WareGuard AI**
[Team name]
[Team members]

> "An AI field intelligence assistant that watches loading and unloading
> footage, explains *why* a handling action is risky, and tells the
> supervisor what to do about it — before the product is damaged, not
> after."

---

## Slide 2 — Problem, Solution & User Journey

**Traditional CCTV**: Camera → Recording → Human review → Incident discovered → Corrective action

**WareGuard AI**: Camera → AI perception → Behaviour understanding → Risk detection → Alert → Intervention → Learning

**Pipeline**: video → tracked objects → behaviour events → explainable risk score → supervisor chat

**Supervisor journey**:
1. Upload or select a shift's footage in the dashboard.
2. Pipeline detects and tracks people + handled material (YOLOv8 + ByteTrack).
3. Behaviour engine flags drop / throw / drag / improper-stack / rough-handling events with a plain-English "why."
4. Risk engine scores each event Low → Critical and rolls the shift into one risk index.
5. Supervisor asks the AI assistant "What was the worst event?" or "What should we do about it?" — gets an answer grounded only in detected events, plus a corrective-action recommendation.
6. Repeat-offender loads and shift trends surface automatically for training/process fixes.

---

## Slide 3 — Technical Architecture & Technology Stack

**Computer vision**: YOLOv8n (Ultralytics), ByteTrack multi-object tracking

**Behaviour intelligence**: custom heuristic engine — Object Detection → Object Tracking → Kinematic feature extraction (smoothed velocity/acceleration, normalised in object-heights/second so it's resolution- and frame-rate-independent) → 5 behaviour detectors → temporal/overlap resolution → Risk classification

**AI/ML**: PyTorch (MPS-accelerated on Apple Silicon), COCO-pretrained detector with a warehouse-object classification layer

**LLM**: optional OpenAI-compatible endpoint for the supervisor assistant; **zero-dependency heuristic responder as default** — the assistant works fully offline, no API key required. LLM answers are grounded only in detected-event JSON, never invented.

**Video processing**: OpenCV (ingestion, HUD overlay rendering, frame scrubbing)

**Edge / cloud infrastructure**: entire behaviour + risk pipeline is **standard-library Python only** (no torch/cv2/numpy dependency) — runs and unit-tests in milliseconds on a machine with no vision stack installed. Detection is swappable (YOLO output, another detector, hand-written fixtures) without the behaviour/risk layer changing. This is what makes edge/offline deployment realistic, not just a slide bullet.

**Front-end**: Streamlit dashboard — video playback + HUD, kinematics charts, structured detection log, risk & safety events panel, AI assistant chat

**Data storage**: structured JSON + CSV logs per shift (`detections_<video>.json/csv`, `events_<video>.json/csv`) — no database dependency, portable, auditable

---

## Slide 4 — Prototype Screenshots & Demo

Include screenshots of:
- Original input video (raw footage)
- Annotated video with HUD overlay: bounding boxes + track trails
- Behaviour detection: an event card showing type, timestamp, confidence
- Risk classification: shift risk index + severity badge (Low/Med/High/Critical)
- Incident replay: frame scrubber jumped to an event's timestamp
- Dashboard: full 5-tab view (Video Playback, Kinematics, Detections Log, Safety Events & Risk, AI Assistant)
- AI assistant: a real Q&A exchange with a recommended corrective action

**Embed a demo video** (3–5 scenarios) — see `docs/demo_video_script.md` in this repo for the exact shot list and narration.

---

## Slide 5 — Impact, Damage Prevention & User Validation

**Reframe** (per the brief's own framing):
- Not: "We detected N damaged products."
- Instead: "We identified [N] high-risk handling events across [N] shift clips and gave the supervisor a specific corrective action for each — before damage occurred."

**Concrete results from this prototype's test run** (7 real Mumbai-branch warehouse clips):
- 1 clip produced a confirmed High-risk event (rough handling during mattress throwing, risk score 58/100, 84% detector confidence) — with an explainable "why" and a recommended corrective action pulled straight from the brief's own Good-Practice table.
- 4 clips produced a confident "no unsafe handling detected" read — genuine clean shifts, not silence.
- 3 clips were honestly flagged as insufficient tracking data rather than false "all clear" — see Responsible AI note below.
- All results, plus a synthetic ground-truth demo scenario (5/5 behaviour types, engine self-test), are reproducible via `python run_analysis.py --simulate demo`.

**Suggested users to validate with**: warehouse supervisor, loading/unloading operator, logistics manager, safety professional.
[Document what they observed and what changed as a result — this is required by the brief; fill in after a validation session.]

**Responsible AI, built in, not bolted on**:
- The system explicitly distinguishes *observed behaviour → potential risk → confirmed damage* — it never reports an empty event list as "all clear" when tracking data was too thin to say so (`data_quality_warning`, tested by `TestDataQualityHonesty`).
- `risk_score` (severity) and `priority_score` (severity × detector confidence) are kept separate so triage never collapses into one misleading number.
- The AI assistant answers only from detected-event data — it is instructed to never invent an event, track ID, or score.
- Runs fully offline / on-device — no footage needs to leave the building for inference.

---

## Notes for a possible Slide 6 (Innovation / Roadmap, if you use the 6-slide max)

- **Edge AI / offline inference**: already true today — the core engine has zero vision-stack dependency.
- **Explainable risk scoring**: every event ships a `why` factor breakdown, not just a number.
- **Roadmap** (be upfront this is future work, not built): orientation classification (vertical-vs-horizontal product handling), dock-level/floor-condition detection, warehouse-specific fine-tuned detector to replace the COCO-proxy classification currently used for real footage.
