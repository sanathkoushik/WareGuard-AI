# WareGuard AI — Demo Video Script

Target: 3-5 minutes, 5 scenarios (brief asks for 3-5). Screen-record the
Streamlit dashboard at `localhost:8501`. Each scenario below is a real,
already-verified result from this project — no need to re-run anything,
just navigate and narrate.

Setup before recording: `streamlit run dashboard/app.py`, open in browser,
sidebar → confirm you're on **default** threshold profile.

---

## Scene 1 — The hero shot: a real detected risk event (45s)

1. Sidebar → **Select Available Video** → `Throwing Mattresses.mp4`.
2. Say: *"WareGuard AI ingests real warehouse footage — no staged demo clip.
   This is unedited footage of mattresses being handled during unloading."*
3. Click **Video Playback & HUD** tab. Point at the annotated video —
   bounding boxes + track trails on the person and the handled cargo.
4. Click **Safety Events & Risk** tab.
5. Say: *"The system flagged this as a High-risk rough-handling event —
   58 out of 100, 84% confidence — and explains exactly why: a sharp jerk
   in motion, 43 heights-per-second-squared, with a worker within 1.3
   box-heights of the impact."*
6. Point at the risk factor breakdown (the "why" line under the event).

## Scene 2 — Confident clean, not silence (30s)

1. Switch video → `Stepping on cartons, vertical product kept horizontally, heavy product kept on top.mp4`.
2. Safety Events & Risk tab.
3. Say: *"Not every clip has an incident. Here the system explicitly
   confirms no unsafe handling was detected — this is a genuine
   zero-incident read, not the system staying quiet."*
4. Point at the green "No unsafe handling detected" banner and the
   Low/0-events risk card.

## Scene 3 — Responsible AI: honest about its own limits (30s)

1. Switch video → `Dock level, dragging cupboard.mp4`.
2. Safety Events & Risk tab.
3. Say: *"This is the important part. Here the tracker couldn't hold onto
   most of the cargo objects long enough to measure their motion reliably.
   Instead of guessing, WareGuard AI says so directly: 'insufficient
   tracking data' — not a false all-clear. We built this distinction in
   deliberately, because a warehouse safety tool that fakes confidence is
   worse than one that admits uncertainty."*
4. Point at the amber "data quality" warning card.

## Scene 4 — The AI assistant, live (45s)

1. Switch back to `Throwing Mattresses.mp4`.
2. Click **AI Assistant** tab.
3. Type live: `What was the worst event and what should we do about it?`
4. Say while it answers: *"This runs fully offline by default — no LLM
   API key needed — and it only ever answers from the events this
   pipeline actually detected, never invented information."*
5. Read the answer aloud as it appears: event id, severity, risk score,
   the "why," and the **recommended action** — note this recommendation
   text is pulled directly from the challenge brief's own Good-Practice
   table ("Handle every product carefully and in a controlled manner,
   particularly at transfer points.").

## Scene 5 — All 5 behaviour types in one pass (45s)

1. Terminal (or narrate over a pre-recorded terminal clip):
   `python run_analysis.py --simulate demo`
2. Say: *"To prove all five behaviour detectors work end to end, here's a
   ground-truth simulated shift — no video needed, this is our own
   engine's self-test. Drop, throw, drag, improper stacking, and rough
   handling — all five flagged, scored, and ranked by priority in under a
   second."*
3. Point at the terminal output: risk index 74/100 Critical, 5 events,
   each with its own explainable score.

---

## Closing line (10s)

*"Seven real warehouse clips, one offline AI assistant, zero cloud
dependency, and every single risk claim comes with an explanation and a
recommended fix. That's WareGuard AI."*

---

### Recording notes
- Record at 1920x1080 if possible; the dashboard is not mobile-optimised.
- Pause 1-2s on each risk/score number so it's readable at video speed.
- If time is short, cut Scene 5 first (it's the least visual) before
  cutting Scene 3 (Responsible AI is a judged criterion — keep it).
- Total budget check: Scenes 1+2+3+4+5 ≈ 3m15s narrated, fits the "short
  demo video" ask on Slide 4 without padding.
