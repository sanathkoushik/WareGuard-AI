"""
WareGuard AI - Interactive Intelligence Dashboard
Streamlit-based inspection app for warehouse video intelligence, telemetry, and tracking analysis.
"""
import os
import json
import sys
import time
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import streamlit as st

# `streamlit run` puts this file's own directory (dashboard/) on sys.path,
# not the project root - so top-level packages (assistant, behavior, risk,
# detection) resolve only once the root is added explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant import WarehouseAssistant
from behavior import BehaviorEngine
from behavior.thresholds import PROFILES
from risk import RiskEngine
from risk.export import assessment_to_assistant_context

try:  # streamlit run puts dashboard/ on sys.path, not the repo root
    from dashboard.events_panel import render_events_tab, SEEK_FRAME_KEY, SEEK_TIME_KEY
except ImportError:
    from events_panel import render_events_tab, SEEK_FRAME_KEY, SEEK_TIME_KEY

try:
    from dashboard.fleet_panel import render_fleet_tab
except ImportError:
    from fleet_panel import render_fleet_tab

SEVERITY_COLORS = {
    "Critical": "#f85149",
    "High": "#d29922",
    "Medium": "#58a6ff",
    "Low": "#3fb950",
}

# Configure Streamlit page
st.set_page_config(
    page_title="WareGuard AI — Warehouse Video Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich cyberpunk / modern industrial dashboard styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #8892b0;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(22, 27, 34, 0.7);
        border: 1px solid rgba(56, 139, 253, 0.2);
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-person {
        background-color: rgba(31, 111, 235, 0.2);
        color: #58a6ff;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #1f6feb;
    }
    .badge-box {
        background-color: rgba(210, 153, 34, 0.2);
        color: #e3b341;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #d29922;
    }
    .badge-severity {
        padding: 3px 10px;
        border-radius: 4px;
        font-weight: 600;
        margin-right: 6px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw_videos"
PROCESSED_DIR = BASE_DIR / "data" / "processed_videos"
LOGS_DIR = BASE_DIR / "data" / "logs"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Header Section
st.markdown('<div class="main-header">🛡️ WareGuard AI — Warehouse Video Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated object detection, multi-object tracking, and kinematic analysis for warehouse safety.</div>', unsafe_allow_html=True)

# Sidebar: Video Selection & Pipeline Runner
VIDEO_SELECT_KEY = "wg_video_select"
PENDING_SELECT_KEY = "wg_pending_video_select"
AUTO_RUN_KEY = "wg_auto_run"

with st.sidebar:
    st.header("⚙️ Video Selection & Config")

    raw_files = list(RAW_DIR.glob("*.mp4")) + list(RAW_DIR.glob("*.avi")) + list(RAW_DIR.glob("*.mov"))
    video_stems = {f.stem for f in raw_files}

    # display label -> {"stem": ..., "filename": ... or None}
    video_choices = {f.name: {"stem": f.stem, "filename": f.name} for f in raw_files}

    # Some committed detection logs (real footage, scored in advance) have no
    # matching video file - large raw clips aren't checked into git (see
    # .gitignore). Surface them too so that demo data isn't invisible after a
    # fresh clone; the video tabs below already degrade gracefully when the
    # source file is missing, showing analysis from the saved log alone.
    log_only_stems = sorted(
        p.stem[len("detections_"):]
        for p in LOGS_DIR.glob("detections_*.json")
        if p.stem[len("detections_"):] not in video_stems
    )
    for stem in log_only_stems:
        video_choices[f"{stem}  (log only — source video not included)"] = {
            "stem": stem, "filename": None
        }

    video_options = list(video_choices.keys())

    # A webcam capture or upload completed on the previous run wants this
    # dropdown to open pre-selected on it - written into session state before
    # the widget is created, same pattern the "jump to event" seek uses.
    pending = st.session_state.pop(PENDING_SELECT_KEY, None)
    if pending and pending in video_choices:
        st.session_state[VIDEO_SELECT_KEY] = pending

    selected_label = None
    if video_options:
        selected_label = st.selectbox(
            "Select Available Video",
            options=video_options,
            index=0,
            key=VIDEO_SELECT_KEY,
        )
    else:
        st.warning("No video files found in `data/raw_videos`.")

    st.markdown("---")
    st.subheader("📤 Upload New Video")
    uploaded_file = st.file_uploader("Upload warehouse clip (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        save_path = RAW_DIR / uploaded_file.name
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved: {uploaded_file.name}")
        video_choices[uploaded_file.name] = {"stem": Path(uploaded_file.name).stem, "filename": uploaded_file.name}
        selected_label = uploaded_file.name

    st.markdown("---")
    st.subheader("📷 Live Camera Capture")
    st.caption("Record straight from a webcam attached to this machine — no pre-recorded footage needed.")
    cam_index = st.number_input("Camera index", min_value=0, max_value=4, value=0, step=1)
    cam_duration = st.slider("Recording length (seconds)", min_value=3, max_value=30, value=8)
    capture_button = st.button("🔴 Record from Webcam", use_container_width=True)

    if capture_button:
        from utils.webcam_capture import record_webcam_clip
        capture_filename = f"webcam_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        capture_path = RAW_DIR / capture_filename
        preview_slot = st.empty()
        try:
            def _on_frame(frame_bgr, elapsed, duration):
                preview_slot.image(
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                    caption=f"Recording… {elapsed:.1f}/{duration:.0f}s",
                    use_container_width=True,
                )

            record_webcam_clip(
                capture_path,
                duration_s=float(cam_duration),
                camera_index=int(cam_index),
                on_frame=_on_frame,
            )
            preview_slot.empty()
            st.success(f"✅ Captured {cam_duration}s clip: {capture_filename}")
            # Pre-select it and immediately run detection on rerun - "point
            # webcam, click record, see it flagged" in one motion.
            st.session_state[PENDING_SELECT_KEY] = capture_filename
            st.session_state[AUTO_RUN_KEY] = True
            st.rerun()
        except Exception as exc:
            preview_slot.empty()
            st.error(
                f"❌ Webcam capture failed: {exc}\n\n"
                "Common causes: no webcam attached, it's in use by another "
                "application, or this environment has no camera at all "
                "(e.g. a headless server)."
            )

    st.markdown("---")
    st.subheader("🚀 Pipeline Settings")
    model_choice = st.selectbox("YOLOv8 Model", ["yolov8n.pt", "yolov8s.pt"], index=0)
    conf_thresh = st.slider("Detection Confidence", min_value=0.1, max_value=0.9, value=0.30, step=0.05)

    run_button = st.button("▶️ Run Detection Pipeline", type="primary", use_container_width=True)

    st.markdown("---")
    st.subheader("🧠 Behavior & Risk Profile")
    threshold_profile = st.selectbox(
        "Threshold Profile",
        options=list(PROFILES),
        index=list(PROFILES).index("default"),
        help="Sensitive flags more, strict flags fewer - see behavior/thresholds.py",
    )

st.markdown("---")
with st.expander("🏆 Cross-Shift Leaderboard — riskiest shifts across all analyzed footage", expanded=False):
    render_fleet_tab(LOGS_DIR, profile=threshold_profile)
st.markdown("---")

if selected_label:
    entry = video_choices[selected_label]
    stem = entry["stem"]
    input_video_path = RAW_DIR / (entry["filename"] or f"{stem}.mp4")
    has_source_video = entry["filename"] is not None
    output_video_path = PROCESSED_DIR / f"annotated_{stem}.mp4"
    json_log_path = LOGS_DIR / f"detections_{stem}.json"
    csv_log_path = LOGS_DIR / f"detections_{stem}.csv"

    # If user clicked Run Detection Pipeline (directly, or via an
    # auto-triggered run right after a webcam capture)
    if run_button or st.session_state.pop(AUTO_RUN_KEY, False):
        if not has_source_video:
            st.sidebar.error("Selected entry has no source video to process.")
        else:
            with st.spinner("Processing video with YOLOv8 & ByteTrack..."):
                try:
                    from detection.pipeline import DetectionPipeline
                    pipeline = DetectionPipeline(model_path=model_choice, conf_threshold=conf_thresh)
                    result = pipeline.process_video(
                        input_video_path=input_video_path,
                        output_video_path=output_video_path,
                        save_json=True,
                        save_csv=True,
                        render_video=True
                    )
                    st.success("✅ Detection pipeline completed successfully!")
                except Exception as exc:
                    st.error(
                        f"❌ Detection pipeline failed: {exc}\n\n"
                        "Common causes: the model weights couldn't be downloaded "
                        "(no network on first run) or the video file is corrupt/unreadable."
                    )

    # Check if processed logs exist
    has_logs = json_log_path.exists()
    metadata = {}
    summary = {}
    detections_data = []

    if has_logs:
        with open(json_log_path, "r", encoding="utf-8") as f:
            log_json = json.load(f)
            metadata = log_json.get("video_metadata", {})
            summary = log_json.get("summary", {})
            detections_data = log_json.get("detections", [])

    # Phases 2-3: behavior detection + risk scoring, run straight off the
    # detection log. Standard-library only and cheap, so it's safe to
    # recompute on every rerun rather than caching (see run_analysis.py).
    behavior_report = None
    assessment = None
    if has_logs:
        try:
            behavior_report = BehaviorEngine(thresholds=threshold_profile).analyze_json(json_log_path)
            assessment = RiskEngine().assess(behavior_report)
        except Exception as exc:
            st.sidebar.error(f"Behavior/risk analysis failed: {exc}")

    # Proactive alerting: a supervisor shouldn't have to open the Safety
    # Events tab to learn a shift had a critical incident. Toast once per
    # event (tracked per-video in session state so a rerun doesn't re-fire
    # the same toast), and keep a persistent banner up for the whole shift.
    if assessment is not None:
        critical_events = [e for e in assessment.events if e.severity == "Critical"]
        high_events = [e for e in assessment.events if e.severity == "High"]

        alerted_key = f"wg_alerted_{stem}"
        if alerted_key not in st.session_state:
            st.session_state[alerted_key] = set()
        for e in critical_events:
            eid = e.event_id or f"trk{e.track_id}"
            if eid not in st.session_state[alerted_key]:
                st.toast(f"Critical: {e.description}", icon="🚨")
                st.session_state[alerted_key].add(eid)

        if critical_events:
            st.error(
                f"🚨 **{len(critical_events)} critical safety event"
                f"{'s' if len(critical_events) != 1 else ''} detected in this shift.** "
                "See the **Safety Events & Risk** tab for details.",
                icon="🚨",
            )
        elif high_events:
            st.warning(
                f"⚠️ **{len(high_events)} high-severity event"
                f"{'s' if len(high_events) != 1 else ''} detected in this shift.** "
                "See the **Safety Events & Risk** tab for details.",
                icon="⚠️",
            )

    # Top KPI Metrics Row
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Frames</div>
            <div class="metric-value">{metadata.get('total_frames', '-')}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Duration</div>
            <div class="metric-value">{metadata.get('duration_seconds', '-')}s</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Inference FPS</div>
            <div class="metric-value">{metadata.get('processing_fps', '-')}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Unique Tracks</div>
            <div class="metric-value">{summary.get('unique_tracks', '-')}</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Detections</div>
            <div class="metric-value">{summary.get('total_detections', '-')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Dashboard Tabs
    tab_video, tab_kinematics, tab_table, tab_events, tab_assistant = st.tabs([
        "📹 Video Playback & HUD",
        "📈 Kinematics & Velocity Analytics",
        "📋 Detections & Trajectory Log",
        "🚨 Safety Events & Risk",
        "🤖 AI Assistant"
    ])

    with tab_video:
        col_v1, col_v2 = st.columns(2)

        with col_v1:
            st.subheader("Raw Input Video")
            if input_video_path.exists():
                st.video(str(input_video_path))
            else:
                st.info("Input video file not found.")

        with col_v2:
            st.subheader("Annotated Video (HUD + Trackers)")
            if output_video_path.exists():
                # Provide native video playback
                st.video(str(output_video_path))
            else:
                st.info("Annotated video not generated yet. Click 'Run Detection Pipeline' in the sidebar to process.")

        # Interactive Frame-by-Frame Inspector
        if output_video_path.exists():
            st.markdown("---")
            st.subheader("🔍 Interactive Frame Scrubber")
            cap = cv2.VideoCapture(str(output_video_path))
            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = metadata.get("fps") or cap.get(cv2.CAP_PROP_FPS) or 30.0
            if total_f > 0:
                # A "Jump to this event" click on the Safety Events tab writes
                # SEEK_FRAME_KEY into session state; apply it to the slider's
                # own state before the widget is created so it actually moves.
                frame_slider_key = "wg_frame_slider"
                seek_frame = st.session_state.pop(SEEK_FRAME_KEY, None)
                st.session_state.pop(SEEK_TIME_KEY, None)
                if seek_frame is not None:
                    st.session_state[frame_slider_key] = max(0, min(int(seek_frame), total_f - 1))

                slider_kwargs = dict(min_value=0, max_value=total_f - 1, step=1, key=frame_slider_key)
                if frame_slider_key not in st.session_state:
                    slider_kwargs["value"] = min(50, total_f - 1)
                frame_slider = st.slider("Select Frame Index", **slider_kwargs)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_slider)
                ret, frame_img = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption=f"Frame #{frame_slider} (t = {frame_slider/video_fps:.2f}s)", use_container_width=True)
            cap.release()

    with tab_kinematics:
        st.subheader("📊 Track Kinematics & Motion Signals")
        if csv_log_path.exists():
            df = pd.read_csv(csv_log_path)
            if not df.empty:
                col_k1, col_k2 = st.columns(2)
                with col_k1:
                    st.write("**Object Speed (px/frame) Over Time**")
                    speed_chart_df = df.pivot_table(index="timestamp", columns="track_id", values="speed", fill_value=0)
                    st.line_chart(speed_chart_df)

                with col_k2:
                    st.write("**Vertical Acceleration ($a_y$) Over Time**")
                    accel_chart_df = df.pivot_table(index="timestamp", columns="track_id", values="acceleration_y", fill_value=0)
                    st.line_chart(accel_chart_df)

                st.write("**Track Kinematic Summaries**")
                track_summary_table = df.groupby(["track_id", "class_name"]).agg(
                    First_Frame=("frame", "min"),
                    Last_Frame=("frame", "max"),
                    Max_Speed=("speed", "max"),
                    Max_Vert_Accel=("acceleration_y", "max"),
                    Avg_Confidence=("confidence", "mean")
                ).reset_index()
                st.dataframe(track_summary_table, use_container_width=True)
            else:
                st.info("CSV log is empty.")
        else:
            st.info("Run the detection pipeline to generate kinematic data.")

    with tab_table:
        st.subheader("📋 Structured Frame Detections Log")
        if csv_log_path.exists():
            df_full = pd.read_csv(csv_log_path)
            if not df_full.empty:
                # Filtering options
                c_filter1, c_filter2 = st.columns(2)
                with c_filter1:
                    class_filter = st.multiselect("Filter by Class", options=df_full["class_name"].unique(), default=df_full["class_name"].unique())
                with c_filter2:
                    track_filter = st.multiselect("Filter by Track ID", options=sorted(df_full["track_id"].unique()), default=[])

                filtered_df = df_full[df_full["class_name"].isin(class_filter)]
                if track_filter:
                    filtered_df = filtered_df[filtered_df["track_id"].isin(track_filter)]

                st.dataframe(filtered_df, use_container_width=True, height=400)

                # Download buttons
                st.download_button(
                    label="⬇️ Download Detections CSV",
                    data=df_full.to_csv(index=False),
                    file_name=f"detections_{stem}.csv",
                    mime="text/csv"
                )
            else:
                st.info("Log table is empty.")
        else:
            st.info("No detections log found. Please run the pipeline first.")

    with tab_events:
        render_events_tab(json_log_path, output_video_path, logs_dir=LOGS_DIR)

    with tab_assistant:
        st.subheader("🤖 Ask the AI Assistant")
        st.caption(
            "Ask a question about this shift — e.g. \"What was the worst "
            "event?\" or \"Any repeat offenders?\""
        )

        if assessment is None:
            st.info(
                "Run the detection pipeline and confirm behavior/risk analysis "
                "succeeded (see the sidebar) to enable the assistant."
            )
        else:
            assistant = WarehouseAssistant(assessment_to_assistant_context(assessment))

            # A fresh WarehouseAssistant is constructed every rerun, so its
            # last_source/last_error defaults reset each time - the true
            # outcome of the last *actual* call has to be persisted in
            # session state (per shift) to survive the rerun below.
            status_key = f"wg_llm_status_{stem}"
            status = st.session_state.get(status_key, {"source": "unconfigured", "error": None})

            if not assistant.client.available:
                st.caption(
                    "⚙️ No LLM configured (set `WAREGUARD_LLM_API_KEY` or "
                    "`OPENAI_API_KEY`, optionally via a `.env` file) — answering "
                    "with the built-in heuristic responder."
                )
            elif status["source"] == "llm":
                st.caption(f"🧠 Answering with LLM model `{assistant.client.model}` — last call succeeded.")
            elif status["source"] == "fallback":
                st.caption(
                    f"⚠️ LLM call failed ({status['error']}) — falling back to the "
                    f"heuristic responder. Model configured: `{assistant.client.model}`."
                )
            else:
                st.caption(
                    f"🧠 LLM configured (`{assistant.client.model}`) — ask a "
                    "question to confirm it responds."
                )

            chat_key = f"wg_chat_{stem}"
            if chat_key not in st.session_state:
                st.session_state[chat_key] = []

            for role, text in st.session_state[chat_key]:
                with st.chat_message(role):
                    st.markdown(text)

            question = st.chat_input("Ask about this shift...")
            if question:
                st.session_state[chat_key].append(("user", question))
                with st.spinner("Thinking..."):
                    answer = assistant.ask(question)
                st.session_state[status_key] = {"source": assistant.last_source, "error": assistant.last_error}
                st.session_state[chat_key].append(("assistant", answer))
                st.rerun()
else:
    st.info("Please select or upload a video clip in the sidebar.")
