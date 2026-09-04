"""
WareGuard AI - Interactive Intelligence Dashboard
Streamlit-based inspection app for warehouse video intelligence, telemetry, and tracking analysis.
"""
import os
import json
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import streamlit as st

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
with st.sidebar:
    st.header("⚙️ Video Selection & Config")

    raw_files = list(RAW_DIR.glob("*.mp4")) + list(RAW_DIR.glob("*.avi")) + list(RAW_DIR.glob("*.mov"))
    video_options = [f.name for f in raw_files]

    selected_video_name = None
    if video_options:
        selected_video_name = st.selectbox(
            "Select Available Video",
            options=video_options,
            index=0
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
        selected_video_name = uploaded_file.name

    st.markdown("---")
    st.subheader("🚀 Pipeline Settings")
    model_choice = st.selectbox("YOLOv8 Model", ["yolov8n.pt", "yolov8s.pt"], index=0)
    conf_thresh = st.slider("Detection Confidence", min_value=0.1, max_value=0.9, value=0.30, step=0.05)

    run_button = st.button("▶️ Run Detection Pipeline", type="primary", use_container_width=True)

if selected_video_name:
    input_video_path = RAW_DIR / selected_video_name
    stem = input_video_path.stem
    output_video_path = PROCESSED_DIR / f"annotated_{stem}.mp4"
    json_log_path = LOGS_DIR / f"detections_{stem}.json"
    csv_log_path = LOGS_DIR / f"detections_{stem}.csv"

    # If user clicked Run Detection Pipeline
    if run_button:
        with st.spinner("Processing video with YOLOv8 & ByteTrack..."):
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
    tab_video, tab_kinematics, tab_table, tab_events = st.tabs([
        "📹 Video Playback & HUD",
        "📈 Kinematics & Velocity Analytics",
        "📋 Detections & Trajectory Log",
        "🚨 Safety Events & Risk"
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
            if total_f > 0:
                frame_slider = st.slider("Select Frame Index", min_value=0, max_value=total_f - 1, value=50, step=1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_slider)
                ret, frame_img = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption=f"Frame #{frame_slider} (t = {frame_slider/30.0:.2f}s)", use_container_width=True)
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
        try:
            from dashboard.events_panel import render_events_tab
        except ImportError:  # streamlit run puts dashboard/ on sys.path
            from events_panel import render_events_tab
        render_events_tab(json_log_path, output_video_path, logs_dir=LOGS_DIR)
else:
    st.info("Please select or upload a video clip in the sidebar.")
