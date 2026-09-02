"""
WareGuard AI - Detection & Tracking CLI Runner
Usage:
    python run_detection.py --input data/raw_videos/sample_warehouse.mp4
    python run_detection.py --input path/to/video.mp4 --model yolov8s.pt --conf 0.35
"""
import argparse
import sys
from pathlib import Path
from detection.pipeline import DetectionPipeline
from config import DEFAULT_MODEL, CONF_THRESHOLD, DEFAULT_TRACKER


def main():
    parser = argparse.ArgumentParser(
        description="WareGuard AI - Warehouse Video Intelligence Detection & Tracking Runner"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input video file (e.g. data/raw_videos/sample.mp4)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path for annotated output video file (optional)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"YOLOv8 model weights (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=CONF_THRESHOLD,
        help=f"Confidence threshold (default: {CONF_THRESHOLD})"
    )
    parser.add_argument(
        "--tracker", "-t",
        type=str,
        default=DEFAULT_TRACKER,
        help=f"Tracker configuration (default: {DEFAULT_TRACKER})"
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video rendering (process logs only)"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable JSON log export"
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Disable CSV log export"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input video file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    print("==================================================")
    print("        🛡️  WAREGUARD AI - DETECTION PIPELINE     ")
    print("==================================================")
    print(f" Input Video     : {input_path}")
    print(f" Model           : {args.model}")
    print(f" Tracker         : {args.tracker}")
    print(f" Confidence Thresh: {args.conf}")
    print(f" Render Video    : {not args.no_video}")
    print("--------------------------------------------------")

    pipeline = DetectionPipeline(
        model_path=args.model,
        conf_threshold=args.conf,
        tracker=args.tracker
    )

    result = pipeline.process_video(
        input_video_path=args.input,
        output_video_path=args.output,
        save_json=not args.no_json,
        save_csv=not args.no_csv,
        render_video=not args.no_video
    )

    print("==================================================")
    print("             🎉 PROCESSING COMPLETED              ")
    print("==================================================")
    print(f" Total Frames Processed : {result['metadata']['total_frames']}")
    print(f" Processing Time        : {result['metadata']['processing_time_sec']}s ({result['metadata']['processing_fps']} FPS)")
    print(f" Total Detections       : {result['total_detections']}")
    print(f" Unique Tracks          : {result['unique_tracks']}")
    if result["output_video"]:
        print(f" Annotated Video        : {result['output_video']}")
    if result["json_log"]:
        print(f" JSON Detection Log     : {result['json_log']}")
    if result["csv_log"]:
        print(f" CSV Detection Log      : {result['csv_log']}")
    print("==================================================")


if __name__ == "__main__":
    main()
