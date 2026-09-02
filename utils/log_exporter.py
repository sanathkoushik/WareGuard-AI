"""
WareGuard AI - Log Exporter Utility
Handles export of structured object detections and track trajectories to JSON and CSV formats.
"""
import json
import csv
from pathlib import Path
from typing import Dict, List, Any


def export_detections_json(
    output_path: Path or str,
    metadata: Dict[str, Any],
    detections: List[Dict[str, Any]],
    track_summaries: Dict[int, Dict[str, Any]] = None
) -> str:
    """
    Exports video metadata and per-frame detections to a structured JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute summary statistics
    class_counts: Dict[str, int] = {}
    for d in detections:
        cname = d.get("class_name", "unknown")
        class_counts[cname] = class_counts.get(cname, 0) + 1

    payload = {
        "video_metadata": metadata,
        "summary": {
            "total_detections": len(detections),
            "unique_tracks": len(set(d.get("track_id", -1) for d in detections if d.get("track_id") is not None)),
            "class_counts": class_counts,
            "track_summaries": track_summaries or {}
        },
        "detections": detections
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return str(output_path)


def export_detections_csv(
    output_path: Path or str,
    detections: List[Dict[str, Any]]
) -> str:
    """
    Exports per-frame detections to a CSV file for tabular analysis.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "frame",
        "timestamp",
        "track_id",
        "class_id",
        "class_name",
        "confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "center_x",
        "center_y",
        "velocity_x",
        "velocity_y",
        "speed",
        "acceleration_y"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for d in detections:
            bbox = d.get("bbox", [0, 0, 0, 0])
            center = d.get("center", [0, 0])
            velocity = d.get("velocity", [0.0, 0.0])

            row = {
                "frame": d.get("frame", 0),
                "timestamp": round(d.get("timestamp", 0.0), 3),
                "track_id": d.get("track_id", -1),
                "class_id": d.get("class_id", -1),
                "class_name": d.get("class_name", ""),
                "confidence": round(d.get("confidence", 0.0), 3),
                "bbox_x1": round(bbox[0], 1) if len(bbox) > 0 else 0,
                "bbox_y1": round(bbox[1], 1) if len(bbox) > 1 else 0,
                "bbox_x2": round(bbox[2], 1) if len(bbox) > 2 else 0,
                "bbox_y2": round(bbox[3], 1) if len(bbox) > 3 else 0,
                "center_x": round(center[0], 1) if len(center) > 0 else 0,
                "center_y": round(center[1], 1) if len(center) > 1 else 0,
                "velocity_x": round(velocity[0], 2) if len(velocity) > 0 else 0,
                "velocity_y": round(velocity[1], 2) if len(velocity) > 1 else 0,
                "speed": round(d.get("speed", 0.0), 2),
                "acceleration_y": round(d.get("acceleration_y", 0.0), 2)
            }
            writer.writerow(row)

    return str(output_path)
