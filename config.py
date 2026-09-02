"""
WareGuard AI - Configuration Module
Central settings for object detection, tracking, paths, and warehouse parameters.
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
PROCESSED_VIDEOS_DIR = DATA_DIR / "processed_videos"
LOGS_DIR = DATA_DIR / "logs"
DOCS_DIR = BASE_DIR / "docs"

# Ensure directories exist
for p in [RAW_VIDEOS_DIR, PROCESSED_VIDEOS_DIR, LOGS_DIR, DOCS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# YOLO Detection & Tracking Settings
DEFAULT_MODEL = "yolov8n.pt"  # Lightweight model for CPU inference
CONF_THRESHOLD = 0.30          # Confidence threshold
IOU_THRESHOLD = 0.45           # NMS IOU threshold
DEFAULT_TRACKER = "bytetrack.yaml"

# Target COCO Classes for Warehouse Operations
# 0: person, 24: backpack, 26: handbag, 28: suitcase, 39: bottle, 56: chair, 63: laptop
WAREHOUSE_CLASSES = {
    0: "person",
    24: "package",   # mapped from backpack
    26: "package",   # mapped from handbag
    28: "box",       # mapped from suitcase
    39: "item",      # mapped from bottle
}

# Color mapping for visualization (BGR format)
CLASS_COLORS = {
    "person": (255, 144, 30),      # Bright Blue/Cyan
    "box": (0, 165, 255),          # Bright Orange
    "package": (0, 215, 255),      # Gold
    "item": (180, 105, 255),       # Pink
    "unknown": (200, 200, 200),    # Gray
}

# Video Output Settings
DEFAULT_FPS = 30
OUTPUT_CODEC = "mp4v"
