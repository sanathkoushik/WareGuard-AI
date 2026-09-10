"""
WareGuard AI - Live Webcam Capture

Records a short clip straight from a local webcam via OpenCV so the pipeline
can be demonstrated live instead of only on pre-recorded footage. No new
dependency: cv2 is already required by the detection pipeline.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import cv2

from config import DEFAULT_FPS, OUTPUT_CODEC


def record_webcam_clip(
    output_path: Path,
    duration_s: float = 8.0,
    camera_index: int = 0,
    fps: float = DEFAULT_FPS,
    on_frame: Optional[Callable[["cv2.typing.MatLike", float, float], None]] = None,
) -> Path:
    """Records `duration_s` seconds from a local webcam to `output_path`.

    `on_frame(frame_bgr, elapsed_s, duration_s)` is called after every frame
    is written, letting a caller (e.g. the dashboard) show a live preview
    while the recording is in progress.

    Raises RuntimeError if the camera can't be opened or no frames land.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. Make sure a webcam "
            "is connected, not in use by another application, and that this "
            "process has camera permission."
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    cam_fps = cap.get(cv2.CAP_PROP_FPS)
    if not cam_fps or cam_fps <= 0 or cam_fps != cam_fps:  # 0, None, or NaN
        cam_fps = fps

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
    writer = cv2.VideoWriter(str(output_path), fourcc, cam_fps, (width, height))

    frames_written = 0
    start = time.time()
    try:
        while True:
            elapsed = time.time() - start
            if elapsed >= duration_s:
                break
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frames_written += 1
            if on_frame is not None:
                on_frame(frame, elapsed, duration_s)
    finally:
        cap.release()
        writer.release()

    if frames_written == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("No frames were captured from the webcam.")

    return output_path
