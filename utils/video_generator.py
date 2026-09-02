"""
WareGuard AI - Synthetic Warehouse Video Generator
Generates realistic simulated warehouse video clips for offline testing and pipeline validation.
Simulates key warehouse handling events:
  1. Box drop (vertical acceleration, impact with ground)
  2. Box drag (horizontal movement on floor plane)
  3. Box stacking (overlapping bounding boxes, tilt/improper alignment)
  4. Worker normal carrying
"""
import math
import cv2
import numpy as np
from pathlib import Path


def draw_warehouse_background(frame: np.ndarray, width: int, height: int):
    """Draws a warehouse floor, wall, shelf racks, and dock markings."""
    floor_y = int(height * 0.65)

    # Wall (top) - light slate industrial grey
    frame[:floor_y, :] = (65, 60, 55)
    # Floor (bottom) - polished warehouse concrete
    frame[floor_y:, :] = (120, 115, 110)

    # Floor grid lines for perspective
    for x in range(0, width, 120):
        cv2.line(frame, (x, floor_y), (int((x - width / 2) * 1.8 + width / 2), height), (95, 90, 85), 2)
    for y in range(floor_y, height, 40):
        cv2.line(frame, (0, y), (width, y), (105, 100, 95), 1)

    # Shelf Rack on the left
    cv2.rectangle(frame, (40, 80), (220, floor_y), (40, 80, 140), 4) # Orange steel uprights
    cv2.line(frame, (40, 200), (220, 200), (40, 80, 140), 3)
    cv2.line(frame, (40, 320), (220, 320), (40, 80, 140), 3)

    # Background stored pallets/boxes on shelves
    cv2.rectangle(frame, (55, 120), (120, 195), (60, 110, 160), -1)
    cv2.rectangle(frame, (135, 130), (205, 195), (50, 90, 130), -1)
    cv2.rectangle(frame, (60, 240), (140, 315), (70, 120, 170), -1)

    # Yellow caution hazard stripes along dock edge
    stripe_w = 30
    for sx in range(0, width, stripe_w * 2):
        pts = np.array([
            [sx, floor_y - 12],
            [sx + stripe_w, floor_y - 12],
            [sx + stripe_w - 15, floor_y],
            [sx - 15, floor_y]
        ], np.int32)
        cv2.fillPoly(frame, [pts], (20, 210, 240)) # Yellow caution

    # Warehouse banner text
    cv2.putText(frame, "WAREHOUSE BAY 04 - UNLOADING ZONE", (width // 2 - 220, 45),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (180, 190, 200), 1, cv2.LINE_AA)


def draw_worker(frame: np.ndarray, x: int, y: int, color=(240, 160, 40)):
    """Draws a worker silhouette with safety vest and hardhat."""
    # Hardhat (Yellow)
    cv2.ellipse(frame, (x, y - 65), (14, 10), 0, 180, 360, (30, 220, 250), -1)
    # Head
    cv2.circle(frame, (x, y - 55), 12, (180, 200, 220), -1)
    # Safety Vest Torso (Hi-vis Orange/Green with reflective silver stripes)
    cv2.rectangle(frame, (x - 18, y - 42), (x + 18, y + 10), (0, 140, 255), -1)
    cv2.line(frame, (x - 18, y - 20), (x + 18, y - 20), (230, 230, 230), 3) # Reflective band
    cv2.line(frame, (x - 18, y - 5), (x + 18, y - 5), (230, 230, 230), 3)
    # Legs (Navy pants)
    cv2.line(frame, (x - 10, y + 10), (x - 10, y + 65), (90, 50, 30), 6)
    cv2.line(frame, (x + 10, y + 10), (x + 10, y + 65), (90, 50, 30), 6)
    # Boots
    cv2.rectangle(frame, (x - 16, y + 60), (x - 6, y + 68), (20, 20, 20), -1)
    cv2.rectangle(frame, (x + 6, y + 60), (x + 16, y + 68), (20, 20, 20), -1)


def draw_cardboard_box(frame: np.ndarray, x: int, y: int, w: int = 70, h: int = 60, rotation_deg: float = 0.0):
    """Draws a realistic cardboard package with shipping label, tape, and barcode."""
    box_color = (60, 130, 185) # Kraft cardboard brown in BGR
    tape_color = (30, 95, 140)

    # Box body
    rect = ((x, y), (w, h), rotation_deg)
    box_pts = cv2.boxPoints(rect)
    box_pts = np.int32(box_pts)
    cv2.drawContours(frame, [box_pts], 0, box_color, -1)
    cv2.drawContours(frame, [box_pts], 0, (40, 90, 140), 2) # Border

    # Center tape
    x1, y1 = int(x - w // 2), int(y - h // 2)
    cv2.line(frame, (x, y1 + 5), (x, y1 + h - 5), tape_color, 4)

    # White shipping label
    label_w, label_h = int(w * 0.4), int(h * 0.35)
    lx1 = int(x - label_w // 2)
    ly1 = int(y - label_h // 2)
    cv2.rectangle(frame, (lx1, ly1), (lx1 + label_w, ly1 + label_h), (245, 245, 245), -1)
    # Barcode stripes
    for bx in range(lx1 + 4, lx1 + label_w - 4, 4):
        cv2.line(frame, (bx, ly1 + 3), (bx, ly1 + label_h - 4), (30, 30, 30), 1)


def generate_sample_warehouse_video(
    output_path: str = "data/raw_videos/sample_warehouse.mp4",
    fps: int = 30,
    duration_sec: int = 10,
    width: int = 1280,
    height: int = 720
) -> str:
    """
    Generates a full synthetic warehouse video demonstrating:
      - Normal carry (frames 0 - 60)
      - Box DROP event (frames 60 - 120)
      - Box DRAG event (frames 130 - 210)
      - Stacking / Stack creation (frames 220 - 300)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    total_frames = fps * duration_sec
    floor_y = int(height * 0.65)

    # Box physics variables for drop simulation
    drop_y = 0.0
    drop_vy = 0.0
    gravity = 1.4

    for f in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        draw_warehouse_background(frame, width, height)

        # Worker 1 position (walks across warehouse)
        w1_x = int(300 + (f * 3.5) % (width - 450))
        w1_y = floor_y + 10
        draw_worker(frame, w1_x, w1_y)

        # -------------------------------------------------------------
        # 1. DROP SCENARIO (Box ID #101) - Drops between frame 50 and 110
        # -------------------------------------------------------------
        if f < 50:
            # Held by worker
            b1_x = w1_x + 25
            b1_y = w1_y - 25
            draw_cardboard_box(frame, b1_x, b1_y, w=75, h=65)
        elif 50 <= f < 120:
            # Falling suddenly with gravity!
            if f == 50:
                drop_y = float(w1_y - 25)
                drop_vy = 0.0
            drop_vy += gravity
            drop_y += drop_vy
            target_ground = float(floor_y + 45)
            if drop_y >= target_ground:
                drop_y = target_ground
                drop_vy = -drop_vy * 0.25 # slight ground bounce then rest
                if abs(drop_vy) < 1.0:
                    drop_vy = 0.0
            b1_x = 475 # Worker moved on, box dropped here
            draw_cardboard_box(frame, b1_x, int(drop_y), w=75, h=65)
        else:
            # Box sitting on the floor at rest
            draw_cardboard_box(frame, 475, floor_y + 45, w=75, h=65)

        # -------------------------------------------------------------
        # 2. DRAG SCENARIO (Box ID #102) - Dragged along floor (frames 130 to 220)
        # -------------------------------------------------------------
        if f >= 130:
            drag_progress = min(1.0, (f - 130) / 80.0)
            b2_x = int(550 + drag_progress * 280)
            b2_y = floor_y + 50 # On floor plane
            draw_cardboard_box(frame, b2_x, b2_y, w=80, h=60)
            # Drag dust/marks indicator if moving
            if 130 <= f <= 210:
                cv2.line(frame, (550, floor_y + 78), (b2_x - 30, floor_y + 78), (90, 85, 80), 2)

        # -------------------------------------------------------------
        # 3. STACKING SCENARIO (Box ID #103 on base box #104)
        # -------------------------------------------------------------
        # Base box on pallet
        base_stack_x = 960
        base_stack_y = floor_y + 40
        draw_cardboard_box(frame, base_stack_x, base_stack_y, w=90, h=70)

        # Top box (stacked improperly with overhang tilt after frame 200)
        if f >= 200:
            top_progress = min(1.0, (f - 200) / 40.0)
            top_stack_y = int((floor_y - 30) * top_progress + (floor_y - 80) * (1.0 - top_progress))
            # Has an overhang of 35 pixels to the right + slight tilt angle
            draw_cardboard_box(frame, base_stack_x + 35, top_stack_y, w=85, h=65, rotation_deg=8.0)

        out.write(frame)

    out.release()
    return str(output_path)


if __name__ == "__main__":
    generated_path = generate_sample_warehouse_video()
    print(f"Synthetic warehouse video generated at: {generated_path}")
