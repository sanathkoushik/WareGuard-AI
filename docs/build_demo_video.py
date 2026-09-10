"""
WareGuard AI - Demo video assembler.

Builds a short, honest demo video from real dashboard screenshots captured
live on 2026-09-10, plus two generated frames (title/closing) and one
generated terminal-output frame for the simulated ground-truth self-test.

This is a captioned screenshot-sequence video (silent, no voice narration) -
not a live screen recording. It is built this way because the environment
this was generated in has no microphone/screen-capture access; every number
and result shown is real, pulled from the same JSON logs the dashboard reads.

Run: python docs/build_demo_video.py
Requires: ffmpeg on PATH (brew install ffmpeg)
Output: docs/demo_video.mp4
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent.parent
SHOT_DIR = Path(
    "/var/folders/0j/twlpw84d2bq09_nw0rvgydhr0000gn/T/claude-chrome-screenshots-QCE6V6"
)
FRAME_DIR = BASE_DIR / "docs" / "_video_frames"
OUT_PATH = BASE_DIR / "docs" / "demo_video.mp4"

W, H = 1920, 1080
BG = (10, 13, 19)
CARD_BG = (18, 22, 31)
CARD_BORDER = (35, 41, 54)
ACCENT = (91, 155, 245)
TEXT = (236, 239, 244)
MUTED = (135, 143, 158)
GREEN = (76, 201, 142)
ORANGE = (240, 165, 60)
RED = (224, 107, 107)

FONT_DIR = Path("/System/Library/Fonts/Supplemental")
F_REG = str(FONT_DIR / "Arial.ttf")
F_BOLD = str(FONT_DIR / "Arial Bold.ttf")
F_MONO = "/System/Library/Fonts/SFNSMono.ttf"


def font(path, size):
    return ImageFont.truetype(path, size)


def canvas():
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def wrap_text(draw, text, f, max_width):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=f) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def scene_chrome(draw, idx, total, kicker, caption):
    """Top kicker/scene-counter + bottom caption bar, shared by every frame."""
    draw.rectangle([0, 0, W, 4], fill=ACCENT)
    draw.text((60, 32), f"SCENE {idx} / {total}", font=font(F_BOLD, 20), fill=ACCENT)
    draw.text((60, 62), kicker.upper(), font=font(F_REG, 16), fill=MUTED)
    draw.text((1920 - 260, 32), "WAREGUARD AI", font=font(F_BOLD, 18), fill=MUTED)

    if caption:
        bar_h = 130
        draw.rectangle([0, H - bar_h, W, H], fill=(13, 17, 24))
        draw.line([(0, H - bar_h), (W, H - bar_h)], fill=CARD_BORDER, width=2)
        lines = wrap_text(draw, caption, font(F_REG, 26), W - 160)
        y = H - bar_h + (bar_h - len(lines) * 34) // 2
        for line in lines:
            draw.text((80, y), line, font=font(F_REG, 26), fill=TEXT)
            y += 34


def paste_screenshot(img, path, top_pad=120, bottom_pad=150, side_pad=140):
    shot = Image.open(path).convert("RGB")
    max_w = W - 2 * side_pad
    max_h = H - top_pad - bottom_pad
    scale = min(max_w / shot.width, max_h / shot.height)
    new_w, new_h = int(shot.width * scale), int(shot.height * scale)
    shot = shot.resize((new_w, new_h), Image.LANCZOS)
    x = (W - new_w) // 2
    y = top_pad + (max_h - new_h) // 2
    draw = ImageDraw.Draw(img)
    draw.rectangle([x - 3, y - 3, x + new_w + 3, y + new_h + 3], outline=CARD_BORDER, width=2)
    img.paste(shot, (x, y))


def draw_shield(d, cx, cy, w, h, fill, outline):
    hw = w / 2
    pts = [
        (cx - hw, cy - h * 0.42), (cx, cy - h * 0.5), (cx + hw, cy - h * 0.42),
        (cx + hw, cy - h * 0.05), (cx, cy + h * 0.5), (cx - hw, cy - h * 0.05),
    ]
    d.polygon(pts, fill=fill, outline=outline, width=3)


def frame_title():
    img, d = canvas()
    d.rectangle([0, 0, W, 6], fill=ACCENT)
    draw_shield(d, W // 2, 330, 130, 170, (20, 29, 46), ACCENT)
    cx, cy = W // 2, 330
    d.line([(cx - 38, cy + 2), (cx - 10, cy + 30), (cx + 42, cy - 34)],
           fill=ACCENT, width=10, joint="curve")
    d.text((W // 2, 470), "WareGuard AI", font=font(F_BOLD, 80), fill=TEXT, anchor="mm")
    d.line([(W // 2 - 200, 540), (W // 2 + 200, 540)], fill=ACCENT, width=3)
    d.text((W // 2, 600), "AI Field Intelligence for Warehouse Handling",
            font=font(F_REG, 32), fill=MUTED, anchor="mm")
    d.text((W // 2, 680), "Prototype demo — 6 real & simulated scenarios",
            font=font(F_REG, 24), fill=ACCENT, anchor="mm")
    return img


def frame_closing():
    img, d = canvas()
    d.rectangle([0, 0, W, 6], fill=ACCENT)
    lines = [
        "7 real warehouse clips. 1 offline AI assistant.",
        "Zero cloud dependency. Every risk claim ships",
        "an explanation and a recommended fix.",
    ]
    y = 420
    for line in lines:
        d.text((W // 2, y), line, font=font(F_REG, 40), fill=TEXT, anchor="mm")
        y += 60
    d.text((W // 2, 660), "That's WareGuard AI.", font=font(F_BOLD, 44), fill=ACCENT, anchor="mm")
    return img


def frame_terminal(lines):
    img, d = canvas()
    scene_chrome(d, 6, 6, "Simulated ground-truth self-test",
                 "All 5 behaviour detectors proven end-to-end — no video needed, sub-second.")
    term_top, term_bottom = 130, 900
    term_left, term_right = 200, 1720
    d.rectangle([term_left, term_top, term_right, term_bottom], fill=(8, 10, 14), outline=CARD_BORDER, width=2)
    d.rectangle([term_left, term_top, term_right, term_top + 40], fill=(16, 19, 26))
    for i, c in enumerate([RED, ORANGE, GREEN]):
        d.ellipse([term_left + 18 + i * 24, term_top + 14, term_left + 18 + i * 24 + 12, term_top + 26], fill=c)
    d.text((term_left + 110, term_top + 11), "python run_analysis.py --simulate demo",
            font=font(F_MONO, 16), fill=MUTED)

    fmono = font(F_MONO, 17)
    y = term_top + 60
    for line, color in lines:
        d.text((term_left + 26, y), line, font=fmono, fill=color)
        y += 26
    return img


SEV_COLOR = {"CRIT": RED, "HIGH": ORANGE, "MED ": (240, 220, 130)}


def build_terminal_lines():
    raw = [
        ("Critical risk shift (74/100): 5 events, 2 critical, 2 high over 0.3 min.", TEXT),
        ("", TEXT),
        (" risk index    : 74/100  (Critical)", ACCENT),
        (" by severity   : Medium=1 High=2 Critical=2", MUTED),
        (" by type       : drop=1 throw=1 drag=1 improper_stack=1 rough_handling=1", MUTED),
        ("", TEXT),
        ("------------------------------------------------------------------", (60, 66, 80)),
        (" EVENTS (highest priority first)", MUTED),
        ("------------------------------------------------------------------", (60, 66, 80)),
        (" [CRIT] EVT-0003  throw           risk=100.0  conf=0.97  t=7.73-8.40s", RED),
        ("        launched at 4.7 m/s; travelled 3.0 box-widths through the air", MUTED),
        (" [CRIT] EVT-0001  drop            risk= 75.8  conf=0.96  t=3.37-4.00s", RED),
        ("        fell 4.3 box-heights, impact at 5.0 m/s (100% of free fall)", MUTED),
        (" [HIGH] EVT-0002  drag            risk= 65.5  conf=0.96  t=5.23-7.23s", ORANGE),
        ("        dragged 7.4 box-widths along the floor over 2.0s", MUTED),
        (" [HIGH] EVT-0005  rough_handling  risk= 69.9  conf=0.87  t=12.87-13.10s", ORANGE),
        ("        jerk 22.9 h/s^2, worker 0.2 box-heights away", MUTED),
        (" [MED ] EVT-0004  improper_stack  risk= 35.6  conf=0.74  t=9.43-11.93s", (240, 220, 130)),
        ("        45% overhang, left standing 2.5s", MUTED),
    ]
    return raw


def main():
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    for f in FRAME_DIR.glob("*.png"):
        f.unlink()

    scenes = []  # (image, duration_seconds)

    scenes.append((frame_title(), 3.5))

    img, d = canvas()
    scene_chrome(d, 1, 6, "Real footage, not staged",
                 "Throwing Mattresses.mp4 — unedited warehouse footage. 65 tracks, 1,615 detections.")
    paste_screenshot(img, SHOT_DIR / "screenshot-1788969742251-7.jpg")
    scenes.append((img, 5.0))

    img, d = canvas()
    scene_chrome(d, 2, 6, "High-risk event, explained",
                 "Risk 58/100, 84% confidence — a \"why\" breakdown and a recommended action, not just a number.")
    paste_screenshot(img, SHOT_DIR / "screenshot-1788969807433-9.png", top_pad=120, bottom_pad=150, side_pad=430)
    scenes.append((img, 6.0))

    img, d = canvas()
    scene_chrome(d, 3, 6, "Confident clean, not silence",
                 "Stepping on cartons... clip: a genuine zero-incident read — the system says so explicitly.")
    paste_screenshot(img, SHOT_DIR / "screenshot-1789039384162-12.png", top_pad=120, bottom_pad=150, side_pad=430)
    scenes.append((img, 5.0))

    img, d = canvas()
    scene_chrome(d, 4, 6, "Responsible AI: honest limits",
                 "Dock level clip: tracking too fragmented to trust — flagged \"unknown, not safe,\" never a false all-clear.")
    paste_screenshot(img, SHOT_DIR / "screenshot-1789039480341-13.png", top_pad=120, bottom_pad=150, side_pad=430)
    scenes.append((img, 5.5))

    img, d = canvas()
    scene_chrome(d, 5, 6, "Offline AI assistant",
                 "No LLM key needed — answers grounded only in detected events, with a recommended corrective action.")
    paste_screenshot(img, SHOT_DIR / "screenshot-1788969855634-10.png", top_pad=120, bottom_pad=150, side_pad=430)
    scenes.append((img, 5.5))

    scenes.append((frame_terminal(build_terminal_lines()), 6.5))
    scenes.append((frame_closing(), 4.0))

    for i, (img, _dur) in enumerate(scenes):
        img.save(FRAME_DIR / f"frame_{i:02d}.png")

    concat_path = FRAME_DIR / "concat.txt"
    with open(concat_path, "w") as f:
        for i, (_img, dur) in enumerate(scenes):
            f.write(f"file 'frame_{i:02d}.png'\n")
            f.write(f"duration {dur}\n")
        # ffmpeg concat demuxer quirk: last entry's duration is ignored
        # unless the file is repeated once more.
        f.write(f"file 'frame_{len(scenes) - 1:02d}.png'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_path),
        "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(OUT_PATH),
    ]
    result = subprocess.run(cmd, cwd=str(FRAME_DIR), capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise SystemExit(result.returncode)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
