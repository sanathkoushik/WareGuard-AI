"""
WareGuard AI - Submission deck generator.

Builds the GEG AI Video Intelligence for Warehouse Handling submission deck
(.pptx) matching the brief's required 5-6 slide structure. Screenshots are
real captures from a live run of this dashboard on 2026-09-09, not mockups.

Design system: dark, single-accent, restrained (the "Linear/Vercel/Stripe"
school of tech deck - flat color, hairline dividers, letter-spaced kickers,
data as stat tiles rather than bullet walls) rather than gradient-heavy
template design. Chosen because it survives PowerPoint/Keynote/LibreOffice
rendering differences better than gradients or shadows do, and reads as more
premium at hackathon-deck scale.

Run: python docs/build_deck.py
Output: docs/WareGuard_AI_Submission_Deck.pptx
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BASE_DIR / "docs" / "WareGuard_AI_Submission_Deck.pptx"

SCREENSHOT_DIR = Path(
    "/var/folders/0j/twlpw84d2bq09_nw0rvgydhr0000gn/T/claude-chrome-screenshots-QCE6V6"
)
SHOT_OVERVIEW = SCREENSHOT_DIR / "screenshot-1788969742251-7.jpg"
SHOT_EVENT_CARD = SCREENSHOT_DIR / "screenshot-1788969807433-9.png"
SHOT_ASSISTANT = SCREENSHOT_DIR / "screenshot-1788969855634-10.png"

VIDEO_PATH = BASE_DIR / "docs" / "demo_video.mp4"
VIDEO_POSTER = BASE_DIR / "docs" / "_video_frames" / "frame_00.png"

TOTAL_SLIDES = 6

# ---------------------------------------------------------------- palette
# One accent, semantic colors reserved for status only.
BG = RGBColor(0x0A, 0x0D, 0x13)
CARD_BG = RGBColor(0x12, 0x16, 0x1F)
CARD_BORDER = RGBColor(0x23, 0x29, 0x36)
HAIRLINE = RGBColor(0x1D, 0x22, 0x2C)
ACCENT = RGBColor(0x5B, 0x9B, 0xF5)
ACCENT_DIM = RGBColor(0x2C, 0x3B, 0x57)
ACCENT_SOFT_BG = RGBColor(0x14, 0x1D, 0x2E)
TEXT = RGBColor(0xEC, 0xEF, 0xF4)
MUTED = RGBColor(0x87, 0x8F, 0x9E)
FAINT = RGBColor(0x54, 0x5B, 0x69)
GREEN = RGBColor(0x4C, 0xC9, 0x8E)
ORANGE = RGBColor(0xF0, 0xA5, 0x3C)
RED = RGBColor(0xE0, 0x6B, 0x6B)

FONT = "Calibri"
FONT_LIGHT = "Calibri Light"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
MARGIN_X = Inches(0.6)
CONTENT_W = SLIDE_W - 2 * MARGIN_X


# --------------------------------------------------------------------------
# low-level helpers
# --------------------------------------------------------------------------

def spaced(s: str) -> str:
    """Fake letter-tracking for kicker labels: 'flow' -> 'F L O W'."""
    return " ".join(s.upper())


def _no_shadow(shape):
    shape.shadow.inherit = False


def _solid(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color


def rect(slide, left, top, width, height, fill=None, line=None, line_w=0.75, radius=None):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius is not None else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, left, top, width, height)
    if radius is not None:
        shape.adjustments[0] = radius
    if fill is not None:
        _solid(shape, fill)
    else:
        shape.fill.background()
    if line is not None:
        shape.line.color.rgb = line
        shape.line.width = Pt(line_w)
    else:
        shape.line.fill.background()
    _no_shadow(shape)
    return shape


def oval(slide, left, top, width, height, fill=None, line=None, line_w=0.75):
    shape = slide.shapes.add_shape(MSO_SHAPE.OVAL, left, top, width, height)
    if fill is not None:
        _solid(shape, fill)
    else:
        shape.fill.background()
    if line is not None:
        shape.line.color.rgb = line
        shape.line.width = Pt(line_w)
    else:
        shape.line.fill.background()
    _no_shadow(shape)
    return shape


def add_text(
    slide, text, left, top, width, height, size=18, color=TEXT, bold=False,
    align=PP_ALIGN.LEFT, font=FONT, line_spacing=1.15, anchor=MSO_ANCHOR.TOP,
    letter_spacing_pt=None, italic=False,
):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.color.rgb = color
        run.font.name = font
        if letter_spacing_pt is not None:
            rPr = run._r.get_or_add_rPr()
            rPr.set("spc", str(int(letter_spacing_pt * 100)))
    return box


def add_bullets(slide, items, left, top, width, height, size=14, color=TEXT, gap_pt=9, font=FONT):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap_pt)
        p.line_spacing = 1.2
        r1 = p.add_run()
        r1.text = "–  "  # en dash, quieter than a bullet dot
        r1.font.size = Pt(size)
        r1.font.color.rgb = ACCENT
        r1.font.bold = True
        r1.font.name = font
        r2 = p.add_run()
        r2.text = item
        r2.font.size = Pt(size)
        r2.font.color.rgb = color
        r2.font.name = font
    return box


def new_presentation() -> Presentation:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def blank_slide(prs, page_num, section=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = rect(slide, 0, 0, SLIDE_W, SLIDE_H, fill=BG)
    slide.shapes._spTree.remove(bg._element)
    slide.shapes._spTree.insert(2, bg._element)
    add_chrome(slide, page_num, section)
    return slide


def add_chrome(slide, page_num, section):
    """Consistent footer across every slide: hairline + wordmark + page count."""
    rect(slide, MARGIN_X, Inches(7.08), CONTENT_W, Pt(0.9), fill=HAIRLINE)
    add_text(
        slide, "WAREGUARD AI", MARGIN_X, Inches(7.14), Inches(4), Inches(0.3),
        size=9, color=FAINT, bold=True, letter_spacing_pt=1.2,
    )
    if section:
        add_text(
            slide, section, Inches(4.6), Inches(7.14), Inches(4.13), Inches(0.3),
            size=9, color=FAINT, align=PP_ALIGN.CENTER, letter_spacing_pt=1.0,
        )
    add_text(
        slide, f"{page_num:02d} / {TOTAL_SLIDES:02d}", MARGIN_X, Inches(7.14),
        CONTENT_W, Inches(0.3), size=9, color=FAINT, align=PP_ALIGN.RIGHT,
        letter_spacing_pt=1.0,
    )


def add_header(slide, kicker, title, subtitle=None):
    add_text(slide, spaced(kicker), MARGIN_X, Inches(0.42), CONTENT_W, Inches(0.32),
              size=12.5, color=ACCENT, bold=True, letter_spacing_pt=1.5)
    add_text(slide, title, MARGIN_X, Inches(0.74), CONTENT_W, Inches(0.75),
              size=29, color=TEXT, bold=True, font=FONT_LIGHT)
    y = Inches(1.5)
    if subtitle:
        add_text(slide, subtitle, MARGIN_X, y, CONTENT_W, Inches(0.35), size=13, color=MUTED)
        y = y + Inches(0.4)
    rect(slide, MARGIN_X, y, Inches(0.55), Pt(2.2), fill=ACCENT)
    return y + Inches(0.25)


def card(slide, left, top, width, height, accent_bar=True, radius=0.045):
    box = rect(slide, left, top, width, height, fill=CARD_BG, line=CARD_BORDER, line_w=1, radius=radius)
    if accent_bar:
        rect(slide, left, top + Inches(0.16), Pt(2.6), height - Inches(0.32), fill=ACCENT)
    return box


def stat_tile(slide, left, top, width, height, number, label, color=TEXT, note=None):
    card(slide, left, top, width, height, accent_bar=False)
    rect(slide, left, top, width, Pt(2.4), fill=color)
    add_text(slide, number, left + Inches(0.22), top + Inches(0.2), width - Inches(0.44), Inches(0.7),
              size=30, color=color, bold=True, font=FONT_LIGHT)
    add_text(slide, label, left + Inches(0.22), top + Inches(0.9), width - Inches(0.44), Inches(0.6),
              size=12.5, color=TEXT, bold=True, line_spacing=1.15)
    if note:
        add_text(slide, note, left + Inches(0.22), top + height - Inches(0.5),
                  width - Inches(0.44), Inches(0.42), size=10, color=MUTED, line_spacing=1.1)


def chevron_flow(slide, y, steps, height=Inches(0.6)):
    n = len(steps)
    denom = n - 0.18 * (n - 1)
    box_w = Emu(int(CONTENT_W / denom))
    overlap = Emu(int(box_w * 0.18))
    x = MARGIN_X
    for i, label in enumerate(steps):
        shp = slide.shapes.add_shape(MSO_SHAPE.CHEVRON, x, y, box_w, height)
        shp.adjustments[0] = 0.55
        _solid(shp, ACCENT_SOFT_BG if i % 2 == 0 else CARD_BG)
        shp.line.color.rgb = ACCENT_DIM
        shp.line.width = Pt(1)
        _no_shadow(shp)
        # Push later chevrons in front so the interlock reads correctly.
        slide.shapes._spTree.remove(shp._element)
        slide.shapes._spTree.append(shp._element)
        tf = shp.text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.margin_left = Inches(0.08)
        tf.margin_right = Inches(0.22)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = label
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = TEXT
        run.font.name = FONT
        x = Emu(int(x + box_w - overlap))
    return y + height


def framed_screenshot(slide, path, left, top, width, height, caption=None):
    chrome_h = Inches(0.28)
    frame = card(slide, left, top, width, height + chrome_h, accent_bar=False, radius=0.03)
    rect(slide, left, top, width, chrome_h, fill=RGBColor(0x0D, 0x11, 0x18))
    for i, dot_color in enumerate([RED, ORANGE, GREEN]):
        oval(slide, left + Inches(0.14 + i * 0.18), top + Inches(0.09), Inches(0.1), Inches(0.1), fill=dot_color)

    img_top = top + chrome_h
    img_area_h = height
    if path.exists():
        pic = slide.shapes.add_picture(str(path), left, img_top, height=img_area_h)
        if pic.width > width:
            scale = width / pic.width
            pic.width = int(pic.width * scale)
            pic.height = int(pic.height * scale)
        pic.left = int(left + (width - pic.width) / 2)
        pic.top = int(img_top)
    else:
        add_text(slide, f"[missing: {path.name}]", left, img_top + img_area_h / 2 - Inches(0.15),
                  width, Inches(0.3), size=11, color=MUTED, align=PP_ALIGN.CENTER)

    bottom = top + chrome_h + height
    if caption:
        add_text(slide, caption, left, bottom + Inches(0.1), width, Inches(0.55),
                  size=11, color=MUTED, align=PP_ALIGN.CENTER, line_spacing=1.15)
    return bottom


# --------------------------------------------------------------------------
# Slide 1 - Solution & Team
# --------------------------------------------------------------------------

def build_slide_1(prs):
    slide = blank_slide(prs, 1)

    add_text(slide, spaced("GEG AI Video Intelligence Challenge"), 0, Inches(1.55), SLIDE_W,
              Inches(0.3), size=11.5, color=MUTED, align=PP_ALIGN.CENTER, letter_spacing_pt=1.5)

    ring = oval(slide, Inches(5.52), Inches(1.95), Inches(2.3), Inches(2.3), line=ACCENT_DIM, line_w=1.25)
    badge = oval(slide, Inches(5.67), Inches(2.1), Inches(2.0), Inches(2.0), fill=ACCENT_SOFT_BG, line=ACCENT, line_w=1.5)
    tf = badge.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = "\U0001F6E1"
    run.font.size = Pt(52)

    add_text(slide, "WareGuard AI", 0, Inches(4.4), SLIDE_W, Inches(0.95),
              size=46, color=TEXT, bold=True, align=PP_ALIGN.CENTER, font=FONT_LIGHT)
    rect(slide, Inches(6.167), Inches(5.28), Inches(1.0), Pt(2.4), fill=ACCENT)
    add_text(slide, "An AI field intelligence assistant for warehouse loading & unloading",
              0, Inches(5.42), SLIDE_W, Inches(0.4), size=15, color=ACCENT,
              align=PP_ALIGN.CENTER)

    add_text(
        slide,
        "Watches warehouse footage, explains why a handling action is risky, and tells the\n"
        "supervisor what to do about it — before the product is damaged, not after.",
        Inches(1.8), Inches(5.85), Inches(9.73), Inches(0.75),
        size=13.5, color=MUTED, align=PP_ALIGN.CENTER, line_spacing=1.35,
    )

    tcard = card(slide, Inches(4.17), Inches(6.55), Inches(5.0), Inches(0.85), accent_bar=False, radius=0.15)
    add_text(slide, "Team PixelDock", Inches(4.17), Inches(6.63), Inches(5.0), Inches(0.32),
              size=13, color=TEXT, align=PP_ALIGN.CENTER, bold=True)
    add_text(slide, "Atul Bharadwaj  ·  Sanath M Koushik  ·  Tarun", Inches(4.17), Inches(6.92),
              Inches(5.0), Inches(0.32), size=11, color=MUTED, align=PP_ALIGN.CENTER)


# --------------------------------------------------------------------------
# Slide 2 - Problem, Solution & User Journey
# --------------------------------------------------------------------------

def build_slide_2(prs):
    slide = blank_slide(prs, 2, "PROBLEM & SOLUTION")
    y = add_header(slide, "From surveillance to intelligence",
                    "Problem, Solution & User Journey")

    add_text(slide, "TRADITIONAL CCTV", MARGIN_X, y + Inches(0.05), Inches(4), Inches(0.28),
              size=11, color=RED, bold=True, letter_spacing_pt=1.2)
    y2 = chevron_flow(slide, y + Inches(0.38), ["Camera", "Recording", "Human review",
                                                  "Incident discovered", "Corrective action"])

    add_text(slide, "WAREGUARD AI", MARGIN_X, y2 + Inches(0.22), Inches(4), Inches(0.28),
              size=11, color=GREEN, bold=True, letter_spacing_pt=1.2)
    y3 = chevron_flow(slide, y2 + Inches(0.55), ["Camera", "AI perception", "Behaviour\nunderstanding",
                                                   "Risk detection", "Alert", "Intervention", "Learning"])

    add_text(slide, "SUPERVISOR JOURNEY", MARGIN_X, y3 + Inches(0.3), Inches(4), Inches(0.3),
              size=11, color=ACCENT, bold=True, letter_spacing_pt=1.2)
    journey = [
        "Select or upload a shift's footage in the dashboard.",
        "Pipeline detects and tracks people + handled material (YOLOv8 + ByteTrack).",
        "Behaviour engine flags drop / throw / drag / improper-stack / rough-handling events, each with a plain-English “why.”",
        "Risk engine scores every event Low → Critical and rolls the shift into one index.",
        "Supervisor asks the AI assistant a question — gets an answer grounded only in detected events, plus a recommended fix.",
        "Repeat-offender loads and shift trends surface automatically for training and process fixes.",
    ]
    add_bullets(slide, journey, MARGIN_X, y3 + Inches(0.68), CONTENT_W, Inches(1.9), size=12.5, gap_pt=6)


# --------------------------------------------------------------------------
# Slide 3 - Technical Architecture & Stack
# --------------------------------------------------------------------------

def build_slide_3(prs):
    slide = blank_slide(prs, 3, "ARCHITECTURE")
    y = add_header(slide, "How it's built", "Technical Architecture & Technology Stack")

    cols = [
        ("\U0001F441", "Computer Vision", [
            "YOLOv8n (Ultralytics)",
            "ByteTrack multi-object tracking",
            "OpenCV ingestion + HUD overlay",
        ]),
        ("\U0001F9E0", "Behaviour Intelligence", [
            "Custom heuristic engine, stdlib-only",
            "Kinematics normalised in object-heights/sec",
            "5 detectors: drop, throw, drag, stack, rough",
            "Temporal + overlap resolution",
        ]),
        ("\U0001F916", "AI / ML", [
            "PyTorch, MPS-accelerated (Apple Silicon)",
            "COCO-pretrained detector +",
            "warehouse-object classification layer",
        ]),
        ("\U0001F4AC", "LLM / Assistant", [
            "Optional OpenAI-compatible endpoint",
            "Offline heuristic responder by default",
            "Grounded only in detected-event JSON",
        ]),
        ("\U0001F5A5", "Front-end", [
            "Streamlit dashboard, 5 tabs",
            "Video + HUD, kinematics, risk panel,",
            "detections log, AI chat",
        ]),
        ("\U0001F5C4", "Data & Edge / Cloud", [
            "Structured JSON + CSV logs per shift",
            "No database dependency",
            "Behaviour/risk core: zero vision-stack deps",
            "— runs fully on-device",
        ]),
    ]

    col_w = Inches(3.94)
    row_h = Inches(2.15)
    gap = Inches(0.18)
    start_x = MARGIN_X
    start_y = y + Inches(0.1)
    for i, (icon, title, items) in enumerate(cols):
        col, row = i % 3, i // 3
        x = start_x + col * (col_w + gap)
        yy = start_y + row * (row_h + Inches(0.18))
        card(slide, x, yy, col_w, row_h)
        add_text(slide, icon, x + Inches(0.2), yy + Inches(0.14), Inches(0.5), Inches(0.4), size=17)
        add_text(slide, title, x + Inches(0.62), yy + Inches(0.18), col_w - Inches(0.8), Inches(0.35),
                  size=13.5, color=TEXT, bold=True)
        add_bullets(slide, items, x + Inches(0.28), yy + Inches(0.62), col_w - Inches(0.5),
                    row_h - Inches(0.75), size=10.5, gap_pt=4)


# --------------------------------------------------------------------------
# Slide 4 - Screenshots & Demo
# --------------------------------------------------------------------------

def build_slide_4(prs):
    slide = blank_slide(prs, 4, "LIVE RESULTS")
    y = add_header(slide, "Not staged", "Prototype Screenshots & Demo",
                    subtitle="Real footage: Throwing Mattresses.mp4 — a genuine warehouse handling clip, unedited.")

    img_h = Inches(2.35)
    top = y + Inches(0.05)
    w = Inches(3.94)
    gap = Inches(0.18)
    framed_screenshot(slide, SHOT_OVERVIEW, MARGIN_X, top, w, img_h,
                       caption="Detection + tracking overview — 65 tracks, 1,615 detections")
    framed_screenshot(slide, SHOT_EVENT_CARD, MARGIN_X + w + gap, top, w, img_h,
                       caption="Risk event card: score, confidence, “why,” recommended action")
    framed_screenshot(slide, SHOT_ASSISTANT, MARGIN_X + 2 * (w + gap), top, w, img_h,
                       caption="AI assistant — offline, grounded in detected events only")

    video_y = top + img_h + Inches(0.78)
    video_w, video_h = Inches(2.4), Inches(1.35)
    vcard = card(slide, MARGIN_X, video_y, CONTENT_W, video_h, accent_bar=True)
    if VIDEO_PATH.exists():
        poster = VIDEO_POSTER if VIDEO_POSTER.exists() else None
        slide.shapes.add_movie(
            str(VIDEO_PATH), MARGIN_X + Inches(0.2), video_y + Inches(0.14),
            video_w, video_h - Inches(0.28),
            poster_frame_image=str(poster) if poster else None,
            mime_type="video/mp4",
        )
    add_text(
        slide, "Demo video — 6 scenarios, real dashboard runs\n(silent, captioned — click to play in PowerPoint)",
        MARGIN_X + Inches(2.8), video_y + Inches(0.18), Inches(5.6), Inches(0.7),
        size=13, color=TEXT, bold=True, line_spacing=1.3,
    )
    add_text(
        slide, "If it doesn't play here: docs/demo_video.mp4",
        MARGIN_X + Inches(2.8), video_y + Inches(0.85), Inches(5.6), Inches(0.35),
        size=10.5, color=MUTED, italic=True,
    )


# --------------------------------------------------------------------------
# Slide 5 - Impact & Validation
# --------------------------------------------------------------------------

def build_slide_5(prs):
    slide = blank_slide(prs, 5, "IMPACT & VALIDATION")
    y = add_header(slide, "Reframed, per the brief", "Damage Prevention & User Validation")

    add_text(
        slide,
        "“We identified 1 high-risk handling event across 7 real shift clips and gave the "
        "supervisor a specific corrective action — before damage occurred.”",
        MARGIN_X, y + Inches(0.02), CONTENT_W, Inches(0.65), size=14.5, color=TEXT,
        italic=True, line_spacing=1.3,
    )

    tiles_y = y + Inches(0.85)
    tile_w = Inches(2.92)
    tile_h = Inches(1.75)
    gap = Inches(0.16)
    stat_tile(slide, MARGIN_X, tiles_y, tile_w, tile_h, "1", "High-risk event confirmed", RED,
              note="Rough handling, risk 58/100, 84% confidence")
    stat_tile(slide, MARGIN_X + (tile_w + gap), tiles_y, tile_w, tile_h, "4", "Confident clean shifts", GREEN,
              note="Genuine “no unsafe handling” reads")
    stat_tile(slide, MARGIN_X + 2 * (tile_w + gap), tiles_y, tile_w, tile_h, "3", "Honest data-quality flags", ORANGE,
              note="Insufficient tracking — never a false all-clear")
    stat_tile(slide, MARGIN_X + 3 * (tile_w + gap), tiles_y, tile_w, tile_h, "5 / 5", "Behaviours proven", ACCENT,
              note="Simulated ground-truth self-test, sub-second")

    add_text(
        slide,
        "12 scenario demonstrations total — 7 real clips + 5 simulated behaviour types "
        "(drop, throw, drag, improper-stack, rough-handling). Brief requires ≥10.",
        MARGIN_X, tiles_y + tile_h + Inches(0.1), CONTENT_W, Inches(0.3), size=11.5, color=ACCENT, italic=True,
    )

    val_y = tiles_y + tile_h + Inches(0.5)
    lcard = card(slide, MARGIN_X, val_y, Inches(7.9), Inches(1.75))
    add_text(slide, "VALIDATION STATUS — HONESTLY LABELLED", MARGIN_X + Inches(0.3), val_y + Inches(0.16),
              Inches(7.3), Inches(0.3), size=11, color=TEXT, bold=True, letter_spacing_pt=0.8)
    add_text(slide, "DONE TODAY (internal, real)", MARGIN_X + Inches(0.3), val_y + Inches(0.52),
              Inches(3.7), Inches(0.28), size=10, color=GREEN, bold=True, letter_spacing_pt=0.6)
    done = [
        "Ran all 7 real clips + simulated self-test end to end",
        "Found + fixed a real bug: 0 cargo tracks classified on real footage",
        "63 / 63 automated tests passing after the fix",
    ]
    add_bullets(slide, done, MARGIN_X + Inches(0.3), val_y + Inches(0.84), Inches(3.75), Inches(0.85), size=10, gap_pt=4)

    add_text(slide, "NOT YET DONE (needs real users)", MARGIN_X + Inches(4.15), val_y + Inches(0.52),
              Inches(3.5), Inches(0.28), size=10, color=ORANGE, bold=True, letter_spacing_pt=0.6)
    pending = [
        "Real supervisor / operator review of alerts + recommendations",
        "Precision / recall against labelled ground truth",
        "Measured business impact (financial, incident reduction)",
    ]
    add_bullets(slide, pending, MARGIN_X + Inches(4.15), val_y + Inches(0.84), Inches(3.55), Inches(0.85), size=10, gap_pt=4)

    rcard = card(slide, Inches(9.05), val_y, Inches(3.7), Inches(1.75), accent_bar=True)
    add_text(slide, "RESPONSIBLE AI, BUILT IN", Inches(9.35), val_y + Inches(0.16), Inches(3.2), Inches(0.3),
              size=10.5, color=GREEN, bold=True, letter_spacing_pt=1.0)
    add_text(
        slide,
        "Never reports empty events as “all clear.” Severity and priority kept separate. "
        "Assistant answers only from detected data. Runs fully offline.",
        Inches(9.35), val_y + Inches(0.5), Inches(3.2), Inches(1.2), size=10, color=MUTED, line_spacing=1.25,
    )


# --------------------------------------------------------------------------
# Slide 6 (optional) - Innovation / Roadmap
# --------------------------------------------------------------------------

def criteria_row(slide, x, y, w, weight, name, status, color, note):
    oval(slide, x, y + Inches(0.05), Inches(0.11), Inches(0.11), fill=color)
    add_text(slide, f"{name}  ({weight})", x + Inches(0.24), y - Inches(0.06), w - Inches(0.24), Inches(0.26),
              size=11, color=TEXT, bold=True)
    add_text(slide, status, x + Inches(0.24), y + Inches(0.19), w - Inches(0.24), Inches(0.22),
              size=9.5, color=color, bold=True)
    add_text(slide, note, x + Inches(0.24), y + Inches(0.42), w - Inches(0.24), Inches(0.55),
              size=9.5, color=MUTED, line_spacing=1.1)


def build_slide_6(prs):
    slide = blank_slide(prs, 6, "INNOVATION & ROADMAP")
    y = add_header(slide, "What's real vs. what's next", "Innovation, Roadmap & Judging-Criteria Fit")

    col_w = Inches(3.92)
    gap = Inches(0.18)
    card_h = Inches(4.75)
    x1 = MARGIN_X
    x2 = x1 + col_w + gap
    x3 = x2 + col_w + gap

    card(slide, x1, y, col_w, card_h)
    add_text(slide, "ALREADY TRUE TODAY", x1 + Inches(0.28), y + Inches(0.2), col_w - Inches(0.5), Inches(0.3),
              size=12, color=GREEN, bold=True, letter_spacing_pt=0.6)
    today = [
        "Edge AI / offline inference — zero vision-stack dependency in the core engine",
        "Explainable risk scoring — every event ships a “why” breakdown, not just a number",
        "Recommended actions pulled from this brief's own Good-Practice table",
        "Data-quality honesty — never reports “all clear” on unusable tracking",
        "63 / 63 automated tests passing",
    ]
    add_bullets(slide, today, x1 + Inches(0.28), y + Inches(0.68), col_w - Inches(0.5), Inches(3.9), size=10.5, gap_pt=11)

    card(slide, x2, y, col_w, card_h)
    add_text(slide, "JUDGING CRITERIA — SELF-ASSESSMENT", x2 + Inches(0.28), y + Inches(0.2),
              col_w - Inches(0.5), Inches(0.3), size=11, color=ACCENT, bold=True, letter_spacing_pt=0.4)
    rows = [
        ("15%", "Innovation & Creativity", "STRONG", GREEN,
         "Honest risk gating, brief-sourced recommendations"),
        ("20%", "Technical Execution", "STRONG", GREEN,
         "End-to-end pipeline, real bugs found + fixed today"),
        ("20%", "AI + Video Integration", "STRONG", GREEN,
         "Detection → behaviour → risk → LLM, fully wired"),
        ("10%", "UX & User Feedback", "PARTIAL", ORANGE,
         "Dashboard dogfooded; no real supervisor feedback yet"),
        ("20%", "Damage Prevention & Impact", "PARTIAL", ORANGE,
         "Reframed + actionable; financial impact not quantified"),
        ("15%", "Presentation Quality", "STRONG", GREEN,
         "This deck + a 6-scenario demo video"),
    ]
    ry = y + Inches(0.64)
    row_h = Inches(0.665)
    for weight, name, status, color, note in rows:
        criteria_row(slide, x2 + Inches(0.28), ry, col_w - Inches(0.5), weight, name, status, color, note)
        ry += row_h

    card(slide, x3, y, col_w, card_h)
    add_text(slide, "ROADMAP, HONESTLY LABELLED", x3 + Inches(0.28), y + Inches(0.2),
              col_w - Inches(0.5), Inches(0.3), size=12, color=ORANGE, bold=True, letter_spacing_pt=0.4)
    roadmap = [
        "Orientation classification (vertical-vs-horizontal handling)",
        "Dock-level / floor-condition detection",
        "Warehouse-specific fine-tuned detector, replacing the COCO-proxy classification on real footage",
        "Multi-camera tracking, PPE compliance detection",
        "Precision/recall against labelled ground truth",
    ]
    add_bullets(slide, roadmap, x3 + Inches(0.28), y + Inches(0.68), col_w - Inches(0.5), Inches(3.9), size=10.5, gap_pt=11)


def main():
    prs = new_presentation()
    build_slide_1(prs)
    build_slide_2(prs)
    build_slide_3(prs)
    build_slide_4(prs)
    build_slide_5(prs)
    build_slide_6(prs)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_PATH))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
