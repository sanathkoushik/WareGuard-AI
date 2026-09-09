"""
WareGuard AI - Submission deck generator.

Builds the GEG AI Video Intelligence for Warehouse Handling submission deck
(.pptx) from docs/submission_deck_content.md's content, matching the brief's
required 5-6 slide structure. Screenshots are real captures from a live run
of this dashboard on 2026-09-09, not mockups.

Run: python docs/build_deck.py
Output: docs/WareGuard_AI_Submission_Deck.pptx
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_PATH = BASE_DIR / "docs" / "WareGuard_AI_Submission_Deck.pptx"

SCREENSHOT_DIR = Path(
    "/var/folders/0j/twlpw84d2bq09_nw0rvgydhr0000gn/T/claude-chrome-screenshots-QCE6V6"
)
SHOT_OVERVIEW = SCREENSHOT_DIR / "screenshot-1788969742251-7.jpg"
SHOT_EVENT_CARD = SCREENSHOT_DIR / "screenshot-1788969807433-9.png"
SHOT_ASSISTANT = SCREENSHOT_DIR / "screenshot-1788969855634-10.png"

# ---------------------------------------------------------------- palette
BG = RGBColor(0x0B, 0x0E, 0x14)
CARD_BG = RGBColor(0x15, 0x19, 0x22)
BLUE = RGBColor(0x5B, 0x9B, 0xF5)
LIGHT_BLUE = RGBColor(0x9C, 0xC7, 0xFA)
TEXT = RGBColor(0xE8, 0xEC, 0xF2)
MUTED = RGBColor(0x9A, 0xA3, 0xB2)
ORANGE = RGBColor(0xF5, 0xA6, 0x3E)
GREEN = RGBColor(0x4C, 0xC9, 0x8E)
RED = RGBColor(0xE5, 0x6B, 0x6B)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def new_presentation() -> Presentation:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    return prs


def blank_slide(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG
    bg.line.fill.background()
    bg.shadow.inherit = False
    # Push background behind everything added after it.
    slide.shapes._spTree.remove(bg._element)
    slide.shapes._spTree.insert(2, bg._element)
    return slide


def add_text(
    slide,
    text,
    left,
    top,
    width,
    height,
    size=18,
    color=TEXT,
    bold=False,
    align=PP_ALIGN.LEFT,
    font="Calibri",
    line_spacing=1.15,
    anchor=MSO_ANCHOR.TOP,
):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
        run.font.name = font
    return box


def add_bullets(
    slide, items, left, top, width, height, size=16, color=TEXT,
    bullet_color=BLUE, gap_pt=8, font="Calibri",
):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap_pt)
        p.line_spacing = 1.15
        run = p.add_run()
        run.text = f"•  {item}"
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.name = font
    return box


def add_pill(slide, text, left, top, width, height, bg_color, text_color=BG):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.adjustments[0] = 0.5
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.fill.background()
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.word_wrap = False
    tf.margin_left = 0
    tf.margin_right = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = text_color
    return shape


def add_kicker_title(slide, kicker, title):
    add_text(slide, kicker, Inches(0.6), Inches(0.35), Inches(8), Inches(0.4),
             size=14, color=BLUE, bold=True)
    add_text(slide, title, Inches(0.6), Inches(0.68), Inches(11.5), Inches(0.8),
             size=30, color=TEXT, bold=True)


def add_picture_framed(slide, path, left, top, width, height, caption=None):
    if path.exists():
        pic = slide.shapes.add_picture(str(path), left, top, height=height)
        # Center horizontally within the allotted width if narrower.
        if pic.width > width:
            scale = width / pic.width
            pic.width = int(pic.width * scale)
            pic.height = int(pic.height * scale)
        pic.left = int(left + (width - pic.width) / 2)
        frame = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, pic.left - Emu(9525), pic.top - Emu(9525),
            pic.width + Emu(19050), pic.height + Emu(19050),
        )
        frame.fill.background()
        frame.line.color.rgb = RGBColor(0x2A, 0x30, 0x3D)
        frame.line.width = Pt(1)
        frame.shadow.inherit = False
        bottom = pic.top + pic.height
    else:
        placeholder = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
        placeholder.fill.solid()
        placeholder.fill.fore_color.rgb = CARD_BG
        placeholder.line.color.rgb = RGBColor(0x2A, 0x30, 0x3D)
        placeholder.shadow.inherit = False
        tf = placeholder.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = f"[screenshot missing: {path.name}]"
        run.font.size = Pt(12)
        run.font.color.rgb = MUTED
        bottom = top + height
    if caption:
        add_text(slide, caption, left, bottom + Inches(0.08), width, Inches(0.35),
                  size=12, color=MUTED, align=PP_ALIGN.CENTER)


# --------------------------------------------------------------------------
# Slide 1 - Solution & Team
# --------------------------------------------------------------------------

def build_slide_1(prs):
    slide = blank_slide(prs)
    shield = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(5.67), Inches(1.7), Inches(2), Inches(2))
    shield.fill.solid()
    shield.fill.fore_color.rgb = RGBColor(0x1B, 0x2A, 0x44)
    shield.line.color.rgb = BLUE
    shield.line.width = Pt(2)
    shield.shadow.inherit = False
    tf = shield.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = "\U0001F6E1"
    run.font.size = Pt(48)

    add_text(slide, "WareGuard AI", 0, Inches(3.85), SLIDE_W, Inches(1.0),
             size=44, color=TEXT, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, "AI Video Intelligence for Warehouse Handling", 0, Inches(4.55),
             SLIDE_W, Inches(0.5), size=18, color=LIGHT_BLUE, align=PP_ALIGN.CENTER)

    add_text(
        slide,
        "An AI field intelligence assistant that watches loading and unloading footage,\n"
        "explains why a handling action is risky, and tells the supervisor what to do about\n"
        "it — before the product is damaged, not after.",
        Inches(1.5), Inches(5.15), Inches(10.33), Inches(1.1),
        size=15, color=MUTED, align=PP_ALIGN.CENTER, line_spacing=1.3,
    )

    add_text(slide, "[Team name]", Inches(0), Inches(6.55), SLIDE_W, Inches(0.35),
             size=14, color=TEXT, align=PP_ALIGN.CENTER, bold=True)
    add_text(slide, "[Team members]", Inches(0), Inches(6.9), SLIDE_W, Inches(0.35),
             size=13, color=MUTED, align=PP_ALIGN.CENTER)


# --------------------------------------------------------------------------
# Slide 2 - Problem, Solution & User Journey
# --------------------------------------------------------------------------

def build_slide_2(prs):
    slide = blank_slide(prs)
    add_kicker_title(slide, "PROBLEM → SOLUTION → JOURNEY",
                      "From CCTV surveillance to operational intelligence")

    def flow_row(y, steps, color):
        n = len(steps)
        gap = Inches(0.18)
        total_w = Inches(12.1)
        box_w = Emu(int((total_w - gap * (n - 1)) / n))
        x = Inches(0.6)
        for i, step in enumerate(steps):
            box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, box_w, Inches(0.6))
            box.adjustments[0] = 0.15
            box.fill.solid()
            box.fill.fore_color.rgb = CARD_BG
            box.line.color.rgb = color
            box.line.width = Pt(1.25)
            box.shadow.inherit = False
            tf = box.text_frame
            tf.word_wrap = True
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER
            run = p.add_run()
            run.text = step
            run.font.size = Pt(11.5)
            run.font.color.rgb = TEXT
            run.font.bold = True
            if i < n - 1:
                arrow = slide.shapes.add_textbox(x + box_w, y, gap, Inches(0.6))
                atf = arrow.text_frame
                atf.vertical_anchor = MSO_ANCHOR.MIDDLE
                ap = atf.paragraphs[0]
                ap.alignment = PP_ALIGN.CENTER
                arun = ap.add_run()
                arun.text = "→"
                arun.font.size = Pt(16)
                arun.font.color.rgb = MUTED
            x = x + box_w + gap

    add_text(slide, "Traditional CCTV", Inches(0.6), Inches(1.68), Inches(4), Inches(0.3),
             size=13, color=RED, bold=True)
    flow_row(Inches(2.0), ["Camera", "Recording", "Human review",
                            "Incident discovered", "Corrective action"], RED)

    add_text(slide, "WareGuard AI", Inches(0.6), Inches(2.85), Inches(4), Inches(0.3),
             size=13, color=GREEN, bold=True)
    flow_row(Inches(3.17), ["Camera", "AI perception", "Behaviour\nunderstanding",
                             "Risk detection", "Alert", "Intervention", "Learning"], GREEN)

    add_text(slide, "Supervisor journey", Inches(0.6), Inches(4.15), Inches(4), Inches(0.35),
              size=16, color=BLUE, bold=True)
    journey = [
        "Select or upload a shift's footage in the dashboard.",
        "Pipeline detects and tracks people + handled material (YOLOv8 + ByteTrack).",
        "Behaviour engine flags drop / throw / drag / improper-stack / rough-handling events, each with a plain-English “why.”",
        "Risk engine scores every event Low → Critical and rolls the shift into one index.",
        "Supervisor asks the AI assistant a question — gets an answer grounded only in detected events, plus a recommended corrective action.",
        "Repeat-offender loads and shift trends surface automatically for training and process fixes.",
    ]
    add_bullets(slide, journey, Inches(0.6), Inches(4.55), Inches(12.1), Inches(2.7), size=14.5, gap_pt=7)


# --------------------------------------------------------------------------
# Slide 3 - Technical Architecture & Stack
# --------------------------------------------------------------------------

def build_slide_3(prs):
    slide = blank_slide(prs)
    add_kicker_title(slide, "ARCHITECTURE", "Technical architecture & technology stack")

    cols = [
        ("Computer Vision", [
            "YOLOv8n (Ultralytics)",
            "ByteTrack multi-object tracking",
            "OpenCV ingestion + HUD overlay",
        ]),
        ("Behaviour Intelligence", [
            "Custom heuristic engine, stdlib-only",
            "Kinematics normalised in object-heights/sec",
            "5 detectors: drop, throw, drag, improper-stack, rough-handling",
            "Temporal + overlap resolution",
        ]),
        ("AI / ML", [
            "PyTorch, MPS-accelerated (Apple Silicon)",
            "COCO-pretrained detector +",
            "warehouse-object classification layer",
        ]),
        ("LLM / Assistant", [
            "Optional OpenAI-compatible endpoint",
            "Zero-dependency heuristic responder",
            "by default — works fully offline",
            "Grounded only in detected-event JSON",
        ]),
        ("Front-end", [
            "Streamlit dashboard",
            "Video + HUD, kinematics charts,",
            "risk panel, AI chat — 5 tabs",
        ]),
        ("Data & Edge/Cloud", [
            "Structured JSON + CSV logs per shift",
            "No database dependency",
            "Behaviour/risk core has zero vision-stack",
            "dependency — runs fully on-device",
        ]),
    ]

    col_w = Inches(3.95)
    row_h = Inches(2.55)
    gap = Inches(0.2)
    start_x = Inches(0.6)
    start_y = Inches(1.75)
    for i, (title, items) in enumerate(cols):
        col = i % 3
        row = i // 3
        x = start_x + col * (col_w + gap)
        y = start_y + row * (row_h + Inches(0.2))
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, col_w, row_h)
        card.adjustments[0] = 0.05
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = RGBColor(0x2A, 0x30, 0x3D)
        card.shadow.inherit = False
        add_text(slide, title, x + Inches(0.2), y + Inches(0.15), col_w - Inches(0.4), Inches(0.4),
                 size=15, color=BLUE, bold=True)
        add_bullets(slide, items, x + Inches(0.2), y + Inches(0.6), col_w - Inches(0.4),
                    row_h - Inches(0.75), size=11.5, gap_pt=5)


# --------------------------------------------------------------------------
# Slide 4 - Screenshots & Demo
# --------------------------------------------------------------------------

def build_slide_4(prs):
    slide = blank_slide(prs)
    add_kicker_title(slide, "LIVE RESULTS", "Prototype screenshots & demo")
    add_text(
        slide,
        "Real footage, not staged: Throwing Mattresses.mp4, a genuine warehouse handling clip.",
        Inches(0.6), Inches(1.4), Inches(12.1), Inches(0.35), size=13, color=MUTED,
    )

    img_h = Inches(3.55)
    y = Inches(1.85)
    add_picture_framed(slide, SHOT_OVERVIEW, Inches(0.5), y, Inches(4.0), img_h,
                        caption="Detection + tracking overview — 65 tracks, 1615 detections")
    add_picture_framed(slide, SHOT_EVENT_CARD, Inches(4.7), y, Inches(4.0), img_h,
                        caption="Risk event card: score, confidence, “why,” recommended action")
    add_picture_framed(slide, SHOT_ASSISTANT, Inches(8.9), y, Inches(4.0), img_h,
                        caption="AI assistant — offline, grounded in detected events only")

    add_text(
        slide,
        "Demo video (3-5 scenarios): see docs/demo_video_script.md — [embed / link recording here]",
        Inches(0.6), Inches(6.85), Inches(12.1), Inches(0.4), size=13, color=ORANGE, bold=True,
    )


# --------------------------------------------------------------------------
# Slide 5 - Impact & Validation
# --------------------------------------------------------------------------

def build_slide_5(prs):
    slide = blank_slide(prs)
    add_kicker_title(slide, "IMPACT", "Damage prevention & user validation")

    add_text(slide, "Reframed, per the brief:", Inches(0.6), Inches(1.55), Inches(6), Inches(0.35),
              size=15, color=BLUE, bold=True)
    add_text(slide, "“We identified 1 high-risk handling event across 7 real shift clips and\n"
                     "gave the supervisor a specific corrective action — before damage occurred.”",
             Inches(0.6), Inches(1.95), Inches(6.0), Inches(1.0), size=14, color=TEXT,
             line_spacing=1.25)

    results = [
        "1 real clip → confirmed High-risk event (rough handling, risk 58/100, 84% confidence), explainable + a recommended fix",
        "4 real clips → confident “no unsafe handling” — genuine clean reads, not silence",
        "3 real clips → honestly flagged as insufficient tracking data, not a false all-clear",
        "Simulated ground-truth demo → all 5 behaviour types detected end-to-end in under a second",
    ]
    add_bullets(slide, results, Inches(0.6), Inches(3.05), Inches(6.0), Inches(2.2), size=12.5, gap_pt=8)

    add_text(slide, "Suggested validation users:", Inches(0.6), Inches(5.55), Inches(6), Inches(0.35),
              size=13, color=BLUE, bold=True)
    add_text(slide, "Warehouse supervisor · Loading/unloading operator · Logistics manager · Safety professional\n[Document what they observed and what changed — fill in after validation session.]",
             Inches(0.6), Inches(5.9), Inches(6.0), Inches(1.1), size=12, color=MUTED, line_spacing=1.25)

    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(1.55),
                                   Inches(5.7), Inches(5.5))
    card.adjustments[0] = 0.04
    card.fill.solid()
    card.fill.fore_color.rgb = CARD_BG
    card.line.color.rgb = GREEN
    card.line.width = Pt(1.25)
    card.shadow.inherit = False
    add_text(slide, "Responsible AI — built in, not bolted on", Inches(7.25), Inches(1.75),
              Inches(5.2), Inches(0.4), size=15, color=GREEN, bold=True)
    resp = [
        "Distinguishes observed behaviour → potential risk → confirmed damage — never reports an empty event list as “all clear” when tracking was too thin to say so",
        "risk_score (severity) and priority_score (severity × confidence) kept separate — triage never collapses into one misleading number",
        "AI assistant answers only from detected-event data — instructed to never invent an event, track ID, or score",
        "Runs fully offline / on-device — footage never has to leave the building for inference",
    ]
    add_bullets(slide, resp, Inches(7.25), Inches(2.25), Inches(5.2), Inches(4.6), size=12.5, gap_pt=10)


# --------------------------------------------------------------------------
# Slide 6 (optional) - Innovation / Roadmap
# --------------------------------------------------------------------------

def build_slide_6(prs):
    slide = blank_slide(prs)
    add_kicker_title(slide, "INNOVATION & ROADMAP", "What's built today, what's next")

    add_text(slide, "Already true today", Inches(0.6), Inches(1.6), Inches(5.8), Inches(0.35),
              size=15, color=GREEN, bold=True)
    today = [
        "Edge AI / offline inference — the core behaviour + risk engine has zero vision-stack dependency",
        "Explainable risk scoring — every event ships a “why” factor breakdown, not just a number",
        "Recommended actions pulled directly from this brief's own Good-Practice table",
    ]
    add_bullets(slide, today, Inches(0.6), Inches(2.05), Inches(5.8), Inches(2.4), size=13, gap_pt=10)

    add_text(slide, "Roadmap (honest about what's not built yet)", Inches(6.9), Inches(1.6),
              Inches(5.8), Inches(0.35), size=15, color=ORANGE, bold=True)
    roadmap = [
        "Orientation classification (vertical-vs-horizontal product handling)",
        "Dock-level / floor-condition detection",
        "Warehouse-specific fine-tuned detector to replace the current COCO-proxy classification on real footage",
        "Multi-camera tracking, PPE compliance detection",
    ]
    add_bullets(slide, roadmap, Inches(6.9), Inches(2.05), Inches(5.8), Inches(2.6), size=13, gap_pt=10)


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
