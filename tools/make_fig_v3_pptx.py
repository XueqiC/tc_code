#!/usr/bin/env python3
"""Method figure v3.0: budgeted verifier-guided self-distillation.

Same visual language as pipeline v6: every element is an editable
pptx shape, 16:9 slide, PNG export for the paper. Blue = student,
teal = verifier, orange = teacher (context only), green = certified
deployment, gray = neutral. Top strip contrasts the conventional
imitation path (crossed out); the main loop reads left to right with
verb labels inside spine arrows; the re-price return arrow closes the
loop; a right-side panel shows graduation and deployment.
"""
from pptx import Presentation
from pptx.util import Inches as In, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

BLUE = RGBColor(0x4C, 0x72, 0xB0)
BLUEBG = RGBColor(0xEE, 0xF2, 0xF9)
TEAL = RGBColor(0x5F, 0x9E, 0xA0)
TEALBG = RGBColor(0xEA, 0xF4, 0xF4)
ORANGE = RGBColor(0xDD, 0x84, 0x52)
ORANGEBG = RGBColor(0xFD, 0xF0, 0xE7)
RED = RGBColor(0xC4, 0x4E, 0x52)
GREEN = RGBColor(0x55, 0xA8, 0x68)
GREENBG = RGBColor(0xEC, 0xF5, 0xEE)
GRAY = RGBColor(0x8C, 0x8C, 0x8C)
CARD = RGBColor(0xF7, 0xF6, 0xF2)
TEXT = RGBColor(0x1F, 0x1F, 0x1E)
MUTED = RGBColor(0x6B, 0x6A, 0x63)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

prs = Presentation()
prs.slide_width = In(13.333)
prs.slide_height = In(7.5)
slide = prs.slides.add_slide(prs.slide_layouts[6])
SH = slide.shapes


def box(x, y, w, h, text, fill, line, size=12, bold=False, color=TEXT,
        shape=MSO_SHAPE.ROUNDED_RECTANGLE, align=PP_ALIGN.CENTER):
    s = SH.add_shape(shape, In(x), In(y), In(w), In(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = line
    s.line.width = Pt(1.25)
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = In(0.06)
    tf.margin_top = tf.margin_bottom = In(0.03)
    first = True
    for ln in text.split("\n"):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = ln
        p.alignment = align
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.color.rgb = color
    return s


def label(x, y, w, h, text, size=11, color=MUTED, bold=False,
          align=PP_ALIGN.CENTER):
    t = SH.add_textbox(In(x), In(y), In(w), In(h))
    tf = t.text_frame
    tf.word_wrap = True
    first = True
    for ln in text.split("\n"):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = ln
        p.alignment = align
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.color.rgb = color
            r.font.bold = bold
    return t


def arrow(x, y, w, text="", color=GRAY, size=10):
    a = SH.add_shape(MSO_SHAPE.RIGHT_ARROW, In(x), In(y), In(w), In(0.42))
    a.fill.solid()
    a.fill.fore_color.rgb = color
    a.line.fill.background()
    if text:
        tf = a.text_frame
        tf.word_wrap = False
        p = tf.paragraphs[0]
        p.text = text
        p.alignment = PP_ALIGN.CENTER
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.bold = True
            r.font.color.rgb = WHITE
    return a


# ---- title strip -----------------------------------------------------
label(0.3, 0.12, 12.7, 0.4,
      "Budgeted verifier-guided self-distillation with teacher context scaffolding",
      size=17, color=TEXT, bold=True, align=PP_ALIGN.LEFT)

# ---- conventional path, crossed out ---------------------------------
box(0.35, 0.68, 2.2, 0.62, "Teacher behavior\n(imitation targets)",
    ORANGEBG, ORANGE, size=11)
arrow(2.62, 0.78, 1.05, "MLE", color=GRAY)
box(3.74, 0.68, 2.2, 0.62, "Student parameters\ndrift to teacher style",
    CARD, GRAY, size=11)
cross = SH.add_connector(MSO_CONNECTOR.STRAIGHT, In(0.5), In(1.42),
                         In(5.85), In(0.6))
cross.line.color.rgb = RED
cross.line.width = Pt(2.5)
label(6.05, 0.78, 3.4, 0.5,
      "below the untrained student on every benchmark",
      size=11, color=RED, align=PP_ALIGN.LEFT)

# ---- main loop -------------------------------------------------------
Y = 2.15
box(0.35, Y, 1.95, 1.15,
    "Task demand\nfrom support set\ndeficit δ per direction",
    CARD, GRAY, size=11)
arrow(2.36, Y + 0.36, 0.85, "price", color=GRAY)
box(3.27, Y, 2.0, 1.15,
    "Buy teacher context\nby unlock value\nρ·coverage / cost",
    ORANGEBG, ORANGE, size=11)
arrow(5.33, Y + 0.36, 0.85, "guide", color=ORANGE)
box(6.24, Y, 2.15, 1.15,
    "Student explores\nunguided first,\ncontext only on failure",
    BLUEBG, BLUE, size=11)
arrow(8.45, Y + 0.36, 0.85, "verify", color=TEAL)
box(9.36, Y, 1.85, 1.15, "Verifier\nV(x,τ) ∈ {0,1}",
    TEALBG, TEAL, size=11)
arrow(11.27, Y + 0.36, 0.75, "learn", color=BLUE)
box(12.06, Y, 1.0, 1.15, "Update\nθ", BLUEBG, BLUE, size=11)

# student-provenance banner under the loop
box(6.24, Y + 1.35, 6.82, 0.5,
    "trains only on the student's own trajectories; clipped importance "
    "ratio corrects guided collection to the unguided policy",
    WHITE, BLUE, size=10)

# re-price return arrow
back = SH.add_shape(MSO_SHAPE.LEFT_ARROW, In(0.75), In(4.25), In(11.6),
                    In(0.4))
back.fill.solid()
back.fill.fore_color.rgb = GRAY
back.line.fill.background()
tfb = back.text_frame
pb = tfb.paragraphs[0]
pb.text = ("re-price after every update    ρ rises, deficits shrink, "
           "queries graduate to unguided, purchases stop at max U ≤ 0")
pb.alignment = PP_ALIGN.CENTER
for r in pb.runs:
    r.font.size = Pt(10)
    r.font.bold = True
    r.font.color.rgb = WHITE

# ---- bottom panels ---------------------------------------------------
YB = 5.15
box(0.35, YB, 3.9, 1.7,
    "Failures teach locally\npreference only from first divergence\n"
    "gated by length-normalized NLL",
    CARD, GRAY, size=11)
box(4.45, YB, 4.1, 1.7,
    "Scaffold graduation\nguided share → 0 as competence grows\n"
    "loop tends to on-policy self-improvement",
    BLUEBG, BLUE, size=11)
box(8.75, YB, 4.25, 1.7,
    "Certified deployment\nconformal in-task region on held-out split\n"
    "specialist inside, escalate outside\nno teacher context at deployment",
    GREENBG, GREEN, size=11)

label(0.35, 6.95, 12.6, 0.4,
      "teacher = exploration context only · student = sole provenance "
      "of training data · verifier = sole learning signal",
      size=12, color=TEXT, bold=True)

out = "results/figs/fig_method_v3.pptx"
prs.save(out)
print("saved", out)
