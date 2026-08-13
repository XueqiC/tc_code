"""Pipeline figure v6: one connected end-to-end flow, native PowerPoint.

Every element is an editable pptx shape (rounded rects, block arrows, text
boxes); nothing is a pasted image. 16:9 slide; also exported to PNG for the
paper. Semantics: blue = in-specification, orange/red = out-of-scope,
green = guarantee, gray = neutral structure. Verb labels live inside the
spine arrows so the flow reads left to right without floating text.
"""

from pptx import Presentation
from pptx.util import Inches as In, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

IN1 = RGBColor(0x4C, 0x72, 0xB0)
IN2 = RGBColor(0x5F, 0x9E, 0xA0)
IN3 = RGBColor(0x7B, 0xA7, 0xD7)
INBG = RGBColor(0xEE, 0xF2, 0xF9)
OUT1 = RGBColor(0xDD, 0x84, 0x52)
OUT2 = RGBColor(0xC4, 0x4E, 0x52)
OUTBG = RGBColor(0xFD, 0xF0, 0xE7)
PALE = RGBColor(0xE8, 0xE6, 0xE1)
CARD = RGBColor(0xF7, 0xF6, 0xF2)
GRAY = RGBColor(0x8C, 0x8C, 0x8C)
GREEN = RGBColor(0x55, 0xA8, 0x68)
TEXT = RGBColor(0x1F, 0x1F, 0x1E)
MUTED = RGBColor(0x6B, 0x6A, 0x63)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GATE = RGBColor(0xDF, 0xE5, 0xEF)

prs = Presentation()
prs.slide_width = In(13.333)
prs.slide_height = In(7.5)
slide = prs.slides.add_slide(prs.slide_layouts[6])
SH = slide.shapes
FONT = "Times New Roman"
SPINE = 3.20  # y center of the flow


def no_shadow(shape):
    el = shape._element
    sp = el.spPr
    for tag in ("a:effectLst", "a:effectDag"):
        for e in sp.findall(qn(tag)):
            sp.remove(e)
    # empty effectLst + no style reference renders shadow-free everywhere
    etree.SubElement(sp, qn("a:effectLst"))
    style = el.find(qn("p:style"))
    if style is not None:
        el.remove(style)


def _style_text(tf, size, color, bold=False, italic=False,
                align=PP_ALIGN.CENTER):
    tf.word_wrap = True
    for p in tf.paragraphs:
        p.alignment = align
        for r in p.runs:
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.color.rgb = color
            r.font.bold = bold
            r.font.italic = italic


def box(x, y, w, h, fc, ec, lw=1.2, text=None, size=10, tcolor=None,
        bold=False, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.25,
        dash=None, italic=False):
    s = SH.add_shape(shape, In(x), In(y), In(w), In(h))
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            s.adjustments[0] = radius
        except Exception:
            pass
    if fc is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fc
    if ec is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = ec
        s.line.width = Pt(lw)
        if dash:
            ln = s.line._get_or_add_ln()
            ln.append(ln.makeelement(qn("a:prstDash"), {"val": dash}))
    no_shadow(s)
    if text is not None:
        s.text_frame.text = text
        s.text_frame.margin_left = s.text_frame.margin_right = Emu(9000)
        s.text_frame.margin_top = s.text_frame.margin_bottom = Emu(4500)
        s.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
        _style_text(s.text_frame, size, tcolor or TEXT, bold=bold,
                    italic=italic)
    return s


def label(x, y, w, h, text, size=9, color=MUTED, bold=False, italic=False,
          align=PP_ALIGN.CENTER):
    t = SH.add_textbox(In(x), In(y), In(w), In(h))
    t.text_frame.text = text
    t.text_frame.margin_left = t.text_frame.margin_right = 0
    t.text_frame.margin_top = t.text_frame.margin_bottom = 0
    _style_text(t.text_frame, size, color, bold=bold, italic=italic,
                align=align)
    return t


def conn(x0, y0, x1, y1, color, w=2.2, arrow=True):
    from pptx.enum.shapes import MSO_CONNECTOR
    c = SH.add_connector(MSO_CONNECTOR.STRAIGHT, In(x0), In(y0), In(x1), In(y1))
    c.line.color.rgb = color
    c.line.width = Pt(w)
    ln = c.line._get_or_add_ln()
    if arrow:
        te = ln.makeelement(qn("a:tailEnd"),
                            {"type": "triangle", "w": "med", "len": "med"})
        ln.append(te)
    no_shadow(c)
    return c


def spine_arrow(x, w, text, h=0.72, fs=9):
    a = box(x, SPINE - h / 2, w, h, GRAY, None, shape=MSO_SHAPE.RIGHT_ARROW,
            radius=None, text=text, size=fs, tcolor=WHITE, bold=True)
    a.adjustments[0] = 0.62
    a.adjustments[1] = 0.42
    return a


def tile(x, y, c, s=0.24):
    return box(x, y, s, s, c, WHITE, lw=0.75, radius=0.3)


def chip(cx, cy, w, h, ec, name, size=9.5):
    box(cx - w / 2, cy - h / 2, w, h, INBG, ec, lw=1.6)
    box(cx - 0.012, cy - h / 2 - 0.14, 0.024, 0.14, ec, None,
        shape=MSO_SHAPE.RECTANGLE, radius=None)
    box(cx - 0.05, cy - h / 2 - 0.24, 0.10, 0.10, ec, None,
        shape=MSO_SHAPE.OVAL, radius=None)
    for dx in (-w * 0.18, w * 0.18):
        box(cx + dx - 0.045, cy - h * 0.10 - 0.045, 0.09, 0.09, ec, None,
            shape=MSO_SHAPE.OVAL, radius=None)
    box(cx - w * 0.14, cy + h * 0.12, w * 0.28, 0.03, ec, None,
        shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5)
    label(cx - w / 2 - 0.35, cy + h / 2 + 0.04, w + 0.7, 0.28, name,
          size=size, color=TEXT, bold=True)


def stage_chip(x, n, claim, w=3.2):
    box(x, 0.20, 0.40, 0.40, TEXT, None, shape=MSO_SHAPE.OVAL, radius=None,
        text=n, size=15, tcolor=WHITE, bold=True)
    label(x + 0.42, 0.22, w, 0.38, claim, size=16, color=TEXT, bold=True,
          align=PP_ALIGN.LEFT)

# ---------------------------------------------------------------- v12 body
# Reference-style overview: white ground, dashed panel dividers, small
# hand-drawn-feel bar glyphs, robot chips, inline math, italic labels.

BLUE = RGBColor(0xA8, 0xC4, 0xE4)
BLUED = RGBColor(0x4C, 0x72, 0xB0)
ORAN = RGBColor(0xF2, 0xB9, 0x8F)
GRNB = RGBColor(0x9E, 0xC9, 0x9A)
GRND = RGBColor(0x4E, 0x8A, 0x4A)
GRAYB = RGBColor(0xC9, 0xC9, 0xC9)
REDL = RGBColor(0xE2, 0x9A, 0x9A)
GOLD = RGBColor(0xE8, 0xC5, 0x6A)

def gridlines(x, y, w, n=4, gap=0.16):
    for i in range(n):
        conn(x, y - i * gap, x + w, y - i * gap, RGBColor(0xE4,0xE4,0xE4),
             w=0.9, arrow=False)

def bars(x, y, hs, colors, bw=0.17, gap=0.055, grid=True):
    """Mini bar chart. y = baseline; heights in inches (positive up)."""
    n = len(hs)
    W = n * bw + (n - 1) * gap
    if grid:
        gridlines(x - 0.05, y - 0.02, W + 0.10)
    conn(x - 0.05, y, x + W + 0.05, y, RGBColor(0x55,0x55,0x55), w=1.4,
         arrow=False)
    for i, h in enumerate(hs):
        c = colors[i] if isinstance(colors, (list, tuple)) else colors
        box(x + i * (bw + gap), y - h, bw, h, c,
            RGBColor(0x66,0x66,0x66), lw=1.0, radius=None,
            shape=MSO_SHAPE.RECTANGLE)
    return W

def ital(x, y, w, t, size=14, color=TEXT, align=PP_ALIGN.CENTER, bold=True):
    label(x, y, w, 0.3, t, size=size, color=color, italic=True, bold=bold,
          align=align)

def dashline(x0, y0, x1, y1):
    c = conn(x0, y0, x1, y1, RGBColor(0x9A,0x9A,0x9A), w=1.6, arrow=False)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:prstDash"), {"val": "dash"}))

def arrow(x0, y0, x1, y1, color=RGBColor(0x33,0x33,0x33), w=2.4):
    conn(x0, y0, x1, y1, color, w=w, arrow=True)

def matrix_icon(x, y, s=0.92, cols=(BLUED, GRNB, ORAN), badge=True):
    box(x, y, s, s, WHITE, RGBColor(0x55,0x55,0x55), lw=1.4, radius=0.12)
    cell = s / 3.6
    for r in range(3):
        for cc in range(3):
            col = cols[(r + cc) % 3]
            box(x + 0.07 + cc * cell, y + 0.07 + r * cell, cell * 0.8,
                cell * 0.8, col, WHITE, lw=0.6, radius=None,
                shape=MSO_SHAPE.RECTANGLE)
    if badge:
        box(x + s - 0.16, y - 0.10, 0.30, 0.30, GRND, WHITE, lw=1.2,
            shape=MSO_SHAPE.OVAL, radius=None, text="✓", size=11,
            tcolor=WHITE, bold=True)

def coin_stack(x, y, n=4):
    for i in range(n):
        box(x, y - i * 0.12, 0.34, 0.085, GOLD, RGBColor(0x9A,0x7B,0x2E),
            lw=1.0, shape=MSO_SHAPE.OVAL, radius=None)

# frame
box(0.12, 0.10, 13.09, 7.28, WHITE, RGBColor(0x8A,0x8A,0x8A), lw=1.6,
    radius=0.03)
dashline(6.55, 0.22, 6.55, 7.26)
dashline(6.66, 4.30, 13.10, 4.30)

# ------------------------------------------------ LEFT: demand estimation
ital(0.35, 0.24, 5.4, "Reading the task as skill demand", size=19,
     align=PP_ALIGN.LEFT)

# support set docs
for i, dx in enumerate((0.0, 0.18, 0.36)):
    box(0.50 + dx, 0.82 + dx * 0.4, 1.00, 1.20, WHITE,
        RGBColor(0x77,0x77,0x77), lw=1.2, radius=0.06)
for yy in (1.28, 1.50, 1.72):
    box(1.02, yy, 0.62, 0.055, GRAYB, None, radius=None,
        shape=MSO_SHAPE.RECTANGLE)
ital(0.35, 2.20, 1.9, "Support Set S", size=14)

arrow(1.95, 1.55, 2.55, 1.55)

# student robot
chip(3.15, 1.42, 1.30, 1.15, BLUED, "Student Model", size=13)
label(2.55, 2.30, 1.2, 0.3, "∇", size=18, color=TEXT, bold=True)

arrow(3.85, 1.45, 4.45, 1.45)

# gradient features bars
bars(4.62, 1.95, [0.62, 0.30, 0.50, 0.22], [ORAN]*4)
ital(4.20, 2.14, 2.1, "Gradient Features", size=14)

# dictionary formula + atoms
label(0.30, 2.80, 4.0, 0.36,
      "min‖X − CD‖²  +  λ‖C‖₁", size=19, color=TEXT, bold=False)
arrow(4.60, 2.55, 4.60, 3.30, color=RGBColor(0x33,0x33,0x33))

# atoms row (colored tiles) -> demand bar chart
for i, c in enumerate((BLUED, IN2, IN3, ORAN, REDL)):
    box(0.80 + i * 0.62, 3.42, 0.50, 0.50, c, WHITE, lw=1.2, radius=0.18)
ital(0.45, 4.00, 3.6, "Capability Atoms (dictionary D)", size=14)

arrow(3.98, 3.67, 4.38, 3.67)
bars(4.52, 4.10, [0.66, 0.44, 0.28, 0.10], [BLUED, IN2, IN3, GRAYB])
ital(4.05, 4.30, 2.2, "Task Demand w", size=14)

# quota line
label(0.50, 4.70, 5.8, 0.36,
      "per-skill token quota  =  w · b", size=17, color=TEXT)

# bridge strip (bottom left)
ital(0.35, 5.20, 5.9, "Cross-modal bridge (fit on paid pairs)", size=15,
     align=PP_ALIGN.LEFT)
bars(0.75, 6.45, [0.30, 0.48, 0.22], [GRAYB]*3, grid=False)
ital(0.40, 6.62, 1.6, "Prompt g", size=13)
arrow(1.55, 6.20, 2.30, 6.20)
matrix_icon(2.40, 5.85, badge=False)
ital(2.05, 6.72, 1.7, "Ridge Map", size=13)
arrow(3.25, 6.20, 4.00, 6.20)
bars(4.20, 6.45, [0.55, 0.25, 0.42], [GRNB]*3, grid=False)
ital(3.85, 6.62, 2.1, "Predicted Skills", size=13)

# ---------------------------------------- RIGHT TOP: budgeted acquisition
ital(6.80, 0.24, 6.2, "Spending the budget where demand is unmet",
     size=19, align=PP_ALIGN.LEFT)

chip(7.50, 1.32, 1.30, 1.15, RGBColor(0x8A,0x5A,0xA8), "Teacher API", size=13)
coin_stack(8.25, 1.30)
ital(8.05, 1.50, 0.9, "b", size=15)

# candidate -> predicted skills -> min() gauge
arrow(8.35, 1.35, 8.90, 1.35)
bars(9.05, 1.75, [0.42, 0.20, 0.34], [GRNB]*3, grid=False)
ital(8.65, 1.92, 1.8, "ŝ(x) predicted", size=13)
# quota gauge: two bars remaining
bars(10.55, 1.75, [0.55, 0.15, 0.30], [BLUED, GRAYB, IN2], grid=False)
ital(10.10, 1.92, 2.1, "r  remaining quota", size=13)
label(11.75, 1.05, 1.5, 0.9, "u(x) =\nΣ min(r, ŝ) / t̂", size=15,
      color=TEXT)
arrow(9.90, 1.55, 10.35, 1.55)

# returns gate row
label(6.85, 2.55, 3.4, 0.32, "returns → execution check + gate",
      size=14, color=MUTED, align=PP_ALIGN.LEFT)
box(10.30, 2.40, 0.44, 0.44, GRNB, WHITE, lw=1.2, shape=MSO_SHAPE.OVAL,
    radius=None, text="✓", size=12, tcolor=WHITE, bold=True)
box(10.85, 2.40, 0.44, 0.44, REDL, WHITE, lw=1.2, shape=MSO_SHAPE.OVAL,
    radius=None, text="✗", size=12, tcolor=WHITE, bold=True)
label(11.20, 2.42, 2.0, 0.36, "wasted ≤ 3%", size=14, color=MUTED,
      align=PP_ALIGN.LEFT)

# draining quota before/after
bars(7.30, 3.90, [0.62, 0.40, 0.26], [BLUED, IN2, IN3])
arrow(8.35, 3.60, 9.05, 3.60)
bars(9.25, 3.90, [0.16, 0.10, 0.06], [BLUED, IN2, IN3])
ital(6.95, 4.02, 3.8, "quota drains as verified supply arrives", size=12.5)

# ------------------------------------- RIGHT BOTTOM: distill + certificate
ital(6.80, 4.42, 6.3, "Demand-weighted distillation, certified deployment",
     size=19, align=PP_ALIGN.LEFT)

# weighted rows: doc + weight bar
for i, (wh, yy) in enumerate(((0.42, 4.95), (0.22, 5.45), (0.34, 5.95))):
    box(6.95, yy, 0.72, 0.38, WHITE, RGBColor(0x77,0x77,0x77), lw=1.1,
        radius=0.08)
    box(7.02, yy + 0.10, 0.42, 0.045, GRAYB, None, radius=None,
        shape=MSO_SHAPE.RECTANGLE)
    box(7.80, yy + 0.34 - wh, 0.16, wh, ORAN, RGBColor(0x66,0x66,0x66),
        lw=1.0, radius=None, shape=MSO_SHAPE.RECTANGLE)
ital(6.70, 6.50, 2.2, "loss weights ∝ demand served", size=12)

# probe curve
gridlines(8.75, 6.15, 1.7, n=4)
pts = [(8.80, 6.05), (9.10, 5.75), (9.45, 5.45), (9.80, 5.35),
       (10.15, 5.50)]
for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
    conn(xa, ya, xb, yb, BLUED, w=2.4, arrow=False)
box(9.72, 5.27, 0.16, 0.16, GRND, WHITE, lw=1.2, shape=MSO_SHAPE.OVAL,
    radius=None)
ital(8.45, 6.38, 2.5, "behavioral probe keeps best step", size=12)

# certificate seal + gate
box(10.90, 4.88, 2.22, 1.10, RGBColor(0xEF,0xF7,0xEE), GRND, lw=2.0,
    radius=0.10)
box(11.15, 5.00, 0.66, 0.66, GRND, WHITE, lw=1.6, shape=MSO_SHAPE.OVAL,
    radius=None, text="✓", size=16, tcolor=WHITE, bold=True)
label(11.85, 5.06, 1.30, 0.7,
      "coverage ≥ 1−α\nleakage ≤ ε", size=12.5, color=TEXT,
      align=PP_ALIGN.LEFT)
ital(11.00, 6.06, 2.0, "Certificate", size=14)
label(10.80, 6.45, 2.3, 0.32, "in → student · out → teacher", size=13,
      color=MUTED)
arrow(10.40, 5.42, 10.95, 5.42)

prs.save("paper/figs/fig1_pipeline_v12.pptx")
print("saved pptx")
