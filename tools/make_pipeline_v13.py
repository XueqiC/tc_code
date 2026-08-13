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

# -------------------------------------------------------------- v13 body
# Denser reference-style overview: everything bold dark, numbered stages,
# focal highlighted acquisition rule, named atoms, cycle arrow.

BLUE = RGBColor(0xA8, 0xC4, 0xE4)
BLUED = RGBColor(0x4C, 0x72, 0xB0)
ORAN = RGBColor(0xF2, 0xB9, 0x8F)
GRNB = RGBColor(0x9E, 0xC9, 0x9A)
GRND = RGBColor(0x3E, 0x7A, 0x3A)
GRAYB = RGBColor(0xC9, 0xC9, 0xC9)
REDL = RGBColor(0xE2, 0x9A, 0x9A)
GOLD = RGBColor(0xE8, 0xC5, 0x6A)
DARK = RGBColor(0x11, 0x11, 0x11)
PALEY = RGBColor(0xFF, 0xF6, 0xDE)
PALEB = RGBColor(0xF0, 0xF5, 0xFB)

def gridlines(x, y, w, n=4, gap=0.16):
    for i in range(n):
        conn(x, y - i * gap, x + w, y - i * gap, RGBColor(0xE0,0xE0,0xE0),
             w=0.9, arrow=False)

def bars(x, y, hs, colors, bw=0.22, gap=0.07, grid=True):
    n = len(hs)
    W = n * bw + (n - 1) * gap
    if grid:
        gridlines(x - 0.05, y - 0.02, W + 0.10)
    conn(x - 0.05, y, x + W + 0.05, y, DARK, w=1.6, arrow=False)
    for i, h in enumerate(hs):
        c = colors[i] if isinstance(colors, (list, tuple)) else colors
        box(x + i * (bw + gap), y - h, bw, h, c, RGBColor(0x44,0x44,0x44),
            lw=1.1, radius=None, shape=MSO_SHAPE.RECTANGLE)
    return W

def ital(x, y, w, t, size=15.9, color=DARK, align=PP_ALIGN.CENTER):
    label(x, y, w, 0.32, t, size=size, color=color, italic=True, bold=True,
          align=align)

def dashline(x0, y0, x1, y1):
    c = conn(x0, y0, x1, y1, RGBColor(0x8A,0x8A,0x8A), w=1.8, arrow=False)
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:prstDash"), {"val": "dash"}))

def arrow(x0, y0, x1, y1, color=DARK, w=2.6):
    conn(x0, y0, x1, y1, color, w=w, arrow=True)

def stagenum(x, y, n):
    box(x, y, 0.50, 0.50, DARK, None, shape=MSO_SHAPE.OVAL, radius=None,
        text=str(n), size=22, tcolor=WHITE, bold=True)

def matrix_icon(x, y, s=1.05, name=None):
    box(x, y, s, s, WHITE, DARK, lw=1.6, radius=0.12)
    cell = s / 3.6
    cols = (BLUED, GRNB, ORAN)
    for r in range(3):
        for cc in range(3):
            box(x + 0.08 + cc * cell, y + 0.08 + r * cell, cell * 0.8,
                cell * 0.8, cols[(r + cc) % 3], WHITE, lw=0.6, radius=None,
                shape=MSO_SHAPE.RECTANGLE)
    if name:
        ital(x - 0.45, y + s + 0.05, s + 0.9, name, size=14.6)

def coin_stack(x, y, n=4):
    for i in range(n):
        box(x, y - i * 0.14, 0.52, 0.125, GOLD, RGBColor(0x9A,0x7B,0x2E),
            lw=1.1, shape=MSO_SHAPE.OVAL, radius=None)

def docs(x, y, w=1.18, h=1.40, nlines=3):
    for dx in (0.0, 0.18, 0.36):
        box(x + dx, y + dx * 0.4, w, h, WHITE, RGBColor(0x44,0x44,0x44),
            lw=1.3, radius=0.06)
    for i in range(nlines):
        box(x + 0.52, y + 0.46 + i * 0.26, 0.74, 0.065, GRAYB, None,
            radius=None, shape=MSO_SHAPE.RECTANGLE)

# frame + dividers
box(0.12, 0.10, 13.09, 7.28, WHITE, RGBColor(0x77,0x77,0x77), lw=1.8,
    radius=0.03)
dashline(6.55, 0.22, 6.55, 7.26)
dashline(6.66, 4.42, 13.10, 4.42)

# ============================== LEFT: demand ==============================
stagenum(0.32, 0.26, 1)
ital(0.95, 0.32, 5.4, "Reading the task as skill demand", size=20,
     align=PP_ALIGN.LEFT)

docs(0.50, 0.92)
ital(0.25, 2.52, 2.3, "Support Set S", size=14.6)
arrow(2.16, 1.62, 2.60, 1.62)
label(2.42, 0.98, 0.9, 0.34, "∇", size=27, color=DARK, bold=True)

chip(3.30, 1.55, 1.55, 1.35, BLUED, "Student Model", size=16)
arrow(4.12, 1.55, 4.55, 1.55)

bars(4.68, 2.10, [0.74, 0.36, 0.60, 0.27], [ORAN]*4)
ital(4.35, 2.42, 2.0, "Gradient Features", size=14.6)

# sparse dictionary formula in pale box
box(0.45, 2.92, 3.95, 0.60, PALEB, BLUED, lw=1.6, radius=0.18,
    text="min‖X − CD‖²  +  λ‖C‖₁", size=20.7, tcolor=DARK, bold=True)
ital(4.50, 3.06, 2.0, "sparse dictionary", size=14.6)
arrow(2.40, 3.54, 2.40, 3.76)

# atoms with names
NAMES = ("a₁", "a₂", "a₃", "a₄", "a₅")
for i, (c, nm) in enumerate(zip((BLUED, IN2, IN3, ORAN, REDL), NAMES)):
    x0 = 0.55 + i * 0.86
    box(x0, 3.84, 0.66, 0.66, c, WHITE, lw=1.4, radius=0.18)
    ital(x0 - 0.20, 4.54, 1.05, nm, size=13.5)
ital(0.60, 4.88, 3.8, "Capability Atoms (dictionary D)", size=15.9)
arrow(4.90, 4.16, 5.18, 4.16)

bars(5.28, 4.55, [0.78, 0.52, 0.33, 0.12], [BLUED, IN2, IN3, GRAYB])
ital(5.22, 4.95, 1.3, "Task Demand w", size=13.5)

# quota line with coin
box(0.60, 5.28, 4.40, 0.58, PALEY, RGBColor(0xB8,0x8A,0x2E), lw=1.6,
    radius=0.18, text="per-skill token quota  =  w · b", size=19.5,
    tcolor=DARK, bold=True)
coin_stack(5.25, 5.70, n=3)

# bridge strip
box(0.40, 6.02, 5.90, 1.30, CARD, RGBColor(0x9A,0x9A,0x9A), lw=1.3,
    radius=0.10)
ital(0.55, 6.06, 5.5, "Cross-modal bridge (fit on paid pairs)", size=15,
     align=PP_ALIGN.LEFT)
bars(0.85, 7.02, [0.30, 0.48, 0.22], [GRAYB]*3, grid=False)
ital(0.50, 7.06, 1.7, "Prompt g", size=12)
arrow(1.95, 6.75, 2.40, 6.75)
matrix_icon(2.50, 6.35, s=0.80)
ital(2.22, 7.06, 1.4, "Ridge Map", size=12)
arrow(3.50, 6.75, 3.95, 6.75)
bars(4.10, 7.02, [0.55, 0.25, 0.42], [GRNB]*3, grid=False)
ital(3.90, 7.06, 1.8, "Predicted Skills", size=12)
label(5.42, 6.48, 0.95, 0.6, "legal before\npaying", size=12,
      color=DARK, bold=True, italic=True)

# ====================== RIGHT TOP: acquisition loop ======================
stagenum(6.75, 0.26, 2)
ital(7.40, 0.32, 5.9, "Spending the budget where demand is unmet",
     size=22, align=PP_ALIGN.LEFT)

chip(7.60, 1.50, 1.55, 1.35, RGBColor(0x8A,0x5A,0xA8), "Teacher API",
     size=16)
coin_stack(8.48, 1.35)
ital(8.50, 1.58, 0.9, "b", size=19)

arrow(9.10, 1.50, 9.42, 1.50)
box(9.50, 1.00, 0.92, 1.08, WHITE, RGBColor(0x44,0x44,0x44), lw=1.3,
    radius=0.06)
for yy in (1.24, 1.48, 1.72):
    box(9.64, yy, 0.62, 0.06, GRAYB, None, radius=None,
        shape=MSO_SHAPE.RECTANGLE)
ital(9.25, 2.16, 1.4, "candidate x", size=14)
arrow(10.48, 1.50, 10.78, 1.50)
bars(10.88, 1.95, [0.50, 0.24, 0.40], [GRNB]*3, grid=False)
ital(10.62, 2.16, 1.7, "ŝ(x) predicted", size=14)
arrow(11.85, 1.50, 12.10, 1.50)
bars(12.18, 1.95, [0.62, 0.18, 0.36], [BLUED, GRAYB, IN2], grid=False)
ital(11.95, 2.16, 1.5, "r remaining", size=13.5)

# focal rule
box(7.30, 2.45, 4.30, 0.62, PALEY, RGBColor(0xB8,0x8A,0x2E), lw=2.2,
    radius=0.18, text="buy  argmax  u(x) = Σ min(r, ŝ(x)) ∕ t̂",
    size=19.5, tcolor=DARK, bold=True)
# cycle arrow back to teacher
conn(7.30, 2.76, 6.95, 2.76, DARK, w=2.2, arrow=False)
conn(6.95, 2.76, 6.95, 1.75, DARK, w=2.2, arrow=False)
arrow(6.95, 1.75, 7.05, 1.62)
ital(11.70, 2.52, 1.6, "each round", size=14)

# returns + gate + drain
label(6.90, 3.20, 3.5, 0.34, "returns → exec ✓ + gate",
      size=15, color=DARK, bold=True, align=PP_ALIGN.LEFT)
box(10.40, 3.10, 0.54, 0.54, GRNB, WHITE, lw=1.3, shape=MSO_SHAPE.OVAL,
    radius=None, text="✓", size=18.3, tcolor=WHITE, bold=True)
box(11.05, 3.10, 0.54, 0.54, REDL, WHITE, lw=1.3, shape=MSO_SHAPE.OVAL,
    radius=None, text="✗", size=18.3, tcolor=WHITE, bold=True)
label(11.72, 3.18, 1.6, 0.36, "wasted ≤ 3%", size=14.5, color=DARK,
      bold=True, align=PP_ALIGN.LEFT)

bars(7.60, 4.25, [0.60, 0.40, 0.25], [BLUED, IN2, IN3])
arrow(8.85, 4.00, 9.35, 4.00)
bars(9.50, 4.25, [0.15, 0.10, 0.06], [BLUED, IN2, IN3])
ital(10.35, 3.86, 2.9, "quota drains as verified", size=14.6)
ital(10.35, 4.10, 2.9, "supply arrives", size=14.6)

# ============== RIGHT BOTTOM: distill + certify ==============
stagenum(6.75, 4.56, 3)
ital(7.38, 4.64, 3.1, "Weighted distillation", size=18,
     align=PP_ALIGN.LEFT)
stagenum(10.42, 4.56, 4)
ital(11.00, 4.64, 2.25, "Certified deployment", size=16.5,
     align=PP_ALIGN.LEFT)

for i, (wh, yy) in enumerate(((0.50, 5.15), (0.26, 5.70), (0.42, 6.25))):
    box(6.90, yy, 0.90, 0.46, WHITE, RGBColor(0x44,0x44,0x44), lw=1.2,
        radius=0.08)
    box(7.00, yy + 0.13, 0.54, 0.06, GRAYB, None, radius=None,
        shape=MSO_SHAPE.RECTANGLE)
    box(7.92, yy + 0.42 - wh, 0.22, wh, ORAN, RGBColor(0x44,0x44,0x44),
        lw=1.1, radius=None, shape=MSO_SHAPE.RECTANGLE)
ital(6.72, 6.88, 2.2, "loss ∝ demand", size=12.5)
arrow(8.30, 5.85, 8.72, 5.85)
chip(9.45, 5.85, 1.35, 1.18, BLUED, "Student", size=15)

# probe curve
gridlines(7.35, 7.10, 1.55, n=3)
pts = [(7.38, 7.03), (7.70, 6.82), (8.05, 6.62), (8.35, 6.56),
       (8.65, 6.68)]
for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
    conn(xa, ya, xb, yb, BLUED, w=2.6, arrow=False)
box(8.27, 6.48, 0.17, 0.17, GRND, WHITE, lw=1.2, shape=MSO_SHAPE.OVAL,
    radius=None)
ital(7.15, 7.12, 2.2, "probe keeps best step", size=12)

# certificate + gate
box(10.55, 5.05, 2.45, 1.10, RGBColor(0xEC,0xF6,0xEB), GRND, lw=2.2,
    radius=0.10)
box(10.70, 5.18, 0.78, 0.78, GRND, WHITE, lw=1.6, shape=MSO_SHAPE.OVAL,
    radius=None, text="✓", size=22, tcolor=WHITE, bold=True)
label(11.48, 5.22, 1.45, 0.72, "coverage ≥ 1−α\nleakage ≤ ε", size=15.9,
      color=DARK, bold=True, align=PP_ALIGN.LEFT)
ital(10.70, 6.20, 2.2, "Certificate", size=15.9)
box(10.70, 6.52, 0.60, 0.60, GATE, DARK, lw=1.4,
    shape=MSO_SHAPE.DIAMOND, radius=None)
label(11.35, 6.52, 1.75, 0.3, "in → student", size=14.6, color=GRND,
      bold=True, align=PP_ALIGN.LEFT)
label(11.35, 6.80, 1.75, 0.3, "out → teacher", size=14.6,
      color=RGBColor(0x8A,0x5A,0xA8), bold=True, align=PP_ALIGN.LEFT)
arrow(10.15, 6.82, 10.62, 6.82)

prs.save("paper/figs/fig1_pipeline_v13.pptx")
print("saved")
