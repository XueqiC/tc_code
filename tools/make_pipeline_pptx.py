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


# v8: block-per-stage layout filling the 16:9 canvas; key concepts drawn
# as physical metaphors (shelved library, checklist spec, funnel selection,
# vertical toll gate). One running example: an SQL-analytics agent.

def skill(x, y, name, color, w=0.62, h=0.28, fs=8):
    tc = WHITE if color != PALE else MUTED
    return box(x, y, w, h, color, WHITE, lw=0.75, radius=0.45, text=name,
               size=fs, tcolor=tc, bold=True)


CODE_FONT = "Consolas"


def code_label(x, y, w, h, text, size=7.5, color=TEXT):
    tt = label(x, y, w, h, text, size=size, color=color, align=PP_ALIGN.LEFT)
    for p in tt.text_frame.paragraphs:
        for r in p.runs:
            r.font.name = CODE_FONT
    return tt


SKILL_COLOR = {"JOIN": IN1, "COUNT": IN2, "GROUP BY": IN3, "FILTER": IN2,
               "SORT": IN3, "PLOT": OUT1, "REGEX": OUT1, "RECURSION": OUT2}

BLOCK_BG = RGBColor(0xFC, 0xFB, 0xF9)


def block(x, w, title):
    box(x, 0.42, w, 6.10, BLOCK_BG, GRAY, lw=1.4, radius=0.06)
    box(x + 0.14, 0.56, w - 0.28, 0.52, TEXT, None, radius=0.3, text=title,
        size=13, tcolor=WHITE, bold=True)


B1, B2, B3, B4 = 0.15, 3.42, 6.69, 10.21
W1, W2, W3, W4 = 3.05, 3.05, 3.30, 2.97
block(B1, W1, "1 · What must a student learn?")
block(B2, W2, "2 · A library of skill atoms")
block(B3, W3, "3 · Distill exactly to spec")
block(B4, W4, "4 · Certified deployment")
for gx in (3.20, 6.47, 9.99):
    a = box(gx - 0.04, 3.28, 0.32, 0.55, GRAY, None,
            shape=MSO_SHAPE.RIGHT_ARROW, radius=None)
    a.adjustments[0] = 0.55
    a.adjustments[1] = 0.55

# ================================================================ block 1
for i in range(2, -1, -1):
    box(0.42 + i * 0.12, 1.35 + i * 0.12, 1.95, 1.42, CARD, GRAY, lw=1.2)
label(0.80, 1.60, 1.9, 0.9, '"How many members\ndoes each club\nhave?"',
      size=9.5, color=TEXT, align=PP_ALIGN.LEFT)
label(2.30, 2.55, 0.5, 0.3, "×k", size=13, color=MUTED, bold=True)
label(0.40, 1.10, 2.2, 0.28, "k example queries", size=10.5, color=MUTED,
      italic=True, align=PP_ALIGN.LEFT)
box(1.42, 3.10, 0.18, 0.42, IN1, None, shape=MSO_SHAPE.DOWN_ARROW,
    radius=None)
label(1.68, 3.16, 1.5, 0.3, "one backward pass", size=9.5, color=IN1,
      bold=True, align=PP_ALIGN.LEFT)
label(0.40, 3.62, 2.4, 0.3, "it reads what each query is missing:",
      size=10, color=MUTED, align=PP_ALIGN.LEFT)
needs = [["JOIN", "COUNT"], ["GROUP BY"], ["JOIN", "FILTER"]]
for r, chips_row in enumerate(needs):
    y = 3.98 + r * 0.62
    box(0.48, y, 0.30, 0.42, CARD, GRAY, lw=0.9)
    for j, name in enumerate(chips_row):
        skill(0.90 + j * 0.86, y + 0.06, name, SKILL_COLOR[name],
              w=0.80 if len(name) > 5 else 0.62)
label(0.35, 5.95, 2.7, 0.34, "the gradient names the missing skills",
      size=10, color=IN1, bold=True)

# ================================================================ block 2
# shelved library: chips resting on shelf lines
shelves = [["JOIN", "GROUP BY", "FILTER"], ["COUNT", "SORT", "…"],
           ["PLOT", "REGEX", "RECURSION"]]
spec_names = {"JOIN", "GROUP BY", "FILTER", "COUNT", "SORT"}
for s_i, row in enumerate(shelves):
    sy = 1.95 + s_i * 0.78
    for j, name in enumerate(row):
        x = 3.62 + j * 0.94
        c = SKILL_COLOR.get(name, PALE)
        box(x, sy, 0.86, 0.34, c, WHITE, lw=0.75, radius=0.45, text=name,
            size=8, tcolor=WHITE if c != PALE else MUTED, bold=True)
        if name in spec_names:
            box(x - 0.05, sy - 0.05, 0.96, 0.44, None, IN1, lw=1.3,
                dash="dash", radius=0.4)
    conn(3.56, sy + 0.40, 6.32, sy + 0.40, GRAY, w=1.6, arrow=False)
label(3.56, 1.28, 2.8, 0.5,
      "a sparse dictionary over many traces\nstocks the shelves with named atoms",
      size=9.5, color=MUTED)
# checklist card = the specification
box(3.70, 4.42, 2.50, 1.72, WHITE, IN1, lw=1.6, radius=0.10)
label(3.82, 4.54, 2.3, 0.32, "Task Specification", size=11, color=IN1,
      bold=True, align=PP_ALIGN.LEFT)
label(3.94, 4.92, 2.2, 0.32, "✓ JOIN    ✓ GROUP BY", size=10, color=IN1,
      bold=True, align=PP_ALIGN.LEFT)
label(3.94, 5.24, 2.2, 0.32, "✓ FILTER  ✓ COUNT  ✓ SORT", size=10,
      color=IN1, bold=True, align=PP_ALIGN.LEFT)
label(3.94, 5.56, 2.2, 0.32, "✗ PLOT   ✗ REGEX", size=10, color=OUT2,
      bold=True, align=PP_ALIGN.LEFT)
label(3.82, 5.90, 2.3, 0.30, "= what the k queries need, no more",
      size=8.5, color=MUTED, align=PP_ALIGN.LEFT)

# ================================================================ block 3
chip(7.62, 1.72, 1.00, 0.76, IN1, "Teacher LLM", size=10)
box(7.55, 2.36, 0.16, 0.34, GRAY, None, shape=MSO_SHAPE.DOWN_ARROW,
    radius=None)
label(7.76, 2.38, 1.0, 0.28, "traces", size=9.5, color=MUTED,
      align=PP_ALIGN.LEFT)
traces = [("SELECT c, COUNT(*)\n FROM a JOIN b …", "JOIN", True),
          ("df.groupby('city')\n .dues.sum()", "GROUP BY", True),
          ("plt.plot(revenue)", "PLOT", False)]
for i, (code, chipname, keep) in enumerate(traces):
    y = 2.62 + i * 0.86
    ec = IN1 if keep else GRAY
    box(6.90, y, 1.62, 0.78, CARD if keep else PALE, ec, lw=1.3)
    code_label(7.00, y + 0.06, 1.5, 0.42, code, size=7)
    skill(7.00, y + 0.48, chipname, SKILL_COLOR[chipname], w=0.80, h=0.22,
          fs=6.5)
    if keep:
        conn(8.56, y + 0.36, 8.98, 3.40 + i * 0.12, IN1, w=2.2)
    else:
        conn(6.84, y + 0.84, 8.66, y - 0.06, OUT2, w=2.2, arrow=False)
label(6.82, 5.26, 2.0, 0.55, "rejected: supplies\nPLOT, off the spec",
      size=9, color=OUT2, bold=True)
# funnel (two slanted sides) + lambda
conn(8.92, 2.65, 9.28, 4.25, GRAY, w=2.6, arrow=False)
conn(9.90, 2.65, 9.54, 4.25, GRAY, w=2.6, arrow=False)
label(9.44, 2.72, 0.6, 0.3, "λ", size=14, color=OUT2, bold=True)
label(8.96, 2.32, 1.0, 0.3, "budget", size=9, color=MUTED)
box(9.32, 4.28, 0.18, 0.34, IN1, None, shape=MSO_SHAPE.DOWN_ARROW,
    radius=None)
chip(9.28, 5.18, 0.94, 0.74, IN1, "Student LLM", size=9.5)
# absorption meter row at the block foot
box(6.90, 5.92, 1.00, 0.22, WHITE, GRAY, lw=1.0)
box(6.93, 5.95, 0.85, 0.16, IN1, None, radius=0.3)
box(7.92, 5.88, 0.28, 0.28, GREEN, WHITE, lw=1.2, shape=MSO_SHAPE.OVAL,
    radius=None, text="✓", size=9, tcolor=WHITE, bold=True)
label(6.80, 6.20, 2.0, 0.26, "train until absorbed, stop", size=8.5,
      color=MUTED, align=PP_ALIGN.LEFT)

# ================================================================ block 4
label(10.40, 1.22, 2.0, 0.3, "live requests", size=10.5, color=MUTED,
      italic=True, align=PP_ALIGN.LEFT)
box(10.44, 1.55, 1.28, 0.62, INBG, IN1, lw=1.3)
code_label(10.52, 1.62, 1.2, 0.48, '"Avg dues\n per club?"', size=7.5,
           color=IN1)
box(11.82, 1.55, 1.28, 0.62, OUTBG, OUT1, lw=1.3)
code_label(11.90, 1.62, 1.2, 0.48, '"Plot a bar\n chart"', size=7.5,
           color=OUT1)
# horizontal toll bar with an opening under the blue card
box(10.40, 3.24, 0.58, 0.32, GATE, IN1, lw=1.3)
box(11.76, 3.24, 1.38, 0.32, GATE, IN1, lw=1.3)
box(11.16, 2.72, 0.44, 0.44, GREEN, WHITE, lw=1.5, shape=MSO_SHAPE.OVAL,
    radius=None, text="90%", size=9, tcolor=WHITE, bold=True)
label(11.50, 3.64, 1.75, 0.5, "conformal gate:\nguaranteed, not tuned",
      size=8.5, color=MUTED, align=PP_ALIGN.LEFT)
conn(11.08, 2.20, 11.08, 4.05, IN1, w=2.8)          # through the opening
conn(12.46, 2.20, 12.46, 3.24, OUT1, w=2.8)          # blocked at the bar
conn(12.46, 3.28, 13.02, 4.42, OUT1, w=2.8)          # deflected
chip(11.08, 4.75, 0.94, 0.74, IN1, "Student agent", size=9.5)
box(12.42, 4.48, 0.72, 0.5, OUTBG, OUT1, lw=1.4, text="refuse",
    size=9, tcolor=OUT1, bold=True)
box(10.42, 5.70, 2.56, 0.68, WHITE, GREEN, lw=1.6,
    text="Certified, both sides:\nin-task ≥ 0.93 · off-task ≤ 0.25",
    size=9.5, tcolor=TEXT)

# ================================================================ legend
lx = 3.4
box(lx - 0.25, 6.72, 7.6, 0.55, CARD, None, radius=0.5)
skill(lx, 6.86, "JOIN", IN1, w=0.55, h=0.27, fs=7.5)
label(lx + 0.65, 6.86, 1.85, 0.3, "= skill the task needs", size=11,
      color=TEXT, align=PP_ALIGN.LEFT)
skill(lx + 2.60, 6.86, "PLOT", OUT1, w=0.55, h=0.27, fs=7.5)
label(lx + 3.25, 6.86, 1.95, 0.3, "= out-of-scope skill", size=11,
      color=TEXT, align=PP_ALIGN.LEFT)
skill(lx + 5.15, 6.86, "…", PALE, w=0.55, h=0.27, fs=7.5)
label(lx + 5.80, 6.86, 1.5, 0.3, "= other atoms", size=11, color=TEXT,
      align=PP_ALIGN.LEFT)

prs.save("paper/figs/fig1_pipeline.pptx")

# strip shadow effects from the theme itself: some renderers (LibreOffice)
# apply theme effect styles even when shapes carry no style reference
import re, zipfile, shutil

src = "paper/figs/fig1_pipeline.pptx"
tmp = src + ".tmp"
with zipfile.ZipFile(src) as zin, zipfile.ZipFile(tmp, "w",
                                                  zipfile.ZIP_DEFLATED) as zout:
    for item in zin.namelist():
        data = zin.read(item)
        if item.startswith("ppt/theme/") and item.endswith(".xml"):
            x = data.decode("utf-8")
            x = re.sub(r"<a:outerShdw[^>]*>.*?</a:outerShdw>", "", x,
                       flags=re.S)
            x = re.sub(r"<a:outerShdw[^>]*/>", "", x)
            data = x.encode("utf-8")
        zout.writestr(item, data)
shutil.move(tmp, src)
print("saved paper/figs/fig1_pipeline.pptx (theme shadows stripped)")
