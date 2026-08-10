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


# named skill chips replace anonymous tiles: the whole figure is one
# concrete running example (an SQL-analytics agent)

def skill(x, y, name, color, w=0.62, h=0.27, fs=7.5):
    return box(x, y, w, h, color, WHITE, lw=0.75, radius=0.45, text=name,
               size=fs, tcolor=WHITE, bold=True)


CODE_FONT = "Consolas"


def code_label(x, y, w, h, text, size=7.5, color=TEXT):
    tt = label(x, y, w, h, text, size=size, color=color, align=PP_ALIGN.LEFT)
    for p in tt.text_frame.paragraphs:
        for r in p.runs:
            r.font.name = CODE_FONT
    return tt


# ---------------------------------------------------------------- headers
stage_chip(0.25, "1", "What must a student learn?")
stage_chip(3.60, "2", "A library of skill atoms")
stage_chip(6.90, "3", "Distill exactly to spec")
stage_chip(10.15, "4", "Certified deployment")

# ---------------------------------------------------------------- node A
for i in range(2, -1, -1):
    box(0.30 + i * 0.11, 2.30 + i * 0.11, 1.72, 1.40, CARD, GRAY, lw=1.2)
label(0.30, 1.92, 2.0, 0.32, "k example queries", size=12, color=MUTED,
      italic=True, align=PP_ALIGN.LEFT)
label(0.66, 2.78, 1.62, 0.9,
      '"How many members\ndoes each club\nhave?"', size=9.5,
      color=TEXT, align=PP_ALIGN.LEFT)
label(1.72, 3.52, 0.5, 0.34, "×k", size=13, color=MUTED, bold=True)

spine_arrow(2.18, 0.78, "backward\npass", fs=8)

# ---------------------------------------------------------------- node B
box(3.00, 1.85, 1.74, 2.55, WHITE, GRAY, lw=1.2)
label(3.00, 1.50, 1.74, 0.30, "each query needs:", size=10, color=MUTED)
SKILL_COLOR = {"JOIN": IN1, "COUNT": IN2, "GROUP BY": IN3, "FILTER": IN2,
               "SORT": IN3, "PLOT": OUT1, "REGEX": OUT1, "RECURSION": OUT2}
needs = [["JOIN", "COUNT"], ["GROUP BY"], ["JOIN", "FILTER"]]
for r, chips_row in enumerate(needs):
    y = 2.12 + r * 0.74
    box(3.12, y, 0.30, 0.44, CARD, GRAY, lw=0.9)
    for j, name in enumerate(chips_row):
        skill(3.50 + j * 0.60, y + 0.08, name, SKILL_COLOR[name],
              w=0.56 if len(name) < 7 else 0.72)
label(2.92, 4.50, 1.9, 0.34, "the gradient names them", size=10.5,
      color=IN1, bold=True)

spine_arrow(4.82, 0.70, "sparse\ncoding")

# ---------------------------------------------------------------- node C
box(5.58, 1.60, 2.10, 2.95, WHITE, GRAY, lw=1.2)
label(5.58, 1.22, 2.10, 0.30, "skill atom library", size=11, color=MUTED)
lib = [(n, SKILL_COLOR.get(n, PALE)) for n in
       ["JOIN", "GROUP BY", "FILTER", "COUNT", "SORT", "PLOT",
        "REGEX", "RECURSION", "…"]]
spec_names = {"JOIN", "GROUP BY", "FILTER", "COUNT", "SORT"}
for i, (name, c) in enumerate(lib):
    r, cc = divmod(i, 2)
    x = 5.76 + cc * 0.92
    y = 1.86 + r * 0.55
    tc = WHITE if c != PALE else MUTED
    box(x, y, 0.82, 0.34, c, WHITE, lw=0.75, radius=0.45, text=name,
        size=8, tcolor=tc, bold=True)
    if name in spec_names:
        box(x - 0.05, y - 0.05, 0.92, 0.44, None, IN1, lw=1.3,
            dash="dash", radius=0.4)
box(5.42, 4.68, 2.42, 0.62, INBG, IN1, lw=1.4,
    text="specification = the atoms\nyour k queries need", size=10,
    tcolor=IN1, bold=True)

spine_arrow(7.78, 0.64, "score by\nsupply")

# ---------------------------------------------------------------- node D
chip(9.16, 1.30, 1.05, 0.80, IN1, "Teacher LLM", size=10.5)
box(9.09, 1.98, 0.16, 0.36, GRAY, None, shape=MSO_SHAPE.DOWN_ARROW,
    radius=None)
label(9.30, 2.00, 1.0, 0.28, "traces", size=10, color=MUTED,
      align=PP_ALIGN.LEFT)
traces = [("SELECT c, COUNT(*)\n  FROM a JOIN b …", ["JOIN"], True),
          ("df.groupby('city')\n  .dues.sum()", ["GROUP BY"], True),
          ("plt.plot(revenue)", ["PLOT"], False)]
for i, (code, chips_row, keep) in enumerate(traces):
    y = 2.42 + i * 0.86
    ec = IN1 if keep else GRAY
    box(8.48, y, 1.62, 0.72, CARD if keep else PALE, ec, lw=1.3)
    code_label(8.58, y + 0.06, 1.5, 0.42, code, size=7)
    for j, name in enumerate(chips_row):
        skill(8.58 + j * 0.62, y + 0.46, name, SKILL_COLOR[name],
              w=0.66, h=0.22, fs=6.5)
    if not keep:
        conn(8.42, y + 0.78, 10.18, y - 0.06, OUT2, w=2.2, arrow=False)
label(8.30, 5.02, 2.0, 0.6, "rejected: supplies PLOT,\nnot in the spec (λ)",
      size=9.5, color=OUT2, bold=True)

spine_arrow(10.22, 0.60, "train,\nstop early")

# ---------------------------------------------------------------- node E
chip(11.32, 2.35, 1.02, 0.82, IN1, "Student LLM", size=10.5)
box(10.82, 3.55, 1.06, 0.24, WHITE, GRAY, lw=1.1)
box(10.85, 3.58, 0.90, 0.18, IN1, None, radius=0.3)
box(11.90, 3.49, 0.30, 0.30, GREEN, WHITE, lw=1.3, shape=MSO_SHAPE.OVAL,
    radius=None, text="✓", size=10, tcolor=WHITE, bold=True)
label(10.62, 3.86, 1.7, 0.28, "demand absorbed", size=9.5, color=MUTED)

# deployment: two live requests routed by the gate
label(10.50, 4.32, 1.55, 0.30, "live requests:", size=10, color=MUTED)
box(10.50, 4.62, 1.42, 0.52, INBG, IN1, lw=1.2)
code_label(10.58, 4.68, 1.3, 0.42, '"Avg dues\n per club?"', size=7.5,
           color=IN1)
box(10.50, 5.24, 1.42, 0.52, OUTBG, OUT1, lw=1.2)
code_label(10.58, 5.30, 1.3, 0.42, '"Plot a bar\n chart"', size=7.5,
           color=OUT1)

# ---------------------------------------------------------------- node F
gx = 12.42
box(gx, 4.10, 0.20, 1.95, GATE, IN1, lw=1.2)
box(gx + 0.68, 4.10, 0.20, 1.95, GATE, IN1, lw=1.2)
box(gx - 0.07, 3.84, 1.02, 0.28, GATE, IN1, lw=1.2)
box(gx + 0.08, 3.62, 0.44, 0.44, GREEN, WHITE, lw=1.5, shape=MSO_SHAPE.OVAL,
    radius=None, text="90%", size=9, tcolor=WHITE, bold=True)
label(12.10, 3.28, 1.25, 0.32, "conformal gate", size=9, color=MUTED)
conn(11.96, 4.88, 12.40, 4.88, IN1, w=2.6)
conn(11.96, 5.50, 12.40, 5.78, OUT1, w=2.6)
box(12.32, 4.30, 0.98, 0.5, INBG, IN1, lw=1.4, text="→ Student",
    size=8.5, tcolor=IN1, bold=True)
box(12.32, 5.85, 0.98, 0.5, OUTBG, OUT1, lw=1.4, text="refuse",
    size=8.5, tcolor=OUT1, bold=True)
box(11.98, 1.55, 1.30, 1.30, WHITE, GREEN, lw=1.6,
    text="Certified:\nin-task\n≥ 0.93\noff-task\n≤ 0.25", size=8.5,
    tcolor=TEXT)
conn(11.85, 2.35, 12.00, 2.20, GREEN, w=1.6, arrow=False)

# captions under the spine
for cx, txt in [(1.25, "one backward pass at the\nstudent's init reads each\nquery's missing skills"),
                (3.85, "a sparse dictionary over\nmany teacher traces factors\ngradients into named atoms"),
                (6.60, "the atoms the k queries\nactivate form the task\nspecification"),
                (9.30, "traces are scored by the atoms\nthey supply; budget filled from\nthe top; stop once absorbed")]:
    label(cx - 1.05, 5.85, 2.15, 0.75, txt, size=9.5, color=MUTED)

# ---------------------------------------------------------------- legend
lx = 2.9
box(lx - 0.25, 6.78, 8.2, 0.5, CARD, None, radius=0.5)
skill(lx, 6.90, "JOIN", IN1, w=0.5, h=0.26, fs=7)
label(lx + 0.60, 6.90, 1.9, 0.3, "= skill the task needs", size=11,
      color=TEXT, align=PP_ALIGN.LEFT)
skill(lx + 2.55, 6.90, "PLOT", OUT1, w=0.5, h=0.26, fs=7)
label(lx + 3.15, 6.90, 2.0, 0.3, "= out-of-scope skill", size=11,
      color=TEXT, align=PP_ALIGN.LEFT)
box(lx + 5.05, 6.90, 0.5, 0.26, PALE, WHITE, lw=0.75, radius=0.45,
    text="…", size=7, tcolor=MUTED, bold=True)
label(lx + 5.65, 6.90, 1.6, 0.3, "= other atoms", size=11, color=TEXT,
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
