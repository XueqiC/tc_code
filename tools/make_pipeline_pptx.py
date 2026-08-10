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


def spine_arrow(x, w, text, h=0.72):
    a = box(x, SPINE - h / 2, w, h, GRAY, None, shape=MSO_SHAPE.RIGHT_ARROW,
            radius=None, text=text, size=9, tcolor=WHITE, bold=True)
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


# ---------------------------------------------------------------- headers
stage_chip(0.25, "1", "What must a student learn?")
stage_chip(3.60, "2", "A library of skill atoms")
stage_chip(6.90, "3", "Distill exactly to spec")
stage_chip(10.15, "4", "Certified deployment")

# ---------------------------------------------------------------- node A
for i in range(3):
    box(0.30 + i * 0.11, 2.30 + i * 0.11, 1.62, 1.35, CARD, GRAY, lw=1.2)
label(0.30, 1.92, 2.0, 0.32, "k example queries", size=12, color=MUTED,
      italic=True, align=PP_ALIGN.LEFT)
label(0.55, 2.72, 1.5, 0.8, '"How many members\nper club?"', size=9,
      color=TEXT, align=PP_ALIGN.LEFT)
label(1.52, 3.42, 0.5, 0.34, "×k", size=13, color=MUTED, bold=True)

spine_arrow(2.14, 0.78, "backward\npass")

# ---------------------------------------------------------------- node B
box(2.98, 1.85, 1.66, 2.55, WHITE, GRAY, lw=1.2)
rows = [[IN1, IN2, PALE, PALE], [IN1, PALE, IN3, PALE], [PALE, IN2, IN3, PALE]]
for r, cols in enumerate(rows):
    y = 2.10 + r * 0.72
    box(3.10, y, 0.30, 0.44, CARD, GRAY, lw=0.9)
    for j, c in enumerate(cols):
        tile(3.50 + j * 0.28, y + 0.07, c, s=0.26)
label(2.88, 4.50, 1.9, 0.34, "needed skills light up", size=11,
      color=IN1, bold=True)

spine_arrow(4.70, 0.72, "sparse\ncoding")

# ---------------------------------------------------------------- node C
box(5.48, 1.60, 1.98, 2.95, WHITE, GRAY, lw=1.2)
atoms = [IN1, PALE, OUT1, PALE, IN2,
         PALE, IN3, PALE, OUT2, PALE,
         OUT1, PALE, IN1, PALE, IN2]
for i, c in enumerate(atoms):
    r, cc = divmod(i, 5)
    x = 5.65 + cc * 0.34
    y = 1.86 + r * 0.85
    tile(x, y, c, s=0.30)
    if c in (IN1, IN2, IN3):
        box(x - 0.05, y - 0.05, 0.40, 0.40, None, IN1, lw=1.3,
            dash="dash", radius=0.25)
label(5.48, 1.22, 1.6, 0.30, "skill atom library", size=11, color=MUTED,
      align=PP_ALIGN.LEFT)
label(6.90, 1.62, 1.5, 0.28, "▸ fires on JOIN", size=9.5, color=TEXT,
      bold=True, align=PP_ALIGN.LEFT)
box(5.30, 4.68, 2.34, 0.62, INBG, IN1, lw=1.4,
    text="specification = the atoms\nyour k queries need", size=10,
    tcolor=IN1, bold=True)

spine_arrow(7.52, 0.70, "score by\nsupply")

# ---------------------------------------------------------------- node D
chip(8.98, 1.30, 1.05, 0.80, IN1, "Teacher LLM", size=10.5)
box(8.91, 1.98, 0.16, 0.40, GRAY, None, shape=MSO_SHAPE.DOWN_ARROW,
    radius=None)
label(9.12, 2.04, 1.0, 0.28, "traces", size=10, color=MUTED,
      align=PP_ALIGN.LEFT)
cards = [([IN1, IN2, PALE], True), ([IN1, IN3, IN2], True),
         ([OUT1, IN1, PALE], False)]
for i, (comp, keep) in enumerate(cards):
    y = 2.50 + i * 0.78
    ec = IN1 if keep else GRAY
    box(8.28, y, 1.42, 0.62, CARD if keep else PALE, ec, lw=1.3)
    for j, c in enumerate(comp):
        tile(8.42 + j * 0.33, y + 0.16, c, s=0.30)
    if not keep:
        conn(8.20, y + 0.68, 9.80, y - 0.06, OUT2, w=2.2, arrow=False)
label(8.02, 4.92, 2.0, 0.6, "rejected: carries orange\n(λ penalizes off-scope)",
      size=9.5, color=OUT2, bold=True)

spine_arrow(9.82, 0.66, "train,\nstop early")

# ---------------------------------------------------------------- node E
chip(11.02, SPINE - 0.42, 1.02, 0.82, IN1, "Student LLM", size=10.5)
box(10.48, 4.42, 1.06, 0.24, WHITE, GRAY, lw=1.1)
box(10.51, 4.45, 0.90, 0.18, IN1, None, radius=0.3)
box(11.56, 4.36, 0.30, 0.30, GREEN, WHITE, lw=1.3, shape=MSO_SHAPE.OVAL,
    radius=None, text="✓", size=10, tcolor=WHITE, bold=True)
label(10.28, 4.74, 1.7, 0.28, "demand absorbed", size=9.5, color=MUTED)

spine_arrow(11.66, 0.58, "deploy")

# ---------------------------------------------------------------- node F
gx = 12.30
box(gx, 2.10, 0.22, 2.45, GATE, IN1, lw=1.2)
box(gx + 0.74, 2.10, 0.22, 2.45, GATE, IN1, lw=1.2)
box(gx - 0.07, 1.82, 1.10, 0.30, GATE, IN1, lw=1.2)
box(gx + 0.10, 1.62, 0.46, 0.46, GREEN, WHITE, lw=1.5, shape=MSO_SHAPE.OVAL,
    radius=None, text="90%", size=9.5, tcolor=WHITE, bold=True)
label(11.55, 1.08, 1.9, 0.55, "conformal gate: coverage\nguaranteed, not tuned",
      size=9.5, color=MUTED)
conn(12.42, 2.95, 12.85, 2.50, IN1, w=2.8)
conn(12.42, 3.70, 12.85, 4.22, OUT1, w=2.8)
box(12.55, 1.98, 0.76, 0.66, INBG, IN1, lw=1.5, text="in-spec →\nStudent",
    size=9, tcolor=IN1, bold=True)
box(12.55, 4.28, 0.76, 0.66, OUTBG, OUT1, lw=1.5, text="refuse /\nescalate",
    size=9, tcolor=OUT1, bold=True)
box(10.35, 5.42, 2.85, 0.85, WHITE, GREEN, lw=1.6,
    text="Certified, both sides:\nin-task ≥ 0.93, off-task ≤ 0.25",
    size=10.5, tcolor=TEXT)

# captions under the spine: the full clause for each transition
for cx, txt in [(1.25, "one backward pass at the\nstudent's init reads each\nquery's missing skills"),
                (3.80, "a sparse dictionary over\nmany teacher traces factors\ngradients into skill atoms"),
                (6.45, "the atoms the k queries\nactivate form the task\nspecification"),
                (9.15, "traces are scored by the atoms\nthey supply; the budget is filled\nfrom the top; stop once the\nin-spec demand is absorbed")]:
    label(cx - 1.05, 5.85, 2.15, 0.75, txt, size=9.5, color=MUTED)

# ---------------------------------------------------------------- legend
lx = 3.6
box(lx - 0.25, 6.78, 7.4, 0.5, CARD, None, radius=0.5)
tile(lx, 6.90, IN1, s=0.26)
label(lx + 0.36, 6.90, 1.75, 0.3, "skill the task needs", size=11,
      color=TEXT, align=PP_ALIGN.LEFT)
tile(lx + 2.30, 6.90, OUT1, s=0.26)
label(lx + 2.66, 6.90, 1.7, 0.3, "out-of-scope skill", size=11, color=TEXT,
      align=PP_ALIGN.LEFT)
tile(lx + 4.55, 6.90, PALE, s=0.26)
label(lx + 4.91, 6.90, 1.2, 0.3, "inactive", size=11, color=TEXT,
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
