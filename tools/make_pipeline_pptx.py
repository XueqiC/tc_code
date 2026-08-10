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


# v9: SkillOpt-Fig2 register — one central capability-space scene with
# contrasting outcomes, plus an analogy panel and a benefits fan.

from pptx.oxml.ns import qn as _qn


def alpha_fill(shape, pct):
    """Make the solid fill partially transparent (pct = opacity %)."""
    sf = shape.fill.fore_color._xFill
    clr = sf.find(_qn("a:srgbClr"))
    a = clr.makeelement(_qn("a:alpha"), {"val": str(int(pct * 1000))})
    clr.append(a)


def skill(x, y, name, color, w=0.62, h=0.28, fs=8):
    tc = WHITE if color != PALE else MUTED
    return box(x, y, w, h, color, WHITE, lw=0.75, radius=0.45, text=name,
               size=fs, tcolor=tc, bold=True)


def pill(x, y, w, text, color=TEXT, fs=9.5, bg=RGBColor(0xEC, 0xEA, 0xE4)):
    return box(x, y, w, 0.34, bg, None, radius=0.5, text=text, size=fs,
               tcolor=color, bold=True)


CODE_FONT = "Consolas"
SKILL_COLOR = {"JOIN": IN1, "COUNT": IN2, "GROUP BY": IN3, "FILTER": IN2,
               "SORT": IN3, "PLOT": OUT1, "REGEX": OUT1, "RECURSION": OUT2}

# ================================================================ scene
# teacher capability: big soft region (left ~2/3 of canvas)
teach = box(0.55, 0.85, 8.9, 6.0, OUTBG, None, shape=MSO_SHAPE.OVAL,
            radius=None)
alpha_fill(teach, 45)
chip(1.55, 1.75, 0.92, 0.70, OUT1, "Teacher LLM  ❄", size=9.5)
label(1.05, 0.62, 3.2, 0.3, "teacher capability (frozen, expensive)",
      size=10.5, color=OUT1, bold=True, align=PP_ALIGN.LEFT)

# out-of-scope skills live in the teacher blob, outside the boundary
for (x, y, name) in [(6.6, 1.7, "PLOT"), (7.6, 2.6, "REGEX"),
                     (6.9, 3.9, "RECURSION"), (7.7, 5.0, "…"),
                     (5.9, 5.6, "…")]:
    skill(x, y, name, SKILL_COLOR.get(name, PALE), w=0.9 if len(name) > 6
          else 0.62)

# task boundary: thick dashed contour around the needed-skill cluster
bnd = box(1.35, 2.35, 4.35, 3.75, WHITE, IN1, lw=3.0, dash="dash",
          shape=MSO_SHAPE.OVAL, radius=None)
bnd.fill.background()

# our student: blue region hugging the boundary from inside
stu = box(1.55, 2.55, 3.95, 3.35, INBG, None, shape=MSO_SHAPE.OVAL,
          radius=None)
alpha_fill(stu, 80)
for (x, y, name) in [(2.2, 3.1, "JOIN"), (3.4, 2.9, "GROUP BY"),
                     (2.0, 4.1, "FILTER"), (3.3, 3.9, "COUNT"),
                     (2.6, 4.9, "SORT")]:
    skill(x, y, name, SKILL_COLOR[name], w=0.92 if len(name) > 6 else 0.66)
chip(4.35, 5.1, 0.85, 0.66, IN1, "Student  🔥", size=9)

# baseline: dashed gray blob that spills across the boundary
base = box(3.3, 1.35, 5.3, 3.3, PALE, GRAY, lw=2.0, dash="sysDash",
           shape=MSO_SHAPE.OVAL, radius=None)
alpha_fill(base, 30)

# callout pills anchored on the scene (SkillOpt-style)
pill(0.85, 6.55, 3.6, "our student: capability ends at the boundary",
     color=IN1, fs=9, bg=INBG)
conn(2.6, 6.52, 3.1, 5.95, IN1, w=1.6)
pill(5.15, 0.62, 3.55, "budget-matched baseline: leaks out of scope",
     color=OUT2, fs=9)
conn(6.9, 0.99, 6.6, 1.55, OUT2, w=1.6)
label(5.05, 2.62, 1.3, 0.5, "leakage", size=9.5, color=OUT2, bold=True,
      italic=True)
conn(5.6, 2.75, 6.35, 2.75, OUT2, w=2.0)

# boundary estimation: k queries -> gradient reading -> boundary
for i in range(2, -1, -1):
    box(0.60 + i * 0.09, 3.95 + i * 0.09, 1.05, 0.78, CARD, GRAY, lw=1.0)
label(0.66, 4.06, 1.0, 0.7, '"members\nper club?"', size=7, color=TEXT,
      align=PP_ALIGN.LEFT)
label(0.42, 3.60, 1.6, 0.3, "k queries", size=9.5, color=MUTED, italic=True,
      align=PP_ALIGN.LEFT)
conn(1.35, 4.35, 1.85, 4.35, IN1, w=2.0)
pill(0.55, 5.62, 2.9, "boundary read from gradients, k≈5", color=IN1,
     fs=8.5, bg=INBG)

# gate on the boundary + certificate seal
box(5.28, 3.95, 0.44, 0.44, GREEN, WHITE, lw=1.5, shape=MSO_SHAPE.OVAL,
    radius=None, text="90%", size=8.5, tcolor=WHITE, bold=True)
label(5.78, 3.98, 1.45, 0.55, "conformal gate\non the boundary", size=8,
      color=GREEN, bold=True, align=PP_ALIGN.LEFT)

# distillation flow along the bottom of the scene
label(2.4, 6.95, 6.0, 0.35,
      "distill: select only traces whose atoms lie inside the boundary "
      "(λ rejects the rest) · train until absorbed, then stop",
      size=9.5, color=MUTED)

# ================================================================ side rail
RX = 9.75
# mini mechanism: query -> backward pass -> named atoms
box(RX, 0.62, 3.35, 1.55, WHITE, GRAY, lw=1.2, radius=0.08)
box(RX + 0.14, 0.76, 0.82, 0.62, CARD, GRAY, lw=1.0)
label(RX + 0.18, 0.84, 0.78, 0.5, '"query"', size=7.5, color=TEXT)
conn(RX + 1.02, 1.07, RX + 1.38, 1.07, IN1, w=1.8)
skill(RX + 1.44, 0.78, "JOIN", IN1, w=0.6, h=0.26, fs=7)
skill(RX + 2.10, 0.78, "COUNT", IN2, w=0.72, h=0.26, fs=7)
label(RX + 1.40, 1.12, 1.9, 0.3, "one backward pass", size=8, color=IN1,
      bold=True, align=PP_ALIGN.LEFT)
label(RX + 0.14, 1.62, 3.1, 0.5,
      "the gradient names the skills a query\nneeds — before any training",
      size=8.5, color=MUTED, align=PP_ALIGN.LEFT)

# analogy panel (SkillOpt's table, ours)
box(RX, 2.42, 3.35, 2.85, RGBColor(0xF2, 0xF1, 0xEC), None, radius=0.06)
label(RX + 0.15, 2.56, 3.05, 0.32, "Capability-matching, one representation",
      size=10, color=TEXT, bold=True, align=PP_ALIGN.LEFT)
rows = [("task boundary", "atom support of k queries"),
        ("capability", "atom supply of the diet"),
        ("selection", "budgeted coverage, λ"),
        ("stopping", "demand absorbed"),
        ("guarantee", "conformal + binomial")]
for i, (a, b) in enumerate(rows):
    y = 2.95 + i * 0.44
    label(RX + 0.15, y, 1.15, 0.4, a, size=9, color=TEXT, bold=True,
          align=PP_ALIGN.LEFT)
    label(RX + 1.32, y, 0.25, 0.4, "→", size=9, color=MUTED)
    label(RX + 1.60, y, 1.72, 0.4, b, size=9, color=MUTED,
          align=PP_ALIGN.LEFT)

# benefits fan (bottom right)
box(RX + 0.05, 5.55, 0.75, 0.85, INBG, IN1, lw=1.4, radius=0.12,
    text="🎓", size=16)
for i, txt in enumerate(["exact in-task ability, tied to spec",
                         "certified leakage ≤ 0.25",
                         "same performance at ¼ budget"]):
    y = 5.48 + i * 0.48
    box(RX + 1.0, y, 2.35, 0.36, RGBColor(0xDCE, 0xE5 % 256, 0xEF)
        if False else RGBColor(0xDC, 0xE5, 0xEF), None, radius=0.5,
        text=txt, size=8.5, tcolor=TEXT, bold=True)
    conn(RX + 0.82, 5.95, RX + 0.98, y + 0.18, GRAY, w=1.2, arrow=False)

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
