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


# v11: fewer, bigger, one focal element per block; the demand ledger is
# the visual hero. Minimal text, no sub-caption rows, no legend bar.

SKILL_COLOR = {"JOIN": IN1, "COUNT": IN2, "GROUP BY": IN3,
               "PLOT": OUT1, "REGEX": OUT2}
PALE = RGBColor(0xE8, 0xE6, 0xE1)
CARD = RGBColor(0xF7, 0xF6, 0xF2)
INBG = RGBColor(0xEE, 0xF2, 0xF9)
IN1 = RGBColor(0x4C, 0x72, 0xB0)
IN2 = RGBColor(0x5F, 0x9E, 0xA0)
IN3 = RGBColor(0x7B, 0xA7, 0xD7)
OUT1 = RGBColor(0xDD, 0x84, 0x52)
OUT2 = RGBColor(0xC4, 0x4E, 0x52)
GREEN = RGBColor(0x55, 0xA8, 0x68)


def skill(x, y, name, color, w=0.95, h=0.42, fs=11):
    return box(x, y, w, h, color, WHITE, lw=1.0, radius=0.45, text=name,
               size=fs, tcolor=WHITE, bold=True)


BLOCK_BG = RGBColor(0xFC, 0xFB, 0xF9)


def block(x, w, title):
    box(x, 0.42, w, 6.60, BLOCK_BG, GRAY, lw=1.4, radius=0.06)
    box(x + 0.14, 0.56, w - 0.28, 0.56, TEXT, None, radius=0.3, text=title,
        size=14, tcolor=WHITE, bold=True)


B1, B2, B3, B4 = 0.15, 3.02, 5.89, 10.31
W1, W2, W3, W4 = 2.65, 2.65, 4.20, 2.90
block(B1, W1, "Gradient Fingerprints")
block(B2, W2, "Capability Atoms")
block(B3, W3, "Market-Clearing Selection")
block(B4, W4, "Certified Agent")
for gx in (2.80, 5.67, 10.09):
    a = box(gx - 0.03, 3.35, 0.36, 0.62, GRAY, None,
            shape=MSO_SHAPE.RIGHT_ARROW, radius=None)
    a.adjustments[0] = 0.55
    a.adjustments[1] = 0.55

# ---- block 1: one query -> one barcode fingerprint
box(0.55, 1.60, 1.85, 1.30, CARD, GRAY, lw=1.3)
label(0.70, 1.80, 1.6, 0.95, '"How many members\ndoes each club\nhave?"', size=10,
      color=TEXT, align=PP_ALIGN.LEFT)
box(1.38, 3.10, 0.18, 0.55, IN1, None, shape=MSO_SHAPE.DOWN_ARROW,
    radius=None)
label(0.35, 3.72, 2.3, 0.3, "one backward pass", size=11, color=IN1,
      bold=True)
stripes = [IN1, PALE, IN3, IN2, PALE, IN1, PALE, IN2, IN3, PALE, IN1, PALE]
for i, c in enumerate(stripes):
    box(0.66 + i * 0.145, 4.35, 0.11, 1.10, c, None, radius=0.3)
box(0.55, 4.22, 1.92, 1.38, None, GRAY, lw=1.2, radius=0.08)
label(0.35, 5.80, 2.3, 0.35, "what must be learned", size=11, color=MUTED,
      italic=True)

# ---- block 2: named atoms + demand gauges (the ledger)
label(3.14, 1.28, 2.4, 0.3, "recurring skills", size=11, color=MUTED,
      italic=True)
atoms = [("JOIN", IN1, 0.92), ("GROUP BY", IN3, 0.60), ("COUNT", IN2, 0.35),
         ("PLOT", OUT1, 0.0)]
for i, (name, c, dem) in enumerate(atoms):
    y = 1.72 + i * 1.05
    skill(3.22, y, name, c, w=1.18, h=0.46, fs=11)
    box(4.52, y + 0.06, 1.00, 0.34, WHITE, GRAY, lw=1.0, radius=0.2)
    if dem > 0:
        box(4.55, y + 0.09, 0.94 * dem, 0.28, c, None, radius=0.2)
    else:
        label(4.52, y + 0.06, 1.0, 0.34, "not needed", size=8.5, color=MUTED)
label(3.25, 6.00, 2.2, 0.55, "demand ledger\n(read from the k queries)",
      size=11, color=IN1, bold=True)

# ---- block 3 (HERO): traces flow in, ledger drains, budget bar
label(6.05, 1.26, 3.9, 0.34,
      "each pick satisfies the most remaining demand per token",
      size=10.5, color=TEXT, bold=True)
traces = [("SELECT … JOIN …", "JOIN", True),
          ("df.groupby(…)", "GROUP BY", True),
          ("plt.plot(…)", "PLOT", False)]
for i, (code, chipname, keep) in enumerate(traces):
    y = 1.78 + i * 1.02
    ec = IN1 if keep else GRAY
    box(6.10, y, 1.60, 0.82, CARD if keep else PALE, ec, lw=1.3)
    label(6.20, y + 0.08, 1.45, 0.35, code, size=9, color=TEXT,
          align=PP_ALIGN.LEFT)
    skill(6.20, y + 0.44, chipname, SKILL_COLOR[chipname], w=0.88, h=0.28,
          fs=8)
    if keep:
        conn(7.74, y + 0.40, 8.28, 2.60 + i * 0.35, IN1, w=2.6)
    else:
        conn(6.04, y + 0.88, 7.78, y - 0.05, OUT2, w=2.4, arrow=False)
label(6.10, 4.90, 1.9, 0.55, "zero remaining demand\n= zero value",
      size=9, color=OUT2, bold=True)
label(8.35, 1.62, 1.6, 0.3, "ledger drains", size=11, color=IN1, bold=True)
drain = [("JOIN", IN1, 0.92, 0.15), ("GROUP BY", IN3, 0.60, 0.10),
         ("COUNT", IN2, 0.35, 0.05)]
for i, (name, c, before, after) in enumerate(drain):
    y = 2.00 + i * 0.98
    box(8.35, y, 1.45, 0.30, WHITE, GRAY, lw=1.0, radius=0.2)
    box(8.38, y + 0.03, max(1.39 * before, 0.02), 0.24, c, None, radius=0.2)
    box(8.35, y + 0.42, 1.45, 0.30, WHITE, GRAY, lw=1.0, radius=0.2)
    if after > 0:
        box(8.38, y + 0.45, max(1.39 * after, 0.02), 0.24, c, None,
            radius=0.2)
label(8.35, 4.90, 1.6, 0.3, "before / after", size=9.5, color=MUTED)
box(6.15, 5.55, 3.65, 0.42, WHITE, GRAY, lw=1.2, radius=0.25)
box(6.19, 5.60, 3.30, 0.32, IN1, None, radius=0.25)
label(6.15, 6.06, 3.7, 0.35, "budget spent only inside the ledger",
      size=10.5, color=IN1, bold=True)

# ---- block 4: student + big certificate seal
chip(11.55, 2.35, 1.30, 1.00, IN1, "Student Agent", size=12)
box(11.00, 3.75, 1.35, 1.35, GREEN, WHITE, lw=2.2, shape=MSO_SHAPE.OVAL,
    radius=None)
label(11.00, 4.10, 1.35, 0.4, "CERTIFIED", size=9.5, color=WHITE, bold=True)
label(11.00, 4.42, 1.35, 0.35, "✓", size=14, color=WHITE, bold=True)
label(10.45, 5.30, 2.7, 0.65, "in-task ≥ 0.93\noff-task ≤ 0.25", size=12.5,
      color=TEXT, bold=True)
label(10.42, 6.12, 2.75, 0.5, "conformal gate routes live requests",
      size=9.5, color=MUTED)

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
