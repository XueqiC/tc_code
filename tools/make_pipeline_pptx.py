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


# v10: the real SkillOpt Fig.2 register — dense engineering flowchart.
# Color families: green=data, purple=compute, blue=artifacts/documents,
# yellow=decision gates, red=reject paths, teal=deployment container.

GRN_F, GRN_E = RGBColor(0xE3, 0xEF, 0xDA), RGBColor(0x77, 0x9E, 0x62)
YEL_F, YEL_E = RGBColor(0xFF, 0xF0, 0xC2), RGBColor(0xC9, 0xA2, 0x2A)
PUR_F, PUR_E = RGBColor(0xE6, 0xDC, 0xEF), RGBColor(0x8E, 0x6B, 0xAE)
BLU_F, BLU_E = RGBColor(0xDD, 0xE9, 0xF7), RGBColor(0x4C, 0x72, 0xB0)
RED_F, RED_E = RGBColor(0xF9, 0xD6, 0xD2), RGBColor(0xC0, 0x50, 0x4D)
TEAL = RGBColor(0x2E, 0x9E, 0x97)
GRY_F = RGBColor(0xED, 0xEC, 0xE7)

CODE_FONT = "Consolas"
SKILL_COLOR = {"JOIN": IN1, "GROUP BY": IN3, "FILTER": IN2, "PLOT": OUT1}


def mbox(x, y, w, h, fill, edge, title, sub=None, ts=8.5, ss=6.5):
    """SkillOpt-style module box: bold title + small sub-label strip."""
    box(x, y, w, h, fill, edge, lw=1.3, radius=0.12)
    if sub:
        label(x + 0.03, y + 0.05, w - 0.06, h * 0.55, title, size=ts,
              color=TEXT, bold=True)
        box(x + 0.07, y + h - 0.24, w - 0.14, 0.18,
            WHITE, edge, lw=0.75, radius=0.3, text=sub, size=ss,
            tcolor=MUTED)
    else:
        label(x + 0.03, y + (h - 0.3) / 2, w - 0.06, 0.34, title, size=ts,
              color=TEXT, bold=True)


def doc(x, y, w, h, title, edge=BLU_E, fill=BLU_F, ts=8, stack=False):
    if stack:
        for i in (2, 1):
            box(x + i * 0.05, y + i * 0.05, w, h, fill, edge, lw=1.0,
                shape=MSO_SHAPE.FOLDED_CORNER, radius=None)
    s = box(x, y, w, h, fill, edge, lw=1.2,
            shape=MSO_SHAPE.FOLDED_CORNER, radius=None)
    label(x + 0.02, y + (h - 0.55) / 2, w - 0.04, 0.6, title, size=ts,
          color=TEXT, bold=True)
    return s


def skill(x, y, name, color, w=0.62, h=0.24, fs=7):
    return box(x, y, w, h, color, WHITE, lw=0.7, radius=0.45, text=name,
               size=fs, tcolor=WHITE, bold=True)


# ================================================================ inputs
box(0.18, 0.40, 2.42, 2.02, GRY_F, GRAY, lw=1.2, radius=0.06)
label(0.30, 0.48, 2.1, 0.28, "TASK EVIDENCE", size=9.5, color=TEXT,
      bold=True, align=PP_ALIGN.LEFT)
box(0.32, 0.80, 2.14, 0.48, GRN_F, GRN_E, lw=1.2, radius=0.15,
    text="k example queries", size=8.5, tcolor=TEXT, bold=True)
box(0.32, 1.34, 2.14, 0.48, YEL_F, YEL_E, lw=1.2, radius=0.15,
    text="calibration split: gate τ", size=8.5, tcolor=TEXT, bold=True)
box(0.32, 1.88, 2.14, 0.44, GRY_F, GRAY, lw=1.2, radius=0.15,
    text="held-out eval 🔒 locked", size=8, tcolor=MUTED, bold=True)

chip(0.85, 3.35, 0.85, 0.64, OUT1, "Frozen Teacher ❄", size=8.5)
conn(1.35, 3.72, 1.72, 3.72, GRAY, w=1.8)
doc(1.80, 3.30, 1.05, 0.85, "trace pool\n1080 exec-\nverified", ts=7,
    fill=CARD, edge=GRAY, stack=True)

# ================================================================ top row: boundary estimation
conn(2.62, 1.04, 2.95, 1.04, GRAY, w=1.8)
mbox(2.98, 0.68, 1.42, 0.78, PUR_F, PUR_E, "backward pass\n@ student init",
     sub="LoRA-B sketch + JL", ts=8)
conn(4.42, 1.04, 4.72, 1.04, GRAY, w=1.8)
doc(4.75, 0.66, 0.98, 0.80, "gradient\nfingerprints", ts=7.5)
conn(5.75, 1.04, 6.05, 1.04, GRAY, w=1.8)
mbox(6.08, 0.68, 1.30, 0.78, PUR_F, PUR_E, "sparse\ndictionary",
     sub="128 atoms · lasso", ts=8)
conn(7.40, 1.04, 7.70, 1.04, GRAY, w=1.8)
box(7.73, 0.66, 2.06, 0.80, WHITE, GRAY, lw=1.2, radius=0.10)
skill(7.83, 0.76, "JOIN", IN1)
skill(8.50, 0.76, "GROUP BY", IN3, w=0.86)
skill(7.83, 1.10, "FILTER", IN2, w=0.72)
skill(8.60, 1.10, "PLOT", OUT1, w=0.58)
label(9.24, 1.06, 0.5, 0.3, "…", size=10, color=MUTED)
label(7.75, 0.38, 2.0, 0.26, "named atom library", size=8, color=MUTED)
conn(9.81, 1.04, 10.11, 1.04, GRAY, w=1.8)
doc(10.14, 0.60, 1.30, 0.92, "TASK SPEC\natom support\n+ threshold τ",
    ts=7.5)
label(11.50, 0.60, 1.8, 0.9,
      "the specification:\nwhat the task needs,\nread before any\ntraining",
      size=8, color=MUTED, align=PP_ALIGN.LEFT)

# spec influence (blue arrows down)
conn(10.79, 1.55, 10.79, 3.14, BLU_E, w=1.8)
conn(10.20, 1.30, 4.85, 3.05, BLU_E, w=1.8)
label(6.3, 2.28, 1.6, 0.26, "gates the diet", size=8, color=BLU_E,
      bold=True, italic=True)

# ================================================================ mid row: selective distillation
for i in range(3):
    y = 2.98 + i * 0.62
    conn(2.88, 3.72, 3.28, y + 0.24, GRAY, w=1.4)
    mbox(3.32, y, 1.50, 0.52, PUR_F, PUR_E, f"score trace",
         sub="in-spec − λ·out", ts=7.5)
label(3.95, 4.86, 0.5, 0.3, "⋮", size=12, color=MUTED)
for i in range(3):
    y = 2.98 + i * 0.62
    conn(4.84, y + 0.26, 5.22, 3.55, GRAY, w=1.4)
mbox(5.26, 3.30, 1.42, 0.78, PUR_F, PUR_E, "rank · fill\ntoken budget",
     sub="greedy prefix", ts=8)
conn(6.70, 3.68, 7.00, 3.68, GRAY, w=1.8)
mbox(7.03, 3.30, 1.46, 0.78, PUR_F, PUR_E, "SFT student",
     sub="stop when absorbed", ts=8.5)
conn(8.51, 3.68, 8.81, 3.68, GRAY, w=1.8)
doc(8.84, 3.32, 0.96, 0.74, "candidate\nstudent S₁", ts=7.5)
conn(9.82, 3.68, 10.10, 3.68, GRAY, w=1.8)
# validation diamond
box(10.12, 3.18, 1.34, 1.0, YEL_F, YEL_E, lw=1.4, shape=MSO_SHAPE.DIAMOND,
    radius=None)
label(10.16, 3.42, 1.26, 0.55, "certify\ntwo-sided?", size=7.5, color=TEXT,
      bold=True)
label(11.42, 3.40, 0.5, 0.26, "accept", size=7, color=GRN_E, bold=True)
conn(11.48, 3.68, 11.80, 3.68, GRN_E, w=2.0)
doc(11.83, 3.24, 1.30, 0.88, "deployed\nstudent +\ncertificates", ts=7.5)
label(10.20, 4.24, 0.62, 0.26, "reject", size=7.5, color=RED_E, bold=True)
# reject loop (dashed red back to lambda)
rj = conn(10.55, 4.20, 9.10, 4.78, RED_E, w=1.6)
box(7.55, 4.62, 1.55, 0.44, RED_F, RED_E, lw=1.2, radius=0.15,
    text="tighten λ · re-select", size=8, tcolor=RED_E, bold=True)
rj2 = conn(7.52, 4.82, 4.10, 4.30, RED_E, w=1.6)
for c in (rj, rj2):
    ln = c.line._get_or_add_ln()
    ln.append(ln.makeelement(qn("a:prstDash"), {"val": "dash"}))

# threshold provenance stated locally (no cross-figure line)
label(4.30, 6.72, 1.35, 0.28, "τ from calibration", size=7, color=YEL_E,
      bold=True)

# ================================================================ deployment container (teal dashed)
dep = box(0.30, 5.30, 12.88, 1.72, RGBColor(0xEC, 0xF6, 0xF5), TEAL,
          lw=1.6, radius=0.05)
ln = dep.line._get_or_add_ln()
ln.append(ln.makeelement(qn("a:prstDash"), {"val": "dash"}))
label(0.48, 5.40, 4.0, 0.3, "DEPLOYMENT-TIME ROUTING", size=9.5,
      color=TEAL, bold=True, align=PP_ALIGN.LEFT)
box(0.55, 5.85, 1.30, 0.55, BLU_F, BLU_E, lw=1.1)
label(0.62, 5.90, 1.2, 0.5, '"Avg dues\nper club?"', size=7, color=IN1,
      align=PP_ALIGN.LEFT)
box(0.55, 6.48, 1.30, 0.48, RGBColor(0xFD, 0xF0, 0xE7), OUT1, lw=1.1)
label(0.62, 6.52, 1.2, 0.42, '"Plot a chart"', size=7, color=OUT1,
      align=PP_ALIGN.LEFT)
conn(1.88, 6.12, 2.30, 6.12, GRAY, w=1.6)
conn(1.88, 6.70, 2.30, 6.70, GRAY, w=1.6)
mbox(2.34, 5.95, 1.60, 0.68, PUR_F, PUR_E, "conformal score",
     sub="one backward pass", ts=8)
conn(3.96, 6.28, 4.35, 6.28, GRAY, w=1.6)
box(4.38, 5.92, 1.15, 0.78, YEL_F, YEL_E, lw=1.3, shape=MSO_SHAPE.DIAMOND,
    radius=None)
label(4.40, 6.16, 1.1, 0.4, "≤ τ ?", size=8.5, color=TEXT, bold=True)
label(5.60, 5.86, 0.5, 0.26, "yes", size=7.5, color=GRN_E, bold=True)
conn(5.55, 6.12, 5.95, 6.12, GRN_E, w=2.0)
box(5.98, 5.85, 1.55, 0.55, GRN_F, GRN_E, lw=1.3, radius=0.15,
    text="→ student agent", size=8.5, tcolor=TEXT, bold=True)
label(5.48, 6.74, 0.4, 0.26, "no", size=7.5, color=RED_E, bold=True)
conn(5.55, 6.55, 5.95, 6.68, RED_E, w=2.0)
box(5.98, 6.46, 1.85, 0.5, RED_F, RED_E, lw=1.3, radius=0.15,
    text="refuse · escalate to teacher", size=7.5, tcolor=RED_E, bold=True)
# certificates on the right of the container
box(8.30, 5.85, 2.20, 0.52, WHITE, GREEN, lw=1.4, radius=0.15,
    text="coverage LB 0.93 (in-task)", size=8, tcolor=TEXT, bold=True)
box(8.30, 6.44, 2.20, 0.52, WHITE, GREEN, lw=1.4, radius=0.15,
    text="leakage UB 0.25 (off-task)", size=8, tcolor=TEXT, bold=True)
label(10.60, 5.90, 2.5, 1.0,
      "Clopper-Pearson bounds from\nbehavioral evaluation only —\nno benchmark-specific tuning",
      size=8, color=MUTED, align=PP_ALIGN.LEFT)
conn(12.35, 4.14, 12.42, 5.28, GRAY, w=1.6)

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
