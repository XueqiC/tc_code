"""Pipeline figure v5: schematic, vivid, LLM-paper register (per PI).

No raw data insets; skills are colored tiles, models are chips with faces,
traces are cards. The only numbers are certified ones, in badges.
Semantics: blue family = in-specification skills, orange/red = out-of-scope,
green = guarantee. Output: paper/figs/fig1_mechanism.png.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "mathtext.fontset": "stix"})

# palette
IN1, IN2, IN3 = "#4C72B0", "#5F9EA0", "#7BA7D7"     # in-spec skill tiles
OUT1, OUT2 = "#DD8452", "#C44E52"                    # out-of-scope tiles
PALE, CARD, GRAY = "#e8e6e1", "#f7f6f2", "#8C8C8C"
GREEN, TEXT, MUTED = "#55A868", "#1f1f1e", "#6b6a63"

fig, ax = plt.subplots(figsize=(15.2, 5.1), facecolor="white")
ax.set_xlim(0, 152); ax.set_ylim(0, 51); ax.axis("off")


def rbox(x, y, w, h, fc, ec, lw=1.2, style="round,pad=0.6"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    return p


def tile(x, y, c, s=2.0):
    ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle="round,pad=0.18",
                                fc=c, ec="white", lw=0.8))


def chip(cx, cy, w, h, fc, ec, label, sub=None, fs=10.5):
    """A model chip with antenna + two eyes: friendly but restrained."""
    rbox(cx - w / 2, cy - h / 2, w, h, fc, ec, lw=1.6)
    ax.plot([cx, cx], [cy + h / 2 + 0.4, cy + h / 2 + 1.6], color=ec, lw=1.6)
    ax.add_patch(Circle((cx, cy + h / 2 + 2.2), 0.7, fc=ec, ec="none"))
    for dx in (-w * 0.18, w * 0.18):
        ax.add_patch(Circle((cx + dx, cy + h * 0.16), 0.55, fc=ec, ec="none"))
    ax.plot([cx - w * 0.14, cx + w * 0.14], [cy - h * 0.10, cy - h * 0.10],
            color=ec, lw=1.6, solid_capstyle="round")
    ax.text(cx, cy - h / 2 - 1.6, label, ha="center", va="top", fontsize=fs,
            weight="bold", color=TEXT)
    if sub:
        ax.text(cx, cy - h / 2 - 4.0, sub, ha="center", va="top",
                fontsize=8.5, color=MUTED)


def arrow(x0, y0, x1, y1, color=GRAY, lw=2.4, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=22, lw=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))


def stage(x, n, claim):
    ax.text(x, 49.2, n, fontsize=13, weight="bold", color="white",
            ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.35", fc=TEXT, ec="none"))
    ax.text(x + 2.4, 49.2, claim, fontsize=13.5, weight="bold", color=TEXT,
            ha="left", va="center")


stage(4, "1", "What must a student learn?")
stage(44, "2", "A library of skill atoms")
stage(82, "3", "Distill exactly to spec")
stage(119, "4", "Certified deployment")

for x in (37.5, 76, 113):
    arrow(x - 1.5, 26, x + 3.5, 26, lw=3.0)

# ------------------------------------------------ stage 1: teacher + queries
chip(8, 36, 11, 8, "#eef2f9", IN1, "Teacher LLM", fs=10)
# fanned query cards
for i, (dx, dy) in enumerate([(0.0, 0.0), (1.2, -1.2), (2.4, -2.4)]):
    rbox(17 + dx, 32 + dy, 13, 8, CARD, GRAY, lw=1.0)
ax.text(25.5, 37.2, "×k", fontsize=11, weight="bold", color=MUTED)
ax.text(20 + 2.4, 32.6, '"How many members\n  per club?"', fontsize=7.8,
        family="monospace", color=TEXT, va="bottom")
ax.text(19.5, 42.0, "k example queries", fontsize=9.5, color=MUTED,
        style="italic", ha="center")

arrow(24, 29.2, 24, 24.0, color=IN1, rad=0.0)
ax.text(25.4, 22.5, "one backward pass each:\nthe gradient names\nthe missing skills",
        fontsize=8.8, color=MUTED, va="top")

# fingerprints: card icon -> row of skill tiles
for row, cols in enumerate([[IN1, IN2, PALE, PALE], [IN1, PALE, IN3, PALE],
                            [PALE, IN2, IN3, PALE]]):
    y = 14.5 - row * 4.4
    rbox(13.5, y - 0.4, 3.6, 3.2, CARD, GRAY, lw=0.9)
    arrow(18, y + 1.2, 20.5, y + 1.2, lw=1.4)
    for j, c in enumerate(cols):
        tile(21.5 + j * 3.0, y, c, s=2.4)
ax.text(21.5, 1.0, "needed skills light up (blue)", fontsize=9.0,
        color=IN1, weight="bold")

# ------------------------------------------------ stage 2: atom library
lib_x, lib_y = 42, 12.5
rbox(lib_x - 1.6, lib_y - 1.8, 26, 24, "white", GRAY, lw=1.2)
atoms = [IN1, PALE, OUT1, PALE, IN2, PALE,
         PALE, IN3, PALE, OUT2, PALE, PALE,
         OUT1, PALE, IN1, PALE, PALE, IN2]
for i, c in enumerate(atoms):
    r, ccol = divmod(i, 6)
    tile(lib_x + ccol * 4.0, lib_y + 14.6 - r * 7.0, c, s=2.9)
ax.text(lib_x + 11.4, 39.0, "skill atoms discovered from many traces\n(sparse dictionary over gradients)",
        fontsize=9.0, color=MUTED, ha="center")
# spec outline around the blue atoms (dashed rounded box per blue tile)
for i, c in enumerate(atoms):
    if c in (IN1, IN2, IN3):
        r, ccol = divmod(i, 6)
        ax.add_patch(FancyBboxPatch((lib_x + ccol * 4.0 - 0.5,
                                     lib_y + 14.6 - r * 7.0 - 0.5), 3.9, 3.9,
                                    boxstyle="round,pad=0.15", fc="none",
                                    ec=IN1, lw=1.5, ls=(0, (3, 2))))
# JOIN tag on one atom
ax.annotate("this one fires on\nJOIN clauses", xy=(lib_x + 0.9, lib_y + 15.6),
            xytext=(lib_x - 1.2, lib_y + 21.5), fontsize=8.6, color=TEXT,
            weight="bold",
            arrowprops=dict(arrowstyle="->", color=TEXT, lw=1.1))
rbox(lib_x + 3.2, lib_y - 6.6, 18, 4.2, "#eef2f9", IN1, lw=1.4)
ax.text(lib_x + 12.2, lib_y - 4.4, "specification = the atoms\nyour k queries need",
        fontsize=9.2, weight="bold", color=IN1, ha="center", va="center")

# ------------------------------------------------ stage 3: select + train
# trace cards with composition strips
cards3 = [([IN1, IN2, PALE], True), ([IN1, IN3, IN2], True),
          ([OUT1, IN1, PALE], False), ([OUT2, PALE, OUT1], False)]
for i, (comp, keep) in enumerate(cards3):
    y = 38.5 - i * 6.6
    ec = IN1 if keep else GRAY
    rbox(80, y, 11.5, 5.0, CARD if keep else "#efede8", ec, lw=1.3)
    for j, c in enumerate(comp):
        tile(81.2 + j * 3.1, y + 1.2, c, s=2.2)
    if keep:
        arrow(92.5, y + 2.5, 97.5, 27.5, color=IN1, lw=1.8,
              rad=-0.18 if i == 0 else 0.12)
    else:
        ax.plot([80.4, 91.2], [y + 0.6, y + 4.6], color=OUT2, lw=1.8, alpha=0.75)
ax.text(85.5, 45.3, "teacher traces, scored by\nwhich atoms they supply",
        fontsize=9.0, color=MUTED, ha="center")
ax.text(83.5, 9.6, "rejected: they carry orange atoms\n($\\lambda$ penalizes out-of-scope supply)",
        fontsize=8.6, color=OUT2, ha="center", va="top", weight="bold")

chip(102, 24, 9, 7, "#eef2f9", IN1, "Student LLM", sub="small, task-exact", fs=9.5)
# absorption meter
rbox(96.5, 9.5, 12, 3.2, "white", GRAY, lw=1.1)
ax.add_patch(Rectangle((97.0, 10.1), 10.2, 2.0, fc=IN1, ec="none"))
ax.add_patch(Circle((110.4, 11.1), 1.5, fc=GREEN, ec="white", lw=1.2))
ax.plot([109.7, 110.2, 111.2], [11.1, 10.4, 11.9], color="white", lw=1.6,
        solid_capstyle="round")
ax.text(104.0, 7.2, "train only until the demand\nis absorbed, then stop",
        fontsize=8.8, color=MUTED, ha="center", va="top")

# ------------------------------------------------ stage 4: gate + routing
gx = 121.5
# gate posts + conformal seal
for dx in (0, 7.5):
    rbox(gx + dx, 17, 2.2, 17, "#dfe5ef", IN1, lw=1.2)
rbox(gx - 0.6, 33.5, 10.8, 3.0, "#dfe5ef", IN1, lw=1.2)
ax.add_patch(Circle((gx + 4.8, 35.0), 2.6, fc=GREEN, ec="white", lw=1.6))
ax.text(gx + 4.8, 35.0, "90%", fontsize=8.5, color="white", ha="center",
        va="center", weight="bold")
ax.text(gx + 1.5, 38.6, "conformal gate:\ncoverage guaranteed by\ncalibration, not by tuning",
        fontsize=8.2, color=MUTED, ha="center", va="bottom")

# incoming query cards
rbox(113.8, 23.5, 5.2, 4.0, "#eef2f9", IN1, lw=1.2)
rbox(113.8, 17.0, 5.2, 4.0, "#fdf0e7", OUT1, lw=1.2)
arrow(119.6, 25.5, 131.5, 25.5, color=IN1, lw=1.8)
arrow(119.6, 18.6, 133.0, 11.2, color=OUT1, lw=1.8, rad=0.28)

chip(137.5, 25.5, 8.5, 6.5, "#eef2f9", IN1, "Student", fs=9.5)
rbox(133.5, 7.5, 13.5, 4.6, "#fdf0e7", OUT1, lw=1.3)
ax.text(140.2, 9.8, "refuse / escalate", fontsize=9.0, weight="bold",
        color=OUT1, ha="center", va="center")

# certificate ribbon
rbox(132.5, 38.5, 17.0, 6.4, "white", GREEN, lw=1.5)
ax.text(141.0, 43.0, "Certified, both sides:", fontsize=9.0, weight="bold",
        color=GREEN, ha="center")
ax.text(141.0, 40.4, "in-task $\\geq$ 0.93,  off-task $\\leq$ 0.25",
        fontsize=8.4, color=TEXT, ha="center")

fig.savefig("paper/figs/fig1_mechanism.png", dpi=200, bbox_inches="tight")
print("saved v5")
