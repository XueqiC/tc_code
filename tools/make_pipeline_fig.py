"""Pipeline figure v4 (fig1_mechanism.png).

Four numbered stages, left to right, every inset from logged runs:
  1 Specify   - k example queries (three domains, showing generality) ->
                gradient fingerprints (real rows of features_grad.npz)
  2 Factor    - real sparse codes from a dictionary refit on the pilot
                gradients; support columns = the specification; the JOIN
                atom annotated with its real top-activating snippet
  3 Select    - real per-trace scores against the SQL subspace, budget
                cut; real absorption curve (m17) for the stopping rule
  4 Certify   - real conformal calibration histogram with the quantile
                threshold; router to student vs refuse; certificate numbers

House style: Times, no in-figure titles, stage chips + claim labels only.
"""

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
    }
)

BLUE = "#4C72B0"   # in-specification
ORANGE = "#DD8452" # out-of-scope
GREEN = "#55A868"  # guarantee
GRAY = "#8C8C8C"
TEXT = "#1f1f1e"
MUTED = "#6b6a63"
RNG = np.random.default_rng(0)

# ---------------------------------------------------------------- real data
d = np.load("results/pilot/features_grad.npz", allow_pickle=True)
X, dom = d["X"], d["domain"]
sql = X[dom == "sql"]
insup_rows = X[np.isin(dom, ["sql"])]

# subspace of the specification (same recipe as boundary.fit_subspace)
mu = insup_rows.mean(0)
U, S, _ = np.linalg.svd(insup_rows - mu, full_matrices=False)
k90 = int(np.searchsorted(np.cumsum(S**2) / (S**2).sum(), 0.90)) + 1
P = U[: len(S)].T[:, :k90] if False else None
V = np.linalg.svd(insup_rows - mu, full_matrices=False)[2][:k90]

def in_energy(rows):
    c = rows - mu
    proj = c @ V.T
    return (proj**2).sum(1) / np.maximum((c**2).sum(1), 1e-12)

# dictionary refit for a real codes heatmap (small, deterministic)
from sklearn.decomposition import MiniBatchDictionaryLearning

dict_fit = MiniBatchDictionaryLearning(
    n_components=48, alpha=0.05, transform_algorithm="lasso_lars",
    transform_alpha=0.05, random_state=0, max_iter=200, batch_size=32,
)
sub = np.concatenate([X[dom == "sql"][:60], X[dom == "pandas"][:30],
                      X[dom == "gsm8k-code"][:30]])
codes_all = dict_fit.fit(sub).transform(
    np.concatenate([X[dom == "sql"][:12], X[dom == "pandas"][:6],
                    X[dom == "gsm8k-code"][:6]]))
mass_in = np.abs(codes_all[:12]).mean(0)
mass_out = np.abs(codes_all[12:]).mean(0)
cols_in = np.argsort(-mass_in)[:14]
cols_out = [c for c in np.argsort(-mass_out)[:16] if c not in cols_in][:12]
order = np.concatenate([cols_in, cols_out])
heat = np.abs(codes_all[:, order])
support_cols = np.arange(len(cols_in))

# real trace scores for the selection panel
pool_rows = np.concatenate([X[dom == "sql"][60:80], X[dom == "pandas"][30:44],
                            X[dom == "gsm8k-cot"][:14], X[dom == "alpaca"][:12]])
scores = in_energy(pool_rows) - 0.5 * (1 - in_energy(pool_rows)) * 0.3
rank = np.argsort(-scores)
scores_sorted = scores[rank]
n_sel = 22

# real absorption curve
m17 = json.load(open("results/m17/records.json"))
steps = [r["step"] for r in m17 if r["lr"] == 0.0002]
demand = [r["in_sub_energy"] for r in m17 if r["lr"] == 0.0002]

# real conformal calibration scores (nonconformity = 1 - in-subspace energy)
cal = 1 - in_energy(X[dom == "sql"][80:120])
off = 1 - in_energy(np.concatenate([X[dom == "alpaca"][12:52],
                                    X[dom == "gsm8k-cot"][14:54]]))
tau = np.quantile(cal, 0.9)

# real snippets
rep = json.load(open("results/atoms_pilot/report.json"))
join_snip = rep["interp"]["atom_71"][0][:40] + "…"

# ---------------------------------------------------------------- layout
fig = plt.figure(figsize=(14.2, 4.9), facecolor="white")
gs = fig.add_gridspec(
    2, 4, left=0.035, right=0.985, top=0.80, bottom=0.10,
    wspace=0.42, hspace=0.55, height_ratios=[1, 1],
)

def chip(fig, x, label, claim):
    fig.text(x, 0.955, label, fontsize=15, weight="bold", color="white",
             ha="center", va="center", family="serif",
             bbox=dict(boxstyle="circle,pad=0.32", fc=TEXT, ec="none"))
    fig.text(x + 0.016, 0.955, claim, fontsize=13.5, weight="bold",
             color=TEXT, ha="left", va="center")

chip(fig, 0.045, "1", "Specify from k queries")
chip(fig, 0.295, "2", "Factor into atoms")
chip(fig, 0.545, "3", "Select traces, distill")
chip(fig, 0.775, "4", "Certify and deploy")

for x in (0.272, 0.522, 0.762):
    fig.patches.append(FancyArrowPatch(
        (x - 0.008, 0.5), (x + 0.012, 0.5), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=26, lw=2.2, color=GRAY))

# ---- stage 1: query cards + fingerprint strips
ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
cards = [
    ("SQL",   "How many members does\neach club have? (JOIN…)"),
    ("Data",  "df.groupby('city')\n  .agg({'sales':'sum'})"),
    ("Math",  "def solve():  # 3 workers,\n  5 days, rate = …"),
]
for i, (tag, txt) in enumerate(cards):
    y = 0.97 - i * 0.33
    ax.add_patch(FancyBboxPatch((0.02, y - 0.30), 0.9, 0.30,
                 boxstyle="round,pad=0.015", fc="#f7f6f2", ec=GRAY, lw=1.0,
                 transform=ax.transAxes))
    ax.text(0.06, y - 0.065, tag, fontsize=9.5, weight="bold", color=BLUE,
            transform=ax.transAxes, va="center")
    ax.text(0.06, y - 0.21, txt, fontsize=8.2, color=TEXT, family="monospace",
            transform=ax.transAxes, va="center")
ax.text(0.5, 1.13, "Example queries, any code-acting workload", fontsize=10.5,
        color=MUTED, ha="center", transform=ax.transAxes, style="italic")

axf = fig.add_subplot(gs[1, 0])
rows = np.vstack([X[dom == "sql"][0][:64], X[dom == "pandas"][0][:64],
                  X[dom == "gsm8k-code"][0][:64]])
axf.imshow(rows, aspect="auto", cmap="RdBu_r",
           vmin=-np.abs(rows).max(), vmax=np.abs(rows).max())
axf.set_yticks([0, 1, 2], ["SQL", "Data", "Math"], fontsize=10)
axf.set_xticks([])
axf.set_xlabel("Gradient Fingerprint (First 64 Dims)", fontsize=11,
               weight="bold", color=TEXT)
axf.tick_params(length=0)
for s in axf.spines.values(): s.set_color(GRAY)
axf.text(0.5, 1.16, "One backward pass at the student's init:\nwhat must this student learn?",
         fontsize=9.5, color=MUTED, ha="center", transform=axf.transAxes)

# ---- stage 2: real codes heatmap + JOIN annotation
axh = fig.add_subplot(gs[0, 1])
axh.imshow(heat, aspect="auto", cmap="Blues", vmax=np.quantile(heat, 0.98))
for c in support_cols:
    axh.axvspan(c - 0.5, c + 0.5, color=BLUE, alpha=0.14, lw=0)
axh.axhline(11.5, color=ORANGE, lw=1.6)
axh.set_yticks([5.5, 15], ["k in-spec\nqueries", "other"], fontsize=9)
axh.set_xticks([])
axh.set_xlabel("Capability Atoms (Sparse Codes)", fontsize=11, weight="bold",
               color=TEXT)
axh.tick_params(length=0)
for s in axh.spines.values(): s.set_color(GRAY)
axh.text(0.5, 1.06, "Specification = atoms the k queries activate",
         fontsize=9.5, color=MUTED, ha="center", transform=axh.transAxes)

axj = fig.add_subplot(gs[1, 1]); axj.axis("off")
axj.add_patch(FancyBboxPatch((0.01, 0.42), 0.96, 0.44,
              boxstyle="round,pad=0.02", fc="#eef2f9", ec=BLUE, lw=1.2,
              transform=axj.transAxes))
axj.text(0.05, 0.74, "One atom, read back on data:", fontsize=9.5,
         color=TEXT, transform=axj.transAxes, weight="bold")
axj.text(0.05, 0.55, join_snip, fontsize=7.6, family="monospace", color=TEXT,
         transform=axj.transAxes)
axj.text(0.5, 0.18, "its top activations are exactly JOIN clauses\n(atom support resolves skills: AUROC 0.906)",
         fontsize=9.5, color=MUTED, ha="center", transform=axj.transAxes)

# ---- stage 3: score ranking + absorption stop
axs = fig.add_subplot(gs[0, 2])
colors = [BLUE if i < n_sel else "#c9c7c0" for i in range(len(scores_sorted))]
axs.bar(np.arange(len(scores_sorted)), scores_sorted, color=colors, width=0.86)
axs.axvline(n_sel - 0.5, color=TEXT, lw=1.6, ls="--")
axs.text(n_sel + 0.8, scores_sorted[0] * 0.92, "budget\ncut", fontsize=9.5,
         color=TEXT, weight="bold")
axs.set_xticks([]); axs.set_yticks([])
axs.set_xlabel("Teacher Traces, Ranked by Score", fontsize=11, weight="bold",
               color=TEXT)
for sp in ["top", "right"]: axs.spines[sp].set_visible(False)
for sp in ["left", "bottom"]: axs.spines[sp].set_color(GRAY)
axs.text(0.5, 1.06,
         "score = supply to specified atoms $-$ $\\lambda\\,\\cdot$ out-of-scope supply",
         fontsize=9.5, color=MUTED, ha="center", transform=axs.transAxes)

axm = fig.add_subplot(gs[1, 2])
axm.plot(steps, demand, color=BLUE, lw=2.4, marker="o", ms=5)
stop_i = next(i for i, v in enumerate(demand) if v < 0.05 * demand[0])
axm.axvline(steps[stop_i], color=GREEN, lw=1.8, ls="--")
axm.annotate("stop when\ndemand absorbed", xy=(steps[stop_i], demand[stop_i]),
             xytext=(steps[stop_i] + 14, demand[0] * 0.55), fontsize=9.5,
             color=TEXT, weight="bold",
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.4))
axm.set_xlabel("Training Step", fontsize=11, weight="bold", color=TEXT)
axm.set_ylabel("In-Spec Demand", fontsize=10.5, weight="bold", color=TEXT)
axm.tick_params(labelsize=9, colors=MUTED)
for sp in ["top", "right"]: axm.spines[sp].set_visible(False)

# ---- stage 4: conformal histogram + router
axc = fig.add_subplot(gs[0, 3])
bins = np.linspace(min(cal.min(), off.min()), 1.0, 22)
axc.hist(cal, bins=bins, color=BLUE, alpha=0.75, label="calibration (in-spec)")
axc.hist(off, bins=bins, color=ORANGE, alpha=0.55, label="off-spec")
axc.axvline(tau, color=TEXT, lw=1.8, ls="--")
axc.text(tau + 0.03, axc.get_ylim()[1] * 0.58,
         "$\\tau$ = 90% quantile\n(guaranteed, not tuned)",
         fontsize=9, color=TEXT, weight="bold", va="top")
axc.set_xlabel("Nonconformity Score", fontsize=11, weight="bold", color=TEXT)
axc.set_yticks([])
axc.tick_params(labelsize=9, colors=MUTED)
for sp in ["top", "right"]: axc.spines[sp].set_visible(False)
axc.legend(frameon=False, fontsize=8.2, loc="upper left",
           bbox_to_anchor=(0.0, 1.02), labelcolor=TEXT, handlelength=1.2)

axr = fig.add_subplot(gs[1, 3]); axr.axis("off")
axr.add_patch(FancyBboxPatch((0.01, 0.40), 0.26, 0.30, boxstyle="round,pad=0.02",
              fc="#f7f6f2", ec=GRAY, lw=1.0, transform=axr.transAxes))
axr.text(0.14, 0.55, "incoming\nquery", fontsize=9, ha="center", va="center",
         color=TEXT, transform=axr.transAxes)
axr.add_patch(FancyBboxPatch((0.50, 0.62), 0.48, 0.34, boxstyle="round,pad=0.02",
              fc="#eef2f9", ec=BLUE, lw=1.4, transform=axr.transAxes))
axr.text(0.74, 0.79, "student agent\nin-task 1.000, LB 0.926", fontsize=8.4,
         ha="center", va="center", color=BLUE, weight="bold",
         transform=axr.transAxes)
axr.add_patch(FancyBboxPatch((0.50, 0.06), 0.48, 0.34, boxstyle="round,pad=0.02",
              fc="#fdf0e7", ec=ORANGE, lw=1.4, transform=axr.transAxes))
axr.text(0.74, 0.23, "refuse / escalate\noff-task residue UB 0.249",
         fontsize=8.4, ha="center", va="center", color=ORANGE, weight="bold",
         transform=axr.transAxes)
axr.annotate("", xy=(0.49, 0.79), xytext=(0.29, 0.60),
             arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.6),
             xycoords=axr.transAxes)
axr.annotate("", xy=(0.49, 0.23), xytext=(0.29, 0.50),
             arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.6),
             xycoords=axr.transAxes)
axr.text(0.36, 0.92, "score $\\leq\\tau$", fontsize=8.4, color=BLUE,
         weight="bold", ha="center", transform=axr.transAxes)
axr.text(0.36, 0.33, "score $>\\tau$", fontsize=8.4, color=ORANGE,
         weight="bold", ha="center", transform=axr.transAxes)

fig.savefig("paper/figs/fig1_mechanism.png", dpi=200, bbox_inches="tight")
print("saved paper/figs/fig1_mechanism.png")
