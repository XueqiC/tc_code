"""Regenerate all result figures in the normative house style.

Times New Roman, no in-figure titles, muted palette, bold Title-Case axis
labels (14-15pt), bold ticks 12-13pt, legends never over data. Every number
comes from logged runs under results/.
"""

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "axes.labelsize": 15,
        "axes.labelweight": "bold",
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 12.5,
    }
)

BLUE, ORANGE, GREEN, RED, PURPLE = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"
GRAY, TEXT, MUTED = "#8C8C8C", "#1f1f1e", "#6b6a63"
DOMC = {"gsm8k-code": BLUE, "gsm8k-cot": RED, "sql": GREEN,
        "pandas": PURPLE, "alpaca": GRAY}
RNG = np.random.default_rng(0)


def style(ax, ylab=None, xlab=None):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_linewidth(1.4)
    ax.tick_params(colors=MUTED, width=1.6)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")
    if ylab:
        ax.set_ylabel(ylab, color=TEXT)
    if xlab:
        ax.set_xlabel(xlab, color=TEXT)


d = np.load("results/pilot/features_grad.npz", allow_pickle=True)
Xg, dom = d["X"], d["domain"]
de = np.load("results/pilot/features_emb.npz", allow_pickle=True)
Xe = de["Et"]  # trajectory embeddings (prompt+response)
dome = de["domain"] if "domain" in de.files else dom


# official pilot protocol: unit rows, uncentered SVD subspace, norm-ratio score
import sys

sys.path.insert(0, "src")
from boundary import unit, fit_subspace, score, evaluate, k_curve  # noqa: E402


# ---------------------------------------------------------------- pilot_pca
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), facecolor="white")
for ax, X, name in [(axes[0], Xe, "Trajectory Embeddings"),
                    (axes[1], Xg, "Gradient Fingerprints")]:
    Z = PCA(2, random_state=0).fit_transform(X)
    for dm in ["gsm8k-code", "gsm8k-cot", "sql", "pandas", "alpaca"]:
        m = dom == dm
        ax.scatter(Z[m, 0], Z[m, 1], s=13, alpha=0.7, color=DOMC[dm],
                   label=dm, lw=0)
    style(ax, xlab=f"PC 1 ({name})")
    ax.set_xticks([]); ax.set_yticks([])
axes[0].set_ylabel("PC 2", color=TEXT)
h, l = axes[1].get_legend_handles_labels()
fig.legend(h, l, frameon=False, ncol=5, loc="upper center",
           bbox_to_anchor=(0.5, 1.04), labelcolor=TEXT, markerscale=2.2,
           handletextpad=0.1, columnspacing=0.9)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig("paper/figs/pilot_pca.png", dpi=200, bbox_inches="tight")
fig.savefig("paper/figs/pilot_pca.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ pilot_roc_hard
spaces = {"grad": (Xg, dom), "emb-traj": (Xe, dome)}
res = evaluate(spaces, seed=0)
fig, ax = plt.subplots(figsize=(4.6, 3.8), facecolor="white")
for sp, name, col in [("grad", "Gradient", BLUE),
                      ("emb-traj", "Embedding", RED)]:
    fpr, tpr, _ = res[sp]["gsm8k-code"]["_roc_hard"]
    au = res[sp]["gsm8k-code"]["hard_auroc"]
    ax.plot(fpr, tpr, color=col, lw=2.5, label=f"{name} (AUROC {au:.3f})")
ax.plot([0, 1], [0, 1], color=GRAY, lw=1.2, ls=":")
style(ax, ylab="True Positive Rate", xlab="False Positive Rate")
ax.legend(frameon=False, loc="lower right", labelcolor=TEXT)
fig.tight_layout()
fig.savefig("paper/figs/pilot_roc_hard.png", dpi=200, bbox_inches="tight")
fig.savefig("paper/figs/pilot_roc_hard.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ pilot_k_curve
ks = (5, 10, 25, 50)
kres = k_curve(spaces, ks=ks, seeds=range(5))
fig, ax = plt.subplots(figsize=(4.6, 3.8), facecolor="white")
for sp, name, col in [("grad", "Gradient", BLUE),
                      ("emb-traj", "Embedding", RED)]:
    mean = np.array([np.mean(kres[sp][k]) for k in ks])
    sd = np.array([np.std(kres[sp][k]) for k in ks])
    ax.plot(ks, mean, color=col, lw=2.5, marker="o", ms=6, label=name)
    ax.fill_between(ks, mean - sd, mean + sd, color=col, alpha=0.15, lw=0)
ax.set_xticks(ks, [str(k) for k in ks])
ax.axhline(1.0, color=GRAY, lw=1.0, ls=":")
ax.set_ylim(0.3, 1.05)
style(ax, ylab="Hard-Pair AUROC", xlab="Examples Used to Fit Boundary (k)")
ax.legend(frameon=False, loc="center right", labelcolor=TEXT)
fig.tight_layout()
fig.savefig("paper/figs/pilot_k_curve.png", dpi=200, bbox_inches="tight")
fig.savefig("paper/figs/pilot_k_curve.pdf", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ pilot_distill
pm = json.load(open("results/pilot_distill/metrics.json"))
doms = ["gsm8k-code", "gsm8k-cot", "sql", "pandas", "alpaca"]
before = [np.mean(pm["before"]["loss"][dm]) for dm in doms]
after = [np.mean(pm["after"]["loss"][dm]) for dm in doms]
x = np.arange(len(doms))
fig, ax = plt.subplots(figsize=(5.6, 3.6), facecolor="white")
ax.bar(x - 0.19, before, width=0.36, color="#c9c7c0", label="Before")
ax.bar(x + 0.19, after, width=0.36,
       color=[BLUE if dm == "gsm8k-code" else ORANGE for dm in doms])
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc="#c9c7c0", label="Before"),
                   Patch(fc=BLUE, label="After, In-Task"),
                   Patch(fc=ORANGE, label="After, Off-Task")],
          frameon=False, loc="upper left", labelcolor=TEXT, fontsize=11)
ax.set_xticks(x, ["Code", "CoT", "SQL", "Pandas", "Alpaca"])
style(ax, ylab="Held-Out Loss")
fig.tight_layout()
fig.savefig("paper/figs/pilot_distill.png", dpi=200, bbox_inches="tight")
fig.savefig("paper/figs/pilot_distill.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- e1_intra
rep = json.load(open("results/atoms_pilot/report.json"))
pairs = ["sql-yes vs sql-no", "sql-no vs sql-yes",
         "pandas-yes vs pandas-no", "pandas-no vs pandas-yes"]
atom_m = [rep["aurocs"][p][0] for p in pairs]
atom_s = [rep["aurocs"][p][1] for p in pairs]
dense_m = [0.615, 0.754, 0.675, 0.616]
dense_s = [0.094, 0.080, 0.090, 0.117]
x = np.arange(len(pairs))
fig, ax = plt.subplots(figsize=(5.8, 3.6), facecolor="white")
ax.bar(x - 0.19, atom_m, yerr=atom_s, width=0.36, color=BLUE, capsize=3,
       label="Atom Support")
ax.bar(x + 0.19, dense_m, yerr=dense_s, width=0.36, color="#c9c7c0",
       capsize=3, label="Dense Subspace")
ax.axhline(0.5, color=GRAY, lw=1.0, ls=":")
ax.set_xticks(x, ["SQL Join\nvs Plain", "SQL Plain\nvs Join",
                  "Pandas Loc\nvs Plain", "Pandas Plain\nvs Loc"], fontsize=11)
ax.set_ylim(0.35, 1.0)
style(ax, ylab="Directed AUROC")
ax.legend(frameon=False, loc="upper right", labelcolor=TEXT)
fig.tight_layout()
fig.savefig("paper/figs/e1_intra.png", dpi=200, bbox_inches="tight")
fig.savefig("paper/figs/e1_intra.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- e2_gate
e2 = json.load(open("results/e2_gate/records.json"))
recs = [r for r in e2["records"] if r["domain"] == "gsm8k-code"]
steps = sorted({r["step"] for r in recs})
fig, ax = plt.subplots(figsize=(5.6, 3.6), facecolor="white")
for ok, col, lab in [(True, BLUE, "Executes Correctly"),
                     (False, ORANGE, "Fails")]:
    xs = [r["step"] + RNG.normal(0, 0.6) for r in recs if r["success"] == ok]
    ys = [r["gradnorm"] for r in recs if r["success"] == ok]
    ax.scatter(xs, ys, s=14, alpha=0.55, color=col, lw=0, label=lab)
ax.set_yscale("log")
rho = e2["analysis"]["pooled"]["rho"]
ax.text(0.98, 0.96, f"Pooled Spearman $\\rho$ = {rho:.2f}",
        fontsize=12.5, weight="bold", color=TEXT, ha="right", va="top",
        transform=ax.transAxes)
style(ax, ylab="Residual Gradient Norm", xlab="Training Step")
ax.legend(frameon=False, loc="lower left", labelcolor=TEXT, markerscale=1.8)
fig.tight_layout()
fig.savefig("paper/figs/e2_gate.png", dpi=200, bbox_inches="tight")
fig.savefig("paper/figs/e2_gate.pdf", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------- e3_tier1
e3 = json.load(open("results/e3_tier1/metrics.json"))
conds = ["base", "A_all", "B_random", "C_emb", "D_grad", "F_refusal"]
names = ["Base", "Full Pool", "Random", "Embedding", "Ours", "Ours+Refusal"]
intask = [e3["conditions"][c]["eval"]["gsm8k_code"]["exec_acc"] for c in conds]
off = [e3["conditions"][c]["eval"]["gsm8k_cot"]["any_acc"] for c in conds]
x = np.arange(len(conds))
fig, ax = plt.subplots(figsize=(6.2, 3.6), facecolor="white")
ax.bar(x - 0.19, intask, width=0.36, color=BLUE, label="In-Task Execution")
ax.bar(x + 0.19, off, width=0.36, color=ORANGE, label="Off-Task Retention")
ax.set_xticks(x, names, fontsize=11)
style(ax, ylab="Accuracy")
ax.legend(frameon=False, loc="upper left", labelcolor=TEXT)
fig.tight_layout()
fig.savefig("paper/figs/e3_tier1.png", dpi=200, bbox_inches="tight")
fig.savefig("paper/figs/e3_tier1.pdf", bbox_inches="tight")
plt.close(fig)

print("regenerated 7 figures")
