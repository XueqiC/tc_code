"""Boundary estimation pilot: spectral subspace + conformal, vs embedding baselines.

Feature spaces compared under the SAME pipeline:
  grad        : LoRA-B gradient features (from grad_features.py) — query+trajectory
  emb-traj    : BGE embedding of prompt+response                 — query+trajectory
  emb-query   : BGE embedding of prompt only                     — query only

Per domain T: fit subspace U_T on n_fit samples (top-r by 90% spectral energy),
membership score s(q)=||U_T^T x||/||x||, split-conformal threshold on n_cal
samples (alpha=0.1). Report AUROC (in vs out), empirical coverage, false-accept.
Hard pair: gsm8k-code boundary vs gsm8k-cot probes (identical prompts).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "pilot"
FIGS = ROOT / "results" / "figs"
DOMAINS = ["gsm8k-code", "gsm8k-cot", "pandas", "sql", "alpaca"]
COLORS = {"gsm8k-code": "#2a78d6", "gsm8k-cot": "#eb6834", "pandas": "#1baf7a",
          "sql": "#eda100", "alpaca": "#e87ba4"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # per feature space in fixed order
TEXT, MUTED = "#1f1f1e", "#6b6a63"
N_FIT, N_CAL, N_TEST = 35, 15, 30
ALPHA = 0.1
ENERGY = 0.90


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)


def load_features():
    d = np.load(OUT / "features_grad.npz", allow_pickle=True)
    X, dom = d["X"].astype(np.float64), d["domain"].astype(str)
    spaces = {"grad": (X, dom)}

    texts_q, texts_t, dom_e = [], [], []
    for name in DOMAINS:
        for line in open(ROOT / "data" / "pilot" / f"{name}.jsonl"):
            r = json.loads(line)
            texts_q.append(r["prompt"])
            texts_t.append(r["prompt"] + "\n" + r["response"])
            dom_e.append(name)
    cache = OUT / "features_emb.npz"
    if cache.exists():
        e = np.load(cache, allow_pickle=True)
        Eq, Et = e["Eq"], e["Et"]
    else:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")
        Eq = m.encode(texts_q, normalize_embeddings=True, show_progress_bar=False)
        Et = m.encode(texts_t, normalize_embeddings=True, show_progress_bar=False)
        np.savez_compressed(cache, Eq=Eq, Et=Et)
    dom_e = np.array(dom_e)
    spaces["emb-traj"] = (Et.astype(np.float64), dom_e)
    spaces["emb-query"] = (Eq.astype(np.float64), dom_e)
    return spaces


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def fit_subspace(Xf):
    _, S, Vt = np.linalg.svd(Xf, full_matrices=False)
    energy = np.cumsum(S**2) / np.sum(S**2)
    r = int(np.searchsorted(energy, ENERGY) + 1)
    return Vt[:r].T


def score(U, X):
    return np.linalg.norm(X @ U, axis=1) / (np.linalg.norm(X, axis=1) + 1e-12)


def split_domain(idx_by_dom, dom, rng):
    ids = rng.permutation(idx_by_dom[dom])
    return ids[:N_FIT], ids[N_FIT:N_FIT + N_CAL], ids[N_FIT + N_CAL:N_FIT + N_CAL + N_TEST]


def evaluate(spaces, seed=0):
    rng = np.random.default_rng(seed)
    results = {}
    splits_cache = {}
    for sp_name, (X, dom) in spaces.items():
        X = unit(X)
        idx_by_dom = {d: np.where(dom == d)[0] for d in DOMAINS}
        if seed not in splits_cache:
            splits_cache[seed] = {d: split_domain(idx_by_dom, d, np.random.default_rng(seed + hash(d) % 1000))
                                  for d in DOMAINS}
        res = {}
        for T in ["gsm8k-code", "pandas", "sql"]:
            fit, cal, test_in = splits_cache[seed][T]
            U = fit_subspace(X[fit])
            s_cal = score(U, X[cal])
            n = len(s_cal)
            k = int(np.floor(ALPHA * (n + 1)))
            thr = np.sort(s_cal)[max(k - 1, 0)]
            s_in = score(U, X[test_in])
            out_ids = np.concatenate([splits_cache[seed][d][2] for d in DOMAINS if d != T])
            s_out = score(U, X[out_ids])
            y = np.r_[np.ones_like(s_in), np.zeros_like(s_out)]
            res[T] = {
                "r": U.shape[1],
                "auroc": roc_auc_score(y, np.r_[s_in, s_out]),
                "coverage": float((s_in >= thr).mean()),
                "false_accept": float((s_out >= thr).mean()),
            }
            if T == "gsm8k-code":
                hard_ids = splits_cache[seed]["gsm8k-cot"][2]
                s_hard = score(U, X[hard_ids])
                yh = np.r_[np.ones_like(s_in), np.zeros_like(s_hard)]
                res[T]["hard_auroc"] = roc_auc_score(yh, np.r_[s_in, s_hard])
                res[T]["hard_reject"] = float((s_hard < thr).mean())
                res[T]["_roc_hard"] = roc_curve(yh, np.r_[s_in, s_hard])
        results[sp_name] = res
    return results


def k_curve(spaces, ks=(5, 10, 25, 50), seeds=range(5)):
    out = {sp: {k: [] for k in ks} for sp in spaces}
    for sp_name, (X, dom) in spaces.items():
        X = unit(X)
        idx_by_dom = {d: np.where(dom == d)[0] for d in DOMAINS}
        for seed in seeds:
            rng = np.random.default_rng(100 + seed)
            perm = {d: rng.permutation(idx_by_dom[d]) for d in DOMAINS}
            test_in = perm["gsm8k-code"][50:80]
            hard = perm["gsm8k-cot"][50:80]
            for k in ks:
                U = fit_subspace(X[perm["gsm8k-code"][:k]])
                s_in, s_hard = score(U, X[test_in]), score(U, X[hard])
                y = np.r_[np.ones_like(s_in), np.zeros_like(s_hard)]
                out[sp_name][k].append(roc_auc_score(y, np.r_[s_in, s_hard]))
    return out


def fig_pca(spaces):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor="white")
    for ax, sp_name in zip(axes, ["grad", "emb-traj", "emb-query"]):
        X, dom = spaces[sp_name]
        Xc = unit(X) - unit(X).mean(0)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        P = Xc @ Vt[:2].T
        for d in DOMAINS:
            m = dom == d
            ax.scatter(P[m, 0], P[m, 1], s=14, c=COLORS[d], label=d,
                       alpha=0.75, linewidths=0)
        style(ax)
        ax.set_title({"grad": "gradient features",
                      "emb-traj": "embedding (query+trajectory)",
                      "emb-query": "embedding (query only)"}[sp_name],
                     fontsize=11, color=TEXT)
        ax.set_xticks([]), ax.set_yticks([])
    axes[0].legend(frameon=False, fontsize=8, loc="best", labelcolor=TEXT)
    fig.suptitle("PCA of feature spaces (5 domains)", fontsize=13, color=TEXT)
    fig.tight_layout()
    fig.savefig(FIGS / "pilot_pca.png", dpi=150, bbox_inches="tight")


def fig_roc(results):
    fig, ax = plt.subplots(figsize=(5.4, 4.6), facecolor="white")
    for c, sp in zip(SERIES, ["grad", "emb-traj", "emb-query"]):
        fpr, tpr, _ = results[sp]["gsm8k-code"]["_roc_hard"]
        au = results[sp]["gsm8k-code"]["hard_auroc"]
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{sp}  AUROC={au:.3f}")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    style(ax)
    ax.set_xlabel("false positive rate", fontsize=10, color=TEXT)
    ax.set_ylabel("true positive rate", fontsize=10, color=TEXT)
    ax.set_title("Hard pair: gsm8k-code boundary vs gsm8k-cot probes\n"
                 "(identical questions, different action space)",
                 fontsize=11, color=TEXT)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "pilot_roc_hard.png", dpi=150, bbox_inches="tight")


def fig_k(kres):
    fig, ax = plt.subplots(figsize=(5.6, 4.2), facecolor="white")
    for c, sp in zip(SERIES, ["grad", "emb-traj", "emb-query"]):
        ks = sorted(kres[sp])
        mu = np.array([np.mean(kres[sp][k]) for k in ks])
        sd = np.array([np.std(kres[sp][k]) for k in ks])
        ax.plot(ks, mu, color=c, lw=2, marker="o", ms=5, label=sp)
        ax.fill_between(ks, mu - sd, mu + sd, color=c, alpha=0.15, linewidth=0)
    style(ax)
    ax.set_xlabel("k (few-shot queries defining the boundary)", fontsize=10, color=TEXT)
    ax.set_ylabel("AUROC on hard pair", fontsize=10, color=TEXT)
    ax.set_title("Sample efficiency of boundary estimation", fontsize=11, color=TEXT)
    ax.set_ylim(0.35, 1.02)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(FIGS / "pilot_k_curve.png", dpi=150, bbox_inches="tight")


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    spaces = load_features()
    results = evaluate(spaces)
    kres = k_curve(spaces)
    fig_pca(spaces)
    fig_roc(results)
    fig_k(kres)

    lines = ["# Pilot results", "",
             f"splits: fit={N_FIT} cal={N_CAL} test={N_TEST}, alpha={ALPHA}, "
             f"energy={ENERGY}", "",
             "| space | domain | r | AUROC | coverage | false-accept | hard-AUROC | hard-reject |",
             "|---|---|---|---|---|---|---|---|"]
    for sp in ["grad", "emb-traj", "emb-query"]:
        for T, m in results[sp].items():
            lines.append(
                f"| {sp} | {T} | {m['r']} | {m['auroc']:.3f} | {m['coverage']:.2f} "
                f"| {m['false_accept']:.2f} | {m.get('hard_auroc', float('nan')):.3f} "
                f"| {m.get('hard_reject', float('nan')):.2f} |")
    lines += ["", "## k-curve (hard pair AUROC, mean±std over 5 seeds)", ""]
    for sp in kres:
        row = ", ".join(f"k={k}: {np.mean(v):.3f}±{np.std(v):.3f}"
                        for k, v in sorted(kres[sp].items()))
        lines.append(f"- **{sp}**: {row}")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
