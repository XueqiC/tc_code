#!/usr/bin/env python3
"""Validity gate v2 analysis (user prompt 14): does fingerprint similarity predict cross-cluster transfer?

Inputs: <gate>/clusters.json (A3 centroids, per-cluster single-task prompt hashes, categories) and
<gate>/rates_base.json, rates_c<k>.json (K=8 pass rates per prompt hash).
Reports, for the gain matrix G[i, j] = mean gain on cluster j after training on cluster i:
  * Spearman(P, G) with P = centroid cosine, over all cells and off-diagonal cells, raw and column-centred
  * within-vs-cross: mean diag(G) - mean offdiag(G), and diag rank (fraction of rows where the trained
    cluster is the best-gaining cluster)
  * sign prediction: among off-diagonal cells, does P<0 predict G<0 (accuracy, precision, recall)
  * ranking quality per row: precision@2 and NDCG@3 of P-ranked targets against G
  * baselines for P: category-overlap similarity (Jaccard of category sets), random permutations of P
    (null distribution of Spearman), and a "same-side" indicator (both call-side or both abstain-side)
Writes JSON to --out and prints a summary.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]


def ndcg_at(scores_pred, scores_true, k):
    order = np.argsort(-scores_pred)[:k]
    gains = np.array(scores_true)[order]
    ideal = np.sort(scores_true)[::-1][:k]
    def dcg(g):
        return sum((v) / math.log2(i + 2) for i, v in enumerate(g))
    idcg = dcg(ideal)
    return dcg(gains) / idcg if idcg > 0 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", default="data/bfcl_sft/gate_v2")
    parser.add_argument("--out", default="results/analysis/crcd_gate_v2.json")
    args = parser.parse_args()
    gate = ROOT / args.gate
    cl = json.loads((gate / "clusters.json").read_text())
    base = json.loads((gate / "rates_base.json").read_text())
    ids = sorted(cl, key=int)
    rates = {c: json.loads((gate / f"rates_c{c}.json").read_text()) for c in ids if (gate / f"rates_c{c}.json").is_file()}
    have = [c for c in ids if c in rates]
    n = len(have)
    G = np.full((n, n), np.nan)
    for i, c in enumerate(have):
        for j, d in enumerate(have):
            t = [x for x in cl[d]["single_task_ids"] if x in base and x in rates[c]]
            if t:
                G[i, j] = float(np.mean([rates[c][x] - base[x] for x in t]))
    cent = np.asarray([cl[c]["centroid"] for c in have], dtype=float)
    cent = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-8)
    P = cent @ cent.T
    cats = [set(cl[c]["categories"]) for c in have]
    Pcat = np.array([[len(a & b) / max(len(a | b), 1) for b in cats] for a in cats])
    abst = [any(("irrelevance" in x) or x.startswith("oos") for x in cl[c]["categories"]) for c in have]
    Pside = np.array([[1.0 if abst[i] == abst[j] else 0.0 for j in range(n)] for i in range(n)])
    mask = ~np.isnan(G)
    off = mask & ~np.eye(n, dtype=bool)
    Gc = G - np.nanmean(G, axis=0, keepdims=True)  # column-centred (removes per-target drift)

    def sp(Pm, Gm, m):
        return float(spearmanr(Pm[m], Gm[m]).correlation) if m.sum() > 3 else float("nan")

    rep = {"clusters": have, "n_single_tasks": {c: len(cl[c]["single_task_ids"]) for c in have},
           "gain_matrix": [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in G],
           "P_cosine": [[round(float(v), 4) for v in row] for row in P]}
    for name, Pm in (("A3_cosine", P), ("category_jaccard", Pcat), ("same_side", Pside)):
        rep[name] = {"spearman_all": sp(Pm, G, mask), "spearman_offdiag": sp(Pm, G, off),
                     "spearman_offdiag_colcentred": sp(Pm, Gc, off)}
    # null distribution for A3 by permuting cluster labels of P
    rng = random.Random(0)
    null = []
    for _ in range(2000):
        perm = list(range(n)); rng.shuffle(perm)
        Pp = P[np.ix_(perm, perm)]
        null.append(sp(Pp, Gc, off))
    obs = rep["A3_cosine"]["spearman_offdiag_colcentred"]
    rep["A3_null"] = {"mean": float(np.nanmean(null)), "p_value_ge_obs": float(np.mean([v >= obs for v in null if not np.isnan(v)]))}
    diag = np.array([G[i, i] for i in range(n)])
    rep["within_vs_cross"] = {"mean_diag": float(np.nanmean(diag)), "mean_offdiag": float(np.nanmean(G[off])),
                              "diag_is_row_max": float(np.mean([np.nanargmax(G[i]) == i for i in range(n)])),
                              "diag_is_row_max_colcentred": float(np.mean([np.nanargmax(Gc[i]) == i for i in range(n)]))}
    # sign prediction on off-diagonal cells: P below its median -> predict negative transfer
    thr = float(np.median(P[off]))
    pred_neg = (P < thr) & off
    true_neg = (G < 0) & off
    tp = float(np.sum(pred_neg & true_neg)); fp = float(np.sum(pred_neg & ~true_neg & off)); fn = float(np.sum(~pred_neg & true_neg & off))
    rep["sign_prediction"] = {"threshold_P": thr, "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                              "accuracy": float(np.mean((pred_neg == true_neg)[off]))}
    # ranking quality per row (exclude the diagonal)
    p2, nd3 = [], []
    for i in range(n):
        idx = [j for j in range(n) if j != i and not np.isnan(G[i, j])]
        if len(idx) < 3:
            continue
        pp = P[i, idx]; gg = Gc[i, idx]
        top = np.argsort(-pp)[:2]; best = np.argsort(-gg)[:2]
        p2.append(len(set(top) & set(best)) / 2)
        nd3.append(ndcg_at(pp, gg - gg.min(), 3))
    rep["ranking"] = {"precision_at_2": float(np.mean(p2)), "ndcg_at_3": float(np.nanmean(nd3))}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("gain_matrix", "P_cosine")}, indent=1))
    print("G (rows=trained, cols=measured):")
    for i, c in enumerate(have):
        print(f"  c{c}: " + " ".join("  nan" if np.isnan(v) else f"{v:+.3f}" for v in G[i]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
