#!/usr/bin/env python3
"""Validity gate, step 3: does the differential fingerprint predict transfer?

Inputs: data/bfcl_sft/gate/clusters.json (centroids, task ids per cluster),
rates_base.json and rates_c<k>.json (student pass rate per single-turn task after
training preference-only on cluster k).

Outputs:
  G[c, c']  = mean over tasks of cluster c' of (rate_ck - rate_base)     actual gain
  P[c, c']  = cosine(centroid_c, centroid_c')                            predicted transfer
  Spearman(P, G) over all ordered pairs and over off-diagonal pairs;
  selectivity = mean diag(G) - mean offdiag(G).
The doc's gate: Spearman > 0.30 and top-vs-bottom quartile gain difference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", default="data/bfcl_sft/gate")
    parser.add_argument("--out", default="results/analysis/crcd_gate_v1.json")
    args = parser.parse_args()
    gate = ROOT / args.gate
    clusters = json.loads((gate / "clusters.json").read_text())
    base = json.loads((gate / "rates_base.json").read_text())
    ids = sorted(clusters, key=int)
    rates = {}
    for c in ids:
        f = gate / f"rates_c{c}.json"
        if f.is_file():
            rates[c] = json.loads(f.read_text())
    have = [c for c in ids if c in rates]
    print(f"[gate] clusters measured: {have} / {ids}")
    G = np.full((len(have), len(have)), np.nan)
    for i, c in enumerate(have):
        for j, d in enumerate(have):
            tasks = [t for t in clusters[d]["single_task_ids"] if t in base and t in rates[c]]
            if tasks:
                G[i, j] = float(np.mean([rates[c][t] - base[t] for t in tasks]))
    cent = np.asarray([clusters[c]["centroid"] for c in have])
    cent = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-8)
    P = cent @ cent.T
    from scipy.stats import spearmanr

    mask = ~np.isnan(G)
    off = mask & ~np.eye(len(have), dtype=bool)
    rho_all = spearmanr(P[mask], G[mask]).correlation if mask.sum() > 3 else None
    rho_off = spearmanr(P[off], G[off]).correlation if off.sum() > 3 else None
    diag = np.nanmean(np.diag(G)) if len(have) else None
    offd = np.nanmean(G[off]) if off.sum() else None
    report = {
        "clusters": have,
        "gain_matrix": [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in G],
        "predicted_transfer": [[round(float(v), 4) for v in row] for row in P],
        "spearman_all": None if rho_all is None else round(float(rho_all), 3),
        "spearman_offdiag": None if rho_off is None else round(float(rho_off), 3),
        "mean_diag_gain": None if diag is None else round(float(diag), 4),
        "mean_offdiag_gain": None if offd is None else round(float(offd), 4),
        "n_tasks_per_cluster": {c: len(clusters[c]["single_task_ids"]) for c in have},
    }
    (ROOT / args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("gain_matrix", "predicted_transfer")}, indent=1))
    print("[gate] G (rows = trained on, cols = measured on):")
    for i, c in enumerate(have):
        print(f"  c{c}: " + " ".join(f"{'  nan' if np.isnan(v) else f'{v:+.3f}'}" for v in G[i]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
