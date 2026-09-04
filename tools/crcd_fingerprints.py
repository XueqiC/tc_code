#!/usr/bin/env python3
"""CRCD P0-3 (A-ladder, MVP): fingerprint representations of intervention events.

Four representations of the same event e = (h, a-, a+, dU) using the student's
Adam-preconditioned, randomly projected LoRA gradient (src/features.py):

  A1  whole-text gradient          g(text = h + a+)           (trajectory-level)
  A2  response gradient            g(a+ | h)                  (turn-level, absolute)
  A3  differential gradient        g(a+ | h) - g(a- | h)      (fail -> correct update direction)
  A4  utility-weighted differential dU * A3

MVP validity proxies (the doc's full gate needs post-update gains, tomorrow):
  * category coherence: k-means ARI against seed_category, k = #categories
  * outcome prediction: ridge regression of u_minus (student pass rate) from the
    fingerprint, 5-fold CV Spearman -- does the representation know how hard the
    event is for THIS student?
  * consequentiality prediction (stateful only): dU > 0 vs not, CV AUROC

Usage:
    PYTHONPATH=src python tools/crcd_fingerprints.py --events data/bfcl_sft/pool_events_pref.jsonl \
        --student Qwen/Qwen3.5-4B --out results/analysis/crcd_fingerprints_v1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def kmeans_ari(feats: np.ndarray, labels: list[str], seed: int = 0) -> float:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    k = max(2, len(set(labels)))
    pred = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(feats)
    return float(adjusted_rand_score(labels, pred))


def cv_spearman(feats: np.ndarray, target: np.ndarray, seed: int = 0) -> float:
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    preds = np.zeros_like(target, dtype=float)
    for train, test in KFold(5, shuffle=True, random_state=seed).split(feats):
        model = Ridge(alpha=1.0).fit(feats[train], target[train])
        preds[test] = model.predict(feats[test])
    return float(spearmanr(preds, target).correlation)


def cv_auroc(feats: np.ndarray, target: np.ndarray, seed: int = 0) -> float | None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    if target.sum() < 3 or (1 - target).sum() < 3:
        return None
    preds = np.zeros_like(target, dtype=float)
    for train, test in StratifiedKFold(3, shuffle=True, random_state=seed).split(feats, target):
        model = LogisticRegression(max_iter=1000).fit(feats[train], target[train])
        preds[test] = model.predict_proba(feats[test])[:, 1]
    return float(roc_auc_score(target, preds))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", default="data/bfcl_sft/pool_events_pref.jsonl")
    parser.add_argument("--student", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--proj-dim", type=int, default=64)
    parser.add_argument("--max-prompt-tok", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="results/analysis/crcd_fingerprints_v1.json")
    parser.add_argument("--gate-clusters", type=int, default=8)
    args = parser.parse_args()

    from features import extract_features

    rows = load_rows(ROOT / args.events)
    if args.limit:
        rows = rows[: args.limit]
    labels = [r.get("_seed_category", "") for r in rows]
    u_minus = np.array([float(r.get("_event_u_minus", 0.0)) for r in rows])
    d_u = np.array([float(r.get("_event_dU", 0.0)) for r in rows])
    print(f"[fp] {len(rows)} events, {len(set(labels))} categories", flush=True)

    def feats(prompt_key_rows):
        f = extract_features(args.student, prompt_key_rows, proj_dim=args.proj_dim,
                             max_prompt_tok=args.max_prompt_tok)
        return np.asarray(f, dtype=float)

    g_plus = feats([{"prompt": r["prompt"], "response": r["response"]} for r in rows])
    g_minus = feats([{"prompt": r["prompt"], "response": r["_rejected"]} for r in rows])
    g_whole = feats([{"prompt": "", "response": r["prompt"] + r["response"]} for r in rows])

    def unit(x):
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
        return x / n

    diff = unit(g_plus - g_minus)
    reps = {
        "A1_whole": unit(g_whole),
        "A2_response": unit(g_plus),
        "A3_differential": diff,
        # utility weighting must survive normalisation: scale the unit
        # differential direction by dU so low-utility events shrink toward the
        # origin (v2 run normalised after scaling, which made A4 == A3)
        "A4_utility_differential": diff * np.clip(d_u, 0.0, 1.0)[:, None],
    }
    report = {"n": len(rows), "categories": sorted(set(labels)), "results": {}}
    stateful = np.array([int(r.get("turn_index", 0)) > 0 or "ev1" not in str(r.get("_traj", "")) for r in rows])
    for name, x in reps.items():
        res = {
            "category_ari": kmeans_ari(x, labels),
            "u_minus_spearman": cv_spearman(x, u_minus),
        }
        if stateful.sum() >= 10:
            res["stateful_consequential_auroc"] = cv_auroc(x[stateful], (d_u[stateful] > 0).astype(int))
        report["results"][name] = res
        print(f"[fp] {name}: {res}", flush=True)
    # per-event features + k-means cluster ids (A3, k=8) for the validity-gate
    # experiment (train per cluster, measure cross-cluster gains)
    from sklearn.cluster import KMeans

    k_gate = int(args.gate_clusters)
    a3 = reps["A3_differential"]
    gate_labels = KMeans(n_clusters=k_gate, n_init=10, random_state=0).fit_predict(a3)
    report["gate"] = {
        "k": k_gate,
        "events": [
            {"task_id": r["task_id"], "traj": r.get("_traj"), "category": labels[i],
             "cluster": int(gate_labels[i]), "u_minus": float(u_minus[i]), "dU": float(d_u[i]),
             "a3": [round(float(v), 5) for v in a3[i]]}
            for i, r in enumerate(rows)
        ],
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"[fp] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
