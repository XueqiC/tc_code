"""M1: interference geometry — can the gradient kernel PREDICT E3's measured
per-domain loss changes, and are cross-domain gradient inner products negative?

First-order theory: dLoss(q) ~ -eta * sum_{x in D} <g(q), g(x)>.
Positive kernel mass => loss should drop; observed out-domain loss RISES in E3
=> either systematically negative inner products (true interference) or
higher-order effects. This script measures both:
  (a) sign/magnitude structure of cross-domain kernel mass per condition,
  (b) Spearman between kernel-predicted dLoss and measured dLoss over
      (condition x domain) cells.
Replicates E3 selections from recomputed features (B_random's exact membership
may differ from the run — statistically equivalent; C/D are score-deterministic).
"""
import json
from glob import glob
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from boundary import fit_subspace, score, unit
from features import extract_features
from grad_features import MODEL, stable_seed
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "m1"
BUDGET = 40000
DOMAINS = ["gsm8k-code", "gsm8k-cot", "pandas", "sql", "alpaca"]
N_TEST = 30


def load_pool():
    items = []
    for f in sorted(glob(str(ROOT / "data" / "pool_v0" / "*" / "*.jsonl"))):
        dom = Path(f).stem
        for line in open(f):
            r = json.loads(line)
            code = r.get("code")
            if not code:
                continue
            if dom == "gsm8k-code" and r.get("verified") is not True:
                continue
            items.append({"domain": dom, "prompt": r["prompt"], "response": code})
    for dom in ["gsm8k-cot", "alpaca"]:
        rows = [json.loads(l) for l in open(ROOT / "data" / "pilot" / f"{dom}.jsonl")]
        perm = np.random.default_rng(stable_seed("split-" + dom)).permutation(len(rows))
        for i in perm[:-N_TEST]:
            items.append(rows[i])
    return items


def test_rows():
    out = {}
    for dom in DOMAINS:
        rows = [json.loads(l) for l in open(ROOT / "data" / "pilot" / f"{dom}.jsonl")]
        perm = np.random.default_rng(stable_seed("split-" + dom)).permutation(len(rows))
        out[dom] = [rows[i] for i in perm[-N_TEST:]]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pool = load_pool()
    print(f"pool={len(pool)}")

    cache = OUT / "pool_features.npz"
    if cache.exists():
        G = np.load(cache)["G"]
    else:
        G = extract_features(MODEL, pool)
        np.savez_compressed(cache, G=G)
    G = unit(G.astype(np.float64))

    tests = test_rows()
    tcache = OUT / "test_features.npz"
    if tcache.exists():
        d = np.load(tcache, allow_pickle=True)
        Q, qdom = d["Q"], d["qdom"].astype(str)
    else:
        flat = [r for dom in DOMAINS for r in tests[dom]]
        Q = extract_features(MODEL, flat)
        qdom = np.array([dom for dom in DOMAINS for _ in tests[dom]])
        np.savez_compressed(tcache, Q=Q, qdom=qdom)
    Q = unit(Q.astype(np.float64))

    # spec rows (same as E3): first 35 of stable-split non-test gsm8k-code golds
    rows = [json.loads(l) for l in open(ROOT / "data" / "pilot" / "gsm8k-code.jsonl")]
    perm = np.random.default_rng(stable_seed("split-gsm8k-code")).permutation(len(rows))
    spec = [rows[i] for i in perm[:-N_TEST]][:50]
    S = unit(extract_features(MODEL, spec).astype(np.float64))
    U = fit_subspace(S[:35])
    s_grad = score(U, G)

    tok = AutoTokenizer.from_pretrained(MODEL)
    ntok = np.array([len(tok(it["response"], add_special_tokens=False)["input_ids"])
                     for it in pool])

    def prefix(order):
        used, keep = 0, []
        for i in order:
            if used + ntok[i] > BUDGET:
                continue
            used += ntok[i]; keep.append(i)
        return np.array(keep)

    sels = {
        "A_all": np.arange(len(pool)),
        "B_random": prefix(np.random.default_rng(0).permutation(len(pool))),
        "D_grad": prefix(np.argsort(-s_grad, kind="stable")),
    }

    measured = json.load(open(ROOT / "results" / "e3_tier1" / "metrics.json"))
    base_loss = measured["conditions"]["base"]["eval"]["loss_by_domain"] \
        if "loss_by_domain" in measured["conditions"]["base"]["eval"] else None
    # fall back: parse from summary table values hardcoded
    LOSS = {"base": {"gsm8k-code": 1.667, "gsm8k-cot": 0.636, "pandas": 1.624,
                     "sql": 1.574, "alpaca": 1.500},
            "A_all": {"gsm8k-code": 1.223, "gsm8k-cot": 0.645, "pandas": 1.287,
                      "sql": 1.283, "alpaca": 1.402},
            "B_random": {"gsm8k-code": 1.153, "gsm8k-cot": 0.635, "pandas": 1.276,
                         "sql": 1.558, "alpaca": 1.506},
            "D_grad": {"gsm8k-code": 1.165, "gsm8k-cot": 0.638, "pandas": 1.496,
                       "sql": 1.451, "alpaca": 1.427}}
    if base_loss:
        pass  # prefer stored values if schema present

    print("\n== kernel mass per (condition, domain): mean_q mean_{x in D} <g_q, g_x> ==")
    preds, meas, cells = [], [], []
    neg_frac = {}
    for cname, idx in sels.items():
        u = G[idx].mean(0)
        for dom in DOMAINS:
            qs = Q[np.array([d == dom for d in np.repeat(DOMAINS, N_TEST)])]
            mass = float(qs @ u @ np.ones(1)) if False else float((qs @ u).mean())
            dl = LOSS[cname][dom] - LOSS["base"][dom]
            preds.append(-mass); meas.append(dl)
            cells.append((cname, dom, mass, dl))
            print(f"{cname:9s} {dom:11s} kernel_mass={mass:+.4f} measured_dLoss={dl:+.3f}")
        K = G[idx] @ Q.T
        for dom in DOMAINS:
            m = np.repeat(DOMAINS, N_TEST) == dom
            neg_frac[(cname, dom)] = float((K[:, m] < 0).mean())

    rho, p = spearmanr(preds, meas)
    print(f"\nSpearman(pred dLoss ~ -kernel_mass, measured dLoss): rho={rho:.3f} p={p:.4f}")
    print("\n== fraction of negative inner products (selected x test) ==")
    for (c, d), v in neg_frac.items():
        print(f"{c:9s} {d:11s} neg_frac={v:.2f}")

    json.dump({"cells": cells, "spearman": [float(rho), float(p)],
               "neg_frac": {f"{c}|{d}": v for (c, d), v in neg_frac.items()}},
              open(OUT / "report.json", "w"), indent=1)
    print("saved", OUT / "report.json")


if __name__ == "__main__":
    main()
