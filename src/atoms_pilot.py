"""Capability-atoms pilot: sparse dictionary decomposition of trajectory gradients.

Exploration (2026-08-10, PI-approved): does a sparse atom dictionary
(a) improve intra-domain boundary resolution over dense subspaces, and
(b) yield interpretable skill units?

Uses cached block-level features (features_gradsteps.npz). CPU/GPU-light.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.metrics import roc_auc_score

from boundary import unit
from grad_features import stable_seed

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "atoms_pilot"
N_ATOMS = 128
SPARSITY_ALPHA = 0.05
TOP_ATOMS = 24  # support size per subgroup
N_SEEDS = 5

OFF = {"gsm8k-code": 0, "gsm8k-cot": 120, "pandas": 240, "sql": 360, "alpaca": 480}


def load_blocks():
    d = np.load(ROOT / "results" / "pilot" / "features_gradsteps.npz",
                allow_pickle=True)
    return unit(d["X_steps"].astype(np.float64)), d["row_id"], d["block_idx"]


def subgroup_rows():
    labels = {}
    for name, kw in [("sql", "join"), ("pandas", "loc[")]:
        rows = [json.loads(l) for l in open(ROOT / "data" / "pilot" / f"{name}.jsonl")]
        for i, r in enumerate(rows):
            labels[OFF[name] + i] = f"{name}-{'yes' if kw in r['response'].lower() else 'no'}"
    return labels


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, rid, bidx = load_blocks()
    print(f"blocks={len(X)} dim={X.shape[1]}")

    dico = MiniBatchDictionaryLearning(
        n_components=N_ATOMS, alpha=SPARSITY_ALPHA, batch_size=256,
        max_iter=150, fit_algorithm="lars", transform_algorithm="lasso_lars",
        transform_alpha=SPARSITY_ALPHA, random_state=0, n_jobs=4)
    codes = dico.fit(X).transform(X)
    A = np.abs(codes)
    active_per_block = (A > 1e-8).sum(1)
    print(f"atoms={N_ATOMS} mean_active_atoms_per_block={active_per_block.mean():.1f}")

    labels = subgroup_rows()
    groups = {}
    for g, l in labels.items():
        groups.setdefault(l, []).append(g)

    def row_code(g):
        m = rid == g
        return A[m].sum(0)  # pooled atom usage of a row's blocks

    pairs = [("sql-yes", "sql-no"), ("sql-no", "sql-yes"),
             ("pandas-yes", "pandas-no"), ("pandas-no", "pandas-yes")]
    report = {}
    for T, P in pairs:
        aurocs = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(1000 + stable_seed(T + P) % 1000 + seed)
            t = rng.permutation(groups[T]); p = rng.permutation(groups[P])[:30]
            fit_rows, test_in = t[:20], t[30:45]
            # discriminative support: atoms overused by T's fit rows relative
            # to the complement subgroup's fit-sized sample
            other = rng.permutation(groups[P])[:20]
            use_T = np.mean([row_code(g) for g in fit_rows], 0)
            use_O = np.mean([row_code(g) for g in other], 0)
            support = np.argsort(-(use_T - use_O))[:TOP_ATOMS]
            def s(g):
                c = row_code(g)
                return c[support].sum() / (c.sum() + 1e-12)
            si = [s(g) for g in test_in]; sp = [s(g) for g in p]
            y = np.r_[np.ones(len(si)), np.zeros(len(sp))]
            aurocs.append(roc_auc_score(y, np.r_[si, sp]))
        report[f"{T} vs {P}"] = (float(np.mean(aurocs)), float(np.std(aurocs)))
        print(f"{T} vs {P}: AUROC {np.mean(aurocs):.3f}±{np.std(aurocs):.3f}")

    # interpretability probe: top blocks for the most join-discriminative atoms
    sql_rows = [json.loads(l) for l in open(ROOT / "data" / "pilot" / "sql.jsonl")]
    yes = [g for g in groups["sql-yes"]]
    no = [g for g in groups["sql-no"]]
    disc = np.mean([row_code(g) for g in yes], 0) - np.mean([row_code(g) for g in no], 0)
    top_atoms = np.argsort(-disc)[:3]
    interp = {}
    for a in top_atoms:
        top_blocks = np.argsort(-A[:, a])[:5]
        snippets = []
        for b in top_blocks:
            g = int(rid[b])
            if 360 <= g < 480:
                resp = sql_rows[g - 360]["response"]
                lines = [l for l in resp.splitlines() if l.strip()]
                k = min(int(bidx[b]), len(lines) - 1)
                snippets.append(lines[k][:90])
            else:
                snippets.append(f"<row {g} dom-block>")
        interp[f"atom_{a}"] = snippets
        print(f"atom {a} (join-discriminative): {snippets[:3]}")

    json.dump({"aurocs": report, "interp": interp,
               "config": {"n_atoms": N_ATOMS, "alpha": SPARSITY_ALPHA,
                          "top_atoms": TOP_ATOMS}},
              open(OUT / "report.json", "w"), indent=1)
    print("saved", OUT / "report.json")


if __name__ == "__main__":
    main()
