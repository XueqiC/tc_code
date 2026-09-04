#!/usr/bin/env python3
"""Fit the capability-atom dictionary on event fingerprints and report coverage (CRCD spec §3.3, §6.1).

Steps: label-free model selection over K x lambda_z on a train/dev split of events (grouped by
prompt hash when available) -> refit on all events -> held-out coverage of the selected dictionary
and of the same-K baselines (PCA/SVD, k-means dictionary, random unit-norm dictionary) on the dev
split -> sparsity and atom-usage statistics -> per-atom category / side profile from zbar = |z|.

Input: fingerprints json (gate.events[*].a3) or .npz with keys psi (n, d), state_hash (n,)
[+ optional category, cluster, side].  Writes results/analysis/atoms_fit_<tag>.json (dictionary U
included, rounded) and appends to logs/t2_atoms.log.

Usage:
    PYTHONPATH=src .venv/bin/python tools/atoms_fit.py --features results/analysis/crcd_fingerprints_v4.json \
        --pool data/bfcl_sft/pool_events_pref_v2.jsonl --tag gate_v2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas import atoms  # noqa: E402


def log(msg: str, fh=None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def load(features: Path, pool: Path | None) -> dict:
    feats = atoms.load_features(features)
    psi = np.asarray(feats["psi"], dtype=np.float64)
    n = psi.shape[0]
    if "meta" in feats:
        cats = [m["category"] for m in feats["meta"]]
        clusters = [int(m["cluster"]) for m in feats["meta"]]
        hashes, sides = [None] * n, [None] * n
        if pool is not None and pool.is_file():
            rows = [json.loads(l) for l in pool.read_text().splitlines() if l.strip()]
            if len(rows) == n:
                hashes = [hashlib.sha1(r["prompt"].encode("utf-8")).hexdigest()[:16] for r in rows]
                sides = [1 if "<tool_call>" in r["response"] else 0 for r in rows]
    else:
        hashes = list(feats["state_hash"])
        cats = list(feats["category"]) if "category" in feats else ["unknown"] * n
        clusters = [int(c) for c in feats["cluster"]] if "cluster" in feats else [-1] * n
        sides = [int(s) for s in feats["side"]] if "side" in feats else [None] * n
    return {"psi": psi, "hash": hashes, "category": cats, "cluster": clusters, "side": sides}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="results/analysis/crcd_fingerprints_v4.json")
    ap.add_argument("--pool", default="data/bfcl_sft/pool_events_pref_v2.jsonl")
    ap.add_argument("--tag", default="gate_v2")
    ap.add_argument("--out", default=None, help="default results/analysis/atoms_fit_<tag>.json")
    ap.add_argument("--log", default="logs/t2_atoms.log")
    ap.add_argument("--ks", default="4,8,16,32")
    ap.add_argument("--lambdas", default="0,0.01,0.02,0.05,0.1,0.2")
    ap.add_argument("--criterion", default="sparse_recon_1se", choices=["sparse_recon_1se", "sparse_recon", "coverage"])
    ap.add_argument("--dev-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-save-dict", action="store_true", help="omit U from the json")
    args = ap.parse_args()
    out_path = ROOT / (args.out or f"results/analysis/atoms_fit_{args.tag}.json")
    (ROOT / args.log).parent.mkdir(parents=True, exist_ok=True)
    fh = open(ROOT / args.log, "a")
    log(f"=== atoms_fit start: features={args.features} tag={args.tag} seed={args.seed}", fh)

    ev = load(ROOT / args.features, ROOT / args.pool if args.pool else None)
    psi = ev["psi"]
    n, d = psi.shape
    groups = ev["hash"] if all(h is not None for h in ev["hash"]) else None
    log(f"n={n} d={d} groups={'prompt-hash' if groups else 'none'} unique_groups={len(set(groups)) if groups else n}", fh)
    Ks = [int(x) for x in args.ks.split(",")]
    lams = [float(x) for x in args.lambdas.split(",")]
    sel = atoms.select_model(psi, Ks, lams, seed=args.seed, dev_frac=args.dev_frac, groups=groups,
                             criterion=args.criterion)
    log(f"grid ({args.criterion}); dev split n_train={sel['n_train']} n_dev={sel['n_dev']}:", fh)
    for g in sel["grid"]:
        log(f"  K={g['K']:2d} lam={g['lambda_z']:<5} dev_cov={g['dev_coverage']:.3f} train_cov={g['train_coverage']:.3f} "
            f"dev_sparse_err={g['dev_sparse_recon_error']:.3f}±{g['dev_sparse_recon_se']:.3f} nnz={g['mean_nnz_per_event']:.2f} "
            f"used={g['atoms_used']} conv={g['converged']}", fh)
    K, lam = int(sel["best"]["K"]), float(sel["best"]["lambda_z"])
    log(f"selected K={K} lambda_z={lam}; argmin K={sel['argmin']['K']} lambda_z={sel['argmin']['lambda_z']}", fh)

    train, dev = np.asarray(sel["train_idx"]), np.asarray(sel["dev_idx"])
    # same-K baselines on the dev split (fit on train)
    base = {}
    fd_tr = atoms.fit(psi[train], K, lam, seed=args.seed)
    base["sparse_dict"] = {"dev_coverage": atoms.reconstruction_coverage(fd_tr.U, psi[dev]),
                           "dev_sparse_recon_error": atoms.sparse_reconstruction_error(fd_tr.U, psi[dev], lam),
                           "mean_nnz_per_event": float(fd_tr.nnz_per_event.mean())}
    pca = atoms.fit_pca(psi[train], K, center=False)
    base["pca_svd"] = {"dev_coverage": atoms.reconstruction_coverage(pca["U"], psi[dev]),
                       "explained_variance_ratio_train": [float(x) for x in pca["explained_variance_ratio"]],
                       "mean_nnz_per_event": float(K)}
    km = atoms.fit_kmeans(psi[train], K, seed=args.seed)
    base["kmeans_dict"] = {"dev_coverage": atoms.reconstruction_coverage(km["U"], psi[dev]),
                           "dev_hard_recon_error": float(np.sum((psi[dev] - atoms.encode_kmeans(km, psi[dev]) @ km["U"].T) ** 2) / np.sum(psi[dev] ** 2)),
                           "mean_nnz_per_event": 1.0}
    Ur = atoms.random_dictionary(d, K, seed=args.seed + 1)
    base["random_dict"] = {"dev_coverage": atoms.reconstruction_coverage(Ur, psi[dev]),
                           "dev_sparse_recon_error": atoms.sparse_reconstruction_error(Ur, psi[dev], lam)}
    for k, v in base.items():
        log(f"  dev coverage {k:<12} = {v['dev_coverage']:.4f}", fh)
    # coverage as a function of K (span only), sparse vs PCA, on dev
    cov_curve = []
    for Kc in Ks:
        fK = atoms.fit(psi[train], Kc, lam, seed=args.seed)
        pK = atoms.fit_pca(psi[train], Kc, center=False)
        rK = atoms.random_dictionary(d, Kc, seed=args.seed + 1)
        cov_curve.append({"K": Kc, "sparse_dict": atoms.reconstruction_coverage(fK.U, psi[dev]),
                          "pca_svd": atoms.reconstruction_coverage(pK["U"], psi[dev]),
                          "random_dict": atoms.reconstruction_coverage(rK, psi[dev])})

    # final fit on all events
    fd = atoms.fit(psi, K, lam, seed=args.seed)
    Z = fd.Z
    zbar = atoms.atom_relevance(Z)
    log(f"final fit: iters={fd.n_iter} converged={fd.converged} obj={fd.objective_history[-1]:.4f} "
        f"coverage_all={atoms.reconstruction_coverage(fd.U, psi):.4f} mean_nnz={fd.nnz_per_event.mean():.2f} "
        f"atoms_used={(fd.atom_usage > 0).sum()}", fh)
    # per-atom profile: relevance-weighted category / cluster / side distribution
    cats = sorted(set(ev["category"]))
    profile = []
    for k in range(K):
        w = zbar[:, k]
        tot = float(w.sum())
        if tot <= 0:
            profile.append({"atom": k, "total_relevance": 0.0, "top_categories": [], "top_clusters": [], "call_share": None})
            continue
        by_cat = defaultdict(float); by_cl = defaultdict(float); call = 0.0; side_tot = 0.0
        for i in range(n):
            by_cat[ev["category"][i]] += w[i]; by_cl[ev["cluster"][i]] += w[i]
            if ev["side"][i] is not None:
                side_tot += w[i]; call += w[i] * ev["side"][i]
        profile.append({"atom": k, "total_relevance": tot, "n_active": int((w > 1e-10).sum()),
                        "top_categories": sorted(((c, round(v / tot, 3)) for c, v in by_cat.items()), key=lambda x: -x[1])[:4],
                        "top_clusters": sorted(((int(c), round(v / tot, 3)) for c, v in by_cl.items()), key=lambda x: -x[1])[:4],
                        "call_share": (call / side_tot) if side_tot > 0 else None})
        log(f"  atom {k:2d}: rel={tot:6.2f} active={profile[-1]['n_active']:3d} cats={profile[-1]['top_categories'][:3]} "
            f"clusters={profile[-1]['top_clusters'][:3]} call_share={profile[-1]['call_share']}", fh)
    # category purity of atoms vs k-means (analysis-only labels)
    top_atom = np.argmax(zbar, axis=1)
    def purity(assign):
        tot = 0
        for a in set(assign):
            idx = [i for i in range(n) if assign[i] == a]
            tot += max(sum(1 for i in idx if ev["category"][i] == c) for c in cats)
        return tot / n
    km_all = atoms.fit_kmeans(psi, K, seed=args.seed)
    purity_stats = {"sparse_top_atom": purity(top_atom.tolist()), "kmeans": purity(km_all["labels"].tolist()),
                    "majority_category_baseline": max(ev["category"].count(c) for c in cats) / n}
    log(f"category purity (analysis only): {purity_stats}", fh)

    out = {"features": args.features, "tag": args.tag, "seed": args.seed, "n": n, "d": d,
           "feature_note": ("A3 = Adam-preconditioned, JL-projected, L2-normalised differential LoRA-B gradients; "
                            "NOT Fisher-whitened. Rerun with --features <psi.npz> (keys psi, state_hash)."),
           "selection": {k: v for k, v in sel.items() if k not in ("train_idx", "dev_idx")},
           "dev_baselines_same_K": base, "dev_coverage_vs_K": cov_curve,
           "final_fit": fd.summary(), "coverage_all_events": atoms.reconstruction_coverage(fd.U, psi),
           "objective_history": [float(x) for x in fd.objective_history],
           "atom_profile": profile, "category_purity": purity_stats,
           "Z": [[round(float(x), 6) for x in row] for row in Z],
           "zbar_mean_per_atom": [float(x) for x in zbar.mean(axis=0)]}
    if not args.no_save_dict:
        out["U"] = [[round(float(x), 6) for x in row] for row in fd.U.T]  # K rows of length d
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float))
    log(f"wrote {out_path.relative_to(ROOT)}", fh)
    fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
