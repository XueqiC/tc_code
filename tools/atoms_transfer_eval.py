#!/usr/bin/env python3
"""Continuous first-order transfer prediction vs measured gains (CRCD spec §3.4), gate-v2 test bed.

Prediction for (trained cluster i, held-out event j):  P_ij = z_j^T U^T U z_i  with the sparse
dictionary U fit on the event fingerprints (K, lambda chosen label-free on a train/dev split of
events), z_j = encode(psi_j), z_i = encode(centroid of cluster i).  Baselines: PCA-projected cosine
(and dot), raw-feature cosine, hard k-means dictionary, random unit-norm dictionary,
category-Jaccard, same-side indicator, random scores.

Actual gain: G[i, j] = rates_c<i>[sha1(prompt_j)[:16]] - rates_base[...]  (K=8 pass rates per prompt;
only single-turn events whose prompt hash is present).  Off-diagonal pairs = event j not in cluster i.

Metrics (off-diagonal pairs): Spearman; column-centred (per-event) and row-centred (per-cluster)
Spearman; NDCG@1/3/5 of positive transfer per trained cluster; MSE after one scalar calibration
fitted leave-one-cluster-out; AUROC / balanced accuracy (LOCO threshold) for the sign of transfer;
group-bootstrap CIs (over prompt hashes, 1000 reps); label-permutation test (1000); partial Spearman
controlling for event category, trained cluster, log response length, same-side and category-Jaccard.
Also a prompt-level (deduplicated) run and a post-hoc (K, lambda) sensitivity sweep.

Inputs: --features  results/analysis/crcd_fingerprints_v4.json (gate.events[*].a3)  or  a .npz with
        keys psi (n, d), state_hash (n,) [+ optional category, cluster, response_len, side, traj].
        A3 here = Adam-preconditioned, JL-projected, L2-normalised LoRA-B differential gradients
        (NOT Fisher-whitened); rerun with --features data/fingerprints/<x>.npz once psi is available.

Usage:
    PYTHONPATH=src .venv/bin/python tools/atoms_transfer_eval.py \
        --features results/analysis/crcd_fingerprints_v4.json --gate data/bfcl_sft/gate_v2 \
        --pool data/bfcl_sft/pool_events_pref_v2.jsonl --out results/analysis/atoms_transfer_gate_v2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas import atoms  # noqa: E402

BOOT_METRICS = ("spearman_offdiag", "spearman_colcentred", "spearman_rowcentred", "spearman_doublecentred",
                "ndcg@1", "ndcg@3", "ndcg@5", "auroc_sign", "balanced_acc_sign", "mse_ratio_calibrated")


def log(msg: str, fh=None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


def prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------
# data assembly
# ----------------------------------------------------------------------------------------------
def load_events(features_path: Path, pool_path: Path, fp_json: Path | None) -> dict:
    """Return psi (n, d) and per-event metadata: hash, cluster, category, side, resp_len, traj."""
    rows = [json.loads(l) for l in pool_path.read_text().splitlines() if l.strip()]
    row_hash = [prompt_hash(r["prompt"]) for r in rows]
    feats = atoms.load_features(features_path)
    if "meta" in feats:  # fingerprints json: positionally aligned with the pool rows (same order)
        meta = feats["meta"]
        if len(meta) != len(rows):
            raise ValueError(f"{len(meta)} fingerprint events vs {len(rows)} pool rows")
        for m, r in zip(meta, rows):
            if m["traj"] != r["_traj"]:
                raise ValueError("fingerprint events and pool rows are not aligned (traj mismatch)")
        hashes = row_hash
        clusters = [int(m["cluster"]) for m in meta]
        cats = [m["category"] for m in meta]
        trajs = [m["traj"] for m in meta]
        resp = [r["response"] for r in rows]
        source = "fingerprints_json"
    else:  # npz: join by state_hash to the pool rows (first match) for metadata not in the file
        hashes = list(feats["state_hash"])
        by_hash = {}
        for r, h in zip(rows, row_hash):
            by_hash.setdefault(h, r)
        fp_meta_by_hash = {}
        if fp_json is not None and fp_json.is_file():
            fp = atoms.load_fingerprints_json(fp_json)
            for m, h in zip(fp["meta"], row_hash):
                fp_meta_by_hash.setdefault(h, m)
        n = len(hashes)

        def col(key, default=None):
            return list(feats[key]) if key in feats else [default] * n

        clusters = col("cluster")
        cats = col("category")
        trajs = col("traj")
        resp = [None] * n
        for k, h in enumerate(hashes):
            r = by_hash.get(h)
            m = fp_meta_by_hash.get(h)
            if resp[k] is None and r is not None:
                resp[k] = r["response"]
            if cats[k] is None:
                cats[k] = r["_seed_category"] if r is not None else (m["category"] if m else "unknown")
            if clusters[k] is None and m is not None:
                clusters[k] = int(m["cluster"])
            if trajs[k] is None:
                trajs[k] = r["_traj"] if r is not None else (m["traj"] if m else h)
        if any(c is None for c in clusters):
            raise ValueError("npz input needs a 'cluster' key or hashes that join to the fingerprints json")
        clusters = [int(c) for c in clusters]
        source = "npz"
    psi = np.asarray(feats["psi"], dtype=np.float64)
    if "side" in feats:
        side = [int(s) for s in feats["side"]]
    else:
        side = [1 if (r is not None and "<tool_call>" in r) else 0 for r in resp]  # 1 = call, 0 = abstain
    if "response_len" in feats:
        resp_len = [float(x) for x in feats["response_len"]]
    else:
        resp_len = [float(len(r)) if r is not None else float("nan") for r in resp]
    return {"psi": psi, "hash": hashes, "cluster": clusters, "category": cats, "side": side,
            "resp_len": resp_len, "traj": trajs, "source": source, "features_path": str(features_path)}


def load_gate(gate: Path) -> dict:
    cl = json.loads((gate / "clusters.json").read_text())
    base = json.loads((gate / "rates_base.json").read_text())
    ids = sorted(cl, key=int)
    rates = {int(c): json.loads((gate / f"rates_c{c}.json").read_text()) for c in ids
             if (gate / f"rates_c{c}.json").is_file()}
    return {"clusters": cl, "base": base, "rates": rates, "ids": [int(c) for c in ids if int(c) in rates]}


def gain_matrix(ev: dict, gate: dict) -> tuple[np.ndarray, np.ndarray]:
    ids = gate["ids"]
    n = len(ev["hash"])
    G = np.full((len(ids), n), np.nan)
    for a, i in enumerate(ids):
        ri = gate["rates"][i]
        for j, h in enumerate(ev["hash"]):
            if h in gate["base"] and h in ri:
                G[a, j] = ri[h] - gate["base"][h]
    valid = ~np.isnan(G)
    off = valid & np.array([[ev["cluster"][j] != i for j in range(n)] for i in ids])
    return G, off


def dedup_by_prompt(ev: dict) -> dict:
    """Prompt-level aggregation: mean (renormalised) fingerprint, cluster set, majority side."""
    groups = defaultdict(list)
    for j, h in enumerate(ev["hash"]):
        groups[h].append(j)
    keys = list(groups)
    psi = np.stack([ev["psi"][groups[h]].mean(axis=0) for h in keys])
    psi = psi / np.maximum(np.linalg.norm(psi, axis=1, keepdims=True), 1e-12)
    clusters = [Counter(ev["cluster"][j] for j in groups[h]).most_common(1)[0][0] for h in keys]
    cluster_sets = [set(ev["cluster"][j] for j in groups[h]) for h in keys]
    return {"psi": psi, "hash": keys, "cluster": clusters, "cluster_sets": cluster_sets,
            "category": [ev["category"][groups[h][0]] for h in keys],
            "side": [int(round(np.mean([ev["side"][j] for j in groups[h]]))) for h in keys],
            "resp_len": [float(np.nanmean([ev["resp_len"][j] for j in groups[h]])) for h in keys],
            "traj": [ev["traj"][groups[h][0]] for h in keys], "source": ev["source"] + "+dedup",
            "n_events_per_prompt": [len(groups[h]) for h in keys]}


# ----------------------------------------------------------------------------------------------
# score matrices
# ----------------------------------------------------------------------------------------------
def cosine_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
    return An @ Bn.T


def build_scores(ev: dict, gate: dict, sel: dict, seed: int, fh=None) -> tuple[dict, dict]:
    psi = ev["psi"]
    ids = gate["ids"]
    n = psi.shape[0]
    K, lam = int(sel["best"]["K"]), float(sel["best"]["lambda_z"])
    # cluster centroids from the fingerprints (compare against clusters.json)
    cent = np.stack([psi[[j for j in range(n) if ev["cluster"][j] == i]].mean(axis=0) for i in ids])
    ref = np.asarray([gate["clusters"][str(i)]["centroid"] for i in ids], dtype=float)
    cent_cos = [float(np.dot(cent[a], ref[a]) / (np.linalg.norm(cent[a]) * np.linalg.norm(ref[a]) + 1e-12))
                for a in range(len(ids))] if ref.shape[1] == psi.shape[1] else None
    members = {i: [j for j in range(n) if ev["cluster"][j] == i] for i in ids}
    info = {"K": K, "lambda_z": lam, "centroid_cos_vs_clusters_json": cent_cos}

    fd = atoms.fit(psi, K, lam, seed=seed)
    Zev = atoms.encode(fd.U, psi, lam)
    Zc = atoms.encode(fd.U, cent, lam)
    Zmean = np.stack([Zev[members[i]].mean(axis=0) for i in ids])
    S = {}
    S["sparse_dict"] = atoms.transfer_prediction(fd.U, Zc, Zev)
    S["sparse_dict_meanz"] = atoms.transfer_prediction(fd.U, Zmean, Zev)
    S["sparse_dict_cos"] = cosine_rows(Zc @ fd.U.T, Zev @ fd.U.T)
    info["sparse_fit"] = fd.summary()
    info["sparse_coverage_all"] = atoms.reconstruction_coverage(fd.U, psi)
    info["sparse_event_nnz_mean"] = float((np.abs(Zev) > 1e-10).sum(axis=1).mean())
    info["sparse_centroid_nnz"] = [int((np.abs(z) > 1e-10).sum()) for z in Zc]
    # PCA at the same K
    pca_c = atoms.fit_pca(psi, K, center=True)
    S["pca_cos"] = cosine_rows(atoms.encode_pca(pca_c, cent), atoms.encode_pca(pca_c, psi))
    S["pca_dot"] = atoms.encode_pca(pca_c, cent) @ atoms.encode_pca(pca_c, psi).T
    pca_u = atoms.fit_pca(psi, K, center=False)
    S["pca_cos_uncentred"] = cosine_rows(atoms.encode_pca(pca_u, cent), atoms.encode_pca(pca_u, psi))
    info["pca_coverage_all"] = atoms.reconstruction_coverage(pca_u["U"], psi)
    # raw cosine
    S["raw_cos"] = cosine_rows(cent, psi)
    # k-means dictionary at the same K
    km = atoms.fit_kmeans(psi, K, seed=seed)
    S["kmeans_dict"] = atoms.transfer_prediction(km["U"], atoms.encode_kmeans(km, cent), atoms.encode_kmeans(km, psi))
    info["kmeans_coverage_all"] = atoms.reconstruction_coverage(km["U"], psi)
    # random unit-norm dictionary at the same K and lambda
    Ur = atoms.random_dictionary(psi.shape[1], K, seed=seed + 1)
    # random atoms are ~orthogonal to the data in 8192-d, so lasso codes at lambda>0 are all zero;
    # use the unregularised projection onto the random K-subspace (least-squares codes) instead
    S["random_dict"] = atoms.transfer_prediction(Ur, atoms.least_squares_codes(Ur, cent), atoms.least_squares_codes(Ur, psi))
    info["random_dict_coverage_all"] = atoms.reconstruction_coverage(Ur, psi)
    # category Jaccard and same side
    cats_i = {i: set(ev["category"][j] for j in members[i]) for i in ids}
    S["category_jaccard"] = np.array([[len(cats_i[i] & {ev["category"][j]}) / len(cats_i[i] | {ev["category"][j]})
                                       for j in range(n)] for i in ids], dtype=float)
    side_i = {i: int(round(np.mean([ev["side"][j] for j in members[i]]))) for i in ids}
    S["same_side"] = np.array([[1.0 if side_i[i] == ev["side"][j] else 0.0 for j in range(n)] for i in ids])
    rng = np.random.default_rng(seed + 7)
    S["random"] = rng.standard_normal((len(ids), n))
    info["cluster_sides"] = {str(i): side_i[i] for i in ids}
    info["cluster_categories"] = {str(i): sorted(cats_i[i]) for i in ids}
    log(f"scores built: K={K} lambda={lam} coverage sparse={info['sparse_coverage_all']:.3f} "
        f"pca={info['pca_coverage_all']:.3f} kmeans={info['kmeans_coverage_all']:.3f} "
        f"random={info['random_dict_coverage_all']:.3f}; mean nnz/event={info['sparse_event_nnz_mean']:.2f}", fh)
    return S, info


# ----------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def _centre(M: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    Mm = np.where(mask, M, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(Mm, axis=axis, keepdims=True)
    return M - np.nan_to_num(mu)


def _ndcg(scores: np.ndarray, rel: np.ndarray, k: int) -> float:
    if rel.max() <= 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")[:k]
    disc = 1.0 / np.log2(np.arange(2, 2 + k))
    dcg = float(np.sum(rel[order] * disc[: len(order)]))
    ideal = np.sort(rel)[::-1][:k]
    idcg = float(np.sum(ideal * disc[: len(ideal)]))
    return dcg / idcg if idcg > 0 else float("nan")


def _auroc(scores: np.ndarray, y: np.ndarray) -> float:
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def _best_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Threshold (predict positive if score >= t) maximising balanced accuracy."""
    order = np.argsort(-scores, kind="stable")
    s, yy = scores[order], y[order]
    npos, nneg = max(int((yy == 1).sum()), 1), max(int((yy == 0).sum()), 1)
    tp = np.cumsum(yy == 1)
    fp = np.cumsum(yy == 0)
    bal = 0.5 * (tp / npos + (nneg - fp) / nneg)
    # only cut between distinct score values
    ok = np.r_[s[:-1] != s[1:], True]
    bal = np.where(ok, bal, -1)
    b = int(np.argmax(bal))
    if bal[b] <= 0.5 and (yy == 1).sum() > 0:
        pass
    return float(s[b])


def evaluate(S: np.ndarray, G: np.ndarray, off: np.ndarray, in_mask: np.ndarray | None = None) -> dict:
    """All metrics for one score matrix on the given pair mask.  Rows = trained clusters."""
    n_cl = S.shape[0]
    out = {}
    m = off
    s, g = S[m], G[m]
    out["n_pairs"] = int(m.sum())
    out["spearman_offdiag"] = _spearman(s, g)
    valid = ~np.isnan(G)
    Gc, Sc = _centre(G, valid, 0), _centre(S, valid, 0)          # per-event (column) centring
    out["spearman_colcentred"] = _spearman(Sc[m], Gc[m])
    Gr, Sr = _centre(G, m, 1), _centre(S, m, 1)                  # per-cluster (row) centring
    out["spearman_rowcentred"] = _spearman(Sr[m], Gr[m])
    Gd, Sd = _centre(Gc, m, 1), _centre(Sc, m, 1)                # both (removes event and cluster main effects)
    out["spearman_doublecentred"] = _spearman(Sd[m], Gd[m])
    # NDCG of positive transfer per trained cluster
    for k in (1, 3, 5):
        vals = []
        for i in range(n_cl):
            idx = np.where(m[i])[0]
            if len(idx) < k:
                continue
            vals.append(_ndcg(S[i, idx], np.maximum(G[i, idx], 0.0), k))
        out[f"ndcg@{k}"] = float(np.nanmean(vals)) if vals and not np.all(np.isnan(vals)) else float("nan")
    # LOCO scalar calibration
    se_pred, se_zero, se_const = [], [], []
    for i in range(n_cl):
        tr = m.copy(); tr[i] = False
        te = np.zeros_like(m); te[i] = m[i]
        if tr.sum() < 4 or te.sum() == 0:
            continue
        st, gt = S[tr], G[tr]
        eta = float(np.dot(st, gt) / max(np.dot(st, st), 1e-12))
        pred = eta * S[te]
        se_pred.append((pred - G[te]) ** 2)
        se_zero.append(G[te] ** 2)
        se_const.append((gt.mean() - G[te]) ** 2)
    if se_pred:
        mse = float(np.concatenate(se_pred).mean())
        mse0 = float(np.concatenate(se_zero).mean())
        msec = float(np.concatenate(se_const).mean())
        out.update({"mse_calibrated": mse, "mse_zero": mse0, "mse_const": msec,
                    "mse_ratio_calibrated": mse / mse0 if mse0 > 0 else float("nan")})
    else:
        out.update({"mse_calibrated": float("nan"), "mse_zero": float("nan"), "mse_const": float("nan"),
                    "mse_ratio_calibrated": float("nan")})
    # sign of transfer (zeros excluded)
    nz = m & (G != 0)
    y = (G > 0).astype(int)
    out["n_sign_pairs"] = int(nz.sum())
    out["frac_positive"] = float(y[nz].mean()) if nz.sum() else float("nan")
    out["auroc_sign"] = _auroc(S[nz], y[nz]) if nz.sum() else float("nan")
    tp = tn = fp = fn = 0
    for i in range(n_cl):
        tr = nz.copy(); tr[i] = False
        te = np.zeros_like(nz); te[i] = nz[i]
        if te.sum() == 0 or tr.sum() < 4 or len(np.unique(y[tr])) < 2:
            continue
        t = _best_threshold(S[tr], y[tr])
        pred = (S[te] >= t).astype(int)
        yt = y[te]
        tp += int(((pred == 1) & (yt == 1)).sum()); tn += int(((pred == 0) & (yt == 0)).sum())
        fp += int(((pred == 1) & (yt == 0)).sum()); fn += int(((pred == 0) & (yt == 1)).sum())
    if tp + fn > 0 and tn + fp > 0:
        out["balanced_acc_sign"] = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
    else:
        out["balanced_acc_sign"] = float("nan")
    return out


def partial_spearman(S: np.ndarray, G: np.ndarray, off: np.ndarray, X: np.ndarray) -> float:
    """Spearman partial correlation: residualise the ranks of S and G on covariates X (with intercept)."""
    s, g = rankdata(S[off]), rankdata(G[off])
    Xc = np.column_stack([np.ones(len(s)), X])
    beta_s, *_ = np.linalg.lstsq(Xc, s, rcond=None)
    beta_g, *_ = np.linalg.lstsq(Xc, g, rcond=None)
    rs, rg = s - Xc @ beta_s, g - Xc @ beta_g
    if np.std(rs) < 1e-9 or np.std(rg) < 1e-9:
        return float("nan")
    return float(np.corrcoef(rs, rg)[0, 1])


def covariates(ev: dict, gate: dict, S: dict, off: np.ndarray) -> np.ndarray:
    n_cl, n = off.shape
    cats = sorted(set(ev["category"]))
    cat_idx = {c: k for k, c in enumerate(cats)}
    rows = []
    for i in range(n_cl):
        for j in range(n):
            if not off[i, j]:
                continue
            oh_cat = np.zeros(len(cats)); oh_cat[cat_idx[ev["category"][j]]] = 1.0
            oh_cl = np.zeros(n_cl); oh_cl[i] = 1.0
            rl = ev["resp_len"][j]
            rows.append(np.concatenate([oh_cat[1:], oh_cl[1:], [math.log1p(rl) if rl == rl else 0.0],
                                        [S["same_side"][i, j], S["category_jaccard"][i, j]]]))
    return np.asarray(rows)


def bootstrap(S: dict, G: np.ndarray, off: np.ndarray, hashes: list, n_boot: int, seed: int,
              methods: list) -> dict:
    """Group bootstrap over prompt hashes (events sharing a prompt share a gain)."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for j, h in enumerate(hashes):
        if off[:, j].any() or (~np.isnan(G[:, j])).any():
            groups[h].append(j)
    keys = list(groups)
    stats = {mname: {k: [] for k in BOOT_METRICS} for mname in methods}
    for _ in range(n_boot):
        pick = rng.integers(len(keys), size=len(keys))
        cols = np.concatenate([groups[keys[p]] for p in pick])
        Gb, offb = G[:, cols], off[:, cols]
        for mname in methods:
            r = evaluate(S[mname][:, cols], Gb, offb)
            for k in BOOT_METRICS:
                stats[mname][k].append(r.get(k, float("nan")))
    out = {}
    for mname in methods:
        out[mname] = {}
        for k in BOOT_METRICS:
            v = np.asarray(stats[mname][k], dtype=float)
            v = v[~np.isnan(v)]
            out[mname][k] = {"lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)),
                             "sd": float(v.std()), "n": int(len(v))} if len(v) else None
    return out


def permutation_test(S: dict, G: np.ndarray, off: np.ndarray, n_perm: int, seed: int, methods: list,
                     observed: dict) -> dict:
    """Label permutation: shuffle which feature row (score column) is attached to which measured event."""
    rng = np.random.default_rng(seed)
    n = G.shape[1]
    null = {m: {"spearman_offdiag": [], "spearman_colcentred": [], "ndcg@3": [], "auroc_sign": []} for m in methods}
    for _ in range(n_perm):
        perm = rng.permutation(n)
        for m in methods:
            r = evaluate(S[m][:, perm], G, off)
            for k in null[m]:
                null[m][k].append(r[k])
    out = {}
    for m in methods:
        out[m] = {}
        for k, vals in null[m].items():
            v = np.asarray(vals, dtype=float); v = v[~np.isnan(v)]
            obs = observed[m].get(k, float("nan"))
            out[m][k] = {"null_mean": float(v.mean()) if len(v) else None,
                         "null_sd": float(v.std()) if len(v) else None,
                         "p_value_ge_obs": float(np.mean(v >= obs)) if len(v) and obs == obs else None}
    return out


# ----------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------
def run_level(ev: dict, gate: dict, sel: dict, args, fh, label: str) -> dict:
    G, off = gain_matrix(ev, gate)
    if "cluster_sets" in ev:  # prompt-level: off-diagonal if cluster i is not among the prompt's clusters
        off = (~np.isnan(G)) & np.array([[i not in cs for cs in ev["cluster_sets"]] for i in gate["ids"]])
    log(f"[{label}] events={len(ev['hash'])} with_gain={int((~np.isnan(G)).any(axis=0).sum())} "
        f"offdiag_pairs={int(off.sum())} in_cluster_pairs={int(((~np.isnan(G)) & ~off).sum())}", fh)
    S, info = build_scores(ev, gate, sel, args.seed, fh)
    methods = list(S)
    res = {m: evaluate(S[m], G, off) for m in methods}
    X = covariates(ev, gate, S, off)
    for m in methods:
        res[m]["partial_spearman_ctrl"] = partial_spearman(S[m], G, off, X)
    log(f"[{label}] bootstrap x{args.n_boot} ...", fh)
    ci = bootstrap(S, G, off, ev["hash"], args.n_boot, args.seed, methods)
    log(f"[{label}] permutation x{args.n_perm} ...", fh)
    perm = permutation_test(S, G, off, args.n_perm, args.seed + 1, methods, res)
    valid = ~np.isnan(G)
    ctx = {"mean_gain_in_cluster": float(np.nanmean(G[valid & ~off])) if (valid & ~off).any() else None,
           "mean_gain_off_cluster": float(np.nanmean(G[off])) if off.any() else None,
           "frac_offdiag_positive": float(np.mean(G[off] > 0)), "frac_offdiag_zero": float(np.mean(G[off] == 0)),
           "frac_offdiag_negative": float(np.mean(G[off] < 0)),
           "per_cluster": {str(i): {"n_off_events": int(off[a].sum()), "n_in_events": int((valid[a] & ~off[a]).sum()),
                                    "mean_gain_off": float(np.nanmean(G[a, off[a]])) if off[a].any() else None,
                                    "mean_gain_in": float(np.nanmean(G[a, valid[a] & ~off[a]])) if (valid[a] & ~off[a]).any() else None}
                           for a, i in enumerate(gate["ids"])}}
    return {"info": info, "metrics": res, "bootstrap_ci": ci, "permutation": perm, "gain_context": ctx,
            "methods": methods, "n_events": len(ev["hash"]), "n_offdiag_pairs": int(off.sum())}


def sensitivity_sweep(ev: dict, gate: dict, sel: dict, args, fh) -> list:
    """Post-hoc: transfer metrics of sparse_dict for every (K, lambda) fit on all events.  NOT used
    for selection (would tune on the test bed); reported to show how fragile the headline is."""
    G, off = gain_matrix(ev, gate)
    psi = ev["psi"]
    ids = gate["ids"]
    n = psi.shape[0]
    cent = np.stack([psi[[j for j in range(n) if ev["cluster"][j] == i]].mean(axis=0) for i in ids])
    out = []
    for g in sel["grid"]:
        K, lam = int(g["K"]), float(g["lambda_z"])
        fd = atoms.fit(psi, K, lam, seed=args.seed)
        P = atoms.transfer_prediction(fd.U, atoms.encode(fd.U, cent, lam), atoms.encode(fd.U, psi, lam))
        r = evaluate(P, G, off)
        pca = atoms.fit_pca(psi, K, center=True)
        rp = evaluate(cosine_rows(atoms.encode_pca(pca, cent), atoms.encode_pca(pca, psi)), G, off)
        out.append({"K": K, "lambda_z": lam, "mean_nnz_per_event": float(fd.nnz_per_event.mean()),
                    "sparse_dict": {k: r[k] for k in ("spearman_offdiag", "spearman_colcentred", "spearman_doublecentred", "ndcg@3", "auroc_sign", "mse_ratio_calibrated")},
                    "pca_cos": {k: rp[k] for k in ("spearman_offdiag", "spearman_colcentred", "spearman_doublecentred", "ndcg@3", "auroc_sign", "mse_ratio_calibrated")}})
        log(f"  sweep K={K:2d} lam={lam:<5} nnz={fd.nnz_per_event.mean():4.2f} sparse rho={r['spearman_offdiag']:+.3f} "
            f"col={r['spearman_colcentred']:+.3f} ndcg3={r['ndcg@3']:.3f} auroc={r['auroc_sign']:.3f} | "
            f"pca rho={rp['spearman_offdiag']:+.3f} col={rp['spearman_colcentred']:+.3f}", fh)
    return out


def fmt_table(level: dict) -> str:
    cols = ["spearman_offdiag", "spearman_colcentred", "spearman_rowcentred", "spearman_doublecentred", "ndcg@1", "ndcg@3", "ndcg@5",
            "mse_ratio_calibrated", "auroc_sign", "balanced_acc_sign", "partial_spearman_ctrl"]
    short = ["rho_off", "rho_col", "rho_row", "rho_dbl", "ndcg@1", "ndcg@3", "ndcg@5", "mse/mse0", "auroc", "bal_acc", "rho_partial"]
    lines = [f"{'method':<20}" + "".join(f"{c:>22}" for c in short) + f"{'perm_p':>8}"]
    for m in level["methods"]:
        r, ci = level["metrics"][m], level["bootstrap_ci"][m]
        cells = []
        for c in cols:
            v = r.get(c, float("nan"))
            if c in ci and ci[c]:
                cells.append(f"{v:+.3f} [{ci[c]['lo']:+.2f},{ci[c]['hi']:+.2f}]")
            else:
                cells.append(f"{v:+.3f}" if v == v else "   nan")
        p = level["permutation"][m]["spearman_offdiag"]["p_value_ge_obs"]
        lines.append(f"{m:<20}" + "".join(f"{c:>22}" for c in cells) + f"{p if p is not None else float('nan'):>8.3f}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="results/analysis/crcd_fingerprints_v4.json",
                    help="fingerprints json (gate.events[*].a3) or .npz with psi/state_hash")
    ap.add_argument("--fingerprints-json", default="results/analysis/crcd_fingerprints_v4.json",
                    help="metadata source (cluster labels) when --features is an .npz")
    ap.add_argument("--gate", default="data/bfcl_sft/gate_v2")
    ap.add_argument("--pool", default="data/bfcl_sft/pool_events_pref_v2.jsonl")
    ap.add_argument("--out", default="results/analysis/atoms_transfer_gate_v2.json")
    ap.add_argument("--log", default="logs/t2_atoms.log")
    ap.add_argument("--ks", default="4,8,16,32")
    ap.add_argument("--lambdas", default="0,0.01,0.02,0.05,0.1,0.2")
    ap.add_argument("--criterion", default="sparse_recon_1se", choices=["sparse_recon_1se", "sparse_recon", "coverage"])
    ap.add_argument("--dev-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--no-dedup", action="store_true")
    args = ap.parse_args()

    (ROOT / args.log).parent.mkdir(parents=True, exist_ok=True)
    fh = open(ROOT / args.log, "a")
    log(f"=== atoms_transfer_eval start: features={args.features} gate={args.gate} seed={args.seed}", fh)
    ev = load_events(ROOT / args.features, ROOT / args.pool, ROOT / args.fingerprints_json)
    gate = load_gate(ROOT / args.gate)
    n_gain = sum(h in gate["base"] for h in ev["hash"])
    log(f"events={len(ev['hash'])} dim={ev['psi'].shape[1]} source={ev['source']} with_gain={n_gain} "
        f"unique_prompts={len(set(ev['hash']))} abstain_events={sum(1 - s for s in ev['side'])}", fh)

    Ks = [int(x) for x in args.ks.split(",")]
    lams = [float(x) for x in args.lambdas.split(",")]
    log(f"model selection (label-free, {args.criterion}) over K={Ks} lambda={lams} ...", fh)
    sel = atoms.select_model(ev["psi"], Ks, lams, seed=args.seed, dev_frac=args.dev_frac,
                             groups=ev["hash"], criterion=args.criterion)
    for g in sel["grid"]:
        log(f"  K={g['K']:2d} lam={g['lambda_z']:<5} dev_cov={g['dev_coverage']:.3f} dev_sparse_err={g['dev_sparse_recon_error']:.3f}"
            f"±{g['dev_sparse_recon_se']:.3f} nnz={g['mean_nnz_per_event']:.2f}", fh)
    log(f"selected K={sel['best']['K']} lambda={sel['best']['lambda_z']} (argmin K={sel['argmin']['K']} lambda={sel['argmin']['lambda_z']})", fh)
    sel_out = {k: v for k, v in sel.items() if k not in ("train_idx", "dev_idx")}

    out = {"features": str(args.features), "feature_source": ev["source"], "gate": args.gate, "seed": args.seed,
           "feature_note": ("A3 = Adam-preconditioned, JL-projected (64/module x 128 LoRA-B modules), L2-normalised "
                            "differential LoRA-B gradients; NOT Fisher-whitened. Rerun with --features <psi.npz> for Fisher-whitened psi."),
           "n_events": len(ev["hash"]), "dim": int(ev["psi"].shape[1]), "n_events_with_gain": n_gain,
           "n_unique_prompts": len(set(ev["hash"])), "selection": sel_out}
    out["event_level"] = run_level(ev, gate, sel, args, fh, "event")
    print("\n== event level (off-diagonal pairs; 95% group-bootstrap CI over prompts; perm_p = label-permutation p for rho_off)")
    print(fmt_table(out["event_level"]))
    if not args.no_dedup:
        evd = dedup_by_prompt(ev)
        out["prompt_level"] = run_level(evd, gate, sel, args, fh, "prompt")
        print("\n== prompt level (events sharing a prompt merged)")
        print(fmt_table(out["prompt_level"]))
    if not args.no_sweep:
        log("post-hoc (K, lambda) sensitivity sweep (not used for selection):", fh)
        out["sensitivity_sweep"] = sensitivity_sweep(ev, gate, sel, args, fh)
    (ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out).write_text(json.dumps(out, indent=1, default=float))
    fh.write("\n" + fmt_table(out["event_level"]) + "\n")
    if "prompt_level" in out:
        fh.write("\n[prompt level]\n" + fmt_table(out["prompt_level"]) + "\n")
    log(f"wrote {args.out}", fh)
    fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
