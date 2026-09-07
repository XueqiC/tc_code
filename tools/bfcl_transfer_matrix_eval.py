#!/usr/bin/env python3
"""Evaluate transfer predictors against the measured single-event transfer matrix (CRCD Block I, T6).

Target: T[i, j] = change of event j's per-token margin (log pi(y^T_j|s_j)/n - log pi(y^S_j|s_j)/n) after
S optimizer steps on event i alone (tools/bfcl_transfer_matrix.py).  Pairs = off-diagonal (i != j) with
row i measured.

Predictors (all (n, n), rows = updated event i, columns = affected event j):
  psi_dot        psi_i . psi_j            Fisher-whitened fingerprints (data/fingerprints/bfcl_r2_v1_meta.npz)
  raw_dot        unwhitened sketch_diff_raw dot product
  sparse_dict    z_j^T U^T U z_i, sparse dictionary (src/bfas/atoms.fit on psi, K=32, lambda=0.02, seed 0
                 = the selection in results/analysis/atoms_fit_gate_v2_fisher.json)
  pca_cos        cosine of the PCA-K codes (K = 32, centred PCA)
  kmeans_same    same k-means cluster on psi (K = 8, seeded)
  gate_cluster   same gate-v2 cluster (npz key 'cluster')
  category_same  same _seed_category
  side_same      both call / both abstain ('<tool_call>' in the teacher response)
  prompt_same    same prompt hash
  random         seeded Gaussian noise

Metrics on off-diagonal pairs: Spearman (raw, row-centred, column-centred, double-centred), sign AUROC
(T_ij > 0), NDCG@5 per row (relevance = max(T_ij, 0)); the same on the different-prompt subset.
Two-way group bootstrap by prompt (rows and columns resampled from the same prompt draw) for CIs; a
permutation test that applies one event permutation to both axes of the predictor (500).  Structure of T:
symmetry corr(T_ij, T_ji), spectrum of the measured square block.  Cluster aggregate: mean_{i in c} T_ij
averaged per prompt vs the gate-v2 cluster -> prompt pass-rate gains (data/bfcl_sft/gate_v2).

Usage:
  PYTHONPATH=src .venv/bin/python tools/bfcl_transfer_matrix_eval.py \
      --matrix results/analysis/transfer_matrix_bfcl_r2.npz --out results/analysis/transfer_matrix_bfcl_r2_eval.json

ALFWorld (T9, 2026-09-04): the same evaluation on results/analysis/transfer_matrix_alfworld_v1.npz with
--fingerprints data/fingerprints/alfworld_v1_v1.npz (aligned by the sha1 of the prompt = ``state_hash``;
no gate clusters, no arm gains).  Extra label predictors when the matrix meta carries them:
  task_same      same game / task id (meta 'task_id')
  teacher_cmd_same  same teacher command (meta 'teacher_command' = the pool's _event_good)
  turn_same      same turn depth (meta 'event_turn')
  conseq_both    both events consequential (meta 'dU' != 0 for i and j)
--fit-json may point at results/analysis/atoms_fit_alfworld_fisher.json (K, lambda for the sparse dictionary);
--group-by task groups the bootstrap by task id instead of prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr, pearsonr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bfas import atoms  # noqa: E402

MAIN_METRICS = ("spearman_offdiag", "spearman_rowcentred", "spearman_colcentred", "spearman_doublecentred",
                "auroc_sign", "ndcg@5")
BOOT_METRICS = ("spearman_offdiag", "spearman_doublecentred", "auroc_sign", "ndcg@5")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------
# statistics (pure numpy; tested on CPU)
# ----------------------------------------------------------------------------------------------
def offdiag_mask(done: np.ndarray, n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    m[np.asarray(done, dtype=bool)] = True
    m[np.diag_indices(n)] = False
    return m


def centre(M: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    """Subtract the masked mean along ``axis`` (axis=1: row means; axis=0: column means)."""
    Mm = np.where(mask, M, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(Mm, axis=axis, keepdims=True)
    return M - np.nan_to_num(mu)


def double_centre(M: np.ndarray, mask: np.ndarray, n_iter: int = 20) -> np.ndarray:
    """Alternating row/column centring on the masked entries (exact for a full mask, iterative otherwise)."""
    X = M.astype(np.float64)
    for _ in range(n_iter):
        X = centre(X, mask, 1)
        X = centre(X, mask, 0)
    return X


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).correlation)


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def ndcg(scores: np.ndarray, rel: np.ndarray, k: int) -> float:
    if rel.max() <= 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")[:k]
    disc = 1.0 / np.log2(np.arange(2, 2 + k))
    dcg = float(np.sum(rel[order] * disc[: len(order)]))
    ideal = np.sort(rel)[::-1][:k]
    idcg = float(np.sum(ideal * disc[: len(ideal)]))
    return dcg / idcg if idcg > 0 else float("nan")


def evaluate(S: np.ndarray, T: np.ndarray, mask: np.ndarray, jitter_seed: int = 0) -> dict:
    """All pair metrics of predictor S against target T on the boolean pair mask."""
    out = {"n_pairs": int(mask.sum())}
    if mask.sum() < 4:
        return {**out, **{m: float("nan") for m in MAIN_METRICS}}
    s, t = S[mask], T[mask]
    out["spearman_offdiag"] = spearman(s, t)
    Tr, Sr = centre(T, mask, 1), centre(S, mask, 1)
    out["spearman_rowcentred"] = spearman(Sr[mask], Tr[mask])
    Tc, Sc = centre(T, mask, 0), centre(S, mask, 0)
    out["spearman_colcentred"] = spearman(Sc[mask], Tc[mask])
    Td, Sd = double_centre(T, mask), double_centre(S, mask)
    out["spearman_doublecentred"] = spearman(Sd[mask], Td[mask])
    out["auroc_sign"] = auroc(s, (t > 0).astype(int))
    out["frac_positive"] = float(np.mean(t > 0))
    # NDCG@5 per measured row; ties in the predictor are broken by a tiny seeded jitter
    rng = np.random.default_rng(jitter_seed)
    vals = []
    for i in np.where(mask.any(axis=1))[0]:
        idx = np.where(mask[i])[0]
        if len(idx) < 5:
            continue
        sc = S[i, idx] + 1e-9 * rng.standard_normal(len(idx)) * max(float(np.std(S[i, idx])), 1e-6)
        vals.append(ndcg(sc, np.maximum(T[i, idx], 0.0), 5))
    out["ndcg@5"] = float(np.nanmean(vals)) if vals and not np.all(np.isnan(vals)) else float("nan")
    return out


def symmetry_stats(T: np.ndarray, done: np.ndarray) -> dict:
    """corr(T_ij, T_ji) over pairs i<j with both rows measured (diagonal ignored)."""
    idx = np.where(np.asarray(done, dtype=bool))[0]
    a, b = [], []
    for p, i in enumerate(idx):
        for j in idx[p + 1:]:
            if np.isfinite(T[i, j]) and np.isfinite(T[j, i]):
                a.append(T[i, j]); b.append(T[j, i])
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 4:
        return {"n_pairs": int(len(a)), "pearson_T_ij_vs_T_ji": float("nan"), "spearman_T_ij_vs_T_ji": float("nan")}
    B = T[np.ix_(idx, idx)].astype(np.float64)
    off = ~np.eye(len(idx), dtype=bool)
    B0 = np.where(off, B, 0.0)
    sym = 0.5 * (B0 + B0.T); anti = 0.5 * (B0 - B0.T)
    return {"n_pairs": int(len(a)),
            "pearson_T_ij_vs_T_ji": float(pearsonr(a, b)[0]),
            "spearman_T_ij_vs_T_ji": float(spearmanr(a, b).correlation),
            "sign_agreement": float(np.mean(np.sign(a) == np.sign(b))),
            "frac_norm_symmetric_part": float(np.sum(sym ** 2) / max(np.sum(B0 ** 2), 1e-30)),
            "frac_norm_antisymmetric_part": float(np.sum(anti ** 2) / max(np.sum(B0 ** 2), 1e-30))}


def spectrum_stats(T: np.ndarray, done: np.ndarray, ranks=(1, 2, 3, 5, 10, 20)) -> dict:
    """Singular spectrum of the measured square block (diagonal set to the row mean of the off-diagonal)."""
    idx = np.where(np.asarray(done, dtype=bool))[0]
    B = T[np.ix_(idx, idx)].astype(np.float64)
    off = ~np.eye(len(idx), dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rowmean = np.nanmean(np.where(off, B, np.nan), axis=1)
    B = np.where(off, B, rowmean[:, None])
    B = np.nan_to_num(B)
    s = np.linalg.svd(B, compute_uv=False)
    e = s ** 2
    tot = float(e.sum()) if e.sum() > 0 else 1.0
    cum = np.cumsum(e) / tot
    pr = float(e.sum() ** 2 / max(np.sum(e ** 2), 1e-30))
    # noise reference: same-size Gaussian matrix with matched Frobenius norm
    rng = np.random.default_rng(0)
    N = rng.standard_normal(B.shape); N *= np.linalg.norm(B) / np.linalg.norm(N)
    sn = np.linalg.svd(N, compute_uv=False)
    cn = np.cumsum(sn ** 2) / max(float(np.sum(sn ** 2)), 1e-30)
    return {"n": int(len(idx)), "singular_values_top10": [float(x) for x in s[:10]],
            "frac_norm_rank": [float(cum[r - 1]) if r <= len(cum) else 1.0 for r in ranks],
            "frac_norm_rank_ranks": list(ranks),
            "frac_norm_rank_gaussian_reference": [float(cn[r - 1]) if r <= len(cn) else 1.0 for r in ranks],
            "participation_ratio": pr,
            "rank1_row_col_model": _rank1_fit(B, off)}


def _rank1_fit(B: np.ndarray, off: np.ndarray) -> dict:
    """How much of T is explained by additive row + column effects (T_ij ~ a_i + b_j)?"""
    X = np.where(off, B, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(X)
        a = np.nanmean(X, axis=1) - mu
        b = np.nanmean(X, axis=0) - mu
    pred = mu + a[:, None] + b[None, :]
    resid = np.where(off, X - pred, 0.0)
    tot = np.where(off, X - mu, 0.0)
    return {"r2_additive_row_plus_col": float(1.0 - np.sum(resid ** 2) / max(np.sum(tot ** 2), 1e-30)),
            "r2_row_only": float(1.0 - np.sum(np.where(off, X - mu - a[:, None], 0.0) ** 2) / max(np.sum(tot ** 2), 1e-30)),
            "r2_col_only": float(1.0 - np.sum(np.where(off, X - mu - b[None, :], 0.0) ** 2) / max(np.sum(tot ** 2), 1e-30))}


def group_bootstrap(S: dict, T: np.ndarray, done: np.ndarray, groups: np.ndarray, n_boot: int, seed: int,
                    metrics=BOOT_METRICS) -> dict:
    """Two-way group bootstrap by prompt: resample prompt groups with replacement; rows and columns of the
    resampled matrix are the events of the drawn groups (measured rows only for the row side)."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    n = len(groups)
    keys = sorted(set(groups.tolist()))
    members = {k: np.where(groups == k)[0] for k in keys}
    done = np.asarray(done, dtype=bool)
    stats = {m: {k: [] for k in metrics} for m in S}
    for _ in range(n_boot):
        pick = rng.integers(len(keys), size=len(keys))
        cols = np.concatenate([members[keys[p]] for p in pick])
        rows = cols[done[cols]]
        if len(rows) < 2:
            continue
        Tb = T[np.ix_(rows, cols)]
        # a row's own copy (same event index) is the diagonal; other copies of the same event are kept
        mask = np.ones(Tb.shape, dtype=bool)
        mask[rows[:, None] == cols[None, :]] = False
        for m in S:
            r = evaluate(S[m][np.ix_(rows, cols)], Tb, mask)
            for k in metrics:
                stats[m][k].append(r.get(k, float("nan")))
    out = {}
    for m in S:
        out[m] = {}
        for k in metrics:
            v = np.asarray(stats[m][k], dtype=float); v = v[~np.isnan(v)]
            out[m][k] = ({"lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)),
                          "sd": float(v.std()), "n": int(len(v))} if len(v) else None)
    return out


def permutation_test(S: dict, T: np.ndarray, done: np.ndarray, n_perm: int, seed: int,
                     metrics=BOOT_METRICS) -> dict:
    """Null: one random event permutation applied to BOTH axes of the predictor (keeps its structure,
    breaks the alignment with the measured events); one-sided p for larger-is-better metrics."""
    rng = np.random.default_rng(seed)
    n = T.shape[0]
    mask = offdiag_mask(done, n)
    obs = {m: evaluate(S[m], T, mask) for m in S}
    null = {m: {k: [] for k in metrics} for m in S}
    for _ in range(n_perm):
        perm = rng.permutation(n)
        for m in S:
            r = evaluate(S[m][np.ix_(perm, perm)], T, mask)
            for k in metrics:
                null[m][k].append(r.get(k, float("nan")))
    out = {}
    for m in S:
        out[m] = {}
        for k in metrics:
            v = np.asarray(null[m][k], dtype=float); v = v[~np.isnan(v)]
            o = obs[m].get(k, float("nan"))
            out[m][k] = {"observed": o, "null_mean": float(v.mean()) if len(v) else None,
                         "null_sd": float(v.std()) if len(v) else None,
                         "p": float((1 + np.sum(v >= o)) / (1 + len(v))) if len(v) and o == o else None}
    return out


# ----------------------------------------------------------------------------------------------
# predictors
# ----------------------------------------------------------------------------------------------
def _has_labels(meta: dict, key: str) -> bool:
    v = meta.get(key)
    return isinstance(v, list) and len(v) > 0 and all(x is not None for x in v)


def cosine_rows(A: np.ndarray) -> np.ndarray:
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    return An @ An.T


def build_predictors(psi: np.ndarray, raw: np.ndarray, meta: dict, fp: dict, K: int, lam: float,
                     seed: int, kmeans_k: int) -> tuple[dict, dict]:
    n = psi.shape[0]
    info = {}
    S = {}
    S["psi_dot"] = psi @ psi.T
    S["raw_dot"] = raw @ raw.T
    fd = atoms.fit(psi, K, lam, seed=seed)
    Z = atoms.encode(fd.U, psi, lam)
    S["sparse_dict"] = atoms.transfer_prediction(fd.U, Z, Z)          # P[i, j] = z_j^T U^T U z_i
    info["sparse_fit"] = {k: v for k, v in fd.summary().items() if k != "atom_usage"}
    info["sparse_coverage_all"] = atoms.reconstruction_coverage(fd.U, psi)
    pca = atoms.fit_pca(psi, K, center=True)
    S["pca_cos"] = cosine_rows(atoms.encode_pca(pca, psi))
    km = atoms.fit_kmeans(psi, kmeans_k, seed=seed)
    S["kmeans_same"] = (km["labels"][:, None] == km["labels"][None, :]).astype(float)
    info["kmeans_sizes"] = np.bincount(km["labels"], minlength=kmeans_k).tolist()
    if "cluster" in fp:
        cl = np.asarray(fp["cluster"])
        S["gate_cluster"] = (cl[:, None] == cl[None, :]).astype(float)
    cat = np.asarray(meta["category"])
    S["category_same"] = (cat[:, None] == cat[None, :]).astype(float)
    side = np.asarray(meta["side"])
    if len(set(side.tolist())) > 1:          # ALFWorld: every teacher continuation is a command (one side)
        S["side_same"] = (side[:, None] == side[None, :]).astype(float)
    ph = np.asarray(meta["prompt_hash"])
    S["prompt_same"] = (ph[:, None] == ph[None, :]).astype(float)
    # event-derived labels (present in meta when the pool carries them; ALFWorld)
    for key, name in (("task_id", "task_same"), ("teacher_command", "teacher_cmd_same"), ("event_turn", "turn_same")):
        if _has_labels(meta, key):
            v = np.asarray([str(x) for x in meta[key]])
            if len(set(v.tolist())) > 1:
                S[name] = (v[:, None] == v[None, :]).astype(float)
    if _has_labels(meta, "dU"):
        c = (np.asarray(meta["dU"], dtype=float) != 0).astype(float)
        if 0 < c.sum() < n:
            S["conseq_both"] = c[:, None] * c[None, :]
            info["n_consequential"] = int(c.sum())
    S["random"] = np.random.default_rng(seed + 7).standard_normal((n, n))
    # column-only covariates (label-free features of the affected event j): a long student continuation
    # has more log-prob to lose, a poor base margin has more room to improve
    if "n_tok_S" in meta:
        S["len_S_col"] = np.tile(np.asarray(meta["n_tok_S"], dtype=float)[None, :], (n, 1))
    if "base_margin" in meta:
        S["neg_base_margin_col"] = np.tile(-np.asarray(meta["base_margin"], dtype=float)[None, :], (n, 1))
    return S, info


# ----------------------------------------------------------------------------------------------
# cluster aggregate vs gate-v2 arm gains
# ----------------------------------------------------------------------------------------------
def cluster_aggregate(T: np.ndarray, done: np.ndarray, cluster: np.ndarray, hashes: list, gate: Path) -> dict:
    base = json.loads((gate / "rates_base.json").read_text())
    ids = sorted(int(c) for c in json.loads((gate / "clusters.json").read_text()))
    rates = {c: json.loads((gate / f"rates_c{c}.json").read_text()) for c in ids if (gate / f"rates_c{c}.json").is_file()}
    ids = [c for c in ids if c in rates]
    hashes = np.asarray(hashes)
    done = np.asarray(done, dtype=bool)
    uniq = sorted(set(hashes.tolist()))
    cols = {h: np.where(hashes == h)[0] for h in uniq}
    A = np.full((len(ids), len(uniq)), np.nan)   # mean single-event transfer of cluster c onto prompt p
    G = np.full((len(ids), len(uniq)), np.nan)   # measured arm gain
    incl = np.zeros((len(ids), len(uniq)), dtype=bool)  # prompt p has an event in cluster c
    n_rows = {}
    for a, c in enumerate(ids):
        rows = np.where((cluster == c) & done)[0]
        n_rows[c] = int(len(rows))
        for b, h in enumerate(uniq):
            js = cols[h]
            incl[a, b] = bool(np.any(cluster[js] == c))
            if len(rows) and h in base and h in rates[c]:
                sub = T[np.ix_(rows, js)]
                # exclude the self pair (i == j) from the aggregate
                m = np.ones(sub.shape, dtype=bool); m[rows[:, None] == js[None, :]] = False
                if m.any():
                    A[a, b] = float(np.mean(sub[m]))
                    G[a, b] = rates[c][h] - base[h]
    valid = np.isfinite(A) & np.isfinite(G)
    off = valid & ~incl
    res = {"cluster_ids": ids, "measured_rows_per_cluster": n_rows, "n_prompts": len(uniq),
           "n_pairs_all": int(valid.sum()), "n_pairs_off_cluster": int(off.sum()),
           "frac_gain_zero": float(np.mean(G[valid] == 0)) if valid.any() else None}
    for name, m in (("all", valid), ("off_cluster", off)):
        if m.sum() < 4:
            continue
        r = {"spearman": spearman(A[m], G[m]), "pearson": float(pearsonr(A[m], G[m])[0])}
        Ac, Gc = centre(A, m, 0), centre(G, m, 0)
        r["spearman_colcentred"] = spearman(Ac[m], Gc[m])
        Ar, Gr = centre(A, m, 1), centre(G, m, 1)
        r["spearman_rowcentred"] = spearman(Ar[m], Gr[m])
        r["auroc_gain_positive"] = auroc(A[m], (G[m] > 0).astype(int))
        r["auroc_gain_negative_vs_rest"] = auroc(-A[m], (G[m] < 0).astype(int))
        # per cluster: does the ranking of prompts by A match the ranking by G?
        r["per_cluster_spearman"] = {str(c): spearman(A[a][m[a]], G[a][m[a]]) for a, c in enumerate(ids)}
        # per prompt (which cluster helps this prompt most?): only where >= 4 clusters
        pc = [spearman(A[:, b][m[:, b]], G[:, b][m[:, b]]) for b in range(len(uniq)) if m[:, b].sum() >= 4
              and np.std(G[:, b][m[:, b]]) > 0]
        r["per_prompt_spearman_mean"] = float(np.nanmean(pc)) if pc else None
        r["n_prompts_with_varying_gain"] = len(pc)
        res[name] = r
    # cluster-level: mean transfer of cluster c to everything vs mean gain of arm c
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res["cluster_level"] = {"mean_transfer": {str(c): float(np.nanmean(A[a])) for a, c in enumerate(ids)},
                                "mean_gain": {str(c): float(np.nanmean(G[a])) for a, c in enumerate(ids)}}
    mt = [res["cluster_level"]["mean_transfer"][str(c)] for c in ids]
    mg = [res["cluster_level"]["mean_gain"][str(c)] for c in ids]
    res["cluster_level"]["spearman_over_clusters"] = spearman(np.asarray(mt), np.asarray(mg))
    return res


# ----------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", type=Path, default=ROOT / "results/analysis/transfer_matrix_bfcl_r2.npz")
    ap.add_argument("--fingerprints", type=Path, default=ROOT / "data/fingerprints/bfcl_r2_v1_meta.npz")
    ap.add_argument("--fit-json", type=Path, default=ROOT / "results/analysis/atoms_fit_gate_v2_fisher.json")
    ap.add_argument("--gate", type=Path, default=ROOT / "data/bfcl_sft/gate_v2")
    ap.add_argument("--out", type=Path, default=ROOT / "results/analysis/transfer_matrix_bfcl_r2_eval.json")
    ap.add_argument("--target", default="T_mean", choices=["T_mean", "T_sum", "T_teacher", "T_student"],
                    help="T_teacher = per-token change of log p(y^T_j) alone; T_student = -(per-token change of log p(y^S_j))")
    ap.add_argument("--rows", type=int, default=None, help="restrict to the first N measured rows of the seeded row order")
    ap.add_argument("--append-key", default=None, help="merge the result into an existing --out JSON under this key")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kmeans-k", type=int, default=8)
    ap.add_argument("--min-final-loss", type=float, default=0.05, help="rows with a final pair loss above this form the non-saturated subset")
    ap.add_argument("--K", type=int, default=32, help="sparse dictionary size when --fit-json is absent")
    ap.add_argument("--lam", type=float, default=0.02, help="sparse dictionary lambda when --fit-json is absent")
    ap.add_argument("--group-by", default="prompt", choices=["prompt", "task"], help="bootstrap grouping unit")
    args = ap.parse_args(argv)

    with np.load(args.matrix, allow_pickle=True) as f:
        M = {k: f[k] for k in f.files}
    meta = json.loads(str(M["meta"]))
    if args.target == "T_teacher":
        T = (M["logp_T_after"].astype(np.float64) - M["base_logp_T"][None, :]) / M["n_tok_T"][None, :]
    elif args.target == "T_student":
        T = -(M["logp_S_after"].astype(np.float64) - M["base_logp_S"][None, :]) / M["n_tok_S"][None, :]
    else:
        T = M[args.target].astype(np.float64)
    done = M["done"].astype(bool)
    if args.rows is not None:
        keep = np.zeros_like(done); keep[M["row_order"][:args.rows]] = True
        done = done & keep
    n = T.shape[0]
    fp = atoms.load_npz(args.fingerprints)
    if "traj" in fp:
        if list(fp["traj"].astype(str)) != list(meta["traj"]):
            raise SystemExit("fingerprint rows and matrix rows are not aligned (traj mismatch)")
    else:
        # ALFWorld fingerprints carry no traj: align by the sha1 of the prompt (full hash if stored, else prefix)
        fh = [str(h) for h in fp["state_hash"]]
        mh = meta.get("state_hash") or None
        ok = (mh is not None and fh == list(mh)) or (mh is None and [h[:16] for h in fh] == list(meta["prompt_hash"]))
        if not ok or len(fh) != n:
            raise SystemExit("fingerprint rows and matrix rows are not aligned (state_hash mismatch)")
    psi = fp["psi"]
    raw = np.asarray(fp["sketch_diff_raw"], dtype=np.float64)
    if args.fit_json.is_file():
        fit = json.loads(args.fit_json.read_text())
        K, lam = int(fit["final_fit"]["K"]), float(fit["final_fit"]["lambda_z"])
    else:
        K, lam = args.K, args.lam
    log(f"matrix {args.matrix.name}: {int(done.sum())}/{n} rows measured, target {args.target}; K={K} lambda={lam}")

    mask = offdiag_mask(done, n)
    ph = np.asarray(meta["prompt_hash"])
    same_prompt = ph[:, None] == ph[None, :]
    mask_dp = mask & ~same_prompt
    cat = np.asarray(meta["category"]); side = np.asarray(meta["side"])
    t = T[mask]
    diag = np.diag(T)[done]
    desc = {"n_rows": int(done.sum()), "n_cols": n, "n_offdiag_pairs": int(mask.sum()),
            "n_diff_prompt_pairs": int(mask_dp.sum()),
            "self_effect": {"min": float(diag.min()), "median": float(np.median(diag)), "max": float(diag.max()),
                            "frac_positive": float(np.mean(diag > 0))},
            "offdiag": {"mean": float(t.mean()), "sd": float(t.std()), "median": float(np.median(t)),
                        "frac_positive": float(np.mean(t > 0)), "frac_exact_zero": float(np.mean(t == 0)),
                        "abs_p90": float(np.percentile(np.abs(t), 90)), "abs_max": float(np.abs(t).max())},
            "mean_by_stratum": {
                "same_prompt": _masked_mean(T, mask & same_prompt),
                "diff_prompt": _masked_mean(T, mask_dp),
                "same_category_diff_prompt": _masked_mean(T, mask_dp & (cat[:, None] == cat[None, :])),
                "diff_category": _masked_mean(T, mask_dp & (cat[:, None] != cat[None, :])),
                "same_side_diff_prompt": _masked_mean(T, mask_dp & (side[:, None] == side[None, :])),
                "diff_side": _masked_mean(T, mask_dp & (side[:, None] != side[None, :]))},
            "ratio_offdiag_sd_to_median_self_effect": float(t.std() / max(abs(np.median(diag)), 1e-12)),
            "row_seconds_mean": float(np.nanmean(M["row_seconds"])),
            "total_row_hours": float(np.nansum(M["row_seconds"]) / 3600),
            "matrix_meta": {k: v for k, v in meta.items() if not isinstance(v, list)}}
    if "noise_check_delta" in M:
        nz = M["noise_check_delta"].astype(np.float64)
        desc["noise_floor"] = {"what": "per-token margin change under a random-sign LoRA-B perturbation of size lr*steps",
                               "sd": float(nz.std()), "abs_max": float(np.abs(nz).max()),
                               "frac_offdiag_abs_T_gt_2sd": float(np.mean(np.abs(t) > 2 * nz.std())),
                               "frac_self_effect_gt_2sd": float(np.mean(diag > 2 * nz.std())),
                               "rows_with_offdiag_sd_gt_2x_noise": int(np.sum(
                                   [np.std(T[i][mask[i]]) > 2 * nz.std() for i in np.where(done)[0]]))}
    other = "T_sum" if args.target == "T_mean" else "T_mean"
    desc["spearman_target_vs_%s_offdiag" % other] = spearman(T[mask], M[other].astype(np.float64)[mask])
    log(f"self-effect median {desc['self_effect']['median']:+.5f} (frac>0 {desc['self_effect']['frac_positive']:.2f}); "
        f"off-diag mean {desc['offdiag']['mean']:+.5f} sd {desc['offdiag']['sd']:.5f} frac>0 {desc['offdiag']['frac_positive']:.2f}")

    meta["n_tok_S"] = M["n_tok_S"].tolist()
    meta["base_margin"] = (M["base_logp_T"] / M["n_tok_T"] - M["base_logp_S"] / M["n_tok_S"]).tolist()
    S, pinfo = build_predictors(psi, raw, meta, fp, K, lam, args.seed, args.kmeans_k)
    res = {m: evaluate(S[m], T, mask) for m in S}
    res_dp = {m: evaluate(S[m], T, mask_dp) for m in S}
    for m in S:
        log(f"  {m:14s} rho={res[m]['spearman_offdiag']:+.3f} row={res[m]['spearman_rowcentred']:+.3f} "
            f"col={res[m]['spearman_colcentred']:+.3f} dbl={res[m]['spearman_doublecentred']:+.3f} "
            f"auroc={res[m]['auroc_sign']:.3f} ndcg5={res[m]['ndcg@5']:.3f} | diff-prompt rho={res_dp[m]['spearman_offdiag']:+.3f} "
            f"dbl={res_dp[m]['spearman_doublecentred']:+.3f}")
    # stratified: continuous predictors within same-category / different-category, different-prompt pairs
    strata = {"diff_prompt_same_category": mask_dp & (cat[:, None] == cat[None, :]),
              "diff_category": mask_dp & (cat[:, None] != cat[None, :]),
              "diff_prompt_same_side": mask_dp & (side[:, None] == side[None, :])}
    if "task_same" in S:
        strata["diff_prompt_same_task"] = mask_dp & (S["task_same"] > 0)
        strata["diff_task"] = mask_dp & (S["task_same"] == 0)
    if "conseq_both" in S:
        strata["both_consequential"] = mask & (S["conseq_both"] > 0)
    strata = {k: v for k, v in strata.items() if v.sum() >= 4}
    strat = {name: {m: {"spearman": spearman(S[m][mm], T[mm]), "auroc_sign": auroc(S[m][mm], (T[mm] > 0).astype(int)),
                        "n_pairs": int(mm.sum())}
                    for m in ("psi_dot", "raw_dot", "sparse_dict", "pca_cos", "random")}
             for name, mm in strata.items()}

    # robustness: rows whose pair loss did not saturate (final loss > --min-final-loss), i.e. smaller updates
    final_loss = M["loss_trace"][:, -1]
    nonsat = done & (final_loss > args.min_final_loss)
    mask_ns = offdiag_mask(nonsat, n)
    res_ns = {m: evaluate(S[m], T, mask_ns) for m in S} if nonsat.sum() >= 3 else {}
    desc["rows_nonsaturated"] = int(nonsat.sum())
    desc["final_loss"] = {"median": float(np.median(final_loss[done])), "frac_below_0.01": float(np.mean(final_loss[done] < 0.01)),
                          "frac_below_min": float(np.mean(final_loss[done] <= args.min_final_loss))}
    sym = symmetry_stats(T, done)
    spec = spectrum_stats(T, done)
    log(f"symmetry: pearson(T_ij, T_ji)={sym['pearson_T_ij_vs_T_ji']:+.3f} spearman={sym['spearman_T_ij_vs_T_ji']:+.3f}; "
        f"spectrum frac-norm rank1/5/10 = {spec['frac_norm_rank'][0]:.2f}/{spec['frac_norm_rank'][3]:.2f}/{spec['frac_norm_rank'][4]:.2f} "
        f"(gaussian ref {spec['frac_norm_rank_gaussian_reference'][0]:.2f}/{spec['frac_norm_rank_gaussian_reference'][3]:.2f}/"
        f"{spec['frac_norm_rank_gaussian_reference'][4]:.2f}); additive row+col R2={spec['rank1_row_col_model']['r2_additive_row_plus_col']:.2f}")

    groups = ph if args.group_by == "prompt" else np.asarray([str(t) for t in meta["task_id"]])
    log(f"bootstrap x{args.n_boot} (two-way, by {args.group_by}; {len(set(groups.tolist()))} groups) ...")
    ci = group_bootstrap(S, T, done, groups, args.n_boot, args.seed)
    log(f"permutation x{args.n_perm} (joint event permutation) ...")
    perm = permutation_test(S, T, done, args.n_perm, args.seed + 1)
    # paired bootstrap difference psi_dot / sparse_dict minus each label baseline
    log("paired bootstrap differences ...")
    pairs = [("psi_dot", "category_same"), ("psi_dot", "side_same"), ("psi_dot", "prompt_same"),
             ("sparse_dict", "category_same"), ("sparse_dict", "psi_dot"),
             ("psi_dot", "raw_dot"), ("pca_cos", "psi_dot"),
             ("psi_dot", "task_same"), ("psi_dot", "teacher_cmd_same"), ("psi_dot", "turn_same"),
             ("psi_dot", "conseq_both"), ("psi_dot", "kmeans_same")]
    pairs = [p for p in pairs if p[0] in S and p[1] in S]
    diffs = paired_bootstrap_diff(S, T, done, groups, min(args.n_boot, 500), args.seed + 2, pairs=pairs)

    agg = {}
    if "cluster" in fp and (args.gate / "rates_base.json").is_file():
        agg = cluster_aggregate(T, done, np.asarray(fp["cluster"]), meta["prompt_hash"], args.gate)
    if "all" in agg:
        log(f"cluster aggregate vs gate-v2 gains: all rho={agg['all']['spearman']:+.3f} "
            f"col-centred={agg['all']['spearman_colcentred']:+.3f}; off-cluster rho={agg.get('off_cluster', {}).get('spearman', float('nan')):+.3f} "
            f"col-centred={agg.get('off_cluster', {}).get('spearman_colcentred', float('nan')):+.3f} "
            f"(pairs {agg['n_pairs_all']}/{agg['n_pairs_off_cluster']}, gain==0 frac {agg['frac_gain_zero']:.2f})")

    table = []
    for m in S:
        r = res[m]
        table.append({"predictor": m,
                      "spearman_offdiag": r["spearman_offdiag"],
                      "spearman_offdiag_ci": [ci[m]["spearman_offdiag"]["lo"], ci[m]["spearman_offdiag"]["hi"]] if ci[m]["spearman_offdiag"] else None,
                      "spearman_rowcentred": r["spearman_rowcentred"], "spearman_colcentred": r["spearman_colcentred"],
                      "spearman_doublecentred": r["spearman_doublecentred"],
                      "spearman_doublecentred_ci": [ci[m]["spearman_doublecentred"]["lo"], ci[m]["spearman_doublecentred"]["hi"]] if ci[m]["spearman_doublecentred"] else None,
                      "auroc_sign": r["auroc_sign"],
                      "auroc_sign_ci": [ci[m]["auroc_sign"]["lo"], ci[m]["auroc_sign"]["hi"]] if ci[m]["auroc_sign"] else None,
                      "ndcg@5": r["ndcg@5"],
                      "ndcg@5_ci": [ci[m]["ndcg@5"]["lo"], ci[m]["ndcg@5"]["hi"]] if ci[m]["ndcg@5"] else None,
                      "perm_p_spearman": perm[m]["spearman_offdiag"]["p"],
                      "perm_p_doublecentred": perm[m]["spearman_doublecentred"]["p"],
                      "perm_p_auroc": perm[m]["auroc_sign"]["p"],
                      "diff_prompt_spearman": res_dp[m]["spearman_offdiag"],
                      "diff_prompt_doublecentred": res_dp[m]["spearman_doublecentred"],
                      "diff_prompt_auroc": res_dp[m]["auroc_sign"]})
    out = {"matrix": str(args.matrix), "fingerprints": str(args.fingerprints), "target": args.target,
           "seed": args.seed, "n_boot": args.n_boot, "n_perm": args.n_perm, "group_by": args.group_by,
           "description": desc, "predictor_info": pinfo, "table": table, "full": res, "diff_prompt": res_dp,
           "nonsaturated_rows": res_ns,
           "stratified": strat, "bootstrap": ci, "permutation": perm, "paired_bootstrap_diff": diffs,
           "symmetry": sym, "spectrum": spec, "cluster_aggregate_vs_gate_v2": agg}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.append_key:
        existing = json.loads(args.out.read_text()) if args.out.is_file() else {}
        existing[args.append_key] = out
        out = existing
    args.out.write_text(json.dumps(out, indent=1, default=_json_default))
    log(f"wrote {args.out}")
    print(format_table(table))
    return 0


def _masked_mean(T: np.ndarray, m: np.ndarray):
    return float(T[m].mean()) if m.any() else None


def paired_bootstrap_diff(S: dict, T: np.ndarray, done: np.ndarray, groups: np.ndarray, n_boot: int, seed: int,
                          pairs: list, metrics=("spearman_offdiag", "spearman_doublecentred", "auroc_sign")) -> dict:
    """Bootstrap CI of metric(A) - metric(B) on the same resamples."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups); keys = sorted(set(groups.tolist()))
    members = {k: np.where(groups == k)[0] for k in keys}
    done = np.asarray(done, dtype=bool)
    names = sorted(set(a for p in pairs for a in p))
    vals = {p: {k: [] for k in metrics} for p in pairs}
    for _ in range(n_boot):
        pick = rng.integers(len(keys), size=len(keys))
        cols = np.concatenate([members[keys[p]] for p in pick]); rows = cols[done[cols]]
        if len(rows) < 2:
            continue
        Tb = T[np.ix_(rows, cols)]
        mask = np.ones(Tb.shape, dtype=bool); mask[rows[:, None] == cols[None, :]] = False
        r = {m: evaluate(S[m][np.ix_(rows, cols)], Tb, mask) for m in names}
        for a, b in pairs:
            for k in metrics:
                vals[(a, b)][k].append(r[a][k] - r[b][k])
    out = {}
    for (a, b) in pairs:
        out[f"{a} - {b}"] = {}
        for k in metrics:
            v = np.asarray(vals[(a, b)][k], dtype=float); v = v[~np.isnan(v)]
            out[f"{a} - {b}"][k] = ({"mean": float(v.mean()), "lo": float(np.percentile(v, 2.5)),
                                    "hi": float(np.percentile(v, 97.5)), "frac_gt0": float(np.mean(v > 0))} if len(v) else None)
    return out


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    raise TypeError(str(type(o)))


def format_table(table: list) -> str:
    def ci(x):
        return f"[{x[0]:+.3f},{x[1]:+.3f}]" if x else "-"
    lines = [f"{'predictor':14s} {'rho_offdiag (95% CI)':26s} {'row':>6s} {'col':>6s} {'dbl (95% CI)':>24s} "
             f"{'AUROC (CI)':>22s} {'NDCG@5':>7s} {'perm p':>7s} {'diffP rho':>9s}"]
    for r in table:
        lines.append(f"{r['predictor']:14s} {r['spearman_offdiag']:+.3f} {ci(r['spearman_offdiag_ci']):19s} "
                     f"{r['spearman_rowcentred']:+.3f} {r['spearman_colcentred']:+.3f} "
                     f"{r['spearman_doublecentred']:+.3f} {ci(r['spearman_doublecentred_ci']):17s} "
                     f"{r['auroc_sign']:.3f} {ci(r['auroc_sign_ci']):15s} {r['ndcg@5']:7.3f} "
                     f"{r['perm_p_spearman'] if r['perm_p_spearman'] is not None else float('nan'):7.3f} "
                     f"{r['diff_prompt_spearman']:+9.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
