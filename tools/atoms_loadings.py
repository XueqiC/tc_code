#!/usr/bin/env python3
"""Per-event atom loadings ``zbar`` for the mirror trainer (CRCD Block III, spec §3.5 / §5).

Turns a fingerprint file (``psi`` (n, d), ``state_hash`` (n,) = sha1(prompt); produced by
``tools/crcd_fingerprints.py`` / ``src/bfas/fingerprint.py``) into the loadings npz that
``tools/crcd_mirror_train.py --loadings`` consumes:

    Z            (n_pool, K)  non-negative per-event loadings, rows sum to 1 (0 for rows w/o fingerprint)
    state_hash   (n_pool,)    the trainer's lookup key; here the UNIQUE event key
                              ``<_traj>|sha1(prompt)[:16]|sha1(_rejected)[:16]`` (see NOTE)
    event_key    (n_pool,)    same values as state_hash (explicit alias)
    traj         (n_pool,)    ``_traj`` of the pool row (= mirror.event_id_for_row; NOT unique in v3t)
    pool_row     (n_pool,)    row index into the pool jsonl
    fingerprint_state_hash    sha1(prompt) as stored in the fingerprint npz ("" if no fingerprint)
    has_fingerprint (n_pool,) bool
    Z_signed     (n_pool, K)  the raw (signed) codes before |.|^power and normalisation
    U            (d, K)       the basis (PCA components or sparse-dictionary atoms)
    + scalar metadata (method, power, K, lambda_z, explained_variance_ratio, sources, versions)

NOTE on the key (2026-09-04): ``tools/crcd_mirror_train.py`` keys loadings by ``Event.event_id``
= the pool row's ``_traj``.  In pool_events_pref_v3t that id is NOT unique: 302 rows carry only 80
distinct ``_traj`` values (the same task/event index re-mined in rounds 2-4 with different states
and different y^S), so an id-keyed lookup silently assigns 271 rows another row's loadings.  This
file therefore stores under ``state_hash`` the unique key
``event_key_for_row(row) = _traj|sha1(prompt)[:16]|sha1(_rejected)[:16]`` (302/302 unique), which
the trainer must build from its ``Event`` the same way (``event_key_for_event``; patch given in
docs/2026-09-04-bf3-mirror-prep.md).  With the unpatched trainer every row misses -> all-zero Z
-> J0 = NaN -> ValueError at the first dual update, i.e. it fails loudly instead of silently.
The fingerprint file's own sha1(prompt) is kept under ``fingerprint_state_hash``.

Basis choice (``--method``):
  * ``pca``    (default) K uncentred SVD components of the L2-normalised, Fisher-whitened
               fingerprints (``atoms.fit_pca(center=False)``, the same baseline used in
               ``tools/atoms_fit.py``).  On bfcl_r2 the sparse dictionary matched PCA-32 in
               held-out coverage and in transfer prediction, so the whitened gradient subspace
               is used directly and deterministically (no lambda_z, no alternating solver).
  * ``sparse`` ``atoms.fit(psi, K, lambda_z)`` sparse dictionary; codes are the lasso codes.

Loading definition (``--power``):  zbar_ki = |z_ki|^p / sum_j |z_ji|^p  (per event, sums to 1).
  p = 1 is the spec's atom relevance zbar = |z| (``atoms.atom_relevance``), rescaled per event so
  every event carries unit total relevance; p = 2 gives energy shares (for an orthonormal PCA basis
  sum_k z_ki^2 = the fraction of ||psi_i||^2 = 1 explained by the K components).  The trainer then
  normalises over events per atom, W[i,k] = zbar_ki / sum_j zbar_jk, so the per-event
  normalisation only fixes the *relative* weight of atoms inside an event and equalises events.

Alignment: fingerprint row r <-> ``<fingerprints>.index.jsonl`` row r (``{"row": r,
"state_hash": ...}``); pool row i is matched by sha1(prompt) == state_hash (checked, not assumed
from position).  Pool rows without a fingerprint get an all-zero row and a warning (the trainer
also zero-fills missing ids); ``--require-all`` makes that an error.

``--events <jsonl>`` (2026-09-04, T8, ALFWorld): the event file the fingerprints were computed
on (row r <-> fingerprint row r, verified by sha1(prompt) == state_hash).  With it, pool rows are
matched by the UNIQUE event key (``_traj|sha1(prompt)|sha1(_rejected)``) instead of by position /
prompt hash, so a pool that is a subset or a re-ordering of the fingerprinted events (e.g.
``pool_C_conseq`` = 85 of the 385 ALFWorld events, 4 of whose prompts are shared by two events)
aligns exactly, y^S included.  ``--check-pool <jsonl>`` (repeatable) reports how many rows of
another pool the trainer's ``event_key`` lookup would find in the written npz.

Usage:
    PYTHONPATH=src .venv/bin/python tools/atoms_loadings.py \
        --fingerprints data/fingerprints/bfcl_r3t_v1.npz \
        --pool data/bfcl_sft/pool_events_pref_v3t.jsonl --K 32 --out data/atoms/bfcl_r3t_K32.npz
    PYTHONPATH=src .venv/bin/python tools/atoms_loadings.py \
        --fingerprints data/fingerprints/alfworld_v1_v1.npz --events data/alf_sft/events_v1.jsonl \
        --pool data/alf_sft/pool_A_all.jsonl --check-pool data/alf_sft/pool_C_conseq.jsonl \
        --K 32 --out data/atoms/alfworld_v1_K32.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas import atoms  # noqa: E402
from bfas import mirror  # noqa: E402


def sha1_of(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def event_key_for_row(row: dict) -> str:
    """Unique per pool row: ``_traj|sha1(prompt)[:16]|sha1(_rejected)[:16]``."""
    return f"{mirror.event_id_for_row(row)}|{sha1_of(row['prompt'])[:16]}|{sha1_of(row['_rejected'])[:16]}"


def event_key_for_event(ev: mirror.Event) -> str:
    """The same key computed from a trainer ``Event`` (prompt + student candidate text)."""
    y_s = ev.candidates[ev.index_of(mirror.STUDENT)].text
    return f"{ev.event_id}|{sha1_of(ev.prompt)[:16]}|{sha1_of(y_s)[:16]}"


def fit_basis(psi: np.ndarray, K: int, method: str = "pca", lambda_z: float = 0.02,
              seed: int = 0) -> dict:
    """Basis U (d, K) and signed codes Z (n, K) for the fingerprints."""
    psi = np.asarray(psi, dtype=np.float64)
    if method == "pca":
        pca = atoms.fit_pca(psi, K, center=False)
        return {"U": pca["U"], "Z": pca["Z"], "K": int(pca["K"]),
                "explained_variance_ratio": np.asarray(pca["explained_variance_ratio"], dtype=np.float64),
                "lambda_z": 0.0}
    if method == "sparse":
        fd = atoms.fit(psi, K, lambda_z, seed=seed)
        return {"U": fd.U, "Z": fd.Z, "K": int(fd.K), "lambda_z": float(lambda_z),
                "explained_variance_ratio": np.full(int(fd.K), np.nan),
                "fit_summary": fd.summary()}
    raise ValueError(f"unknown method {method!r}")


def normalise_loadings(Z_signed: np.ndarray, power: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    """zbar_ki = |z_ki|^p / sum_j |z_ji|^p per event; all-zero rows stay zero."""
    A = np.abs(np.asarray(Z_signed, dtype=np.float64)) ** float(power)
    s = A.sum(axis=1, keepdims=True)
    out = np.zeros_like(A)
    nz = s[:, 0] > eps
    out[nz] = A[nz] / s[nz]
    return out


def align_pool(pool_rows: list[dict], fp_hashes: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """For each pool row, the fingerprint row index (-1 if absent).

    A fingerprint is per *row* (it depends on y^S, not only on the state), and several pool rows
    share a prompt (v3t: 198 distinct prompts / 302 rows), so a hash lookup would collapse them.
    The primary alignment is therefore positional (fingerprint row i <-> pool row i, as written by
    ``crcd_fingerprints.py``), accepted only when sha1(prompt_i) equals the stored state_hash; rows
    that fail the positional check (pool subsets, re-ordered pools) fall back to the first
    fingerprint row with the same prompt hash and are counted in ``n_hash_fallback``.  The check
    sees only the state, so it is exact only for the pool the fingerprints were computed on, in
    its original order (there the fingerprint tool wrote row i for pool row i); for subsets the
    fallback picks *a* fingerprint of the same state, which may belong to a different y^S.
    Returns (fp_index (n_pool,), event_ids, prompt_hashes).
    """
    lookup: dict[str, int] = {}
    for r, h in enumerate(fp_hashes):
        lookup.setdefault(h, r)
    idx = np.full(len(pool_rows), -1, dtype=np.int64)
    ids, hashes = [], []
    for i, row in enumerate(pool_rows):
        h = sha1_of(row["prompt"])
        hashes.append(h)
        ids.append(mirror.event_id_for_row(row))
        if i < len(fp_hashes) and fp_hashes[i] == h:
            idx[i] = i
        else:
            idx[i] = lookup.get(h, -1)
    return idx, ids, hashes


def align_pool_by_key(pool_rows: list[dict], event_rows: list[dict],
                      fp_hashes: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    """Exact alignment through the fingerprint *source* events.

    ``event_rows[r]`` is the row fingerprint r was computed from (checked: sha1(prompt) equals the
    stored state_hash for every r).  Each pool row is then looked up by its unique event key
    (``event_key_for_row``: _traj + prompt + y^S), which is what the trainer does, so subsets,
    re-orderings and prompts shared by several events all resolve to the right fingerprint.
    Returns (fp_index (n_pool,), event_ids, prompt_hashes) like ``align_pool``.
    """
    if len(event_rows) != len(fp_hashes):
        raise ValueError(f"events file has {len(event_rows)} rows vs {len(fp_hashes)} fingerprints")
    bad = [r for r, (row, h) in enumerate(zip(event_rows, fp_hashes)) if sha1_of(row["prompt"]) != h]
    if bad:
        raise ValueError(f"events file does not match the fingerprint order at rows {bad[:5]} "
                         f"({len(bad)} mismatches)")
    lookup: dict[str, int] = {}
    for r, row in enumerate(event_rows):
        k = event_key_for_row(row)
        if k in lookup:
            raise ValueError(f"events file has duplicate event keys (row {lookup[k]} and {r}): {k}")
        lookup[k] = r
    idx = np.full(len(pool_rows), -1, dtype=np.int64)
    ids, hashes = [], []
    for i, row in enumerate(pool_rows):
        hashes.append(sha1_of(row["prompt"]))
        ids.append(mirror.event_id_for_row(row))
        idx[i] = lookup.get(event_key_for_row(row), -1)
    return idx, ids, hashes


def match_pool(loadings: Path | dict, pool: Path) -> dict:
    """Replicate ``crcd_mirror_train.load_loadings`` for ``pool``: how many rows find a loading row.

    Returns ``{"n_pool", "n_matched", "n_missing", "missing_keys", "rows"}`` where ``rows[i]`` is
    the npz row a pool row maps to (-1 if absent).
    """
    data = np.load(loadings, allow_pickle=True) if isinstance(loadings, (str, Path)) else loadings
    keys = [str(k) for k in data["state_hash"]]
    lookup = {k: i for i, k in enumerate(keys)}
    by_key = "event_key" in (data.files if hasattr(data, "files") else data)
    pool_rows = atoms_pool_rows(pool)
    rows, missing = [], []
    for row in pool_rows:
        ev = mirror.candidates_from_row(row)
        k = event_key_for_event(ev) if by_key else ev.event_id
        j = lookup.get(k, -1)
        rows.append(j)
        if j < 0:
            missing.append(k)
    return {"n_pool": len(pool_rows), "n_matched": len(pool_rows) - len(missing),
            "n_missing": len(missing), "missing_keys": missing, "rows": np.array(rows, dtype=np.int64)}


def build(fingerprints: Path, pool: Path, K: int, method: str, power: float, lambda_z: float,
          seed: int, require_all: bool = False, events: Path | None = None) -> dict:
    fp = atoms.load_npz(fingerprints)
    psi = fp["psi"]
    fp_hashes = [str(h) for h in fp["state_hash"]]
    index_path = fingerprints.with_name(fingerprints.stem + ".index.jsonl")
    if index_path.is_file():
        index = [json.loads(l) for l in index_path.read_text().splitlines() if l.strip()]
        if len(index) != len(fp_hashes):
            raise ValueError(f"{index_path}: {len(index)} rows vs {len(fp_hashes)} fingerprints")
        for r, rec in enumerate(index):
            if int(rec["row"]) != r or str(rec["state_hash"]) != fp_hashes[r]:
                raise ValueError(f"{index_path} row {r} does not match the npz order")
    pool_rows = atoms_pool_rows(pool)
    if events is not None:
        fp_idx, event_ids, prompt_hashes = align_pool_by_key(pool_rows, atoms_pool_rows(events), fp_hashes)
        alignment = "event_key"
    else:
        fp_idx, event_ids, prompt_hashes = align_pool(pool_rows, fp_hashes)
        alignment = "positional+prompt_hash"
    event_keys = [event_key_for_row(r) for r in pool_rows]
    if len(set(event_keys)) != len(event_keys):
        dup = [e for e in set(event_keys) if event_keys.count(e) > 1]
        raise ValueError(f"pool has duplicate event keys (lookup would be ambiguous): {dup[:5]}")
    n_unique_traj = len(set(event_ids))
    missing = [event_keys[i] for i in np.where(fp_idx < 0)[0]]
    n_positional = int(sum(1 for i, j in enumerate(fp_idx) if j == i))
    # the prompt-hash fallback only exists in positional mode; under event_key alignment every
    # match is exact, so re-ordered rows are not "fallbacks"
    n_hash_fallback = 0 if events is not None else int(sum(1 for i, j in enumerate(fp_idx) if j >= 0 and j != i))
    if missing and require_all:
        raise ValueError(f"{len(missing)} pool rows have no fingerprint: {missing[:5]}")
    unused = sorted(set(range(len(fp_hashes))) - set(int(j) for j in fp_idx if j >= 0))

    basis = fit_basis(psi, K, method=method, lambda_z=lambda_z, seed=seed)
    Kf = basis["K"]
    Z_signed = np.zeros((len(pool_rows), Kf))
    has = fp_idx >= 0
    Z_signed[has] = basis["Z"][fp_idx[has]]
    Z = normalise_loadings(Z_signed, power=power)
    zsum = Z_signed[has] ** 2
    return {
        "Z": Z, "Z_signed": Z_signed, "U": basis["U"],
        "state_hash": np.array(event_keys), "event_key": np.array(event_keys),
        "traj": np.array(event_ids), "n_unique_traj": np.int64(n_unique_traj),
        "pool_row": np.arange(len(pool_rows), dtype=np.int64),
        "fingerprint_state_hash": np.array([prompt_hashes[i] if has[i] else "" for i in range(len(pool_rows))]),
        "fingerprint_row": fp_idx, "has_fingerprint": has,
        "K": np.int64(Kf), "method": method, "power": np.float64(power),
        "lambda_z": np.float64(basis["lambda_z"]),
        "explained_variance_ratio": basis["explained_variance_ratio"],
        "energy_captured_mean": np.float64(zsum.sum(axis=1).mean()) if has.any() else np.float64(np.nan),
        "fingerprints": str(fingerprints), "pool": str(pool),
        "events": str(events) if events is not None else "", "alignment": alignment,
        "fingerprint_version": str(fp.get("fingerprint_version", "")),
        "fisher_id": str(fp.get("fisher_id", "")),
        "n_missing": np.int64(len(missing)), "missing_ids": np.array(missing),
        "n_positional": np.int64(n_positional), "n_hash_fallback": np.int64(n_hash_fallback),
        "n_fingerprints_unused": np.int64(len(unused)),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "definition": (f"zbar_ki = |z_ki|^{power:g} / sum_j |z_ji|^{power:g}; z = codes of psi on the "
                       f"{method} basis (K={Kf}); rows without a fingerprint are all-zero"),
    }


def atoms_pool_rows(pool: Path) -> list[dict]:
    rows = []
    with open(pool, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarise(out: dict) -> str:
    Z, has = out["Z"], out["has_fingerprint"]
    Zh = Z[has]
    col = Zh.sum(axis=0)
    n_eff = col ** 2 / np.maximum((Zh ** 2).sum(axis=0), 1e-12)
    lines = [
        f"events={Z.shape[0]} with_fingerprint={int(has.sum())} K={Z.shape[1]} method={out['method']} "
        f"power={float(out['power']):g} energy_captured_mean={float(out['energy_captured_mean']):.4f}",
        f"row sums: min={Zh.sum(axis=1).min():.6f} max={Zh.sum(axis=1).max():.6f}",
        f"mean loading per atom: {np.round(Zh.mean(axis=0), 4).tolist()}",
        f"max loading per atom : {np.round(Zh.max(axis=0), 3).tolist()}",
        f"atom-weighted effective n (sum^2/sum sq): {np.round(n_eff, 1).tolist()}",
        f"top-atom share per event: mean={Zh.max(axis=1).mean():.3f} median={np.median(Zh.max(axis=1)):.3f}",
        f"explained variance ratio: {np.round(out['explained_variance_ratio'], 4).tolist()}",
        f"alignment={out.get('alignment', 'positional+prompt_hash')} "
        f"aligned positionally={int(out['n_positional'])} by-hash-fallback={int(out['n_hash_fallback'])} "
        f"missing={int(out['n_missing'])} unused_fingerprints={int(out['n_fingerprints_unused'])} "
        f"unique_traj_ids={int(out['n_unique_traj'])} (trainer must key by event_key, see docstring)",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fingerprints", default="data/fingerprints/bfcl_r3t_v1.npz")
    ap.add_argument("--pool", default="data/bfcl_sft/pool_events_pref_v3t.jsonl")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--method", default="pca", choices=["pca", "sparse"])
    ap.add_argument("--power", type=float, default=1.0, help="zbar = |z|^p normalised per event")
    ap.add_argument("--lambda-z", type=float, default=0.02, help="sparse method only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--require-all", action="store_true", help="fail if any pool row lacks a fingerprint")
    ap.add_argument("--events", default=None,
                    help="event file the fingerprints were computed on (row r <-> fingerprint r); "
                         "aligns pool rows by the unique event key instead of position/prompt hash")
    ap.add_argument("--check-pool", action="append", default=[],
                    help="report how many rows of this pool the trainer's lookup finds in the written npz")
    ap.add_argument("--out", default=None, help="default data/atoms/<fp stem>_K<K>.npz")
    args = ap.parse_args(argv)
    fp = ROOT / args.fingerprints
    pool = ROOT / args.pool
    events = ROOT / args.events if args.events else None
    out_path = ROOT / (args.out or f"data/atoms/{fp.stem.replace('_v1', '')}_K{args.K}.npz")
    out = build(fp, pool, args.K, args.method, args.power, args.lambda_z, args.seed, args.require_all,
                events=events)
    if int(out["n_missing"]):
        print(f"[loadings][warn] {int(out['n_missing'])} pool rows without fingerprint (zero loadings): "
              f"{out['missing_ids'][:5].tolist()}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    print(summarise(out))
    print(f"[loadings] wrote {out_path}", flush=True)
    for cp in [pool] + [ROOT / p for p in args.check_pool]:
        m = match_pool(out_path, cp)
        print(f"[loadings][check] {cp.relative_to(ROOT) if cp.is_relative_to(ROOT) else cp}: "
              f"trainer lookup matches {m['n_matched']}/{m['n_pool']} rows"
              + (f" (missing e.g. {m['missing_keys'][:3]})" if m["n_missing"] else ""), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
