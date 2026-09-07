"""tools/atoms_loadings.py: synthetic loadings + alignment against the real bfcl_r3t files."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas import mirror  # noqa: E402

_spec = importlib.util.spec_from_file_location("atoms_loadings", ROOT / "tools" / "atoms_loadings.py")
al = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(al)

FP = ROOT / "data/fingerprints/bfcl_r3t_v1.npz"
POOL = ROOT / "data/bfcl_sft/pool_events_pref_v3t.jsonl"
OUT = ROOT / "data/atoms/bfcl_r3t_K32.npz"


def _synthetic(tmp_path: Path, n: int = 12, d: int = 16, K: int = 4, seed: int = 0):
    """A pool whose rows share _traj ids and prompts, plus a matching fingerprint file."""
    rng = np.random.default_rng(seed)
    U = np.linalg.qr(rng.standard_normal((d, K)))[0]
    codes = rng.standard_normal((n, K)) * np.array([3.0, 2.0, 1.0, 0.5])
    psi = codes @ U.T + 0.01 * rng.standard_normal((n, d))
    psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    rows, hashes = [], []
    for i in range(n):
        prompt = f"state-{i // 2}"           # two rows per state
        rows.append({"task_id": f"t{i // 3}", "turn_index": 0, "_traj": f"t{i // 3}#ev1",  # 3 rows per id
                     "prompt": prompt, "response": f"teacher-{i}", "_rejected": f"student-{i}",
                     "_event_u_plus": 1.0, "_event_u_minus": 0.2, "teacher": "teacher_authored_gt"})
        hashes.append(hashlib.sha1(prompt.encode()).hexdigest())
    pool = tmp_path / "pool.jsonl"
    pool.write_text("".join(json.dumps(r) + "\n" for r in rows))
    fp = tmp_path / "fp_v1.npz"
    np.savez(fp, psi=psi.astype(np.float32), state_hash=np.array(hashes), fingerprint_version="synthetic")
    (tmp_path / "fp_v1.index.jsonl").write_text(
        "".join(json.dumps({"row": r, "state_hash": h}) + "\n" for r, h in enumerate(hashes)))
    return rows, psi, U, fp, pool


def test_normalise_loadings_rows_sum_to_one_and_zero_rows_stay_zero():
    Z = np.array([[1.0, -3.0, 0.0], [0.0, 0.0, 0.0], [2.0, 2.0, -4.0]])
    W1 = al.normalise_loadings(Z, power=1)
    assert np.allclose(W1[0], [0.25, 0.75, 0.0])
    assert np.all(W1[1] == 0)
    assert np.allclose(W1.sum(axis=1), [1.0, 0.0, 1.0])
    W2 = al.normalise_loadings(Z, power=2)
    assert np.allclose(W2[2], [4 / 24, 4 / 24, 16 / 24])
    assert np.all(W1 >= 0) and np.all(W2 >= 0)


def test_synthetic_pca_recovers_subspace_and_positional_alignment(tmp_path):
    rows, psi, U_true, fp, pool = _synthetic(tmp_path)
    out = al.build(fp, pool, K=4, method="pca", power=1.0, lambda_z=0.0, seed=0)
    assert out["Z"].shape == (12, 4)
    assert np.allclose(out["Z"].sum(axis=1), 1.0)
    assert np.all(out["Z"] >= 0)
    assert int(out["n_positional"]) == 12 and int(out["n_hash_fallback"]) == 0
    assert int(out["n_missing"]) == 0 and int(out["n_unique_traj"]) == 4
    # rows sharing a prompt must NOT share loadings (fingerprints are per row)
    assert not np.allclose(out["Z"][0], out["Z"][1])
    # the recovered span equals the planted one (PCA-4 on a rank-4 signal)
    P_true = U_true @ U_true.T
    P_fit = out["U"] @ out["U"].T
    assert np.linalg.norm(P_true - P_fit) < 0.05
    assert float(out["energy_captured_mean"]) > 0.99
    # keys the trainer will look up: unique, and reproducible from an Event
    keys = [str(k) for k in out["state_hash"]]
    assert len(set(keys)) == 12
    for i, row in enumerate(rows):
        ev = mirror.candidates_from_row(row)
        assert al.event_key_for_event(ev) == keys[i] == al.event_key_for_row(row)
        assert str(out["traj"][i]) == ev.event_id
        assert int(out["pool_row"][i]) == i
    # loading the file back gives the same arrays
    path = tmp_path / "load.npz"
    np.savez(path, **out)
    data = np.load(path, allow_pickle=True)
    assert np.allclose(data["Z"], out["Z"]) and data["Z"].shape[1] == 4


def test_synthetic_subset_falls_back_to_hash_and_missing_rows_are_zero(tmp_path):
    rows, psi, U_true, fp, pool = _synthetic(tmp_path)
    sub = tmp_path / "sub.jsonl"
    extra = dict(rows[0], prompt="never-fingerprinted", _rejected="student-x", _traj="new#ev1")
    sub.write_text("".join(json.dumps(r) + "\n" for r in [rows[3], rows[0], extra]))
    out = al.build(fp, sub, K=4, method="pca", power=2.0, lambda_z=0.0, seed=0)
    # position 0 holds rows[3] (state-1 != fp row 0's state-0) -> hash fallback to fp row 2;
    # position 1 holds rows[0] whose state equals fp row 1's state (rows 0/1 share a prompt), so the
    # positional check accepts it: the state hash cannot distinguish rows that share a state
    # (documented limitation -- positional alignment is only exact for the pool the fingerprints
    # were computed on, in its original order)
    assert int(out["n_hash_fallback"]) == 1 and int(out["n_positional"]) == 1
    assert int(out["fingerprint_row"][0]) == 2
    assert int(out["n_missing"]) == 1
    assert np.all(out["Z"][2] == 0) and not out["has_fingerprint"][2]
    assert np.allclose(out["Z"][:2].sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        al.build(fp, sub, K=4, method="pca", power=1.0, lambda_z=0.0, seed=0, require_all=True)


def test_duplicate_event_keys_are_rejected(tmp_path):
    rows, psi, U_true, fp, pool = _synthetic(tmp_path)
    dup = tmp_path / "dup.jsonl"
    dup.write_text("".join(json.dumps(r) + "\n" for r in [rows[0], rows[0]]))
    with pytest.raises(ValueError, match="duplicate event keys"):
        al.build(fp, dup, K=4, method="pca", power=1.0, lambda_z=0.0, seed=0)


@pytest.mark.skipif(not (FP.is_file() and POOL.is_file() and OUT.is_file()),
                    reason="real bfcl_r3t files not present")
def test_real_bfcl_r3t_K32_alignment():
    data = np.load(OUT, allow_pickle=True)
    rows = [json.loads(l) for l in POOL.read_text().splitlines() if l.strip()]
    fp = np.load(FP, allow_pickle=True)
    Z = data["Z"]
    assert Z.shape == (len(rows), 32) == (302, 32)
    assert np.all(Z >= 0) and np.allclose(Z.sum(axis=1), 1.0)
    assert bool(data["has_fingerprint"].all())
    keys = [str(k) for k in data["state_hash"]]
    assert len(set(keys)) == 302
    for i, row in enumerate(rows):
        assert keys[i] == al.event_key_for_row(row) == al.event_key_for_event(mirror.candidates_from_row(row))
        assert str(data["traj"][i]) == row["_traj"]
        assert str(data["fingerprint_state_hash"][i]) == hashlib.sha1(row["prompt"].encode()).hexdigest() \
            == str(fp["state_hash"][i])
        assert int(data["fingerprint_row"][i]) == i
    # loadings are the |PCA codes| of the aligned fingerprint rows
    psi = np.asarray(fp["psi"], dtype=np.float64)
    codes = psi @ data["U"]
    assert np.allclose(codes, data["Z_signed"], atol=1e-6)
    assert np.allclose(np.abs(codes) / np.abs(codes).sum(axis=1, keepdims=True), Z, atol=1e-9)
    assert str(data["fingerprint_version"]) == str(fp["fingerprint_version"])
    assert int(data["n_unique_traj"]) == 80  # documented non-uniqueness of _traj in v3t


@pytest.mark.skipif(not (FP.is_file() and POOL.is_file() and OUT.is_file()),
                    reason="real bfcl_r3t files not present")
def test_real_loadings_drive_trainer_math():
    """W column-normalisation and Lambda as the trainer computes them are finite and sane."""
    data = np.load(OUT, allow_pickle=True)
    W = mirror.normalized_loadings(data["Z"])
    assert np.allclose(W.sum(axis=0), 1.0, atol=1e-6)
    Lam = mirror.capability_pressure(W, np.ones(32))
    assert np.all(np.isfinite(Lam)) and Lam.min() > 0
    assert Lam.mean() == pytest.approx(32 / 302, rel=1e-6)


# ---------------------------------------------------------------------------
# --events alignment (T8, 2026-09-04): exact event-key alignment against the fingerprint source
# ---------------------------------------------------------------------------

ALF_FP = ROOT / "data/fingerprints/alfworld_v1_v1.npz"
ALF_EVENTS = ROOT / "data/alf_sft/events_v1.jsonl"
ALF_POOL_A = ROOT / "data/alf_sft/pool_A_all.jsonl"
ALF_POOL_C = ROOT / "data/alf_sft/pool_C_conseq.jsonl"
ALF_OUT = ROOT / "data/atoms/alfworld_v1_K32.npz"


def test_events_alignment_resolves_subsets_reorders_and_shared_prompts(tmp_path):
    rows, psi, U_true, fp, pool = _synthetic(tmp_path)
    # a re-ordered subset in which rows 0 and 1 share a prompt (state-0) but differ in y^S
    sub = tmp_path / "sub.jsonl"
    order = [5, 0, 1, 9]
    sub.write_text("".join(json.dumps(rows[i]) + "\n" for i in order))
    out = al.build(fp, sub, K=4, method="pca", power=1.0, lambda_z=0.0, seed=0, events=pool)
    assert str(out["alignment"]) == "event_key"
    assert int(out["n_missing"]) == 0 and int(out["n_hash_fallback"]) == 0
    assert out["fingerprint_row"].tolist() == order
    full = al.build(fp, pool, K=4, method="pca", power=1.0, lambda_z=0.0, seed=0, events=pool)
    assert np.allclose(out["Z"], full["Z"][order])
    assert not np.allclose(full["Z"][0], full["Z"][1])  # shared prompt, different fingerprint
    # positional/hash alignment gets both shared-prompt rows wrong here: position 1 holds rows[0]
    # but fp row 1 has the same state (accepted positionally -> row 1), position 2 holds rows[1]
    # and falls back to the first fp row with that prompt (-> row 0)
    pos = al.build(fp, sub, K=4, method="pca", power=1.0, lambda_z=0.0, seed=0)
    assert pos["fingerprint_row"].tolist()[1:3] == [1, 0] and out["fingerprint_row"].tolist()[1:3] == [0, 1]
    # the trainer-side lookup finds every subset row in the full-pool npz
    path = tmp_path / "full.npz"
    np.savez(path, **full)
    m = al.match_pool(path, sub)
    assert m["n_matched"] == 4 and m["rows"].tolist() == order
    # rows absent from the events file are reported missing (and rejected by --require-all)
    extra = dict(rows[0], _rejected="student-x", _traj="new#ev1")
    sub2 = tmp_path / "sub2.jsonl"
    sub2.write_text("".join(json.dumps(r) + "\n" for r in [rows[2], extra]))
    out2 = al.build(fp, sub2, K=4, method="pca", power=1.0, lambda_z=0.0, seed=0, events=pool)
    assert int(out2["n_missing"]) == 1 and np.all(out2["Z"][1] == 0)
    assert al.match_pool(path, sub2)["n_matched"] == 1
    with pytest.raises(ValueError):
        al.build(fp, sub2, K=4, method="pca", power=1.0, lambda_z=0.0, seed=0, events=pool, require_all=True)


def test_events_alignment_rejects_mismatched_events_file(tmp_path):
    rows, psi, U_true, fp, pool = _synthetic(tmp_path)
    wrong = tmp_path / "wrong.jsonl"
    wrong.write_text("".join(json.dumps(r) + "\n" for r in rows[::-1]))  # reversed order
    with pytest.raises(ValueError, match="does not match the fingerprint order"):
        al.build(fp, pool, K=4, method="pca", power=1.0, lambda_z=0.0, seed=0, events=wrong)
    short = tmp_path / "short.jsonl"
    short.write_text("".join(json.dumps(r) + "\n" for r in rows[:5]))
    with pytest.raises(ValueError, match="rows vs"):
        al.build(fp, pool, K=4, method="pca", power=1.0, lambda_z=0.0, seed=0, events=short)


_ALF_PRESENT = all(p.is_file() for p in (ALF_FP, ALF_EVENTS, ALF_POOL_A, ALF_POOL_C, ALF_OUT))


@pytest.mark.skipif(not _ALF_PRESENT, reason="real alfworld_v1 files not present")
def test_real_alfworld_K32_alignment_and_pool_lookup():
    data = np.load(ALF_OUT, allow_pickle=True)
    rows = [json.loads(l) for l in ALF_POOL_A.read_text().splitlines() if l.strip()]
    events = [json.loads(l) for l in ALF_EVENTS.read_text().splitlines() if l.strip()]
    fp = np.load(ALF_FP, allow_pickle=True)
    assert len(rows) == len(events) == 385 and rows == events  # pool_A_all is the event file
    Z = data["Z"]
    assert Z.shape == (385, 32) and np.all(Z >= 0) and np.allclose(Z.sum(axis=1), 1.0)
    assert bool(data["has_fingerprint"].all()) and str(data["alignment"]) == "event_key"
    keys = [str(k) for k in data["state_hash"]]
    assert len(set(keys)) == 385 and int(data["n_unique_traj"]) == 385  # _traj IS unique on ALFWorld
    for i, row in enumerate(rows):
        assert keys[i] == al.event_key_for_row(row) == al.event_key_for_event(mirror.candidates_from_row(row))
        assert int(data["fingerprint_row"][i]) == i
        assert str(data["fingerprint_state_hash"][i]) == hashlib.sha1(row["prompt"].encode()).hexdigest() \
            == str(fp["state_hash"][i])
    # 4 prompts are shared by two events each (381 distinct): the key still separates them
    assert len(set(r["prompt"] for r in rows)) == 381
    psi = np.asarray(fp["psi"], dtype=np.float64)
    codes = psi @ data["U"]
    assert np.allclose(codes, data["Z_signed"], atol=1e-6)
    assert np.allclose(np.abs(codes) / np.abs(codes).sum(axis=1, keepdims=True), Z, atol=1e-9)
    # the consequential pool is an exact 85-row subset: the trainer's lookup finds every row
    m = al.match_pool(ALF_OUT, ALF_POOL_C)
    assert m["n_pool"] == 85 and m["n_matched"] == 85
    C = [json.loads(l) for l in ALF_POOL_C.read_text().splitlines() if l.strip()]
    for j, row in zip(m["rows"], C):
        assert rows[int(j)] == row
    # ALFWorld fingerprints are low-dimensional: PC1 dominates (documented in the prep note)
    evr = data["explained_variance_ratio"]
    assert evr[0] > 0.6 and float(Z[:, 0].mean()) > 0.3


@pytest.mark.skipif(not _ALF_PRESENT, reason="real alfworld_v1 files not present")
def test_real_alfworld_loadings_drive_trainer_math_on_both_pools():
    data = np.load(ALF_OUT, allow_pickle=True)
    for pool, n in ((ALF_POOL_A, 385), (ALF_POOL_C, 85)):
        m = al.match_pool(ALF_OUT, pool)
        Z = data["Z"][m["rows"]]  # what load_loadings assembles for this pool
        W = mirror.normalized_loadings(Z)
        assert np.allclose(W.sum(axis=0), 1.0, atol=1e-6)
        Lam = mirror.capability_pressure(W, np.ones(32))
        assert np.all(np.isfinite(Lam)) and Lam.min() > 0
        assert Lam.mean() == pytest.approx(32 / n, rel=1e-6)
