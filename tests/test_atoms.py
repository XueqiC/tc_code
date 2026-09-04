from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas import atoms  # noqa: E402


def planted(n=120, d=40, K=5, noise=0.05, seed=0, nnz=2):
    """Events = sparse combinations of K planted unit-norm atoms + small noise."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((d, K))
    U /= np.linalg.norm(U, axis=0)
    Z = np.zeros((n, K))
    for i in range(n):
        idx = rng.choice(K, size=nnz, replace=False)
        Z[i, idx] = rng.uniform(0.5, 1.5, size=nnz) * rng.choice([-1, 1], size=nnz)
    Psi = Z @ U.T + noise * rng.standard_normal((n, d))
    return Psi, U, Z


def test_unit_norm_columns():
    Psi, _, _ = planted()
    fd = atoms.fit(Psi, K=5, lambda_z=0.05, seed=1)
    assert fd.U.shape == (40, 5)
    np.testing.assert_allclose(np.linalg.norm(fd.U, axis=0), 1.0, atol=1e-9)
    for init in ("random", "samples"):
        f2 = atoms.fit(Psi, K=8, lambda_z=0.1, seed=2, init=init, n_iter=10)
        np.testing.assert_allclose(np.linalg.norm(f2.U, axis=0), 1.0, atol=1e-9)


def test_sparsity_increases_with_lambda():
    Psi, _, _ = planted(noise=0.1)
    nnz = []
    for lam in (0.0, 0.05, 0.2, 0.6):
        fd = atoms.fit(Psi, K=8, lambda_z=lam, seed=0)
        nnz.append(fd.nnz_per_event.mean())
    assert nnz[0] == pytest.approx(8.0)  # lambda = 0: dense codes
    assert nnz[1] < nnz[0] and nnz[2] < nnz[1] and nnz[3] < nnz[2]
    # sparsity also holds for encode() on fresh data with a fixed dictionary
    fd = atoms.fit(Psi, K=8, lambda_z=0.05, seed=0)
    dense = (np.abs(atoms.encode(fd.U, Psi[:10], 0.0)) > 1e-10).sum()
    sparse = (np.abs(atoms.encode(fd.U, Psi[:10], 0.3)) > 1e-10).sum()
    assert sparse < dense


def test_objective_decreases_and_recovers_planted_atoms():
    Psi, U_true, _ = planted(noise=0.02)
    fd = atoms.fit(Psi, K=5, lambda_z=0.02, seed=0)
    hist = np.asarray(fd.objective_history)
    assert hist[-1] <= hist[0]
    # each planted atom is matched (up to sign) by some learned atom
    match = np.abs(U_true.T @ fd.U).max(axis=1)
    assert match.min() > 0.9
    assert atoms.reconstruction_coverage(fd.U, Psi) > 0.95


def test_pca_recovers_planted_low_rank_signal():
    rng = np.random.default_rng(3)
    d, r, n = 30, 3, 200
    B = np.linalg.qr(rng.standard_normal((d, r)))[0]
    Psi = rng.standard_normal((n, r)) @ B.T * 3.0 + 0.05 * rng.standard_normal((n, d))
    pca = atoms.fit_pca(Psi, K=r, center=False)
    assert pca["U"].shape == (d, r)
    np.testing.assert_allclose(pca["U"].T @ pca["U"], np.eye(r), atol=1e-8)
    # subspace overlap: projection of true basis onto learned span ~ identity
    overlap = np.linalg.norm(pca["U"].T @ B, ord="fro") ** 2 / r
    assert overlap > 0.99
    assert atoms.reconstruction_coverage(pca["U"], Psi) > 0.99
    assert float(pca["explained_variance_ratio"].sum()) > 0.99
    # centred PCA encodes with the mean removed
    pc = atoms.fit_pca(Psi, K=r, center=True)
    np.testing.assert_allclose(atoms.encode_pca(pc, Psi).mean(axis=0), 0.0, atol=1e-8)


def test_coverage_in_unit_interval_and_monotone_in_K():
    Psi, _, _ = planted(noise=0.3, seed=5)
    train, dev = atoms.split_events(Psi.shape[0], dev_frac=0.3, seed=0)
    assert len(set(train) & set(dev)) == 0 and len(train) + len(dev) == Psi.shape[0]
    prev = -1.0
    for K in (1, 2, 4, 8):
        pca = atoms.fit_pca(Psi[train], K)
        cov = atoms.reconstruction_coverage(pca["U"], Psi[dev])
        assert 0.0 <= cov <= 1.0
        assert cov >= prev - 1e-9
        prev = cov
    Ur = atoms.random_dictionary(Psi.shape[1], 4, seed=0)
    assert 0.0 <= atoms.reconstruction_coverage(Ur, Psi[dev]) <= 1.0
    # identity-like dictionary covers everything
    full = atoms.fit_pca(Psi, K=Psi.shape[1])
    assert atoms.reconstruction_coverage(full["U"], Psi) == pytest.approx(1.0, abs=1e-9)
    # lambda = 0 sparse fit spans the same subspace as PCA at the same K (coverage matches)
    fd0 = atoms.fit(Psi, K=4, lambda_z=0.0, seed=0)
    assert atoms.reconstruction_coverage(fd0.U, Psi) == pytest.approx(
        atoms.reconstruction_coverage(atoms.fit_pca(Psi, 4)["U"], Psi), abs=2e-3)


def test_encode_and_fit_determinism():
    Psi, _, _ = planted(seed=7)
    f1 = atoms.fit(Psi, K=6, lambda_z=0.05, seed=11)
    f2 = atoms.fit(Psi, K=6, lambda_z=0.05, seed=11)
    np.testing.assert_array_equal(f1.U, f2.U)
    np.testing.assert_array_equal(f1.Z, f2.Z)
    z1 = atoms.encode(f1.U, Psi[:20], 0.05)
    z2 = atoms.encode(f1.U, Psi[:20], 0.05)
    np.testing.assert_array_equal(z1, z2)
    # encode reproduces the training codes for the fitted dictionary
    np.testing.assert_allclose(atoms.encode(f1.U, Psi, 0.05), f1.Z, atol=1e-4)
    # single vector input is accepted
    assert atoms.encode(f1.U, Psi[0], 0.05).shape == (1, 6)
    f3 = atoms.fit(Psi, K=6, lambda_z=0.05, seed=12)
    assert not np.array_equal(f1.U, f3.U)  # seed actually matters
    # encode solves the lasso: matches a slow iterative solution; zero lambda == least squares
    ls = atoms.least_squares_codes(f1.U, Psi[:5])
    np.testing.assert_allclose(atoms.encode(f1.U, Psi[:5], 0.0, n_iter=5000, tol=1e-12), ls, atol=1e-5)


def test_kmeans_random_relevance_transfer_shapes():
    Psi, _, _ = planted(seed=2)
    km = atoms.fit_kmeans(Psi, K=5, seed=0)
    np.testing.assert_allclose(np.linalg.norm(km["U"], axis=0), 1.0, atol=1e-9)
    assert ((np.abs(km["Z"]) > 0).sum(axis=1) == 1).all()  # hard assignment
    km2 = atoms.fit_kmeans(Psi, K=5, seed=0)
    np.testing.assert_array_equal(km["labels"], km2["labels"])
    Ur = atoms.random_dictionary(40, 5, seed=0)
    np.testing.assert_allclose(np.linalg.norm(Ur, axis=0), 1.0, atol=1e-9)
    fd = atoms.fit(Psi, K=5, lambda_z=0.05, seed=0)
    zbar = atoms.atom_relevance(fd.Z)
    assert zbar.shape == fd.Z.shape and (zbar >= 0).all()
    P = atoms.transfer_prediction(fd.U, fd.Z[:3], fd.Z[:7])
    assert P.shape == (3, 7)
    # P_ii = ||U z_i||^2 >= 0 and symmetric between the same rows
    np.testing.assert_allclose(np.diag(P[:3, :3]), np.linalg.norm(fd.Z[:3] @ fd.U.T, axis=1) ** 2)
    np.testing.assert_allclose(P[:3, :3], P[:3, :3].T, atol=1e-10)


def test_select_model_grid_and_npz_roundtrip(tmp_path):
    Psi, _, _ = planted(n=60, d=20, K=3, seed=4)
    sel = atoms.select_model(Psi, Ks=(2, 4), lambdas=(0.0, 0.1), seed=0, dev_frac=0.25,
                             fit_kwargs={"n_iter": 30})
    assert len(sel["grid"]) == 4 and sel["best"] in sel["grid"]
    assert all(0 <= g["dev_coverage"] <= 1 for g in sel["grid"])
    assert sel["n_train"] + sel["n_dev"] == 60
    sel2 = atoms.select_model(Psi, Ks=(2, 4), lambdas=(0.0, 0.1), seed=0, dev_frac=0.25,
                              fit_kwargs={"n_iter": 30})
    assert sel["best"] == sel2["best"]
    # grouped split keeps groups together
    groups = np.repeat(np.arange(20), 3)
    tr, dv = atoms.split_events(60, dev_frac=0.3, seed=1, groups=groups)
    assert not (set(groups[tr]) & set(groups[dv]))
    # npz I/O with the required keys
    p = tmp_path / "psi.npz"
    np.savez(p, psi=Psi.astype(np.float32), state_hash=np.array([f"h{i:03d}" for i in range(60)]))
    d = atoms.load_features(p)
    assert d["psi"].shape == (60, 20) and d["state_hash"][0] == "h000" and d["psi"].dtype == np.float64
    np.savez(tmp_path / "bad.npz", psi=Psi)
    with pytest.raises(KeyError):
        atoms.load_npz(tmp_path / "bad.npz")
