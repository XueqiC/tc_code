"""Saved collect-to-codes contract and training-only Fisher geometry."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bfas.behavior import deltas, whiten  # noqa: E402
from src.bfas.behavior.response import (  # noqa: E402
    OutcomeRecord, ProbeRecord, content_hash, read_responses, write_responses,
)
from tools.behavior_atom import build_codes as codes  # noqa: E402
from tools import behavior_atom_experiment as experiment  # noqa: E402


def dump(path, value):
    path.write_text(json.dumps(value))


def seal(directory):
    dump(directory / "complete.json", dict(status="complete", exit_code=0, artifact_hashes={
        p.relative_to(directory).as_posix(): deltas.file_hash(p)
        for p in directory.rglob("*") if p.is_file() and p.name != "complete.json"}))


@pytest.fixture
def collected(tmp_path):
    run = tmp_path / "pilot"
    run.mkdir()
    # Registration/layout order deliberately differs from sorted parameter names.
    model = torch.nn.Module()
    model.register_parameter("z", torch.nn.Parameter(torch.zeros(2)))
    model.register_parameter("a", torch.nn.Parameter(torch.zeros(3)))
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    vectors = np.array([[1, 2, 0, 0, 0], [0, 1, 1, 0, 0],
                        [0, 0, 0, 2, 0], [2, 4, 0, 0, 0], [0]*5, [0]*5], dtype=np.float32)
    units, updates, outcomes = [], [], []
    ids = [f"s{i}" for i in range(len(vectors))]
    probe = ProbeRecord("p", "probe-parent", "probe-trajectory", content_hash("state"),
                        content_hash("target"), content_hash("student"), "bfcl", "dev", "probe-group")
    for i, (sid, vector) in enumerate(zip(ids, vectors)):
        with torch.no_grad():
            model.z.copy_(torch.from_numpy(vector[:2]))
            model.a.copy_(torch.from_numpy(vector[2:]))
        directory = run / "sources" / sid
        directory.mkdir(parents=True)
        delta = deltas.save_delta(model, base, directory / f"delta_{sid}.npz")
        role = "noop" if i >= 4 else "pilot"
        summary = dict(source_id=sid, role=role, delta_payload_hash=delta.manifest["payload_hash"])
        dump(directory / "summary.json", summary)
        seal(directory)
        updates.append(summary)
        units.append(dict(source_id=sid, parent_task_id=f"g{i}" if i < 4 else "control",
                          group=f"g{i}" if i < 4 else "control", role=role,
                          factor="boundary", family="family-a"))
        outcomes.append(OutcomeRecord(sid, f"run-{sid}", "p", 0, content_hash("raw"), "checker-v1",
                                      0, int(i < 4), source_factor="boundary"))
    shared = dict(identity=dict(source_ids=ids), eval_seeds=[0])
    write_responses(run / "responses.jsonl", [probe], outcomes, shared)
    dump(run / "result.json", dict(stage="pilot", status="complete", updates=updates, shared=shared))
    seal(run)
    source_path, fold_path, fisher_path = [tmp_path / name for name in ("sources.json", "folds.json", "fisher.npz")]
    dump(source_path, dict(units=units))
    dump(fold_path, dict(n_folds=4, assignment=dict(zip(ids[:4], [0, 1, 2, 0])), controls=ids[4:],
                         groups={u["source_id"]: u["group"] for u in units}))
    F = np.array([0., 1., 3., 2., 5.], dtype=np.float32)
    np.savez(fisher_path, fisher=F, layout_hash=deltas.layout_hash(model), version="synthetic-v1", eps=99.)
    return dict(run=run, sources=source_path, folds=fold_path, fisher=fisher_path,
                vectors=vectors, F=F, layout=deltas.parameter_layout(model), ids=ids)


def build(c, output=None, **kwargs):
    return codes.build_codes(c["run"], c["fisher"], c["sources"], c["folds"], output, block_size=2, **kwargs)


def test_artifacts_geometry_residuals_zero_and_response_alignment(collected):
    c = collected
    summary = build(c, pilot_smoke=True)
    out = c["run"] / "codes"
    assert summary["pilot_smoke"] == dict(passed=True, zero_source_ids=["s4", "s5"])
    assert summary["fisher"]["eps_effective"] == 1e-3
    assert summary["fisher"]["archive_eps"] == 99.
    assert summary["fisher"]["normalization"] == "none"
    assert [f["basis_dimension"] for f in summary["folds"]] == [2, 2, 2, 3]
    assert summary["folds"][3]["heldout_source_ids"] == []
    H = np.diag(c["F"].astype(float) + 1e-3)
    for item in summary["folds"] + [summary["transductive"]]:
        path = out / item["codes"]
        with np.load(path, allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        X = arrays["X"]
        train = arrays["train_indices"]
        assert X.shape == (6, item["basis_dimension"])
        np.testing.assert_array_equal(arrays["source_ids"], c["ids"])
        np.testing.assert_array_equal(arrays["folds"], [0, 1, 2, 0, -1, -1])
        np.testing.assert_array_equal(arrays["groups"], arrays["source_groups"])
        np.testing.assert_array_equal(arrays["source_factor"], ["boundary"] * 6)
        np.testing.assert_array_equal(arrays["family"], ["family-a"] * 6)
        C = c["vectors"][train].T @ arrays["decoder_coefficients"]
        np.testing.assert_allclose(C.T @ H @ C, np.eye(C.shape[1]), atol=1e-12)
        np.testing.assert_allclose(X, c["vectors"] @ H @ C, atol=1e-12)
        residual = c["vectors"] - X @ C.T
        np.testing.assert_allclose(residual @ H @ C, 0, atol=1e-12)
        expected = np.sqrt(np.einsum("ij,jk,ik->i", residual, H, residual))
        np.testing.assert_allclose(arrays["residual_norm_h"], expected, atol=1e-12)
        np.testing.assert_array_equal(X[4:], 0)
        np.testing.assert_array_equal(arrays["residual_fraction"][4:], 0)
        np.testing.assert_array_equal(arrays["residual_norm_h"][4:], 0)
        assert item["transductive"] == arrays["transductive"].item()
        decoder = json.loads(path.with_suffix(".decoder.json").read_text())
        reloaded = np.column_stack([deltas.load_delta(r["path"]).flat() for r in decoder["deltas"]])
        np.testing.assert_allclose(reloaded @ np.asarray(decoder["coefficients"]), C)
        # The existing responses loader sees the intended axes, including NA policy.
        data = experiment._jsonl_dataset(out / "responses.jsonl", arrays, {}, {})
        assert data["M"].shape == (1, 6)
        np.testing.assert_array_equal(data["M"][0], [1, 1, 1, 1, 0, 0])
    assert summary["folds"][0]["heldout_residual_fractions"]["s0"] > 0
    assert max(summary["transductive"]["residual_fractions"].values()) < 1e-12
    assert (out / "responses.jsonl").read_bytes() == (c["run"] / "responses.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        build(c)


def test_heldout_delta_cannot_change_training_basis(collected):
    c = collected
    H = whiten.from_fisher(c["F"], c["layout"], eps=.01, layout_hash=deltas.canonical_hash(c["layout"]), block_size=2)
    train = [1, 2]
    first, X, _ = codes.encode_fold(c["vectors"], H, train)
    changed = c["vectors"].copy()
    changed[0] = [1e6, 0, 1e5, 0, -1e6]
    second, new_X, residual = codes.encode_fold(changed, H, train)
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    np.testing.assert_array_equal(first.eigenvalues, second.eigenvalues)
    np.testing.assert_array_equal(X[train], new_X[train])
    assert not np.allclose(X[0], new_X[0]) and residual[0] > 1e6
    code, norm = first.encode(changed[0])
    orthogonal, direct_norm = first.residual(changed[0])
    np.testing.assert_allclose(first.decode(code, residual=orthogonal), changed[0], atol=1e-9)
    assert norm == direct_norm


def test_entirely_zero_training_span(collected):
    c = collected
    H = whiten.from_fisher(c["F"], c["layout"], eps=.01, layout_hash=deltas.canonical_hash(c["layout"]))
    basis, X, residual = codes.encode_fold(c["vectors"], H, [4, 5])
    assert basis.rank == 0 and X.shape == (6, 0)
    np.testing.assert_array_equal(residual[4:], 0)
    assert residual[0] == H.norm(c["vectors"][0])


@pytest.mark.parametrize("dry_run", [False, True])
def test_fisher_layout_mismatch_rejected_before_writes(collected, dry_run):
    c = collected
    np.savez(c["fisher"], fisher=c["F"], layout_hash="wrong")
    with pytest.raises(ValueError, match="Fisher layout hash mismatch"):
        build(c, dry_run=dry_run)
    assert not (c["run"] / "codes").exists()


def test_each_delta_layout_is_checked(collected):
    c = collected
    directory = c["run"] / "sources/s1"
    path = directory / "delta_s1.npz.manifest.json"
    manifest = json.loads(path.read_text())
    manifest["layout"][0]["name"] = "different_parameter"
    manifest["layout_hash"] = deltas.canonical_hash(manifest["layout"])
    dump(path, manifest)
    seal(directory)
    seal(c["run"])
    with pytest.raises(ValueError, match="source delta layout hash mismatch"):
        build(c, dry_run=True)


def test_cli_dry_run_and_relative_damping(collected, capsys):
    c = collected
    args = ["codes", "--run-dir", str(c["run"]), "--fisher", str(c["fisher"]),
            "--sources", str(c["sources"]), "--folds", str(c["folds"]), "--dry-run", "--pilot-smoke"]
    assert experiment.main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["sources"] == 6 and plan["number_of_models"] == plan["max_rollouts"] == 0
    assert plan["delta_payloads_verified"] is False
    assert not (c["run"] / "codes").exists()
    relative = build(c, dry_run=True, eps_mode="relative", eps=.1)
    assert relative["fisher"]["eps_effective"] == pytest.approx(.2)
    for eps in (0, -1, float("nan")):
        with pytest.raises(ValueError, match="eps"):
            build(c, dry_run=True, eps=eps)
    np.savez(c["fisher"], fisher=np.zeros(5), layout_hash=deltas.canonical_hash(c["layout"]))
    with pytest.raises(ValueError, match="positive Fisher median"):
        build(c, dry_run=True, eps_mode="relative")


def test_bad_folds_and_corrupted_payload_rejected(collected):
    c = collected
    original = c["sources"].read_bytes()
    source = json.loads(original)
    source["units"][1]["parent_task_id"] = "g0"
    dump(c["sources"], source)
    with pytest.raises(ValueError, match="group leakage"):
        build(c)
    c["sources"].write_bytes(original)
    path = c["run"] / "sources/s0/delta_s0.npz"
    with path.open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        build(c)
    assert not (c["run"] / "codes").exists()


def test_actual_merge_resolves_shards_by_hash(collected, tmp_path):
    from tools.behavior_atom.gpu_driver import merge

    c = collected
    result = json.loads((c["run"] / "result.json").read_text())
    probes, outcomes, _ = read_responses(c["run"] / "responses.jsonl")
    shards = []
    for i in range(2):
        shard = tmp_path / f"shard_{i}"
        shard.mkdir()
        updates = result["updates"][i::2]
        ids = {u["source_id"] for u in updates}
        for sid in ids:
            shutil.copytree(c["run"] / "sources" / sid, shard / "sources" / sid)
        write_responses(shard / "responses.jsonl", probes, [r for r in outcomes if r.source_id in ids], result["shared"])
        dump(shard / "result.json", {**result, "stage": "collect", "shard": [i, 2], "updates": updates})
        seal(shard)
        shards.append(shard)
    merged = tmp_path / "merged"
    merge(shards, merged)
    assert not (merged / "sources").exists()
    summary = build({**c, "run": merged})
    assert summary["sources"] == 6
    assert set(summary["shard_dirs"]) == set(map(str, shards))
    # Explicit shard locations support results moved out of the sibling tree.
    moved = tmp_path / "elsewhere/moved"
    moved.parent.mkdir()
    shutil.move(shards[0], moved)
    with pytest.raises(ValueError, match="missing merged delta shards"):
        build({**c, "run": merged}, dry_run=True)
    assert build({**c, "run": merged}, dry_run=True, shard_dirs=[moved, shards[1]])["sources"] == 6
