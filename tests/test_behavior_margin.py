"""C16 archive provenance, actual native CE, exact replay and grouped diagnostics."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from src.bfas.behavior import margin as m
from src.bfas.behavior import microupdate as micro
from src.bfas.behavior.deltas import save_delta, tensor_state_hash
from tools.behavior_atom import gpu_driver as driver
from tools import behavior_atom_experiment as cli


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def probe(pid, kind="should_abstain"):
    return dict(probe_id=pid, kind=kind, prompt="serialized prompt",
                expected="abstain" if kind == "should_abstain" else "call", truth="NEVER A TARGET")


def evaluation(pid, text, outcome, **extra):
    return dict(probe_id=pid, output=text, outcome=outcome, errors=[], **extra)


@pytest.fixture
def archive(tmp_path):
    probes = dump(tmp_path / "probes.json", dict(probes=[probe("p"), probe("q", "should_call"), probe("r")]))
    root = tmp_path / "results"
    dump(root / "collect_r2/shard-00000-of-00003/base/evaluations.json", [
        evaluation("p", "base right", 1), evaluation("q", "base wrong", 0), evaluation("r", "", 1)])
    # The winning non-base mode has three occurrences; the base candidate wins
    # even though another correct text is more frequent. q never gains a plus.
    for i, run in enumerate(("collect_r2/shard-00001-of-00003", "joint_call_r1", "joint_abstain_r1", "halfdose_r1")):
        dump(root / run / f"sources/s{i}_right/evaluations.json", [evaluation("p", "other right", 1)])
        dump(root / run / f"sources/s{i}/evaluations.json", [
            evaluation("p", "modal wrong" if i < 3 else "rare wrong", 0),
            evaluation("q", "unverified right", None), evaluation("r", "", 1)])
    # These locations are outside the registered archive selection.
    dump(root / "collect_r2/shard-00001-of-00003/base/evaluations.json", [evaluation("q", "forbidden", 1)])
    dump(root / "pilot_r2/sources/x/evaluations.json", [evaluation("q", "forbidden", 1)])
    return probes, root


def test_pairs_base_preference_mode_na_and_provenance(archive):
    path, root = archive
    result = m.build_pairs(path, root)
    p, q, r = result["probes"]
    assert p["y_plus"] == "base right"
    assert p["y_minus"] == "modal wrong"
    assert p["y_plus_provenance"][0]["state_id"] == "base"
    assert len(p["y_minus_provenance"]) == 3
    assert {x["state_id"] for x in p["y_minus_provenance"]} == {"s0", "s1", "s2"}
    assert q["y_plus"] is None and q["y_plus_provenance"] is None
    assert q["y_minus"] == "base wrong" and q["missing_sides"] == ["plus"]
    assert r["y_plus"] == "" and r["y_minus"] is None
    assert r["missing_sides"] == ["minus"]
    assert result["coverage"]["archive_both_sides"] == 1
    assert result["coverage"]["rejected_archive_records"] == 4
    assert len(result["archive_inputs"]) == 9
    assert "NEVER A TARGET" not in json.dumps(result)
    assert result == m.build_pairs(path, root)
    output = path.with_name("pairs.json")
    m._write(output, result)
    assert '"y_plus": null' in output.read_text() and "NaN" not in output.read_text()


def minimal_archive(tmp_path):
    probes = dump(tmp_path / "probes.json", dict(probes=[probe("p")]))
    root = tmp_path / "results"
    dump(root / "collect_r2/shard-00000-of-00003/base/evaluations.json", [evaluation("p", "", 1)])
    return probes, root


def test_samples_only_fill_missing_sides_and_checker_errors_are_na(tmp_path):
    probes, root = minimal_archive(tmp_path)
    samples = dict(k=3, records=[
        {**evaluation("p", "sample right", 1), "state_id": "base_sample", "sample_index": 0, "checker_version": "v"},
        {**evaluation("p", "sample wrong", 0), "state_id": "base_sample", "sample_index": 1, "checker_version": "v"},
        {**evaluation("p", "checker exploded", 0), "state_id": "base_sample", "sample_index": 2,
         "checker_version": "v", "errors": ["checker:RuntimeError"]}])
    result = m.build_pairs(probes, root, samples=samples)
    p = result["probes"][0]
    assert p["y_plus"] == ""
    assert p["y_minus"] == "sample wrong"
    assert p["y_minus_provenance"][0]["state_id"] == "base_sample"
    assert result["base_sampling"]["checked"] == 2
    assert result["base_sampling"]["supplemented_sides"] == [dict(probe_id="p", side="minus")]
    assert result["coverage"]["archive_both_sides"] == 0 and result["coverage"]["both_sides"] == 1
    samples["records"][0]["state_id"] = "teacher"
    with pytest.raises(ValueError, match="base_sample"):
        m.build_pairs(probes, root, samples=samples)


def test_mode_tie_is_lexical_and_errors_never_supply_a_side(tmp_path):
    probes, root = minimal_archive(tmp_path)
    for sid, text, errors in [("a", "zzz", []), ("b", "aaa", []), ("c", "000", ["parse:ValueError"])]:
        dump(root / f"joint_call_r1/sources/{sid}/evaluations.json", [
            {**evaluation("p", text, 0), "errors": errors}])
    assert m.build_pairs(probes, root)["probes"][0]["y_minus"] == "aaa"


class Tokenizer:
    eos_token_id, pad_token_id = 0, 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [2] * len(text) if text != "minus" else [3]}


class TinyConfig:
    use_cache = False

    def to_dict(self):
        return {"use_cache": self.use_cache, "kind": "tiny"}


class TinyModel(torch.nn.Module):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0., 0., 1., -1.], dtype=dtype))
        self.register_buffer("buffer", torch.ones(1))
        self.config = TinyConfig()

    def forward(self, input_ids, attention_mask=None):
        # Position dependence catches wrong shifted token positions/prompt loss.
        positions = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=self.weight.dtype)
        slopes = torch.tensor([0., 0., 0.125, -0.125], device=input_ids.device, dtype=self.weight.dtype)
        logits = self.weight[None, None, :] + positions[None, :, None] * slopes
        return SimpleNamespace(logits=logits.expand(len(input_ids), -1, -1))


def tiny_student(protocol=None):
    return driver.Student(TinyModel(), Tokenizer(), "fake", None, None)


def protocol(**overrides):
    return driver.frozen_settings(dict(micro_update=dict(device="cpu", prompt_cap=8, **overrides),
                                       evaluation=dict(batch_size=4, head_chunk=2)))


@pytest.fixture(autouse=True)
def capped_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_native_ce_fp32_sum_eos_shift_mask_and_missing_side(monkeypatch):
    student, cfg = tiny_student(), protocol()
    pairs = [dict(probe_id="p", prompt="p", y_plus="++", y_minus="minus"),
             dict(probe_id="q", prompt="q", y_plus="", y_minus=None),
             dict(probe_id="r", prompt="r", y_plus=None, y_minus=None)]
    monkeypatch.setattr(student, "generate", lambda *a, **kw: pytest.fail("scoring must never decode"))
    result = m.score_pairs(student, pairs, cfg)
    for row, p in zip(result[:2], pairs):
        prompt, _ = m._prompt_ids(student.tokenizer, p["prompt"], cfg)
        for side in ("plus", "minus"):
            if p[f"y_{side}"] is None:
                assert row[f"logp_{side}"] is None
                continue
            response = student.tokenizer(p[f"y_{side}"])["input_ids"] + [0]
            ids = torch.tensor([prompt+response])
            logits = student.model(ids).logits[0, len(prompt)-1:-1]
            expected = -F.cross_entropy(logits, torch.tensor(response), reduction="none").float().sum()
            assert row[f"logp_{side}"] == expected.item()
            assert row["counts"][side]["effective_loss_tokens"] == len(response)
    assert result[0]["m"] == result[0]["logp_plus"] - result[0]["logp_minus"]
    assert result[1]["m"] is None and result[2]["m"] is None
    assert result[1]["counts"]["plus"]["effective_loss_tokens"] == 1  # Empty text scores EOS.
    assert result == m.score_pairs(student, pairs, cfg)
    # A long response remains whole, even beyond the driver's training cap.
    long = m.score_pairs(student, [dict(probe_id="long", prompt="x", y_plus="x"*520, y_minus=None)], cfg)
    assert long[0]["counts"]["plus"]["effective_loss_tokens"] == 521
    assert long[0]["counts"]["plus"]["response_tokens_dropped"] == 0


def test_score_matches_driver_likelihood_within_response_cap():
    student, cfg = tiny_student(), protocol()
    p = dict(probe_id="p", prompt="p", y_plus="++", y_minus="minus")
    score = m.score_pairs(student, [p], cfg)[0]
    # Exercise the existing differentiable likelihood, with the same encoder.
    import appworld_train as trainer
    with driver._context(cfg):
        for side in ("plus", "minus"):
            ids, labels = trainer.encode(student.tokenizer, dict(prompt=driver._thinking_off("p"),
                                                               response=p[f"y_{side}"]), device="cpu")
            expected = micro.evaluation_score(student.model, ids, labels)
            assert score[f"logp_{side}"] == expected.item()


@pytest.mark.parametrize("missing_trainables_hash", [False, True])
def test_state_discovery_exact_replay_matrix_and_scoring_outputs(tmp_path, missing_trainables_hash):
    model = TinyModel()
    initial = tmp_path / "initial.pt"
    micro.save_initial_state(model, initial)
    if missing_trainables_hash:
        state = torch.load(initial, weights_only=True)
        del state["trainables_hash"]
        torch.save(state, initial)
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    run = tmp_path / "run"
    for sid, change in [("a", 1.), ("b", -1.), ("zero", 0.)]:
        path = run / sid / f"delta_{sid}.npz"
        path.parent.mkdir(parents=True)
        with torch.no_grad():
            model.weight.copy_(base["weight"])
            model.weight[2] += change
        save_delta(model, base, path)
    states = m.discover_states([run, run], tmp_path)
    assert [s["state_id"] for s in states] == ["base", "a", "b", "zero"]
    bundle = dict(probes=[dict(probe_id="p", prompt="p", y_plus="+", y_minus="minus")],
                  coverage={}, base_sampling={})
    out = tmp_path / "out"
    matrix = m.score_states(bundle, states, protocol(init_state_path=str(initial)), tmp_path, out,
                            student_factory=tiny_student)
    assert matrix["runtime"]["initial_trainables_hash"] == tensor_state_hash(base)
    assert matrix["m"][0] == matrix["m"][3]
    assert matrix["m"][1][0] > matrix["m"][0][0] > matrix["m"][2][0]
    rows = [json.loads(r) for r in (out / "margins.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    summary = m._read(out / "summary.json")
    assert summary["status"] == "scored"
    assert summary["runtime"] == matrix["runtime"]
    assert m._read(out / "margin_matrix.json") == matrix
    with pytest.raises(ValueError, match="duplicate/missing"):
        m.margin_matrix(states, bundle, rows[:-1])


def test_base_student_accepts_legacy_settings_and_restores_buffers(tmp_path):
    initial = tmp_path / "legacy.pt"
    micro.save_initial_state(TinyModel(), initial)
    state = torch.load(initial, weights_only=True)
    del state["trainables_hash"]
    del state["model_settings"]["model_config"]["kind"]
    state["buffers"]["buffer"].fill_(7)
    assert "buffers_hash" not in state
    torch.save(state, initial)
    with pytest.warns(UserWarning, match="model settings differ: model_config.kind"):
        with m._base_student(protocol(init_state_path=str(initial)), student_factory=tiny_student) as (student, _, runtime):
            assert torch.equal(student.model.buffer, state["buffers"]["buffer"])
            assert tensor_state_hash(micro._trainables(student.model)) == tensor_state_hash(state["trainables"])
            assert runtime["init_state_settings_diff"] == ["model_config.kind"]
            assert runtime["initial_trainables_hash"] == tensor_state_hash(state["trainables"])


@pytest.mark.parametrize("mismatch", ["missing_hash_wrong_tensors", "valid_hash_wrong_tensors", "wrong_hash"])
def test_base_student_rejects_trainables_mismatch(tmp_path, mismatch):
    initial = tmp_path / "wrong.pt"
    micro.save_initial_state(TinyModel(), initial)
    state = torch.load(initial, weights_only=True)
    if mismatch == "wrong_hash":
        state["trainables_hash"] = "incorrect"
    else:
        state["trainables"]["weight"].add_(1)
        if mismatch == "missing_hash_wrong_tensors":
            del state["trainables_hash"]
        else:
            state["trainables_hash"] = tensor_state_hash(state["trainables"])
    torch.save(state, initial)
    with pytest.raises(ValueError, match="initial-state trainables hash mismatch"):
        with m._base_student(protocol(init_state_path=str(initial)), student_factory=tiny_student):
            pytest.fail("mismatched initial trainables were accepted")


def test_temperature_samples_are_seeded_restored_base_only(tmp_path, monkeypatch):
    probes, root = minimal_archive(tmp_path)
    bundle = m.build_pairs(probes, root)
    initial = tmp_path / "initial.pt"
    micro.save_initial_state(TinyModel(), initial)
    checked, closed = [], []

    def factory(cfg):
        student = tiny_student()
        student.tokenizer.batch_decode = lambda outputs, **kw: ["call" if int(outputs[0, 0]) else ""]
        student.parse = lambda text: [{"f": {}}] if text else []

        def check(p, calls):
            checked.append(p["probe_id"])
            return dict(valid=not calls)

        def generate(input_ids, **kwargs):
            assert kwargs["do_sample"] and kwargs["temperature"] == 0.8
            assert not torch.is_grad_enabled()
            assert torch.equal(student.model.weight, TinyModel().weight)
            return torch.cat((input_ids, torch.randint(2, (1, 1))), dim=1)

        student.check = check
        student.model.generate = generate
        student._close_checker = lambda: closed.append(True)
        return student

    monkeypatch.setattr(driver, "load_student", factory)
    cfg = protocol(init_state_path=str(initial))
    cfg["eval_seeds"] = [19]
    samples = m.sample_base(probes, bundle, cfg, 8)
    assert samples == m.sample_base(probes, bundle, cfg, 8)
    assert samples["seed"] == 19 and len(samples["records"]) == 8
    assert len(checked) == 16 and len(closed) == 2
    assert {r["state_id"] for r in samples["records"]} == {"base_sample"}
    assert all(r["outcome"] in (0, 1) and r["checker_version"] == "fake" for r in samples["records"])


def diagnostic_fixture():
    pairs = [dict(probe_id=f"p{j}", kind="should_abstain" if j < 3 else "should_call",
                  base_outcome=1 if j != 2 else 0, y_plus="+", y_minus=None if j == 1 else "-",
                  archive_both_sides=j != 1) for j in range(6)]
    bundle = dict(probes=pairs, coverage=dict(probes=6, archive_both_sides=5, both_sides=5,
                                             missing_plus=0, missing_minus=1),
                  base_sampling=dict(enabled=False, attempted=0, checked=0, supplemented_sides=[]))
    ids = ["base"] + list(m.SINGLES)
    values = [[0., np.nan, 0., 0., 0., 0.]] + [[1., np.nan, 1., 1., 1., 1.]]*4
    for single in m.SINGLES:
        for anchor in ("anchor_match", "anchor_random", "abstain_anchor"):
            ids.append(f"joint_{single}_{anchor}")
            values.append([3., np.nan, 2., 1., 0., 3.])
    ids += ["half_src_17", "half_src_04"]
    values += [[1., np.nan, 1., 1., 1., 1.]]*2
    return dict(state_ids=ids, probe_ids=[p["probe_id"] for p in pairs], m=values), bundle


def test_q1_retained_mask_pack_mapping_and_binary_invisible_gains():
    matrix, bundle = diagnostic_fixture()
    bundle["archived_outcomes"] = {s: {"p0": 1} for s in matrix["state_ids"]}
    result = m.analyze_q1(matrix, bundle)
    assert result["retained_probe_ids"] == ["p0", "p1"]
    assert len(result["packs"]) == 12
    for r in result["packs"]:
        assert r["single"] in m.SINGLES
        assert r["both_sides"] == r["na_count"] == 1
        assert r["mean_delta_m"] == 2
        assert r["per_probe_delta_m"] == {"p0": 2., "p1": None}
        assert r["unchanged_binary_count"] == 1 and r["unchanged_binary_mean_delta_m"] == 2
    matrix["state_ids"][1] = "missing_src_17"
    assert m.analyze_q1(matrix, bundle)["packs"][0]["na_count"] == 2


def test_q2_profiles_paired_bootstrap_and_undefined_cosine():
    matrix, bundle = diagnostic_fixture()
    result = m.analyze_q2(matrix, bundle, bootstrap_samples=100, seed=7)
    for r in result["comparisons"]:
        expected = m._profile_metrics(np.array([3., 2., 1., 0., 3.]), np.ones(5))
        assert r["both_sides"] == 5 and r["na_count"] == 1
        for metric, value in expected.items():
            assert r[metric] == pytest.approx(value)
        assert r["bootstrap_ci95"]["l1"][0] <= r["l1"] <= r["bootstrap_ci95"]["l1"][1]
    assert result == m.analyze_q2(matrix, bundle, bootstrap_samples=100, seed=7)
    row = matrix["state_ids"].index("half_src_17")
    matrix["m"][row] = matrix["m"][0].copy()
    q2 = m.analyze_q2(matrix, bundle, bootstrap_samples=30)
    assert q2["comparisons"][0]["cosine"] is None
    assert q2["comparisons"][0]["bootstrap_ci95"]["cosine"] is None


def q3_fixture(signal=True):
    rng = np.random.default_rng(102)
    # Balanced ±1 contrasts, repeated across four disjoint parent groups.
    block = np.array(np.meshgrid(*[[-1., 1.]]*3)).reshape(3, -1).T
    X = np.tile(np.column_stack((np.ones(8), block)), (4, 1))
    X[:, 0] = np.tile(np.linspace(1., 2., 8), 4)
    norms = np.sqrt((X*X).sum(axis=1)+np.tile(np.linspace(0., 1., 8), 4))
    ids, labels = [f"s{i}" for i in range(len(X))], np.repeat(np.arange(4), 8)
    Y = (3*X[:, 0]+norms)[:, None] * np.array([[1., -1., 0.5]])
    if signal:
        Y += (X[:, 2]-X[:, 3])[:, None] * np.array([[8., 3., -5.]])
    Y[:, 2] = np.nan
    Y[2, 1] = np.nan
    folds = []
    for f in range(4):
        # Coordinate rotations vary by fold, as real per-fold bases do.
        rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        folds.append(dict(fold=f, source_ids=ids, source_groups=[f"g{g}" for g in labels], X=X@rotation,
                          delta_norm_h=norms.copy(), train_indices=np.flatnonzero(labels != f),
                          test_indices=np.flatnonzero(labels == f)))
    pairs = [dict(probe_id=f"p{j}", kind="should_call", base_outcome=0, y_plus="+", y_minus="-") for j in range(3)]
    bundle = dict(probes=pairs, coverage=dict(probes=3, archive_both_sides=2, both_sides=2,
                                            missing_plus=1, missing_minus=0),
                  base_sampling=dict(enabled=False, attempted=0, checked=0, supplemented_sides=[]))
    matrix = dict(state_ids=["base"]+ids, probe_ids=[p["probe_id"] for p in pairs],
                  m=np.vstack((np.zeros(3), Y)), pairs_sha256=m._digest(bundle))
    return matrix, bundle, folds


def test_q3_detects_extra_signal_uses_four_folds_and_five_permutations():
    matrix, bundle, folds = q3_fixture()
    result = m.analyze_q3(matrix, bundle, folds, ridge=0.01)
    assert result["status"] == "ok" and result["cells"] == 63
    assert result["exceeds_common_norm_and_all_permutations"]
    assert result["mse"]["common+norm+codes ridge"] < result["mse"]["common+norm"] * 0.01
    assert len(result["mse"]) == 7 and len(result["folds"]) == 4
    assert result["permutation_rank_p"] == 1/6
    assert result == m.analyze_q3(matrix, bundle, folds, ridge=0.01)
    assert all(row[2] is None for row in result["predictions"]["common+norm"])
    # Changing ONLY held-out outcomes must not change that fold's predictions.
    changed = copy.deepcopy(matrix)
    changed["m"][1:9, :2] += 1000
    again = m.analyze_q3(changed, bundle, folds, ridge=0.01)
    for model in result["predictions"]:
        assert result["predictions"][model][:8] == again["predictions"][model][:8]


def test_q3_baseline_signal_and_leaking_groups(tmp_path):
    matrix, bundle, folds = q3_fixture(signal=False)
    result = m.analyze_q3(matrix, bundle, folds)
    assert result["mse"]["common+norm"] < 1e-20
    assert abs(result["mse_improvement"]) < 1e-20
    assert not result["exceeds_common_norm_and_all_permutations"]
    assert m.analyze(matrix, bundle, folds, bootstrap_samples=10)["verdict"] == m.VERDICT_RULES[0]
    broken = copy.deepcopy(folds)
    for f in broken:
        f["source_groups"][8] = "g0"
    with pytest.raises(ValueError, match="group leaks"):
        m.analyze_q3(matrix, bundle, broken)
    with pytest.raises(ValueError, match="five"):
        m.analyze_q3(matrix, bundle, folds, permutation_seeds=[1])


def test_fold_loader_rejects_transductive_and_wrong_partition(tmp_path):
    _, _, folds = q3_fixture()
    registered = dict(n_folds=4, assignment={f"s{i}": i//8 for i in range(32)},
                      groups={f"s{i}": f"g{i//8}" for i in range(32)})
    fold_path = dump(tmp_path / "folds.json", registered)
    for f in folds:
        path = tmp_path / f"fold_{f['fold']}" / "codes.npz"
        path.parent.mkdir()
        np.savez(path, **f, heldout_fold=f["fold"], basis_scope="training_fold", transductive=False,
                 folds=[registered["assignment"][s] for s in f["source_ids"]],
                 decoder_source_ids=[f["source_ids"][i] for i in f["train_indices"]])
    assert len(m.load_fold_codes(tmp_path, fold_path)) == 4
    path = tmp_path / "fold_0/codes.npz"
    with np.load(path) as data:
        fields = dict(data)
    fields["transductive"] = True
    np.savez(path, **fields)
    with pytest.raises(ValueError, match="transductive"):
        m.load_fold_codes(tmp_path, fold_path)
    fields["transductive"] = False
    fields["train_indices"] = np.arange(32)
    np.savez(path, **fields)
    with pytest.raises(ValueError, match="decoder"):
        m.load_fold_codes(tmp_path, fold_path)


def test_report_exact_three_questions_fixed_rules_and_na(tmp_path):
    matrix, bundle, folds = q3_fixture()
    result = m.analyze(matrix, bundle, folds, ridge=0.01, bootstrap_samples=30)
    report = m.report_markdown(result)
    assert sum(line.startswith("## Q") for line in report.splitlines()) == 3
    assert all(rule in report for rule in m.VERDICT_RULES)
    assert result["verdict"] == m.VERDICT_RULES[1]
    assert "NA" in report and "pre-specified NEW-combination" in report
    m._write(tmp_path / "summary.json", result)
    assert "NaN" not in (tmp_path / "summary.json").read_text()
    matrix["m"][:] = np.nan
    result = m.analyze(matrix, bundle, folds, bootstrap_samples=30)
    assert result["verdict"].startswith("INCONCLUSIVE")
    bundle["probes"][0]["y_plus"] = "changed"
    with pytest.raises(ValueError, match="different probe pairs"):
        m.analyze(matrix, bundle, folds)


def test_cli_dispatch_and_cpu_pair_stage_without_torch(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["margin", "--help"])
    assert exc.value.code == 0
    assert "--sample-base" in capsys.readouterr().out
    probes, root = minimal_archive(tmp_path)
    cfg = dict(probes="probes.json", results_root="results", pairs="pairs.json", output_dir="out",
               codes_dir="codes", folds="folds.json", run_dirs=["results"],
               protocol=dict(micro_update=dict(init_state_path="missing.pt")))
    path = dump(tmp_path / "config.json", cfg)
    # Guard imports, not just CUDA calls: archive pairing is truly CPU/stdlib+numpy.
    script = ("import sys, importlib.abc\n"
              "class NoTorch(importlib.abc.MetaPathFinder):\n"
              " def find_spec(self, fullname, *args):\n"
              "  if fullname == 'torch': raise AssertionError('CPU pairs imported torch')\n"
              "sys.meta_path.insert(0, NoTorch())\n"
              "from tools.behavior_atom_experiment import main\n"
              f"raise SystemExit(main(['margin','--config',{str(path)!r},'--step','pairs']))\n")
    run = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert m._read(tmp_path / "pairs.json")["probes"][0]["y_minus"] is None
    assert not (tmp_path / "out").exists()
    assert cli.main(["margin", "--sample-base", "1", "--step", "score"]) == 2


def test_cpu_analysis_cli_writes_real_artifacts_from_fold_npzs(tmp_path):
    matrix, bundle, folds = q3_fixture()
    probes = dump(tmp_path / "probes.json", {"probes": bundle["probes"]})
    bundle["probes_sha256"] = m._hash(probes)
    dump(tmp_path / "pairs.json", bundle)
    matrix.update(pairs_sha256=m._digest(bundle), runtime=dict(initial_trainables_hash="base-hash"),
                  states=[dict(state_id=s, delta_sha256="delta-"+s) for s in matrix["state_ids"]])
    m._write(tmp_path / "out/margin_matrix.json", matrix)
    registered = dict(n_folds=4, assignment={f"s{i}": i//8 for i in range(32)},
                      groups={f"s{i}": f"g{i//8}" for i in range(32)})
    dump(tmp_path / "folds.json", registered)
    for f in folds:
        path = tmp_path / "codes" / f"fold_{f['fold']}" / "codes.npz"
        path.parent.mkdir(parents=True)
        np.savez(path, **f, heldout_fold=f["fold"], basis_scope="training_fold", transductive=False,
                 folds=[registered["assignment"][s] for s in f["source_ids"]],
                 decoder_source_ids=[f["source_ids"][i] for i in f["train_indices"]])
    dump(tmp_path / "codes/summary.json", dict(base_hash="base-hash", deltas=[
        dict(source_id=s, content_hash="delta-"+s) for s in folds[0]["source_ids"]]))
    cfg = dump(tmp_path / "config.json", dict(
        probes="probes.json", pairs="pairs.json", results_root="results", output_dir="out",
        codes_dir="codes", folds="folds.json", run_dirs=["results"],
        protocol=dict(micro_update=dict(init_state_path="missing.pt")),
        analysis=dict(ridge=0.01, bootstrap_samples=10)))
    script = ("import sys, importlib.abc\n"
              "class NoTorch(importlib.abc.MetaPathFinder):\n"
              " def find_spec(self, fullname, *args):\n"
              "  if fullname == 'torch': raise AssertionError('CPU analyze imported torch')\n"
              "sys.meta_path.insert(0, NoTorch())\n"
              "from tools.behavior_atom_experiment import main\n"
              f"raise SystemExit(main(['margin','--config',{str(cfg)!r},'--step','analyze']))\n")
    run = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    summary = m._read(tmp_path / "out/summary.json")
    assert summary["Q3"]["exceeds_common_norm_and_all_permutations"]
    report = (tmp_path / "out/report.md").read_text()
    assert run.stdout == report + "\n"
    assert "## Q1" in report and "## Q2" in report and "## Q3" in report
