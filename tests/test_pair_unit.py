"""C20 CPU-only math, retention, exposure and legacy-invariance contracts."""
from __future__ import annotations

import ast
from collections import Counter
from contextlib import contextmanager
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import appworld_train as legacy
from bfas.arms import (pair_side_rows, pair_unit_loss, pair_unit_margins,
                       same_seed_pair_counts, shuffle_pair_sides)
from bfas.pair_unit import (categorical_kl, configuration, load_anchors, load_pairs,
                            response_kl, step_schedule, train)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith("AW_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("AW_MAX_PROMPT_TOKENS", "9")
    monkeypatch.setenv("AW_TRUNCATE_SIDE", "tail")
    # Tiny CPU tensors run much faster without a large BLAS thread pool.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def raw_pairs(n_seeds=3):
    return [dict(pair_id=f"p{kind}{i}", type=kind, seed_function=f"f{i}", split="train",
                 sides=[dict(prompt=f"state{kind}{i}{side}" * (i + 1),
                             y_plus=f"good{i}{side}", y_minus=f"bad{side}" if i else None,
                             side_id=f"p{kind}{i}:{side}", provenance={"source": i})
                        for side in range(2)])
            for kind in ("1", "3") for i in range(n_seeds)]


def normalized_pairs(tmp_path, n_seeds=3):
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps({"pairs": raw_pairs(n_seeds)}))
    return load_pairs(path)


@pytest.fixture
def imbalanced_pair_source(tmp_path):
    data = raw_pairs(n_seeds=6)
    for i, pair in enumerate(data[:6]):
        pair.update(type="binding", seed_function=(
            "simple_javascript_42" if i < 5 else "live_parallel_multiple_10-9-0"))
    source, split = tmp_path / "pairs.json", tmp_path / "split.json"
    source.write_text(json.dumps(data))
    split.write_text(json.dumps({"train_pair_ids": [p["pair_id"] for p in data]}))
    return source, split


def assert_shuffle_preserves_sides(pairs, shuffled):
    assert len(shuffled) == len(pairs)
    # Count whole side records to cover prompt/label bytes and all provenance,
    # preserving occurrences even when multiple pairs contain the same state.
    for index in range(2):
        records = lambda ps: Counter(json.dumps(p["sides"][index], sort_keys=True) for p in ps)
        assert records(shuffled) == records(pairs)
    by_id = {p["pair_id"]: p for p in pairs}
    for before, after in zip(pairs, shuffled):
        assert before["pair_id"] == after["pair_id"] == after["source_pair_ids"][0]
        assert before["sides"][0] == after["sides"][0]
        assert after["source_pair_ids"][0] != after["source_pair_ids"][1]
        donor = by_id[after["source_pair_ids"][1]]
        assert before["type"] == donor["type"] == after["type"]
        assert after["sides"][1] == donor["sides"][1]
        assert after["pairing"] == "shuffled"


@pytest.mark.parametrize("reduction,expected,gradient", [
    ("worst", [2.25, 0.0], [[-3.0, 0.0], [0.0, 0.0]]),
    ("sum", [2.5, 0.0], [[-3.0, -1.0], [0.0, 0.0]]),
])
def test_joint_objective_math_and_gradient(reduction, expected, gradient):
    margins = torch.tensor([[-0.5, 0.5], [1.5, 2.0]], requires_grad=True)
    loss = pair_unit_loss(margins, gamma=1, reduction=reduction)
    torch.testing.assert_close(loss, torch.tensor(expected))
    loss.sum().backward()
    torch.testing.assert_close(margins.grad, torch.tensor(gradient))


def test_tie_and_configurable_gamma():
    margins = torch.zeros(2, requires_grad=True)
    loss = pair_unit_loss(margins, gamma=2)
    assert loss.item() == 4
    loss.backward()
    torch.testing.assert_close(margins.grad, torch.tensor([-2., -2.]))
    assert pair_unit_loss(margins, gamma=2, reduction="sum").item() == 8


def test_base_centred_margin_missing_and_mixed_rejects():
    plus = torch.tensor([-2., -3.], requires_grad=True)
    base = torch.tensor([-4., -4.])
    reject, base_reject = torch.tensor([-3., 0.]), torch.tensor([-4., 0.])
    torch.testing.assert_close(pair_unit_margins(plus, base, reject, base_reject, beta=.2),
                               torch.tensor([.2, .2]))
    torch.testing.assert_close(pair_unit_margins(plus, base, beta=.5), torch.tensor([1., .5]))
    torch.testing.assert_close(pair_unit_margins(base, base, reject, reject), torch.zeros(2))
    with pytest.raises(ValueError, match="together"):
        pair_unit_margins(plus, base, reject)


@pytest.mark.parametrize("kwargs", [dict(gamma=-1), dict(gamma=float("nan")), dict(reduction="mean")])
def test_invalid_objective(kwargs):
    with pytest.raises(ValueError):
        pair_unit_loss(torch.zeros(2), **kwargs)


@pytest.mark.parametrize("beta", [0, -1, float("inf")])
def test_invalid_beta(beta):
    with pytest.raises(ValueError, match="beta"):
        pair_unit_margins(torch.zeros(2), torch.zeros(2), beta=beta)


def test_two_sides_required():
    with pytest.raises(ValueError, match="exactly two"):
        pair_unit_loss(torch.zeros(3))


def test_shuffle_preserves_labels_provenance_and_groups(tmp_path):
    pairs = normalized_pairs(tmp_path, n_seeds=4)
    original = copy.deepcopy(pairs)
    shuffled = shuffle_pair_sides(pairs, seed=19)
    assert shuffled == shuffle_pair_sides(pairs, seed=19)
    assert pairs == original
    assert shuffled != shuffle_pair_sides(pairs, seed=31)
    assert_shuffle_preserves_sides(pairs, shuffled)
    assert same_seed_pair_counts(shuffled) == {"1": 0, "3": 0}
    for after in shuffled:
        assert after["sides"][0]["seed_function"] != after["sides"][1]["seed_function"]
    shuffled[0]["sides"][1]["y_plus"] = "mutated"
    assert pairs == original


@pytest.mark.parametrize("seeds,expected", [(["a", "a"], 2), (["a", "a", "a"], 3),
                                           (["a", "a", "b"], 1), (["a", "a", "b", "b"], 0)])
def test_shuffle_prefers_cross_seed_and_falls_back_within_type(tmp_path, seeds, expected):
    pairs = normalized_pairs(tmp_path, n_seeds=len(seeds))[:len(seeds)]
    for pair, seed in zip(pairs, seeds):
        pair["seed_function"] = seed
        for side in pair["sides"]:
            side["seed_function"] = seed
    original = copy.deepcopy(pairs)
    for seed in (0, 19, 31):
        shuffled = shuffle_pair_sides(pairs, seed=seed)
        assert shuffled == shuffle_pair_sides(pairs, seed=seed)
        assert_shuffle_preserves_sides(pairs, shuffled)
        assert same_seed_pair_counts(shuffled) == {"1": expected}
    assert pairs == original


def test_shuffle_five_plus_one_binding_pairs(imbalanced_pair_source):
    pairs = load_pairs(*imbalanced_pair_source)
    original = copy.deepcopy(pairs)
    outputs = []
    for seed in (0, 19, 31):
        shuffled = shuffle_pair_sides(pairs, seed=seed)
        assert shuffled == shuffle_pair_sides(pairs, seed=seed)
        assert_shuffle_preserves_sides(pairs, shuffled)
        assert same_seed_pair_counts(shuffled) == {"binding": 4, "3": 0}
        outputs.append(shuffled)
    assert outputs[0] != outputs[1]
    assert pairs == original
    outputs[0][0]["sides"][1]["provenance"]["source"] = "mutated"
    assert pairs == original


def test_singleton_shuffle_fails_without_dropping_sides(tmp_path):
    pairs = normalized_pairs(tmp_path)
    pairs.pop()  # One normal type followed by a singleton type.
    pairs.pop()
    original = copy.deepcopy(pairs)
    with pytest.raises(ValueError, match="type '3': no complete different-pair-id derangement"):
        shuffle_pair_sides(pairs)
    assert pairs == original


@pytest.mark.parametrize("wrapper", [None, "pairs", "confused_pairs"])
def test_loader_and_flattening(tmp_path, wrapper):
    data = raw_pairs()
    data[0]["sides"][0]["y_minus"] = ""
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps({wrapper: data} if wrapper else data))
    pairs = load_pairs(path)
    flat = pair_side_rows(pairs)
    assert len(flat) == 2 * len(data)
    for row, side in zip(flat, [s for p in data for s in p["sides"]]):
        assert row["prompt"] == side["prompt"]
        assert row["response"] == side["y_plus"]
        assert row.get("_rejected") == (side["y_minus"] or None)
        assert row["provenance"] == side["provenance"]


def test_id_file_and_explicit_split(tmp_path):
    data = raw_pairs()
    for pair in data:
        pair.pop("split")
    (tmp_path / "candidates.jsonl").write_text("".join(json.dumps(p) + "\n" for p in data))
    selected = tmp_path / "confused_pairs.json"
    selected.write_text(json.dumps({"confused_pair_ids": [p["pair_id"] for p in data]}))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train_pair_ids": [data[0]["pair_id"]],
                                 "test_pair_ids": [data[1]["pair_id"]]}))
    assert [p["pair_id"] for p in load_pairs(selected, split)] == [data[0]["pair_id"]]
    with pytest.raises(ValueError, match="explicit train split"):
        load_pairs(selected)
    split.write_text(json.dumps({"train": [data[0]["pair_id"]], "test": [data[0]["pair_id"]]}))
    with pytest.raises(ValueError, match="overlap"):
        load_pairs(selected, split)


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p[0].update(sides=[]), "exactly two"),
    (lambda p: p[0]["sides"][0].update(y_plus=""), "y_plus"),
    (lambda p: p[0]["sides"][0].update(y_minus=7), "y_minus"),
    (lambda p: p[0].pop("seed_function"), "seed_function"),
    (lambda p: p[0]["sides"][0].update(seed_function="other"), "share a seed"),
    (lambda p: p[0]["sides"][0].update(_anchor=True), "anchor"),
    (lambda p: p.append(p[0]), "unique"),
])
def test_loader_rejects_invalid_pairs(tmp_path, mutation, match):
    data = raw_pairs()
    mutation(data)
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match=match):
        load_pairs(path)


def test_schedule_exact_pass_and_partial_step_rejection():
    config = configuration({"AW_PAIR_BATCH_UNITS": "2", "AW_PAIR_STEPS": "6"})
    batches = list(step_schedule(5, config, np.random.default_rng(3)))
    assert len(batches) == 6
    assert Counter(i for b in batches for i in b) == Counter({i: 2 for i in range(5)})
    config["steps"] = 5
    with pytest.raises(ValueError, match="multiple of 3"):
        list(step_schedule(5, config, np.random.default_rng(3)))


def test_full_vocabulary_kl_is_not_sampled_logratio():
    policy = torch.tensor([[.8, .2], [.1, .9]], dtype=torch.float64)
    base = torch.tensor([[.5, .5], [.25, .75]], dtype=torch.float64)
    expected = (policy * (policy / base).log()).sum(-1)
    actual = categorical_kl(policy.log(), base.log())
    torch.testing.assert_close(actual.double(), expected)
    assert (actual > 0).all()
    torch.testing.assert_close(categorical_kl(base.log(), base.log()), torch.zeros(2))


class TinyTokenizer:
    eos_token_id = 5

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(char) % 5 for char in text]}


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("base", torch.arange(36, dtype=torch.float32).reshape(6, 6) / 20)
        self.delta = torch.nn.Parameter(torch.zeros(6, 6))
        self.disabled = False
        self.config = SimpleNamespace(use_cache=True)

    def forward(self, input_ids):
        table = self.base if self.disabled else self.base + self.delta
        return SimpleNamespace(logits=table[input_ids])

    @contextmanager
    def disable_adapter(self):
        before = self.disabled
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = before


def anchor_rows():
    return [dict(task_id=f"anchor{i}", prompt="anchor context" + str(i),
                 response=("<tool_call>yes" if i < 2 else "no"), _rejected="reject",
                 _anchor=True, _event_dU=0., teacher="self_anchor", turn_index=0, token_hint=1)
            for i in range(4)]


def test_three_arm_training_tokens_margins_kl_and_updates(tmp_path):
    pairs = normalized_pairs(tmp_path)
    config = configuration({"AW_PAIR_BATCH_UNITS": "2", "AW_DDPO_LR": ".01"})
    result = []
    for arm in "ABC":
        model = TinyPolicy()
        source = shuffle_pair_sides(pairs, seed=0) if arm == "C" else pairs
        cfg = {**config, "mode": "ddpo" if arm == "A" else "pair_unit"}
        path = tmp_path / f"{arm}.jsonl"
        report = train(model, TinyTokenizer(), source, anchor_rows(), seed=0, config=cfg,
                        trace_path=path, trainer=legacy)
        trace = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(trace) == report["optimizer_steps"] == 5
        assert model.config.use_cache is True and model.disabled is False
        assert model.delta.abs().sum() > 0
        for entry in trace:
            assert entry["kl_to_base"] >= -1e-7
            assert entry["tokens"]["diagnostic_input_tokens"] == 2 * entry["tokens"]["train_input_tokens"]
            assert entry["kl_response_tokens"] == entry["tokens"]["train_response_tokens"]
            assert all(len(p["margins"]) == (1 if p["anchor"] else 2) for p in entry["pairs"])
        assert trace[-1]["kl_to_base"] > 0
        assert trace[-1]["cumulative_tokens"] == report["tokens"]
        expected_input = expected_response = 0
        for row in pair_side_rows(source) + anchor_rows():
            for branch in [row] + ([legacy.rejected_row(row)] if row.get("_rejected") else []):
                ids, labels = legacy.encode(TinyTokenizer(), branch, device="cpu")
                expected_input += ids.numel()
                expected_response += int((labels[:, 1:] != -100).sum())
        assert report["tokens"]["train_input_tokens"] == expected_input
        assert report["tokens"]["train_response_tokens"] == expected_response
        result.append((model.delta.detach().clone(), report, trace))
    assert result[0][1] == result[1][1] == result[2][1]
    assert not torch.equal(result[0][0], result[1][0])
    assert not torch.equal(result[1][0], result[2][0])
    # The anchor identities, order and step remain equal despite C's re-pairing.
    anchors = lambda tr: [[p["side_ids"] for p in step["pairs"] if p["anchor"]] for step in tr]
    assert anchors(result[0][2]) == anchors(result[1][2]) == anchors(result[2][2])


def test_run_manifests_record_fallback_and_preserve_ab(imbalanced_pair_source, tmp_path, monkeypatch):
    import bfas.pair_unit as module

    class ExportPolicy(TinyPolicy):
        def merge_and_unload(self):
            return self

        def save_pretrained(self, path, **kwargs):
            (path / "model.safetensors").write_text("stub")

    class ExportTokenizer(TinyTokenizer):
        def save_pretrained(self, path):
            pass

    source, split = imbalanced_pair_source
    monkeypatch.setenv("AW_CC_PAIRS_PATH", str(source))
    monkeypatch.setenv("AW_CC_SPLIT_PATH", str(split))
    monkeypatch.setenv("AW_PAIR_SHUFFLE_SEED", "19")
    anchor_path = tmp_path / "anchors.jsonl"
    anchor_path.write_text("".join(json.dumps(r) + "\n" for r in anchor_rows()))
    monkeypatch.setenv("AW_CC_ANCHOR_PATH", str(anchor_path))
    output_root = tmp_path / "students"
    monkeypatch.setattr(legacy, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(legacy, "load_tokenizer", lambda student: ExportTokenizer())
    monkeypatch.setattr(legacy, "build_lora_model", lambda student, seed: ExportPolicy())
    original = load_pairs(source, split)
    manifests = []
    for arm in "ABC":
        monkeypatch.setenv("AW_DISTILL", "ddpo" if arm == "A" else "pair_unit")
        monkeypatch.setenv("AW_PAIR_PAIRING", "shuffled" if arm == "C" else "correct")
        module.run(SimpleNamespace(selection="full", tag=arm, seed=0, student="tiny"), legacy)
        manifest = json.loads((output_root / arm / "cc_manifest.json").read_text())
        if arm == "C":
            assert manifest["shuffle_same_seed_by_type"] == {"binding": 4, "3": 0}
            assert manifest["pairs"] == shuffle_pair_sides(original, seed=19)
            assert_shuffle_preserves_sides(original, manifest["pairs"])
        else:
            assert "shuffle_same_seed_by_type" not in manifest
            assert manifest["pairs"] == original
        manifests.append(manifest)
    for key in ("exposure_sha256", "optimizer_steps", "tokens", "reference_tokens", "anchor_rows"):
        assert manifests[0][key] == manifests[1][key] == manifests[2][key]


@pytest.mark.parametrize("mode", ["ddpo", "pair_unit"])
def test_anchors_keep_legacy_logistic_objective(tmp_path, mode):
    model = TinyPolicy()
    config = configuration({"AW_DISTILL": mode, "AW_PAIR_BATCH_UNITS": "4"})
    path = tmp_path / f"{mode}.jsonl"
    train(model, TinyTokenizer(), [], anchor_rows(), seed=0, config=config,
          trace_path=path, trainer=legacy)
    trace = json.loads(path.read_text())
    assert trace["loss"] == pytest.approx(float(-F.logsigmoid(torch.tensor(0.))))
    assert all(p["anchor"] for p in trace["pairs"])
    assert trace["tokens"]["anchor_response_tokens"] == trace["tokens"]["train_response_tokens"]


def test_anchor_update_matches_legacy_training(tmp_path, monkeypatch):
    original_encode = legacy.encode
    monkeypatch.setattr(legacy, "encode", lambda tok, row, device="cpu": original_encode(tok, row, device="cpu"))
    baseline, joint = TinyPolicy(), TinyPolicy()
    anchors = anchor_rows()
    legacy.cache_ddpo_reference_log_probs(baseline, TinyTokenizer(), anchors)
    steps = legacy.train_ddpo(baseline, TinyTokenizer(), anchors, 0, 1, .01, .1)
    config = configuration({"AW_DDPO_LR": ".01", "AW_PAIR_BATCH_UNITS": "4"})
    report = train(joint, TinyTokenizer(), [], anchor_rows(), seed=0, config=config,
                    trace_path=tmp_path / "joint.jsonl", trainer=legacy)
    assert steps == report["optimizer_steps"] == 1
    torch.testing.assert_close(baseline.delta, joint.delta, atol=1e-7, rtol=1e-5)


def test_anchor_loader_keeps_only_existing_anchor_rows(tmp_path):
    rows = anchor_rows()
    path = tmp_path / "anchors.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows + [{**rows[0], "_anchor": False}]))
    anchors = load_anchors(path, legacy)
    assert len(anchors) == 4
    assert [r["response"] for r in anchors] == [r["response"] for r in rows]
    assert load_anchors("none", legacy) == []
    rows[0].pop("_rejected")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="nonempty _rejected"):
        load_anchors(path, legacy)


def test_kl_masks_prompt_and_restores_model_state():
    model = TinyPolicy()
    model.train()
    with torch.no_grad():
        model.delta[0, 0] = 1
        model.delta[1, 1] = .5
    ids = torch.tensor([[0, 1, 2, 3, 5]])
    labels = torch.tensor([[-100, -100, -100, 3, 5]])
    value, count = response_kl(model, ids, labels)
    assert count == 2 and value == pytest.approx(0, abs=1e-7)
    assert model.training is True and model.disabled is False


def test_adaptive_rule_and_probe_accounting(tmp_path, monkeypatch):
    monkeypatch.setenv("AW_ANCHOR_ADAPTIVE", "1")
    monkeypatch.setenv("AW_ANCHOR_PROBES", "1")
    monkeypatch.setenv("AW_ANCHOR_PROBE_EVERY", "1")
    # Tightened tolerance ensures the dual rule activates for this tiny fixture.
    monkeypatch.setenv("AW_ANCHOR_EPS", "-1")
    monkeypatch.setenv("AW_ANCHOR_TRACE", str(tmp_path / "adaptive.json"))
    config = configuration({"AW_DDPO_EPOCHS": "2", "AW_PAIR_BATCH_UNITS": "2"})
    path = tmp_path / "steps.jsonl"
    train(TinyPolicy(), TinyTokenizer(), normalized_pairs(tmp_path), anchor_rows(),
          seed=0, config=config, trace_path=path, trainer=legacy)
    trace = [json.loads(line) for line in path.read_text().splitlines()]
    dual = json.loads((tmp_path / "adaptive.json").read_text())
    previous = {"call": 0., "abstain": 0.}
    for entry in dual:
        for side, drift in entry["drift"].items():
            expected = min(4., max(0., previous[side] + .5 * (1. - drift)))
            assert entry["lambda"][side] == pytest.approx(expected)
        previous = entry["lambda"]
    assert trace[0]["tokens"].get("anchor_response_tokens", 0) == 0
    assert trace[-1]["cumulative_tokens"]["anchor_response_tokens"] > 0
    assert all(step["tokens"]["probe_input_tokens"] > 0 for step in trace)


def test_legacy_ddpo_function_byte_for_byte_unchanged():
    # Frozen before C20. Covers the original objective, batching and both anchor paths.
    source = inspect.getsource(legacy.train_ddpo).rstrip()
    assert hashlib.sha256(source.encode()).hexdigest() == "a2800e428a64db5b9537cef42646d1aa1f082506796668fdb0348016653b2ef9"


def test_mode_off_config_invariance(monkeypatch):
    monkeypatch.setenv("AW_PAIR_GAMMA", "invalid but inactive")
    monkeypatch.setenv("AW_PAIR_PAIRING", "invalid but inactive")
    assert legacy.distillation_config() is None
    monkeypatch.setenv("AW_DISTILL", "ddpo")
    assert legacy.distillation_config() == dict(mode="ddpo", epochs=1, learning_rate=5e-6, beta=.1)
    monkeypatch.setenv("AW_DDPO_EPOCHS", "3")
    monkeypatch.setenv("AW_DDPO_LR", "1e-6")
    monkeypatch.setenv("AW_DDPO_BETA", ".2")
    assert legacy.distillation_config() == dict(mode="ddpo", epochs=3, learning_rate=1e-6, beta=.2)


def test_mode_off_main_source_unchanged_except_dispatch():
    # Remove explicit opt-in dispatches and compare to the pre-C20 main body.
    source = inspect.getsource(legacy.main)
    tree = ast.parse(source)
    branches = [node for node in tree.body[0].body if isinstance(node, ast.If)
                and any(marker in ast.get_source_segment(source, node.test)
                        for marker in ("AW_CC_PAIRS_PATH", '"pbsd_agent"'))]
    lines = source.splitlines()
    for branch in reversed(branches):
        del lines[branch.lineno - 1:branch.end_lineno]
    assert hashlib.sha256("\n".join(lines).encode()).hexdigest() == "d20c5376bd520186a22489f7f7f1efe6e57e1603d50f65fede691668d964414c"


def test_mode_on_config_and_dispatch(monkeypatch):
    import bfas.pair_unit as module
    monkeypatch.setenv("AW_DISTILL", "pair_unit")
    monkeypatch.setenv("AW_PAIR_GAMMA", "2")
    monkeypatch.setenv("AW_PAIR_REDUCTION", "sum")
    monkeypatch.setenv("AW_PAIR_PAIRING", "shuffled")
    assert legacy.distillation_config()["gamma"] == 2
    called = []
    monkeypatch.setattr(module, "run", lambda args, trainer: called.append((args.tag, trainer)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    legacy.main(["--selection", "full", "--tag", "test"])
    assert called == [("test", legacy)]


@pytest.mark.parametrize("script", ["tools/cc_three_arms.sh", "scripts/cc_arms_hpg.slurm"])
def test_shell_syntax_and_no_launch(script):
    subprocess.run(["bash", "-n", str(ROOT / script)], check=True)
    text = (ROOT / script).read_text()
    active = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    assert "sbatch" not in active and "srun" not in active
    assert "export HOME=" not in active


def test_slurm_and_campaign_contract():
    text = (ROOT / "scripts/cc_arms_hpg.slurm").read_text()
    for option in ("account=fsu-compsci-dept", "qos=fsu-compsci-dept", "partition=hpg-b200",
                   "gres=gpu:b200:1", "time=04:00:00"):
        assert f"#SBATCH --{option}" in text
    runner = (ROOT / "tools/cc_three_arms.sh").read_text()
    assert 'selection=${CC_ARMS-A B C}' in runner
    assert 'bash tools/bfcl_std_campaign.sh "$CUDA_VISIBLE_DEVICES" "$PORT" "$tag"' in runner
    assert "bfcl_fast_eval" not in runner
    assert "AW_ANCHOR_ADAPTIVE=0" in runner


@pytest.mark.parametrize("include_c", ["0", "1"])
def test_runner_preflight_five_plus_one_binding_pairs(imbalanced_pair_source, include_c):
    source, split = imbalanced_pair_source
    runner = (ROOT / "tools/cc_three_arms.sh").read_text()
    preflight = runner.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"),
               AW_CC_PAIRS_PATH=str(source), AW_CC_SPLIT_PATH=str(split),
               AW_CC_ANCHOR_PATH="none", AW_PAIR_SHUFFLE_SEED="19",
               CC_MIN_PAIRS="12", CC_MIN_TYPES="2")
    result = subprocess.run([sys.executable, "-", include_c], input=preflight, env=env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "train pairs=12 types=2 anchors=0 steps=3" in result.stdout
    prefix = "[cc] arm C shuffle_same_seed_by_type="
    if include_c == "1":
        report = next(line for line in result.stdout.splitlines() if line.startswith(prefix))
        assert json.loads(report[len(prefix):]) == {"binding": 4, "3": 0}
    else:
        assert prefix not in result.stdout


@pytest.fixture
def fake_campaign(tmp_path):
    """Stub every training/eval executable; only the shell control flow is real."""
    for script in ("tools/cc_three_arms.sh", "tools/aw_hpg_common.sh"):
        target = tmp_path / script
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / script, target)
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(f"#!{sys.executable}\n" + '''
import json, os, pathlib, sys
root = pathlib.Path.cwd()
def log(event):
    with (root / 'calls.jsonl').open('a') as handle:
        handle.write(json.dumps(event) + '\\n')
if sys.argv[1] == '-':
    body = sys.stdin.read()
    if 'load_pairs' in body:
        log({'phase': 'preflight', 'shuffle': sys.argv[2]})
    else:
        log({'phase': 'match', 'tags': sys.argv[2:]})
        if os.environ.get('FAKE_MISMATCH') == '1':
            raise SystemExit(1)
else:
    tag = sys.argv[sys.argv.index('--tag') + 1]
    log({'phase': 'train', 'tag': tag, 'mode': os.environ['AW_DISTILL'],
         'pairing': os.environ['AW_PAIR_PAIRING'], 'beta': os.environ['AW_DDPO_BETA'],
         'lr': os.environ['AW_DDPO_LR'], 'adaptive': os.environ['AW_ANCHOR_ADAPTIVE'],
         'steps': os.environ['AW_PAIR_STEPS'], 'anchor': os.environ['AW_CC_ANCHOR_PATH']})
    out = root / 'results/appworld_students' / tag / 'adapter'
    out.mkdir(parents=True)
    (out / 'model.safetensors').write_text('stub')
''')
    fake_python.chmod(0o755)
    campaign = tmp_path / "tools/bfcl_std_campaign.sh"
    campaign.write_text('''#!/bin/bash
set -eu
tag=$3
echo "{\\"phase\\":\\"eval\\",\\"tag\\":\\"$tag\\",\\"gpu\\":\\"$1\\"}" >> calls.jsonl
if [[ ${FAKE_EVAL_FAIL:-0} == 1 ]]; then
  echo "[bfclstd] $tag GENERATE FAILED"
  exit 0
fi
mkdir -p "results/bfcl_std/$tag/scoredir"
printf 'Overall Acc\\n0.5\\n' > "results/bfcl_std/$tag/data_overall.csv"
echo "[bfclstd] $tag OVERALL=0.5"
''')
    return tmp_path, fake_python


def call_fake_runner(fixture, **extra):
    root, executable = fixture
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CC_", "SLURM_"))}
    env.update(CC_PYTHON=str(executable), CUDA_VISIBLE_DEVICES="GPU-assigned", **extra)
    result = subprocess.run(["bash", "tools/cc_three_arms.sh"], cwd=root, env=env,
                            capture_output=True, text=True, timeout=20)
    calls = root / "calls.jsonl"
    records = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    return result, records


def test_runner_three_arms_shared_settings_and_eval_after_matching(fake_campaign):
    result, calls = call_fake_runner(fake_campaign, CC_ARMS="C,A,B", CC_BETA=".2", CC_LR=".001")
    assert result.returncode == 0, result.stdout + result.stderr
    assert [r["phase"] for r in calls] == ["preflight", "train", "train", "train", "match", "eval", "eval", "eval"]
    assert [r["mode"] for r in calls[1:4]] == ["ddpo", "pair_unit", "pair_unit"]
    assert [r["pairing"] for r in calls[1:4]] == ["correct", "correct", "shuffled"]
    assert all(r["beta"] == ".2" and r["lr"] == ".001" and r["adaptive"] == "0" for r in calls[1:4])
    assert len({r["anchor"] for r in calls[1:4]}) == 1
    assert all(r["gpu"] == "GPU-assigned" for r in calls[-3:])


def test_runner_aw_style_arm_selection(fake_campaign):
    result, calls = call_fake_runner(fake_campaign, CC_ARMS="b")
    assert result.returncode == 0, result.stdout + result.stderr
    assert calls[0]["shuffle"] == "0"
    assert [r["tag"] for r in calls if r["phase"] == "train"] == ["cc_v1_B_s0"]


@pytest.mark.parametrize("failure", ["FAKE_EVAL_FAIL", "FAKE_MISMATCH"])
def test_runner_detects_failure_even_when_campaign_exits_zero(fake_campaign, failure):
    result, calls = call_fake_runner(fake_campaign, CC_ARMS="A", **{failure: "1"})
    assert result.returncode != 0
    if failure == "FAKE_MISMATCH":
        assert not any(r["phase"] == "eval" for r in calls)
    else:
        assert "Official BFCL v4 full evaluation failed" in result.stderr
