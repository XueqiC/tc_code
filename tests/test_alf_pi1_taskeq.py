"""CPU oracles for task pooling, unchanged CE, threshold saves and CLI dispatch."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]

from bfas.rtd.baselines import pi1
from bfas.rtd.baselines.paper_losses import span_ce
from bfas.rtd.baselines.paper_train import PaperTrainer, hyperparameters
from bfas.rtd.baselines.pi1_taskeq import TASK_EQUAL_FORMULA, TokenCEState, task_equal_token_weights
from bfas.rtd.persistence import ComputeJournal
from test_alf_pi1_train import Backend, config, row
from tools import alf_pi1_train as cli


CONFIGS = ("taskeq", "all87_tok", "all87_taskeq")


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(name, "1")


def token_config(targets, *, method="pi1_ce_taskeq", budget=512, seed=0):
    cfg = dict(config(), method=method, exposure_tokens=targets,
               supervised_tokens_per_update=budget, training_seed=seed)
    cfg.pop("exposure_passes")
    return cfg


def make_trainer(directory, rows, cfg, *, resume=False):
    backend = Backend()
    # Nonuniform probabilities make row/task-weighting bugs observable.
    with torch.no_grad():
        backend.model.lora_logits.copy_(torch.linspace(-2, 2, 256))
    costs = [len(pi1.encode_teacher_turn(backend.tokenizer, r, cfg["max_context_tokens"])["target_ids"])
             for r in rows]
    plan = pi1.exposure_plan(costs, cfg, cfg["training_seed"])
    identity = dict(config=cfg, seed=cfg["training_seed"], hyperparameters=hyperparameters(cfg, cfg["method"]),
                    bank=dict(sealed_manifest_sha256="synthetic"),
                    target_boundary=pi1.boundary_contract(backend.tokenizer))
    manifest = pi1.prepare_run(directory, identity, rows, plan, resume=resume)
    return PaperTrainer(backend, rows, cfg, cfg["method"], directory,
                        ComputeJournal(directory/"compute.jsonl", cuda=False), manifest=manifest)


def two_tasks(lengths=(10, 90)):
    return [replace(row(i, chr(65+i)*(length-1)), task_id=f"task-{i}", package_id=f"package-{i}")
            for i, length in enumerate(lengths)]


def test_two_task_10_90_token_weights_and_total_mass():
    weights = task_equal_token_weights(["a", "b"], [10, 90])
    assert weights["a"] == 5.0
    assert weights["b"] == pytest.approx(0.5556, abs=0.00005)
    assert weights["b"] == 50/90
    assert 10*weights["a"] + 90*weights["b"] == 100


def test_packages_pool_by_support_task_and_weights_reset_per_update():
    # Two packages of b have 30+60 tokens; package count must not be the divisor.
    assert task_equal_token_weights(["a", "b", "b"], [10, 30, 60]) == {"a": 5., "b": 50/90}
    assert task_equal_token_weights(["b", "b"], [30, 60]) == {"b": 1.}
    assert task_equal_token_weights(["a", "b"], [90, 10]) == {"a": 50/90, "b": 5.}


@pytest.mark.parametrize("length", [10, 50, 90])
def test_equal_lengths_loss_is_plain_ce_bit_for_bit(length):
    values = torch.linspace(-13, -.1, 2*length, dtype=torch.float32)
    weights = task_equal_token_weights(["a", "b"], [length, length])
    per_token = values.new_tensor([weights["a"]]*length + [weights["b"]]*length)
    actual = -(values*per_token).sum()/len(values)
    assert torch.equal(actual, span_ce(values, ["teacher"]*len(values), "pi1_ce"))


def test_equal_task_totals_produce_identical_optimizer_and_loss_trace(tmp_path):
    # Equal task totals with unequal row lengths and several packages per task.
    rows = [replace(row(i, text), task_id=task, package_id=f"p-{i}")
            for i, (text, task) in enumerate([("A"*9, "a"), ("B"*3, "b"), ("C"*5, "b")])]
    ce = make_trainer(tmp_path/"ce", rows, token_config([60, 200], method="pi1_ce"))
    eq = make_trainer(tmp_path/"eq", rows, token_config([60, 200]))
    assert ce.train() == eq.train()
    assert torch.equal(ce.backend.model.lora_logits, eq.backend.model.lora_logits)
    assert (ce.directory/"loss_trace.json").read_bytes() == (eq.directory/"loss_trace.json").read_bytes()
    assert (ce.directory/"training_rows.json").read_bytes() == (eq.directory/"training_rows.json").read_bytes()
    assert (ce.directory/"exposure_schedule.json").read_bytes() == (eq.directory/"exposure_schedule.json").read_bytes()
    for r in rows:
        assert ce.encode(r) == eq.encode(r)


def test_task_equal_actual_adamw_matches_independent_pooled_task_oracle(tmp_path):
    rows = [replace(row(i, text), task_id=task, package_id=f"p-{i}")
            for i, (text, task) in enumerate([("A"*9, "a"), ("B"*29, "b"), ("C"*59, "b")])]
    t = make_trainer(tmp_path/"eq", rows, token_config([100]))
    expected = t.backend.model.lora_logits.detach().clone().requires_grad_()
    hp = t.hp
    optimizer = torch.optim.AdamW([expected], lr=hp["learning_rate"], betas=tuple(hp["adam_betas"]),
        eps=hp["adam_epsilon"], weight_decay=hp["weight_decay"], foreach=False, fused=False)
    by_task = {}
    for r in rows:
        by_task.setdefault(r.task_id, []).extend((*r.target.encode(), 106))
    loss = torch.stack([-expected.log_softmax(0)[ids].mean() for ids in by_task.values()]).mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([expected], hp["gradient_clip"])
    optimizer.step()
    result = t.train()
    assert result["student_commits"] == 1
    assert result["losses"] == pytest.approx([float(loss.detach())], rel=1e-6)
    assert torch.allclose(t.backend.model.lora_logits, expected, rtol=1e-6, atol=1e-10)
    assert t.backend.samples == 0
    assert t.hp["loss_normalization"] == "task_equal_weight_per_update"
    assert t.hp["loss_formula"] == t.manifest["loss_formula"] == TASK_EQUAL_FORMULA
    saved = json.loads((t.directory/"tokens-100/manifest.json").read_text())
    assert saved["hyperparameters"]["loss_formula"] == TASK_EQUAL_FORMULA


def test_task_equal_pass_mode_retains_legacy_schedule_and_endpoint_names(tmp_path):
    rows = two_tasks()
    cfg = dict(config(), method="pi1_ce_taskeq", training_seed=2)
    t = make_trainer(tmp_path/"passes", rows, cfg)
    ce_plan = pi1.exposure_plan([10, 90], dict(cfg, method="pi1_ce"), 2)
    assert json.loads((t.directory/"exposure_schedule.json").read_text()) == ce_plan
    result = t.train()
    assert result["endpoints"] == [3, 10]
    assert result["supervised_tokens"] == 1000
    for passes in (3, 10):
        receipt = json.loads((t.directory/f"pass-{passes}/manifest.json").read_text())
        assert receipt["supervised_token_count"] == 100*passes
        assert receipt["identity"]["endpoint_mode"] == "passes"
        assert receipt["hyperparameters"]["loss_formula"] == TASK_EQUAL_FORMULA


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("budget", [37, 512, 4096])
def test_d0_token_endpoints_equal_pass_3_and_10_at_identical_steps(seed, budget):
    costs = [22]*423 + [343]  # 424 complete turns, including native boundaries.
    assert sum(costs) == 9649
    passes = pi1.exposure_plan(costs, dict(config(), supervised_tokens_per_update=budget), seed)
    tokens = pi1.exposure_plan(costs, token_config([28947, 96490], budget=budget), seed)
    normalized = [dict(b, endpoint=[b["endpoint"]*sum(costs)] if b["endpoint"] else None)
                  for b in passes["batches"]]
    assert tokens["batches"] == normalized
    assert tokens["endpoint_saves"] == [dict(target_supervised_tokens=b["cumulative_tokens"],
        supervised_token_count=b["cumulative_tokens"], optimizer_step_count=i)
        for i, b in enumerate(passes["batches"], 1) if b["endpoint"]]
    assert [s["supervised_token_count"] for s in tokens["endpoint_saves"]] == [28947, 96490]


def test_non_pass_targets_cross_at_first_regular_update_with_seeded_cycles():
    costs, targets, seed = [7, 11, 23], [1, 2, 50, 99, 145], 2
    plan = pi1.exposure_plan(costs, token_config(targets, budget=31), seed)
    indices = [i for b in plan["batches"] for i in b["indices"]]
    rng = np.random.default_rng(seed)
    oracle = []
    while len(oracle) < len(indices):
        oracle.extend(rng.permutation(len(costs)).tolist())
    assert indices == oracle[:len(indices)]
    for save in plan["endpoint_saves"]:
        step = save["optimizer_step_count"]
        previous = plan["batches"][step-2]["cumulative_tokens"] if step > 1 else 0
        assert previous < save["target_supervised_tokens"] <= save["supervised_token_count"]
    assert all(31 <= b["supervised_tokens"] < 31+max(costs) for b in plan["batches"])
    assert plan["batches"][0]["endpoint"] == [1, 2]


@pytest.mark.parametrize("method", ["pi1_ce", "pi1_ce_taskeq"])
def test_token_receipts_multiple_targets_and_resume_after_partial_publication(tmp_path, monkeypatch, method):
    rows, cfg = two_tasks(), token_config([1, 2, 301], method=method, budget=64)
    full = make_trainer(tmp_path/"full", rows, cfg)
    expected = full.train()
    interrupted = make_trainer(tmp_path/"interrupted", rows, cfg)
    publish = TokenCEState.publish_endpoint
    def crash(self, targets):
        publish(self, targets[:1])
        raise InterruptedError("one endpoint published after durable commit")
    monkeypatch.setattr(TokenCEState, "publish_endpoint", crash)
    with pytest.raises(InterruptedError):
        interrupted.train()
    assert (interrupted.directory/"tokens-1").is_dir()
    assert not (interrupted.directory/"tokens-2").exists()
    monkeypatch.setattr(TokenCEState, "publish_endpoint", publish)
    resumed = make_trainer(interrupted.directory, rows, cfg, resume=True)
    assert resumed.train(resume=True) == expected
    assert torch.equal(resumed.backend.model.lora_logits, full.backend.model.lora_logits)
    for save in resumed.manifest["endpoint_saves"]:
        target = save["target_supervised_tokens"]
        receipt = json.loads((resumed.directory/f"tokens-{target}/manifest.json").read_text())
        assert {k: receipt[k] for k in save} == save
        assert receipt["exposure_check"]["matched"]
        assert receipt["exposure_check"]["overshoot_tokens"] == save["supervised_token_count"]-target
        assert receipt["identity"]["method"] == method
        assert receipt["identity"]["endpoint_mode"] == "tokens"
    assert make_trainer(interrupted.directory, rows, cfg, resume=True).train(resume=True) == expected
    for changed in (dict(cfg, method="pi1_ce" if method == "pi1_ce_taskeq" else "pi1_ce_taskeq"),
                    dict(cfg, exposure_tokens=[1, 3, 301])):
        with pytest.raises(ValueError, match="resume identity differs"):
            make_trainer(interrupted.directory, rows, changed, resume=True)


@pytest.mark.parametrize("name", CONFIGS)
def test_registered_configs_preserve_every_other_recipe_key(name):
    cfg = pi1.load_config(ROOT/f"configs/rtd/pi1_alfworld_k32_{name}.yaml")
    source = "pi1_alfworld_k32.yaml" if name == "taskeq" else "pi1_alfworld_k32_smartad_all87.yaml"
    old = pi1.load_config(ROOT/"configs/rtd"/source)
    excluded = {"method", "bank", "training_seeds", "exposure_tokens", "exposure_passes"}
    assert {k: v for k, v in cfg.items() if k not in excluded} == {
        k: v for k, v in old.items() if k not in excluded}
    assert cfg["method"] == ("pi1_ce" if name == "all87_tok" else "pi1_ce_taskeq")
    assert cfg["training_seeds"] == [0, 1, 2]
    assert cfg["exposure_tokens"] == [28947, 96490]
    assert "exposure_passes" not in cfg
    hp = hyperparameters(dict(cfg, training_seed=2), cfg["method"])
    plain = pi1.plain_ce_hyperparameters(cfg, 2)
    extras = {"method", "loss_normalization", "loss_formula", "task_definition"}
    assert {k: v for k, v in hp.items() if k not in extras} == {
        k: v for k, v in plain.items() if k not in extras}


@pytest.mark.parametrize("change", [dict(exposure_tokens=[]), dict(exposure_tokens=[0]),
    dict(exposure_tokens=[True]), dict(exposure_tokens=[1.0]), dict(exposure_tokens=[2, 1]),
    dict(exposure_tokens=[1, 1]), dict(exposure_passes=[3, 10]), dict(method="unknown"),
    dict(training_seeds=[0, 3]), dict(learning_rate=1e-4), dict(supervised_tokens_per_update=256)])
def test_new_config_rejects_ambiguous_endpoints_and_recipe_drift(tmp_path, change):
    cfg = yaml.safe_load((ROOT/"configs/rtd/pi1_alfworld_k32_taskeq.yaml").read_text())
    cfg.update(change)
    path = tmp_path/"bad.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        pi1.load_config(path)


@pytest.mark.parametrize("name", CONFIGS)
def test_real_config_cpu_preflight_and_load_bank_counts(name):
    path = ROOT/f"configs/rtd/pi1_alfworld_k32_{name}.yaml"
    completed = subprocess.run([sys.executable, str(ROOT/"tools/alf_pi1_train.py"),
        "--config", str(path), "--preflight", "--seed", "2"],
        cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), check=True, capture_output=True, text=True)
    report = json.loads(completed.stdout)
    assert report["assertions_passed"] and report["endpoint_mode"] == "tokens"
    assert report["last_8_label_ids"][-1] == 106
    assert report["distinct_tasks"] == 30
    counts = (30, 424, 9649) if name == "taskeq" else (87, 1343, 33436)
    assert (report["bank"]["demonstrations"], report["bank"]["supervised_turns"],
            report["exposure"]["bank_supervised_tokens"]) == counts
    assert report["exposure"]["target_tokens"] == [28947, 96490]
    if name == "taskeq":
        assert [s["supervised_token_count"] for s in report["exposure"]["endpoint_saves"]] == [28947, 96490]


@pytest.mark.parametrize("name", CONFIGS)
def test_cli_check_step_dispatch_and_identity_on_cpu(tmp_path, monkeypatch, capsys, name):
    """Only a tiny CPU backend; GPU selection, loading and journal are test doubles."""
    from bfas.rtd import persistence, runtime
    from bfas.rtd.baselines import paper_train
    from bfas.rtd.benchmarks import alfworld_identity, alfworld_support
    rows = [replace(row(i, text), task_id=f"t-{i%2}") for i, text in enumerate(("A", "BB", "CCC", "DDDD"))]
    cfg = pi1.load_config(ROOT/f"configs/rtd/pi1_alfworld_k32_{name}.yaml")
    cfg["bank_supervised_tokens"] = sum(len(r.target)+1 for r in rows)
    monkeypatch.setattr(pi1, "load_config", lambda path: dict(cfg))
    monkeypatch.setattr(pi1, "load_bank", lambda *args: (rows, dict(sealed_manifest_sha256="synthetic")))
    backend = Backend()
    monkeypatch.setattr(alfworld_support, "FrozenRenderer", lambda path: type("Renderer", (), {
        "adapter": type("Adapter", (), {"_tokenizer": backend.tokenizer})()})())
    monkeypatch.setattr(alfworld_identity, "tokenizer_identity", lambda path: {"hash": "tiny"})
    monkeypatch.setattr(cli, "select_device", lambda uuid: {"uuid": uuid})
    monkeypatch.setattr(cli, "verify_cuda_device", lambda selected, config: selected)
    monkeypatch.setattr(runtime, "load_backend", lambda *args: backend)
    monkeypatch.setattr(persistence, "ComputeJournal", lambda path, **kw: ComputeJournal(path, cuda=False))
    def cpu_trainer(backend, rows, config, method, *args, **kwargs):
        return PaperTrainer(backend, rows, dict(config, training_device="cpu"), method, *args, **kwargs)
    monkeypatch.setattr(paper_train, "PaperTrainer", cpu_trainer)
    model, output = tmp_path/"model", tmp_path/"check"
    model.mkdir()
    (model/"config.json").write_text("{}")
    assert cli.main(["--config", str(ROOT/f"configs/rtd/pi1_alfworld_k32_{name}.yaml"),
        "--seed", "2", "--check-step", "--output", str(output), "--model-path", str(model)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["student_commits"] == 1 and result["checked_rows"] == 4
    manifest = json.loads((output/"manifest.json").read_text())
    assert manifest["identity"]["method"] == cfg["method"]
    assert manifest["identity"]["endpoint_mode"] == "tokens"
    assert "exposure_passes" not in manifest["identity"]["config"]
    assert "src/bfas/rtd/baselines/pi1_taskeq.py" in manifest["identity"]["source_hashes"]
    assert manifest["identity"]["execution_check"]
    assert not torch.cuda.is_initialized()
