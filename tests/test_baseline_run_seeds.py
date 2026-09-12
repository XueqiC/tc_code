"""CPU-only seed repeats: real purchases/training, stubbed model load and eval."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]
from tools import baseline_run
from bfas.rtd.baselines import paper_evaluation, paper_train
from bfas.rtd.baselines.paper_data import TeacherRow, load_purchased
from bfas.rtd.baselines.paper_seeds import seed_training, verify_seed_zero
from bfas.rtd.persistence import ComputeJournal, atomic_json, tree_hash
from test_baseline_run_data import make_bank
from test_baseline_run_training import Backend


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    def forbidden(*args, **kwargs):
        pytest.fail("seed tests must never launch workers or load a real model")
    monkeypatch.setattr(baseline_run, "run_worker", forbidden)
    from bfas.rtd import runtime
    monkeypatch.setattr(runtime, "load_backend", forbidden)
    yield
    torch.set_num_threads(threads)


def cli(bank, prefix, *, seed=None, budget=("--budget-tokens", "33"), method="smartad"):
    args = ["--method", method, "--benchmark", "bfcl", "--bank", str(bank),
            *budget, "--run-dir", str(prefix), "--smoke"]
    return args + (["--seed", str(seed)] if seed is not None else [])


def prepared(bank, prefix, *, seed=0, method="smartad"):
    assert baseline_run.main(cli(bank, prefix, seed=seed, method=method)+["--prepare-only"]) == 0
    directory = prefix.with_name(f"{prefix.name}_s{seed}") if seed else prefix
    return directory, json.loads((directory/"manifest.json").read_text())


@pytest.mark.parametrize("seed", [None, 0, 7])
@pytest.mark.parametrize("budget,names", [
    (("--budget-tokens", "33"), ["curve"]),
    (("--budget-tokens", "15,33"), ["curve_B15", "curve_B33"]),
    (("--budget-fraction", "1"), ["curve"]),
])
def test_seed_paths_receipts_and_purchase_behavior(tmp_path, capsys, seed, budget, names):
    bank, prefix = make_bank(tmp_path), tmp_path/"output"/"curve"
    before = tree_hash(bank)
    assert baseline_run.main(cli(bank, prefix, seed=seed, budget=budget)+["--prepare-only"]) == 0
    seed = seed or 0
    names = [name+f"_s{seed}" if seed else name for name in names]
    assert sorted(p.name for p in prefix.parent.iterdir()) == names
    summaries = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    for name, summary in zip(names, summaries):
        directory = prefix.parent/name
        manifest = json.loads((directory/"manifest.json").read_text())
        assert manifest["seed"] == manifest["config"]["training_seed"] == seed
        assert manifest["hyperparameters"]["seed"] == seed
        assert manifest["hyperparameters"]["gad"]["discriminator_seed"] == seed
        assert summary["seed"] == seed and summary["run_dir"] == str(directory)
        purchase, rows = load_purchased(bank, "bfcl", manifest["budget_fraction"],
                                       budget_tokens=manifest["budget_tokens"])
        assert all(manifest[k] == v for k, v in purchase.items())
        assert json.loads((directory/"purchased_rows.json").read_text()) == [asdict(r) for r in rows]
        assert manifest["purchase_seed"] == 0
        assert manifest["evaluation_protocol"] == paper_evaluation.protocol("bfcl", smoke=True)
        assert not (directory/"metrics.json").exists()
        if seed:
            assert manifest["seed_zero_verification"] == dict(status="no_seed_zero_run", references=[])
    assert tree_hash(bank) == before


@pytest.mark.parametrize("seed", [1, 2**32, 2**64+17])
def test_cli_accepts_any_nonnegative_seed(seed):
    assert baseline_run.arguments(cli(Path("/bank"), Path("/output"), seed=seed)).seed == seed


@pytest.mark.parametrize("seed", [-1, "1.5", "invalid"])
def test_cli_rejects_invalid_seed(seed):
    with pytest.raises(SystemExit) as error:
        baseline_run.arguments(cli(Path("/bank"), Path("/output"), seed=seed))
    assert error.value.code == 2


@pytest.mark.parametrize("method", ["sad", "smartad", "kang", "gad"])
def test_purchases_are_byte_identical_and_reference_is_untouched(tmp_path, method):
    bank, prefix = make_bank(tmp_path), tmp_path/"curve"
    zero, baseline = prepared(bank, prefix, method=method)
    before = tree_hash(zero)
    repeat, manifest = prepared(bank, prefix, seed=9, method=method)
    assert (repeat/"purchased_rows.json").read_bytes() == (zero/"purchased_rows.json").read_bytes()
    for key in ("charges", "purchase_order", "purchased_package_ids", "rows_hash", "evaluation_protocol"):
        assert manifest[key] == baseline[key]
    receipt = manifest["seed_zero_verification"]
    assert receipt["status"] == "purchases_verified_selection_pending"
    assert receipt["references"][0]["run_dir"] == str(zero)
    assert tree_hash(zero) == before


@pytest.mark.parametrize("changed", ["purchased_rows.json", "purchase_order", "charges", "rows_hash", "evaluation_protocol"])
def test_purchase_guard_fails_before_workers(tmp_path, changed):
    bank, prefix = make_bank(tmp_path), tmp_path/"curve"
    zero, manifest = prepared(bank, prefix)
    if changed.endswith(".json"):
        # Same JSON content/hash, different bytes must still fail.
        with (zero/changed).open("a") as stream:
            stream.write("\n")
    else:
        manifest[changed] = "tampered"
        atomic_json(zero/"manifest.json", manifest)
    before = tree_hash(zero)
    with pytest.raises(ValueError, match=f"seed-zero guard: {changed} differs"):
        baseline_run.main(cli(bank, prefix, seed=3))
    failed = json.loads((tmp_path/"curve_s3/manifest.json").read_text())
    assert failed["status"] == "failed" and changed in failed["error"]
    assert not (tmp_path/"curve_s3/train.log").exists()
    assert tree_hash(zero) == before


def test_guard_finds_renamed_seed_zero_in_same_root(tmp_path):
    bank = make_bank(tmp_path)
    zero, _ = prepared(bank, tmp_path/"old_name")
    _, repeat = prepared(bank, tmp_path/"new_name", seed=1)
    assert repeat["seed_zero_verification"]["references"][0]["run_dir"] == str(zero)
    with (zero/"purchased_rows.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="purchased_rows.json differs"):
        prepared(bank, tmp_path/"third_name", seed=2)


@pytest.mark.parametrize("invalid", ["missing", "malformed", "non_object", "wrong_seed", "symlink"])
def test_guard_rejects_invalid_seed_zero_reference(tmp_path, invalid):
    bank, prefix = make_bank(tmp_path), tmp_path/"curve"
    zero, manifest = prepared(bank, prefix)
    if invalid == "missing":
        (zero/"manifest.json").unlink()
    elif invalid in {"malformed", "non_object"}:
        (zero/"manifest.json").write_text("{" if invalid == "malformed" else "[]")
    elif invalid == "wrong_seed":
        manifest["seed"] = 1
        atomic_json(zero/"manifest.json", manifest)
    else:
        moved = tmp_path/"reference"
        zero.rename(moved)
        zero.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="seed-zero guard"):
        prepared(bank, prefix, seed=7)


def test_guard_rechecks_reference_that_appears_after_preparation(tmp_path):
    bank, prefix = make_bank(tmp_path), tmp_path/"curve"
    repeat, manifest = prepared(bank, prefix, seed=7)
    assert manifest["seed_zero_verification"]["status"] == "no_seed_zero_run"
    zero, _ = prepared(bank, prefix)
    with (zero/"purchased_rows.json").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="purchased_rows.json differs"):
        verify_seed_zero(repeat, manifest)


def tiny_train(directory, manifest):
    rows = [TeacherRow(**r) for r in json.loads((directory/"purchased_rows.json").read_text())]
    config = dict(manifest["config"], training_device="cpu")
    trainer = paper_train.PaperTrainer(Backend(), rows, config, manifest["method"], directory,
                                     ComputeJournal(directory/"compute.jsonl", cuda=False), manifest=manifest)
    trainer.train()
    return trainer


@pytest.mark.parametrize("method", ["sad", "smartad", "kang", "gad"])
def test_selection_artifacts_match_across_seeds_with_real_cpu_trainer(tmp_path, method):
    bank, prefix = make_bank(tmp_path), tmp_path/"curve"
    zero, baseline = prepared(bank, prefix, method=method)
    tiny_train(zero, baseline)
    repeat, manifest = prepared(bank, prefix, seed=7, method=method)
    tiny_train(repeat, manifest)
    artifacts = ["purchased_rows.json", "training_rows.json"]
    if method == "smartad":
        artifacts.append("smartad_selection.json")
    for name in artifacts:
        assert (repeat/name).read_bytes() == (zero/name).read_bytes()
    receipt = json.loads((repeat/"manifest.json").read_text())["seed_zero_verification"]
    assert receipt["status"] == "verified"
    assert set(receipt["references"][0]["artifact_sha256"]) == set(artifacts)
    assert (repeat/"exposure_schedule.json").read_bytes() != (zero/"exposure_schedule.json").read_bytes()


@pytest.mark.parametrize("artifact", ["training_rows.json", "smartad_selection.json"])
@pytest.mark.parametrize("missing", [False, True])
def test_selection_guard_fails_before_first_update(tmp_path, monkeypatch, artifact, missing):
    bank, prefix = make_bank(tmp_path), tmp_path/"curve"
    zero, baseline = prepared(bank, prefix)
    tiny_train(zero, baseline)
    if missing:
        (zero/artifact).unlink()
    else:
        (zero/artifact).write_text("[]\n")
    repeat, manifest = prepared(bank, prefix, seed=1)
    def forbidden(*args, **kwargs):
        pytest.fail("selection mismatch must fail before preconditioner/update")
    monkeypatch.setattr(paper_train.PaperTrainer, "step_rule", forbidden)
    with pytest.raises(ValueError, match=f"seed-zero guard: .*{artifact}"):
        tiny_train(repeat, manifest)
    assert not (repeat/"exposure_schedule.json").exists()
    assert not (repeat/"checkpoint").exists()


def test_worker_cli_preserves_seed_and_does_not_suffix_twice(tmp_path, monkeypatch):
    bank, prefix = make_bank(tmp_path), tmp_path/"curve"
    calls = []
    def train(directory, manifest):
        calls.append(("train", directory.name, manifest["seed"], manifest["config"]["training_seed"]))
    def evaluate(root, directory, manifest):
        calls.append(("evaluate", directory.name, manifest["seed"], manifest["config"]["training_seed"]))
    def worker(command, log):
        assert command[command.index("--seed")+1] == "7"
        assert baseline_run.main(command[2:]) == 0
    monkeypatch.setattr(baseline_run, "train_worker", train)
    monkeypatch.setattr(paper_evaluation, "evaluate_run", evaluate)
    monkeypatch.setattr(baseline_run, "run_worker", worker)
    assert baseline_run.main(cli(bank, prefix, seed=7, budget=("--budget-tokens", "15,33"))) == 0
    assert calls == [(phase, f"curve_B{cap}_s7", 7, 7) for cap in (15, 33) for phase in ("train", "evaluate")]


def test_worker_rejects_seed_mismatch(tmp_path):
    bank, prefix = make_bank(tmp_path), tmp_path/"curve"
    directory, _ = prepared(bank, prefix, seed=7)
    with pytest.raises(ValueError, match="worker seed differs"):
        baseline_run.main(cli(bank, directory, seed=0)+["--_phase", "train"])


def test_nonzero_curve_collision_checked_before_any_level(tmp_path):
    bank, prefix = make_bank(tmp_path), tmp_path/"curve"
    (tmp_path/"curve_B33_s7").mkdir()
    with pytest.raises(FileExistsError, match="fresh"):
        baseline_run.main(cli(bank, prefix, seed=7, budget=("--budget-tokens", "15,33")))
    assert not (tmp_path/"curve_B15_s7").exists()


@pytest.mark.parametrize("seed", [0, 7, 2**64+17])
def test_train_worker_seeds_initialization_dropout_and_scoring_on_cpu(tmp_path, monkeypatch, seed):
    from bfas.rtd import hardware, persistence, runtime
    from tools import bfcl_hub_merge_export
    bank, prefix = make_bank(tmp_path), tmp_path/"curve"
    directory, manifest = prepared(bank, prefix, seed=seed)
    model = tmp_path/"model"
    model.mkdir()
    (model/"config.json").write_text("{}")
    monkeypatch.setattr(bfcl_hub_merge_export, "_snapshot_for_model", lambda _: model)
    monkeypatch.setattr(hardware, "hardware_identity", lambda: {"hard": "cpu-stub"})
    monkeypatch.setattr(persistence, "ComputeJournal",
                        lambda path, **kw: ComputeJournal(path, cuda=False))
    def draws():
        return (random.random(), np.random.rand(), torch.nn.Linear(4, 4).weight.detach().clone(),
                torch.nn.Dropout(.5)(torch.ones(64)), torch.rand(4))
    seed_training(seed)
    expected = draws()
    seed_training(seed+1)
    def load(config, given, journal):
        assert config["training_seed"] == seed % (2**64)
        assert given["seed"] == given["config"]["training_seed"] == seed
        actual = draws()
        assert actual[:2] == expected[:2]
        assert all(torch.equal(a, b) for a, b in zip(actual[2:], expected[2:]))
        return "tiny"
    class Trainer:
        def __init__(self, backend, rows, config, method, out, journal, *, manifest):
            assert backend == "tiny" and config["training_seed"] == seed
        def train(self):
            (directory/"checkpoint").mkdir()
            (directory/"checkpoint/tiny.json").write_text("{}")
            return dict(student_commits=0)
    monkeypatch.setattr(runtime, "load_backend", load)
    monkeypatch.setattr(paper_train, "PaperTrainer", Trainer)
    baseline_run.train_worker(directory, manifest)
    assert json.loads((directory/"manifest.json").read_text())["status"] == "trained"


@pytest.mark.parametrize("method", ["sad", "smartad", "kang", "gad"])
def test_training_schedule_and_sampling_repeat_and_vary_by_seed(tmp_path, method):
    samples, schedules, discriminators = [], [], []
    class SamplingBackend(Backend):
        def sample_action(self, prompt, parameters, generator):
            ids = torch.randint(97, 123, (4,), generator=generator).tolist()
            samples[-1].append(ids)
            return SimpleNamespace(action_ids=tuple(ids)+(255,), prompt_ids=self.tokenizer.encode(prompt))
    rows = [TeacherRow(str(i), str(i), str(i), 0, f"prompt{i}", chr(65+i), "bfcl", True) for i in range(4)]
    for index, seed in enumerate((0, 7, 7)):
        directory = tmp_path/f"repeat{index}"
        directory.mkdir()
        config = dict(student="tiny", lora_rank=16, lora_alpha=32, lora_target_modules=["q_proj"],
                      max_context_tokens=256, smoke=True, training_seed=seed)
        samples.append([])
        trainer = paper_train.PaperTrainer(SamplingBackend(), rows, config, method, directory,
                                          ComputeJournal(directory/"compute.jsonl", cuda=False))
        if method == "gad":
            discriminators.append(torch.cat([p.detach().flatten() for p in trainer.discriminator.parameters()]))
        trainer.train()
        batches = json.loads((directory/"exposure_schedule.json").read_text())["batches"]
        schedules.append(batches)
        oracle = random.Random(seed)
        assert batches == [[oracle.randrange(4) for _ in range(2)] for _ in range(2)]
    assert samples[0] != samples[1] == samples[2]
    assert schedules[0] != schedules[1] == schedules[2]
    if method == "gad":
        assert not torch.equal(discriminators[0], discriminators[1])
        assert torch.equal(discriminators[1], discriminators[2])


@pytest.mark.parametrize("seed", [0, 7])
@pytest.mark.parametrize("method", ["sad", "kang"])
def test_metrics_record_seed_without_changing_evaluation(tmp_path, monkeypatch, seed, method):
    from bfas.rtd import hardware
    # Exercise actual receipt/metrics writing, with only the external campaign stubbed.
    directory = tmp_path/(f"curve_s{seed}" if seed else "curve")
    lora, model = directory/"checkpoint/lora", tmp_path/"model"
    lora.mkdir(parents=True)
    model.mkdir()
    (lora/"adapter_config.json").write_text("{}")
    (model/"config.json").write_text("{}")
    manifest = dict(seed=seed, config=dict(training_seed=seed), benchmark="alfworld", method=method,
        checkpoint_sha256=tree_hash(lora.parent), base_checkpoint_hash=tree_hash(model),
        model_path=str(model), hardware={"hard": "cpu-stub"}, teacher_tokens_charged=33, B=40, port=8930)
    monkeypatch.setattr(hardware, "hardware_identity", lambda: manifest["hardware"])
    calls = []
    def campaign(root, policy, out, **kwargs):
        calls.append(kwargs)
        return dict(tasks=140, overall_accuracy_percent=50.)
    monkeypatch.setattr(paper_evaluation, "run_alfworld", campaign)
    result = paper_evaluation.evaluate_run(ROOT, directory, manifest)
    assert calls == [dict(kang=False, port=8930, smoke=False)] + (
        [dict(kang=True, port=8930, smoke=False)] if method == "kang" else [])
    assert result["seed"] == seed and result["protocol"] == paper_evaluation.protocol("alfworld")
    for name in ("metrics.json", "official_metrics.json", "manifest.json"):
        assert json.loads((directory/name).read_text())["seed"] == seed
    if method == "kang":
        assert json.loads((directory/"kang_metrics.json").read_text())["seed"] == seed
