"""Official campaign completeness, interruption, locks, reuse and CPU integration."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import warnings

import pytest

from bfas.rtd.benchmarks import alfworld_evaluation as evaluation
from bfas.rtd.benchmarks import alfworld_identity as identity
from bfas.rtd.evaluation_lock import evaluation_lock, tag_lock_path as bfcl_tag_lock_path
from bfas.rtd.persistence import ComputeJournal, atomic_json, digest, tree_hash
from rtd_alfworld_evaluation_fixtures import (
    FakeBackend, FakeEnv, audit, campaign, changed_harness, put, round_checkpoint, run,
)


def records(c):
    return [json.loads(p.read_text())["record"] for p in sorted((c.directory / "artifacts/tasks").glob("*.json"))]


def test_full_campaign_units_records_and_artifacts(campaign):
    c = campaign
    result = run(c)
    aggregate = result["aggregate"]
    assert aggregate["tasks"] == 140 and aggregate["successes"] == 109
    assert aggregate["success_rate"] == 109 / 140
    assert aggregate["success_rate_percent"] == 100 * 109 / 140
    assert aggregate["units"] == dict(success_rate="fraction_0_1", success_rate_percent="percent_0_100")
    assert aggregate["uncertainty"]["parent_groups"] == 137
    assert result["artifacts_hash"] == tree_hash(c.directory / "artifacts")
    assert len(records(c)) == 140 and all(r["generated_text"] for r in records(c))
    assert len(c.backends) == 1 and c.backends[0].closed and all(e.closed for e in c.envs)
    before = (c.directory / "campaign.json").read_bytes()
    assert run(c, backend_factory=lambda m: pytest.fail("reuse loaded a model")) == result
    assert (c.directory / "campaign.json").read_bytes() == before


@pytest.mark.parametrize("fault", ["missing", "duplicate", "extra", "split", "step_cap", "incomplete", "generated_text"])
def test_invalid_records_never_aggregate(campaign, fault):
    c = campaign
    result = run(c)
    rows = records(c)
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif fault == "extra":
        rows[0]["task_id"] = "extra/task"
    elif fault == "split":
        rows[0]["split"] = "valid_unseen"
    elif fault == "step_cap":
        rows[0]["max_steps"] = 50
    elif fault == "incomplete":
        rows[0]["done"] = False
    else:
        rows[0]["generated_text"] = []
    with pytest.raises(ValueError):
        evaluation.aggregate_records(result["expected"], rows, result["identity"])


@pytest.mark.parametrize("field", ["success", "done", "truncated", "horizon_reached"])
@pytest.mark.parametrize("value", [0, 1, 0.0, 1.0, "true", None])
def test_saved_boolean_types_are_strict(campaign, field, value):
    c = campaign
    ident = identity.campaign_identity(c.manifest)
    expected = c.manifest["evaluation_harness"]["expected"]
    tid = expected["task_ids"][0]
    row = evaluation.official_episode(tid, FakeBackend(), env_factory=FakeEnv, identity=ident)
    row[field] = value
    with pytest.raises(ValueError, match="bool"):
        evaluation.validate_records(expected, [row], ident, complete=False)


@pytest.mark.parametrize("field", ["won", "done"])
@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_environment_boolean_types_are_strict_and_worker_closed(campaign, field, value):
    c = campaign
    env = FakeEnv("fixture/task")
    original = env.step
    env.step = lambda command: dict(original(command), **{field: value})
    with pytest.raises(ValueError, match="bool"):
        evaluation.official_episode("pick_and_place_simple-fixture/trial", FakeBackend(),
            env_factory=lambda tid: env, identity=identity.campaign_identity(c.manifest))
    assert env.closed


def test_horizon_truncation_and_parser_fallback(campaign):
    c = campaign
    env = FakeEnv("fixture/task", won=False, terminal_step=50)
    backend = FakeBackend()
    backend.generate = lambda *a, **k: evaluation.Generation("unparseable thought", 256, True)
    row = evaluation.official_episode("pick_and_place_simple-fixture/trial", backend,
        env_factory=lambda tid: env, identity=identity.campaign_identity(c.manifest))
    assert row["steps"] == 40 and row["truncated"] and row["horizon_reached"]
    assert row["success"] is False and row["done"] is False
    assert all(t["command"] == "look" for t in row["turns"])
    assert row["generated_text"] == ["unparseable thought"] * 40


def test_interrupt_resumes_only_complete_tasks(campaign):
    c = campaign
    count = 0
    def factory(tid):
        nonlocal count
        count += 1
        if count == 3:
            raise evaluation.EnvironmentUnavailable("fixture worker unavailable")
        return c.env_factory(tid)
    with pytest.raises(evaluation.EnvironmentUnavailable):
        run(c, env_factory=factory)
    assert not (c.directory / "campaign.json").exists()
    assert not (c.directory / "artifacts/aggregate.json").exists()
    assert len(records(c)) == 2 and c.backends[0].closed
    saved = {p.name: p.read_bytes() for p in (c.directory / "artifacts/tasks").glob("*.json")}
    result = run(c)
    assert len(c.envs) == 140 and result["aggregate"]["tasks"] == 140
    assert all((c.directory / "artifacts/tasks" / n).read_bytes() == v for n, v in saved.items())


def test_completed_artifacts_without_receipt_recover_without_model(campaign):
    c = campaign
    first = run(c)
    (c.directory / "campaign.json").unlink()
    assert run(c, backend_factory=lambda m: pytest.fail("completed artifacts loaded a model")) == first


@pytest.mark.parametrize("fault", ["weights", "config", "tokenizer", "step_cap", "split", "extra_field", "missing_field",
                                   "artifact", "aggregate_units", "missing_task", "missing_class", "class"])
def test_completed_campaign_rejects_identity_and_corruption(campaign, fault):
    c = campaign
    result = run(c)
    if fault in ("weights", "config", "tokenizer", "step_cap", "split"):
        key = dict(weights="base_checkpoint_hash", config="config_hash", tokenizer="tokenizer_hash",
                   step_cap="max_steps", split="split")[fault]
        result["identity"][key] = "wrong"
    elif fault == "extra_field":
        result["identity"]["extra"] = "value"
    elif fault == "missing_field":
        del result["identity"]["tokenizer_hash"]
    elif fault == "artifact":
        put(c.directory / "artifacts/extra.txt", "corrupted evidence")
    elif fault == "aggregate_units":
        result["aggregate"]["success_rate"] = 77.86
        atomic_json(c.directory / "artifacts/aggregate.json", result["aggregate"])
        result["artifacts_hash"] = tree_hash(c.directory / "artifacts")
    elif fault == "missing_task":
        next((c.directory / "artifacts/tasks").glob("*.json")).unlink()
        result["artifacts_hash"] = tree_hash(c.directory / "artifacts")
    elif fault == "missing_class":
        del result["hardware_class"]
    else:
        result["hardware_class"]["memory"] += 1
    atomic_json(c.directory / "campaign.json", result)
    with pytest.raises(ValueError):
        run(c, backend_factory=lambda m: pytest.fail("corrupt reuse launched a model"))


def test_partial_artifact_hash_and_old_identity_are_not_rebound(campaign):
    c = campaign
    run(c)
    (c.directory / "campaign.json").unlink()
    path = next((c.directory / "artifacts/tasks").glob("*.json"))
    envelope = json.loads(path.read_text())
    envelope["record"]["success"] = not envelope["record"]["success"]
    atomic_json(path, envelope)
    with pytest.raises(ValueError, match="corrupted task"):
        run(c)


def test_audited_reuse_retains_original_scored_identity_and_artifacts(campaign):
    c = campaign
    first = run(c)
    original = (c.directory / "campaign.json").read_bytes()
    current = changed_harness(c)
    supplement = audit(c, c.manifest["evaluation_harness"], current)
    for _ in range(2):
        assert run(c, supplement=supplement, backend_factory=lambda m: pytest.fail("audited reuse loaded model")) == first
    assert (c.directory / "campaign.json").read_bytes() == original
    assert tree_hash(c.directory / "artifacts") == first["artifacts_hash"]
    events = ComputeJournal(c.directory / "audit.jsonl").events
    assert len(events) == 2 and all(e["kind"] == "evaluation_reuse_via_audited_identity" for e in events)
    assert all(e["gpu_seconds"] == e["gpu_reserved_seconds"] == 0 for e in events)
    assert events[0]["stored_harness_hash"] == first["identity"]["evaluation_harness_hash"]
    assert events[0]["current_harness_hash"] == digest(current)


def test_unaudited_or_incomplete_historical_campaign_refused(campaign):
    c = campaign
    run(c)
    current = changed_harness(c)
    with pytest.raises(ValueError, match="audited"):
        run(c)
    supplement = audit(c, c.manifest["evaluation_harness"], current)
    (c.directory / "campaign.json").unlink()
    with pytest.raises(ValueError, match="refusing regeneration"):
        run(c, supplement=supplement)


def test_two_audits_reuse_only_connected_original_campaign(campaign):
    c = campaign
    first = run(c)
    middle = changed_harness(c)
    supplement = audit(c, c.manifest["evaluation_harness"], middle)
    latest = changed_harness(c)
    supplement = audit(c, middle, latest, supplement=supplement)
    assert run(c, supplement=supplement) == first
    corrupted = deepcopy(first)
    corrupted["identity"]["evaluation_harness_hash"] = digest(dict(latest, another_run=True))
    atomic_json(c.directory / "campaign.json", corrupted)
    with pytest.raises(ValueError, match="identity"):
        run(c, supplement=supplement)


def test_input_change_during_generation_prevents_aggregate(campaign):
    c = campaign
    def factory(manifest):
        backend = FakeBackend()
        def close():
            put(c.model / "model.safetensors", "changed while campaign was running")
        backend.close = close
        return backend
    with pytest.raises(ValueError):
        run(c, backend_factory=factory)
    assert not (c.directory / "campaign.json").exists()
    assert not (c.directory / "artifacts/aggregate.json").exists()


@pytest.mark.parametrize("location", ["training", "historical"])
def test_campaign_does_not_write_into_input_or_historical_directory(campaign, location):
    c = campaign
    if location == "training":
        output = c.model
        before = tree_hash(output)
    else:
        output = c.output
        put(output / c.tag / "metrics.json", dict(historical=True))
        before = tree_hash(output)
    with pytest.raises(ValueError, match="output|separate"):
        run(c, output_root=output)
    assert tree_hash(output) == before and not c.backends


def test_base_repair_damage_with_verified_same_class(campaign):
    c = campaign
    base = run(c, tag="base")
    directory = round_checkpoint(c)
    c.manifest = identity.make_manifest(c.root, c.config, data_root=c.data, model_path=c.model,
        hardware=c.hardware, run_directory=directory, round_number=1)
    result = run(c, base_evaluation=c.output / "base", env_factory=lambda tid: FakeEnv(tid, won=True))
    assert sum(r["repaired"] for r in result["repairs_damage"].values()) == 31
    assert sum(r["damaged"] for r in result["repairs_damage"].values()) == 0
    assert base["hardware_class"] == result["hardware_class"]
    saved = json.loads((c.output / "base/campaign.json").read_text())
    del saved["hardware_class"]
    atomic_json(c.output / "base/campaign.json", saved)
    with pytest.raises(ValueError, match="hardware class"):
        run(c, base_evaluation=c.output / "base")


def test_missing_base_class_rejects_new_campaign(campaign):
    c = campaign
    base = run(c, tag="base")
    del base["hardware_class"]
    atomic_json(c.output / "base/campaign.json", base)
    with pytest.raises(ValueError, match="hardware class"):
        run(c, base_evaluation=c.output / "base")
    assert not (c.directory / "campaign.json").exists()
    assert not (c.directory / "artifacts/aggregate.json").exists()
    assert len(c.backends) == 1  # only the base campaign ran


def test_benchmark_tag_lock_namespaces(campaign):
    c = campaign
    alf = evaluation.tag_lock_path(c.root, "same-tag")
    bfcl = bfcl_tag_lock_path(c.root, "same-tag")
    other = evaluation.tag_lock_path(c.root, "other-tag")
    assert alf != bfcl and alf != other
    with evaluation_lock(bfcl, tag="bfcl/same-tag", timeout=0):
        with evaluation_lock(alf, tag="alfworld/same-tag", timeout=0):
            with evaluation_lock(other, tag="alfworld/other-tag", timeout=0):
                with pytest.raises(TimeoutError):
                    with evaluation_lock(alf, tag="alfworld/same-tag", timeout=0):
                        pytest.fail("same tag lock was not exclusive")
    assert alf.exists()
    with evaluation_lock(alf, tag="alfworld/same-tag", timeout=0):
        pass


def test_tag_lock_contends_across_processes(campaign):
    c = campaign
    lock = evaluation.tag_lock_path(c.root, "same")
    code = """
import sys
from bfas.rtd.evaluation_lock import evaluation_lock
try:
    with evaluation_lock(sys.argv[1], tag='child', timeout=0):
        raise SystemExit(2)
except TimeoutError:
    pass
"""
    with evaluation_lock(lock, tag="parent", timeout=0):
        child = subprocess.run([sys.executable, "-c", code, str(lock)], capture_output=True, text=True, timeout=15)
    assert child.returncode == 0, child.stderr


def test_historical_anchors_are_explicit_references(campaign):
    c = campaign
    path = c.root / "results/appworld_students/alfabl_CE_conseq_s0/adapter"
    put(path / "model.safetensors", "fixture historical checkpoint")
    result = evaluation.comparison_anchors(c.root)
    ce, base = result["anchors"]
    assert result["budget_matched"] is False
    assert ce["checkpoint_exists"] and ce["weights_exists"] and not ce["historical_script_checkpoint_exists"]
    assert ce["reported_percent"] == 77.86 and base["reported_percent"] == 7.14
    assert ce["success_rate"] == 109 / 140 and base["success_rate"] == 10 / 140


def test_hf_loader_overlay_and_greedy_generation_are_cpu_testable(campaign, monkeypatch):
    c = campaign
    import torch
    calls = []
    class Model:
        def to(self, device):
            assert device == "cpu"
            return self
        def eval(self):
            return self
        def generate(self, **kwargs):
            cfg = kwargs["generation_config"]
            assert cfg.do_sample is False and cfg.num_beams == 1 and cfg.max_new_tokens == 256
            return torch.tensor([[10, 11, 20, 2]])
    class Loader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append((args, kwargs))
            assert kwargs["local_files_only"] and kwargs["device_map"] == "cpu"
            return Model()
    class Tokenizer:
        eos_token_id, pad_token_id = 2, 0
        def __call__(self, prompt, **kwargs):
            assert kwargs["add_special_tokens"] is False
            return dict(input_ids=torch.tensor([[10, 11]]))
        def decode(self, ids, **kwargs):
            assert ids == [20]
            return "ACTION: look"
    monkeypatch.setattr(evaluation, "FrozenRenderer", lambda path: SimpleNamespace(adapter=SimpleNamespace(_tokenizer=Tokenizer())))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModelForCausalLM=Loader,
        GenerationConfig=lambda **kwargs: SimpleNamespace(**kwargs)))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=Loader))
    directory = round_checkpoint(c)
    manifest = identity.make_manifest(c.root, c.config, data_root=c.data, model_path=c.model,
        hardware=c.hardware, run_directory=directory, round_number=1)
    backend = evaluation.HFBackend(manifest, device="cpu")
    assert calls[1][0][1] == directory / "round-1/lora"
    assert calls[1][1]["torch_device"] == "cpu"
    assert backend.generate("prompt", temperature=0, max_new_tokens=256) == evaluation.Generation("ACTION: look", 2, False)
    backend.max_context = 257
    with pytest.raises(ValueError, match="overflow"):
        backend.generate("prompt", temperature=0, max_new_tokens=256)
    backend.close()
    calls.clear()
    merged = evaluation.HFBackend(c.manifest, device="cpu")
    assert len(calls) == 1 and calls[0][0][0] == str(c.model)
    merged.close()


def test_cli_prepare_and_read_only_validation(campaign, capsys):
    c = campaign
    from tools.rtd_alfworld_evaluate import main
    put(c.root / "hardware.json", c.hardware)
    binding = c.root / "binding.json"
    assert main(["--root", str(c.root), "prepare", "--model", str(c.model),
        "--data-root", str(c.data), "--hardware-json", str(c.root / "hardware.json"), "--out", str(binding)]) == 0
    assert json.loads(binding.read_text()) == c.manifest
    capsys.readouterr()
    run(c)
    before = tree_hash(c.directory)
    assert main(["--root", str(c.root), "validate", "--binding", str(binding),
                 "--output-root", str(c.output), "--tag", c.tag]) == 0
    assert tree_hash(c.directory) == before
    capsys.readouterr()


with warnings.catch_warnings():
    warnings.simplefilter("ignore", pytest.PytestUnknownMarkWarning)
    integration = pytest.mark.integration


@integration
def test_official_loop_two_real_valid_seen_tasks_with_scripted_backend(tmp_path, monkeypatch):
    """Two full CPU environment episodes; no policy model, teacher or GPU smoke."""
    from bfas.rtd.benchmarks.alfworld_support import FrozenRenderer
    from tools.rtd_alfworld_verify import TOKENIZER
    adapter = evaluation.adapter
    if not adapter.ENV_PYTHON.is_file() or not (TOKENIZER / "tokenizer.json").is_file():
        pytest.skip("ALFWorld environment Python or local tokenizer unavailable; no downloads")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    import torch
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda *a, **k: pytest.fail("integration attempted CUDA initialization"))
    monkeypatch.chdir(tmp_path)
    try:
        expected = identity.official_expectations(adapter.DATA)
    except (OSError, ValueError) as exc:
        pytest.skip(f"official valid_seen environment data unavailable: {exc}")
    backend = FakeBackend()
    backend.renderer = FrozenRenderer(TOKENIZER)
    backend.generate = lambda *a, **k: evaluation.Generation("ACTION: look", 3, False)
    ident = dict(split="valid_seen", evaluation_temperature=0., max_steps=40, max_action_tokens=256)
    for tid in expected["task_ids"][:2]:
        reset_seen = []
        class Bridge(evaluation.EvaluationEnvBridge):
            def _read(self):
                state = super()._read()
                if state.get("op") == "state":
                    reset_seen.append(True)
                return state
        try:
            record = evaluation.official_episode(tid, backend, identity=ident,
                env_factory=lambda task: Bridge(task, data_root=adapter.DATA,
                    environment_root=adapter.ROOT / "envs/alfworld"))
        except evaluation.EnvironmentUnavailable as exc:
            if not reset_seen:
                pytest.skip(f"real ALFWorld worker unavailable before reset: {exc}")
            raise
        assert record["task_id"] == tid and record["split"] == "valid_seen"
        assert record["steps"] == 40 or record["done"]
        assert type(record["success"]) is bool
        assert record["generated_text"] == ["ACTION: look"] * record["steps"]
