"""CPU integration of the real micro-update/delta/response pipeline; no downloads."""
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

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from tools.behavior_atom import gpu_driver as driver
from tools import behavior_atom_experiment as cli
from src.bfas.behavior import microupdate as micro
from src.bfas.behavior.deltas import apply_delta, load_delta, tensor_state_hash
from src.bfas.behavior.provenance import repo_commit
from src.bfas.behavior.response import read_responses, content_hash
from tools.behavior_atom.split_check import event_key


class Tokenizer:
    eos_token_id, pad_token_id, bos_token_id = 0, 0, 1
    chat_template = None

    def __call__(self, text, **kwargs):
        # Teachers = token 2, rejected = 3. Prompts use other tokens.
        if text in {"teacher", "student"}:
            return {"input_ids": [2 if text == "teacher" else 3]}
        return {"input_ids": [1] + [4 + ord(c) % 4 for c in text]}

    def get_vocab(self):
        return {str(i): i for i in range(8)}

    def batch_decode(self, outputs, **kwargs):
        return ["call" if int(row[0]) == 2 else "abstain" for row in outputs]


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(8))
        self.register_buffer("fixed_buffer", torch.ones(1))
        self.frozen = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.config = TinyConfig()

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.logits.view(1, 1, -1).expand(*input_ids.shape, 8))


class TinyConfig:
    use_cache = False

    def to_dict(self):
        return {"use_cache": self.use_cache, "kind": "tiny"}


class FakeStudent(driver.Student):
    def __init__(self, *, error_probe=None, failure=None):
        self.batches = []
        self.check_calls = []
        self.error_probe = error_probe
        self.failure = failure
        super().__init__(TinyModel(), Tokenizer(), "fake-checker-v1", self.parse_calls, self.check_calls_fn)

    def parse_calls(self, text):
        return [{"f": {}}] if text == "call" else []

    def check_calls_fn(self, probe, calls):
        self.check_calls.append(probe["probe_id"])
        if probe["probe_id"] == self.error_probe:
            raise RuntimeError("checker unavailable")
        return {"valid": True}

    def generate(self, ids, mask, settings):
        assert settings["temperature"] == 0 and settings["do_sample"] is False
        assert settings["enable_thinking"] is False
        assert ids.device.type == "cpu"
        self.batches.append((ids.clone(), mask.clone()))
        if self.failure:
            raise self.failure("injected generation failure")
        # Real differentiable training moves logit 2 above logit 3; a fixed
        # greedy tie-break yields abstention at exactly the common init.
        token = 2 if self.model.logits[2] > self.model.logits[3] else 3
        return torch.cat((ids, torch.full((len(ids), 1), token, dtype=torch.long)), dim=1)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("status,dirty", [("", 0), (" M tracked.py\n?? untracked/\n", 2)])
def test_repo_commit_prefers_git(tmp_path, monkeypatch, status, dirty):
    monkeypatch.setenv("BEHAVIOR_ATOM_COMMIT", "env-sha")
    calls = []
    def check_output(args, *, cwd, text, stderr):
        assert cwd == tmp_path and text is True and stderr == subprocess.DEVNULL
        calls.append(args)
        return "git-sha\n" if args == ["git", "rev-parse", "HEAD"] else status
    monkeypatch.setattr(subprocess, "check_output", check_output)
    assert repo_commit(tmp_path) == {"commit": "git-sha", "dirty": dirty, "provenance": "git"}
    assert calls == [["git", "rev-parse", "HEAD"], ["git", "status", "--short"]]


@pytest.mark.parametrize("failure", [
    subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"]),
    FileNotFoundError("git"), RuntimeError("Git unavailable"),
])
@pytest.mark.parametrize("env,contents,expected", [
    (" env-sha \n", "file-sha dirty=3\n", {"commit": "env-sha", "dirty": None, "provenance": "env"}),
    ("env-sha dirty=0", None, {"commit": "env-sha", "dirty": 0, "provenance": "env"}),
    (None, "file-sha dirty=3\n", {"commit": "file-sha", "dirty": 3, "provenance": "file"}),
    (" \n", " file-sha\n", {"commit": "file-sha", "dirty": None, "provenance": "file"}),
    (None, None, {"commit": "unknown", "dirty": None, "provenance": "no-git"}),
    (None, " \n", {"commit": "unknown", "dirty": None, "provenance": "no-git"}),
    (None, "file-sha dirty=bad", {"commit": "unknown", "dirty": None, "provenance": "no-git"}),
])
def test_repo_commit_fallbacks(tmp_path, monkeypatch, failure, env, contents, expected):
    def fail(*args, **kwargs):
        raise failure
    monkeypatch.setattr(subprocess, "check_output", fail)
    monkeypatch.delenv("BEHAVIOR_ATOM_COMMIT", raising=False)
    if env is not None:
        monkeypatch.setenv("BEHAVIOR_ATOM_COMMIT", env)
    if contents is not None:
        path = tmp_path / "data/behavior_atom_v1/COMMIT"
        path.parent.mkdir(parents=True)
        path.write_text(contents)
    assert repo_commit(tmp_path) == expected


def test_repo_commit_status_failure_and_unreadable_file(tmp_path, monkeypatch):
    def check_output(args, **kwargs):
        if args == ["git", "rev-parse", "HEAD"]:
            return "git-sha\n"
        raise subprocess.CalledProcessError(128, args)
    monkeypatch.setattr(subprocess, "check_output", check_output)
    monkeypatch.setenv("BEHAVIOR_ATOM_COMMIT", "env-sha")
    assert repo_commit(tmp_path) == {"commit": "env-sha", "dirty": None, "provenance": "env"}
    monkeypatch.delenv("BEHAVIOR_ATOM_COMMIT")
    path = tmp_path / "data/behavior_atom_v1/COMMIT"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff")
    assert repo_commit(tmp_path) == {"commit": "unknown", "dirty": None, "provenance": "no-git"}
    path.unlink()
    path.mkdir()
    assert repo_commit(tmp_path) == {"commit": "unknown", "dirty": None, "provenance": "no-git"}


def fixture_files(tmp_path):
    rows = [dict(prompt=f"long source event prompt before the tail cap {i}", response="teacher", _rejected="student", _traj=f"train_{i}#event") for i in range(6)]
    pool = tmp_path / "pool.jsonl"
    pool.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    def unit(sid, indices, role="formal", kind="event"):
        return dict(source_id=sid, role=role, unit_type=kind, parent_task_id="train_" + sid,
                    group="train_" + sid, factor="boundary", family="simple", rows=[
                        dict(pool_row=i, event_key=event_key(rows[i]), prompt_sha1=driver._sha1(rows[i]["prompt"])[:16],
                             yT_sha1=driver._sha1(rows[i]["response"]), yS_sha1=driver._sha1(rows[i]["_rejected"])) for i in indices])
    units = [unit("pilot", [0], "pilot"), unit("formal_a", [1, 2], kind="pack"),
             unit("formal_b", [3, 4], kind="contrastive"), unit("zero", [5], "noop")]
    probes = [dict(probe_id=f"probe_{i}", task_id=f"probe_task_{i}", prompt="probe " + "x" * size,
                   function=[{"name": "f"}], truth=[{"f": {}}], category="simple_python",
                   factor="boundary", family="simple", expected="abstain" if i == 2 else "call",
                   base_rate=0.25, base_margin=None, parent_task_id=f"held_out_{i}", group=f"held_out_{i}",
                   response="teacher", _rejected="student") for i, size in enumerate((10, 1, 5))]
    for p in probes:
        p["prompt_sha1"] = driver._sha1(p["prompt"])
    source_file, probe_file = tmp_path / "sources.json", tmp_path / "probes.json"
    source_file.write_text(json.dumps({"units": units}))
    probe_file.write_text(json.dumps({"probes": probes}))
    descriptor = {"sources": source_file.name, "probes": probe_file.name, "pool": pool.name}
    data_file = tmp_path / "data.json"
    data_file.write_text(json.dumps(descriptor))
    config = {"run_id": "tiny", "data": "data.json", "_base": str(tmp_path),
              "cost_estimate": {"gpu_hours_per_update": 0.01, "gpu_seconds_per_rollout": 1},
              "protocol": {"micro_update": {"model_id": "tiny", "device": "cpu", "precision": "float32",
                                            "lr": 0.1, "prompt_cap": 32},
                           "evaluation": {"batch_size": 2, "max_new_tokens": 3, "token_budget": 80},
                           "pilot": {"max_abstention_shift": 1.0}}}
    protocol = cli.frozen_protocol({**config, "protocol": driver.frozen_settings(config["protocol"])})
    data = driver.load_inputs(config, descriptor, data_file)
    return config, protocol, data, data_file


def pilot(tmp_path, *, student_factory=None):
    config, protocol, data, data_file = fixture_files(tmp_path)
    out = tmp_path / "pilot"
    driver.run("pilot", config, protocol, data, out, student_factory=student_factory or (lambda _: FakeStudent()))
    config["pilot_pass"] = str(out / "pilot_pass.json")
    return config, protocol, data, data_file, out


def test_pipeline_restores_saves_deltas_scores_noop_and_pass(tmp_path, monkeypatch):
    def no_git(args, **kwargs):
        raise subprocess.CalledProcessError(128, args)
    monkeypatch.setattr(subprocess, "check_output", no_git)
    monkeypatch.setenv("BEHAVIOR_ATOM_COMMIT", "cluster-sha dirty=2")
    commit = {"commit": "cluster-sha", "dirty": 2, "provenance": "env"}
    created, initial_hashes, updates = [], [], []
    original = micro.run_micro_update
    def spy(factory, rows, cfg):
        model = factory()
        initial_hashes.append(tensor_state_hash(micro._trainables(model)))
        assert cfg["batch_size"] == len(rows) and cfg["accumulation"] == 1
        result = original(factory, rows, cfg)
        updates.append(result.manifest)
        return result
    monkeypatch.setattr(micro, "run_micro_update", spy)
    def factory(_):
        student = FakeStudent()
        created.append(student)
        return student
    config, protocol, data, _, out = pilot(tmp_path, student_factory=factory)
    for name in ("manifest.json", "launch.json"):
        manifest = json.loads((out / name).read_text())
        assert {key: manifest[key] for key in commit} == commit
    assert len(created) == 1
    assert len(initial_hashes) == 2 and len(set(initial_hashes)) == 1
    assert tensor_state_hash(micro._trainables(created[0].model)) == initial_hashes[0]
    saved = torch.load(out / "initial_state.pt", weights_only=True)
    for source in ("pilot", "zero"):
        directory = out / "sources" / source
        delta = load_delta(directory / f"delta_{source}.npz")
        model = TinyModel()
        apply_delta(model, delta)
        summary = json.loads((directory / "summary.json").read_text())
        assert summary["updated_trainables_hash"] == tensor_state_hash(micro._trainables(model))
        assert summary["reload_equal"] is True
        assert delta.manifest["base_hash"] == tensor_state_hash(saved["trainables"])
        manifest = json.loads((directory / "update.manifest.json").read_text())
        assert {key: manifest[key] for key in commit} == commit
    assert load_delta(out / "sources/zero/delta_zero.npz").manifest["update_norm"] == 0
    pp, records, _ = read_responses(out / "responses.jsonl")
    assert len(pp) == 3 and len(records) == 6
    assert {r.source_id for r in records} == {"pilot", "zero"}
    assert all(r.difference == 0 for r in records if r.source_id == "zero")
    assert all(r.updated_outcome in {0, 1} for r in records)
    assert all(r.teacher_likelihood is not None and r.student_likelihood is not None for r in records)
    passed = json.loads((out / "pilot_pass.json").read_text())
    assert passed["pass"] and passed["passed"]
    assert passed["frozen_config_hash"] == content_hash(protocol)
    assert passed["floor"]["repeat"]["max_abs_score_change"] == 0
    assert passed["floor"]["zero_delta_equal"] is True
    assert passed["n_changed_probes"] == 3
    assert passed["kl"]["pilot"]["mean"] > 0
    assert cli.require_pilot_pass(config, protocol, {})["passed"]
    assert updates[0]["steps"] == 3
    assert updates[0]["preprocessing"]["prompt_side"] == "tail"
    assert updates[0]["preprocessing"]["source_counts"][0]["chosen"]["prompt_tokens_dropped"] > 0
    # Base, repeated base, zero application, pilot update, noop; same batches.
    batches = created[0].batches
    assert len(batches) == 10
    for i in range(0, len(batches), 2):
        assert torch.equal(batches[i][0], batches[0][0])
        assert torch.equal(batches[i + 1][0], batches[1][0])
    assert "probe_2" not in created[0].check_calls  # abstain never calls checker
    monkeypatch.setenv("BEHAVIOR_ATOM_COMMIT", "different-launch-sha")
    resumed = driver.run("pilot", config, protocol, data, out, resume=True,
                         student_factory=lambda _: pytest.fail("complete resume loaded model"))
    assert resumed["status"] == "complete"


def test_failures_are_na_and_cannot_pass_pilot(tmp_path):
    config, protocol, data, _ = fixture_files(tmp_path)
    out = tmp_path / "errors"
    driver.run("pilot", config, protocol, data, out, student_factory=lambda _: FakeStudent(error_probe="probe_1"))
    _, records, _ = read_responses(out / "responses.jsonl")
    failure = next(r for r in records if r.source_id == "pilot" and r.probe_id == "probe_1")
    assert failure.status == "error" and failure.updated_outcome is None and failure.difference is None
    assert "checker:RuntimeError" in failure.error_type
    matrix = json.loads((out / "response_matrix.json").read_text())
    assert matrix["M"][1][0] is None
    passed = json.loads((out / "pilot_pass.json").read_text())
    assert passed["pass"] is False
    config["pilot_pass"] = str(out / "pilot_pass.json")
    with pytest.raises(ValueError, match="pilot pass"):
        driver.run("collect", config, protocol, data, tmp_path / "refused", dry_run=True)


@pytest.mark.parametrize("failure", [RuntimeError, torch.OutOfMemoryError])
def test_generation_failure_recorded_without_batch_fallback(tmp_path, failure):
    _, protocol, data, _ = fixture_files(tmp_path)
    student = FakeStudent(failure=failure)
    records, policy = driver.evaluate(student, data["probes"], protocol)
    assert len(student.batches) == 2
    assert all(r["outcome"] is None and r["output"] is None for r in records)
    assert all(any(e.startswith("generate:") for e in r["errors"]) for r in records)
    assert all(p is not None for p in policy)


def test_shards_merge_identical_to_full_collect(tmp_path):
    config, protocol, data, _, _ = pilot(tmp_path)
    out = tmp_path / "collect"
    full = driver.run("collect", config, protocol, data, out, student_factory=lambda _: FakeStudent())
    for update in full["updates"]:
        assert update["steps"] == (0 if update["role"] == "noop" else 3)
        if update["role"] != "noop":
            assert update["unit_type"] in {"pack", "contrastive"}
            assert update["effective_loss_tokens"] == 3 * 2 * 2 * 2  # steps, rows, T/S, token+EOS
    parts = tmp_path / "parts"
    for i in range(2):
        driver.run("collect", config, protocol, data, parts, shard=f"{i}/2", student_factory=lambda _: FakeStudent())
    paths = sorted(parts.glob("shard-*"))
    merged = tmp_path / "merged"
    driver.merge(paths, merged)
    assert (merged / "responses.jsonl").read_bytes() == (out / "responses.jsonl").read_bytes()
    assert (merged / "response_matrix.json").read_bytes() == (out / "response_matrix.json").read_bytes()
    driver.merge(paths, merged, resume=True)
    with pytest.raises(ValueError, match="partition"):
        driver.merge(paths[:1], tmp_path / "incomplete")
    assert cli.main(["merge", "--shards", *map(str, paths), "--output-dir", str(tmp_path / "cli_merge")]) == 0
    with open(paths[1] / "responses.jsonl", "a") as stream:
        stream.write("corruption")
    with pytest.raises(ValueError, match="content hash"):
        driver.merge(paths, tmp_path / "corrupted")


def test_partial_resume_after_update_oom_keeps_committed_sources(tmp_path, monkeypatch):
    config, protocol, data, _, _ = pilot(tmp_path)
    original = micro.run_micro_update
    calls, models = [], []
    fail = True
    def runner(factory, rows, cfg):
        calls.append(cfg["source_id"])
        model = factory()
        models.append(model)
        if fail and cfg["source_id"] == "formal_b":
            with torch.no_grad():
                model.logits.add_(99)
            raise torch.OutOfMemoryError("update interrupted")
        return original(factory, rows, cfg)
    monkeypatch.setattr(micro, "run_micro_update", runner)
    out = tmp_path / "resume"
    with pytest.raises(torch.OutOfMemoryError):
        driver.run("collect", config, protocol, data, out, roles=["formal", "noop"],
                   student_factory=lambda _: FakeStudent())
    assert torch.equal(models[-1].logits, torch.zeros(8))
    assert (out / "sources/formal_a/complete.json").is_file()
    assert not (out / "sources/formal_b").exists()
    fail = False
    driver.run("collect", config, protocol, data, out, resume=True, roles=["formal", "noop"],
               student_factory=lambda _: FakeStudent())
    assert calls == ["formal_a", "formal_b", "formal_b", "zero"]
    _, records, _ = read_responses(out / "responses.jsonl")
    assert len(records) == 9


def test_cli_dry_run_real_wiring_and_gate(tmp_path, monkeypatch, capsys):
    config, protocol, data, data_file = fixture_files(tmp_path)
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({k: v for k, v in config.items() if k != "_base"}))
    out = tmp_path / "dry"
    monkeypatch.setattr(driver, "load_student", lambda _: pytest.fail("dry-run loaded model"))
    assert cli.main(["pilot", "--config", str(config_file), "--output-dir", str(out), "--dry-run"]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["number_of_models"] == 1
    assert planned["sources"] == 2 and planned["probes"] == 3
    assert planned["max_rollouts"] == 9 and planned["numerical_check_rollouts"] == 6
    assert planned["cost_estimate"]["gpu_hours"] > 0 and planned["dependencies"]
    assert planned["new_teacher_tokens"] == 0 and not out.exists()
    assert cli.main(["collect", "--config", str(config_file), "--dry-run"]) == 2
    assert "pilot pass" in capsys.readouterr().err
    monkeypatch.setattr(driver, "load_student", lambda _: FakeStudent())
    assert cli.main(["pilot", "--config", str(config_file), "--sources", str(tmp_path / "sources.json"),
                     "--probes", str(tmp_path / "probes.json"), "--pool", str(tmp_path / "pool.jsonl"),
                     "--output-dir", str(out)]) == 0
    assert json.loads((out / "pilot_pass.json").read_text())["pass"]
    config["pilot_pass"] = str(out / "pilot_pass.json")
    config_file.write_text(json.dumps({k: v for k, v in config.items() if k != "_base"}))
    capsys.readouterr()
    assert cli.main(["collect", "--config", str(config_file), "--shard", "0/2", "--dry-run"]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["sources"] == 1 and planned["total_sources"] == 2 and planned["max_rollouts"] == 6


@pytest.mark.parametrize("ell", ["full_sequence", "length_normalized"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batched_likelihood_matches_shared_gradient_scorer(tmp_path, ell, dtype):
    _, protocol, data, _ = fixture_files(tmp_path)
    protocol["micro_update"]["ell"] = protocol["evaluation"]["ell"] = ell
    student = FakeStudent()
    student.model.to(dtype=dtype)
    with torch.no_grad():
        student.model.logits.copy_(torch.arange(8) / 7)
    records, _ = driver.evaluate(student, data["probes"], protocol)
    import appworld_train as trainer
    with driver._context(protocol):
        for p, record in zip(data["probes"], records):
            for field, key in (("teacher", "response"), ("student", "_rejected")):
                encoded = trainer.encode(student.tokenizer, {"prompt": driver._thinking_off(p["prompt"]), "response": p[key]}, device="cpu")
                expected = micro.evaluation_score(student.model, *encoded, ell=ell)
                assert record[field] == pytest.approx(float(expected.detach()))
                assert expected.requires_grad


def test_manifest_content_and_frozen_setting_rejection(tmp_path):
    config, protocol, data, data_file = fixture_files(tmp_path)
    changed = copy.deepcopy(config["protocol"])
    changed["stochastic"] = True
    with pytest.raises(ValueError, match="deterministic"):
        driver.frozen_settings(changed)
    with pytest.raises(ValueError, match="ell"):
        driver.frozen_settings({"evaluation": {"ell": "length_normalized"}})
    for shard in ("1/1", "-1/2", "0/0", "1/9", "bad"):
        with pytest.raises(ValueError, match="shard"):
            driver.select_sources(data, "collect", shard)
    with pytest.raises(ValueError, match="only by collect"):
        driver.select_sources(data, "pilot", "0/2")
    rows = (tmp_path / "pool.jsonl").read_text()
    (tmp_path / "pool.jsonl").write_text(rows.replace("teacher", "altered"))
    with pytest.raises(ValueError, match="hash mismatch"):
        driver.load_inputs(config, json.loads(data_file.read_text()), data_file)


def test_collect_rejects_changed_initial_model_and_resume_corruption(tmp_path):
    config, protocol, data, _, out = pilot(tmp_path)
    def changed(_):
        student = FakeStudent()
        with torch.no_grad():
            student.model.logits[0] = 0.5
        return student
    with pytest.raises(ValueError, match="pilot runtime identity"):
        driver.run("collect", config, protocol, data, tmp_path / "wrong_init", student_factory=changed)
    with open(out / "sources/pilot/delta_pilot.npz", "ab") as stream:
        stream.write(b"broken")
    with pytest.raises(ValueError, match="content hash"):
        driver.run("pilot", config, protocol, data, out, resume=True,
                   student_factory=lambda _: pytest.fail("corrupt resume loaded model"))


def test_likelihood_only_change_cannot_pass(tmp_path):
    config, protocol, data, _ = fixture_files(tmp_path)
    class NoBehaviorChange(FakeStudent):
        def generate(self, ids, mask, settings):
            return torch.cat((ids, torch.full((len(ids), 1), 3, dtype=torch.long)), dim=1)
    out = tmp_path / "unresolved"
    driver.run("pilot", config, protocol, data, out, student_factory=lambda _: NoBehaviorChange())
    passed = json.loads((out / "pilot_pass.json").read_text())
    _, records, _ = read_responses(out / "responses.jsonl")
    assert any(r.teacher_likelihood != r.base_teacher_likelihood for r in records)
    assert passed["measurement_stable"] and not passed["measurement_resolved"]
    assert passed["n_changed_probes"] == 0 and passed["pass"] is False


def test_unstable_floor_and_zero_delta_fail_pilot(tmp_path):
    config, protocol, data, _ = fixture_files(tmp_path)
    class Flaky(FakeStudent):
        def generate(self, ids, mask, settings):
            output = super().generate(ids, mask, settings)
            if len(self.batches) > 2:  # all passes after the first base differ
                output[:, -1] = 2
            return output
    out = tmp_path / "unstable"
    driver.run("pilot", config, protocol, data, out, student_factory=lambda _: Flaky())
    passed = json.loads((out / "pilot_pass.json").read_text())
    assert passed["floor"]["repeat"]["outcome_changes"] == 3
    assert not passed["floor"]["zero_delta_equal"]
    assert not passed["measurement_stable"] and not passed["pass"]


def test_optional_likelihood_texts_remain_na_without_evaluation_failure(tmp_path):
    _, protocol, data, _ = fixture_files(tmp_path)
    for probe in data["probes"]:
        probe.pop("response")
        probe.pop("_rejected")
    records, policy = driver.evaluate(FakeStudent(), data["probes"], protocol)
    assert all(r["teacher"] is None and r["student"] is None and not r["errors"] for r in records)
    assert all(p is not None for p in policy)


@pytest.mark.parametrize("arrives_before_collect", [False, True])
def test_missing_fisher_pilot_collect_and_resume(tmp_path, monkeypatch, arrives_before_collect):
    config, _, _, _ = fixture_files(tmp_path)
    config["protocol"]["fisher"] = "pending_fisher.npz"
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({k: v for k, v in config.items() if k != "_base"}))
    monkeypatch.setattr(driver, "load_student", lambda _: FakeStudent())
    out = tmp_path / "pending_pilot"
    assert cli.main(["pilot", "--config", str(config_file), "--output-dir", str(out)]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["fisher"] == "pending"
    assert manifest["protocol"]["fisher"] == "pending_fisher.npz"
    assert json.loads((out / "pilot_pass.json").read_text())["passed"]

    # The strict consumer used for fit/decompose whitening must still fail
    # until the Fisher exists; collection's sentinel never substitutes for F.
    from src.bfas.behavior.whiten import from_fisher
    fisher_path = tmp_path / "pending_fisher.npz"
    with pytest.raises(FileNotFoundError):
        from_fisher(fisher_path, TinyModel(), eps=1e-6)
    if arrives_before_collect:
        # Actual valid Fisher payload; pilot identity must remain usable.
        from src.bfas.behavior.deltas import layout_hash
        np.savez(fisher_path, fisher=np.ones(8, dtype=np.float32), layout_hash=layout_hash(TinyModel()))
        assert from_fisher(fisher_path, TinyModel(), eps=1e-6).size == 8

    config["pilot_pass"] = str(out / "pilot_pass.json")
    config_file.write_text(json.dumps({k: v for k, v in config.items() if k != "_base"}))
    collect = tmp_path / "pending_collect"
    assert cli.main(["collect", "--config", str(config_file), "--output-dir", str(collect)]) == 0
    collected = json.loads((collect / "manifest.json").read_text())
    assert collected["fisher"] == (content_hash(fisher_path.read_bytes()) if arrives_before_collect else "pending")
    assert collected["identity"]["frozen_config_hash"] == manifest["identity"]["frozen_config_hash"]
    assert (collect / "complete.json").is_file()
    monkeypatch.setattr(driver, "load_student", lambda _: pytest.fail("resume loaded model"))
    assert cli.main(["pilot", "--config", str(config_file), "--output-dir", str(out), "--resume"]) == 0
    assert json.loads((out / "manifest.json").read_text())["fisher"] == "pending"


def test_fisher_tolerance_does_not_hide_other_file_errors(tmp_path):
    config, protocol, data, _ = fixture_files(tmp_path)
    # A directory/permission error is not an absent, pending artifact.
    protocol["fisher"] = str(tmp_path)
    with pytest.raises(IsADirectoryError):
        driver.run("pilot", config, protocol, data, tmp_path / "bad_fisher",
                   student_factory=lambda _: pytest.fail("loaded model before input validation"))


@pytest.mark.parametrize("strict", [False, True])
def test_initial_settings_diff_warns_or_fails_strict(tmp_path, strict):
    config, protocol, data, _ = fixture_files(tmp_path)
    external = tmp_path / "legacy.pt"
    micro.save_initial_state(TinyModel(), external)
    legacy = torch.load(external, weights_only=True)
    legacy["model_settings"]["model_config"].update(
        gradient_checkpointing=False, use_cache=True, deterministic_algorithms=False)
    # Older Fisher states did not include this optional integrity digest.
    legacy.pop("trainables_hash")
    torch.save(legacy, external)
    original = external.read_bytes()
    protocol["micro_update"]["init_state_path"] = str(external)
    out = tmp_path / "settings"
    if strict:
        with pytest.raises(ValueError, match=r"model settings differ:.*gradient_checkpointing.*strict-init-state"):
            driver.run("pilot", config, protocol, data, out, strict_init_state=True,
                       student_factory=lambda _: FakeStudent())
        assert not (out / "base").exists()
    else:
        with pytest.warns(UserWarning, match="model settings differ"):
            result = driver.run("pilot", config, protocol, data, out, student_factory=lambda _: FakeStudent())
        assert result["status"] == "complete"
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["init_state_settings_diff"] == [
            "model_config.deterministic_algorithms", "model_config.gradient_checkpointing", "model_config.use_cache"]
        assert manifest["init_state_created"] is False
        assert json.loads((out / "pilot_pass.json").read_text())["passed"]
        no_load = lambda _: pytest.fail("complete resume loaded model")
        driver.run("pilot", config, protocol, data, out, resume=True, student_factory=no_load)
        with pytest.raises(ValueError, match="strict-init-state"):
            driver.run("pilot", config, protocol, data, out, resume=True,
                       strict_init_state=True, student_factory=no_load)
    assert external.read_bytes() == original


@pytest.mark.parametrize("mismatch", ["layout", "frozen", "trainables"])
@pytest.mark.parametrize("strict", [False, True])
def test_initial_tensor_identity_mismatch_always_fails(tmp_path, mismatch, strict):
    config, protocol, data, _ = fixture_files(tmp_path)
    model = TinyModel()
    with torch.no_grad():
        if mismatch == "layout":
            model.logits = torch.nn.Parameter(torch.zeros(9))
        elif mismatch == "frozen":
            model.frozen.add_(1)
        else:
            model.logits.add_(1)
    external = tmp_path / "wrong.pt"
    micro.save_initial_state(model, external)
    protocol["micro_update"]["init_state_path"] = str(external)
    with pytest.raises(ValueError, match=f"initial-state {mismatch}"):
        driver.run("pilot", config, protocol, data, tmp_path / "wrong", strict_init_state=strict,
                   student_factory=lambda _: FakeStudent())


def test_auto_created_init_cli_roundtrip_resume_and_collect(tmp_path, monkeypatch):
    config, _, _, _ = fixture_files(tmp_path)
    config["protocol"]["micro_update"].update(init_state_path="states/seed37.pt", init_seed=37)
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    external = tmp_path / "states/seed37.pt"
    def factory(protocol):
        assert torch.initial_seed() == protocol["micro_update"]["init_seed"] == 37
        student = FakeStudent()
        with torch.no_grad():
            student.model.logits.fill_(float(torch.rand(())))
        return student
    monkeypatch.setattr(driver, "load_student", factory)
    out = tmp_path / "auto"
    assert driver.main(["pilot", "--config", str(config_file), "--output-dir", str(out), "--strict-init-state"]) == 0
    original = external.read_bytes()
    saved = torch.load(external, weights_only=True)
    local = torch.load(out / "initial_state.pt", weights_only=True)
    assert saved["trainables_hash"] == tensor_state_hash(local["trainables"])
    assert saved["layout_hash"] == local["layout_hash"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["init_state_created"] is True and manifest["init_state_settings_diff"] == []
    assert json.loads((out / "pilot_pass.json").read_text())["passed"]
    monkeypatch.setattr(driver, "load_student", lambda _: pytest.fail("complete resume loaded model"))
    assert driver.main(["pilot", "--config", str(config_file), "--output-dir", str(out), "--resume"]) == 0
    config["pilot_pass"] = str(out / "pilot_pass.json")
    config_file.write_text(json.dumps(config))
    monkeypatch.setattr(driver, "load_student", factory)
    collected = tmp_path / "auto_collect"
    assert driver.main(["collect", "--config", str(config_file), "--output-dir", str(collected)]) == 0
    collect_manifest = json.loads((collected / "manifest.json").read_text())
    assert collect_manifest["init_state_created"] is False
    assert collect_manifest["identity"]["frozen_config_hash"] == manifest["identity"]["frozen_config_hash"]
    assert external.read_bytes() == original


def test_init_state_subcommand_only_builds_seeded_model(tmp_path, monkeypatch, capsys):
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps({"protocol": {"micro_update": {"model_id": "tiny", "device": "cpu"}}}))
    path = tmp_path / "init.pt"
    args = ["init-state", "--config", str(config_file), "--init-state-path", str(path), "--init-seed", "19"]
    monkeypatch.setattr(driver, "load_student", lambda _: pytest.fail("init-state loaded checker/student"))
    monkeypatch.setattr(driver, "load_inputs", lambda *args: pytest.fail("init-state loaded sources/probes"))
    monkeypatch.setattr(driver, "load_model", lambda _: pytest.fail("dry-run built model"))
    assert driver.main([*args, "--dry-run"]) == 0
    assert not path.exists()
    assert json.loads(capsys.readouterr().out)["max_rollouts"] == 0
    def build(protocol):
        assert torch.initial_seed() == protocol["micro_update"]["init_seed"] == 19
        return TinyModel()
    monkeypatch.setattr(driver, "load_model", build)
    rng = torch.get_rng_state().clone()
    assert driver.main(args) == 0
    assert torch.equal(rng, torch.get_rng_state())
    result = json.loads(capsys.readouterr().out)
    assert result["init_state_created"] is True
    assert result["trainables_hash"] == tensor_state_hash(micro._trainables(TinyModel()))
    assert json.loads(path.with_suffix(".pt.manifest.json").read_text()) == result
    original = path.read_bytes()
    assert driver.main(args) == 2
    assert "already exists" in capsys.readouterr().err
    assert path.read_bytes() == original


def add_anchor_pool(tmp_path):
    path = tmp_path / "pool_anchors.jsonl"
    rows = [dict(prompt=f"anchor prompt {i}", response=f"anchor teacher {i}",
                 _rejected=f"anchor student {i}", _traj=f"anchor_{i}#event") for i in range(2)]
    path.write_text("\n\n".join(json.dumps(row) for row in rows) + "\n")
    refs = [dict(pool_row=i, event_key=event_key(row), prompt_sha1=driver._sha1(row["prompt"]),
                 yT_sha1=driver._sha1(row["response"]), yS_sha1=driver._sha1(row["_rejected"]))
            for i, row in enumerate(rows)]
    unit = dict(source_id="anchor", role="anchor", unit_type="pack", parent_task_id="anchor",
                group="anchor", pool_path=path.name, rows=refs)
    sources = json.loads((tmp_path / "sources.json").read_text())
    sources["units"].append(unit)
    (tmp_path / "sources.json").write_text(json.dumps(sources))
    return path, rows


@pytest.mark.parametrize("location", ["row", "unit", "manifest", "repository"])
def test_multi_pool_resolution_precedence_and_content_identity(tmp_path, monkeypatch, location):
    config, _, _, data_file = fixture_files(tmp_path)
    path, rows = add_anchor_pool(tmp_path)
    source_file = tmp_path / "sources.json"
    sources = json.loads(source_file.read_text())
    anchor = sources["units"][-1]
    descriptor = json.loads(data_file.read_text())
    if location == "row":
        anchor["pool_path"] = "missing-unit-pool.jsonl"
        for ref in anchor["rows"]:
            ref["pool_path"] = path.name
        # One pack may mix pools, with each index local to its own file.
        anchor["rows"].append({**sources["units"][0]["rows"][0], "pool_path": "pool.jsonl"})
    elif location == "manifest":
        sources["pool_path"] = path.name
        descriptor.pop("pool")
        anchor.pop("pool_path")
        for unit in sources["units"][:-1]:
            unit["pool_path"] = "pool.jsonl"
    elif location == "repository":
        root = tmp_path / "repo"
        path = root / "data/behavior_atom_v1/pool_anchors.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("\n".join(json.dumps(row) for row in rows))
        monkeypatch.setattr(driver, "ROOT", root)
        anchor["pool_path"] = "data/behavior_atom_v1/pool_anchors.jsonl"
    source_file.write_text(json.dumps(sources))
    prepared = driver.load_inputs(config, descriptor, data_file)
    resolved = prepared["sources"][-1]["resolved_rows"]
    assert [r["response"] for r in resolved[:2]] == [r["response"] for r in rows]
    assert [r["prompt"] for r in resolved[:2]] == [driver._thinking_off(r["prompt"]) for r in rows]
    if location == "row":
        assert resolved[2]["response"] == "teacher"
    assert content_hash(path.read_bytes()) in prepared["input_hashes"]["pools"]
    # Changes outside referenced rows must still invalidate the pool identity.
    path.write_text(path.read_text() + "\n" + json.dumps({"unreferenced": True}) + "\n")
    changed = driver.load_inputs(config, descriptor, data_file)
    assert changed["input_hashes"] != prepared["input_hashes"]


@pytest.mark.parametrize("field", ["prompt", "response", "_rejected"])
def test_multi_pool_sha1_checks(tmp_path, field):
    config, _, _, data_file = fixture_files(tmp_path)
    path, rows = add_anchor_pool(tmp_path)
    rows[0][field] += " corruption"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="content hash mismatch"):
        driver.load_inputs(config, json.loads(data_file.read_text()), data_file)


def test_roles_defaults_explicit_anchor_cli_and_budgets(tmp_path, monkeypatch, capsys):
    config, protocol, _, data_file = fixture_files(tmp_path)
    add_anchor_pool(tmp_path)
    data = driver.load_inputs(config, json.loads(data_file.read_text()), data_file)
    def selected(stage, roles=None):
        return [u["source_id"] for u in driver.select_sources(data, stage, roles=roles)[0]]
    assert selected("pilot") == ["pilot", "zero"]
    assert selected("collect") == ["formal_a", "formal_b"]
    assert selected("pilot", "pilot+noop+anchor") == ["pilot", "zero", "anchor"]
    assert selected("collect", ["formal", "anchor"]) == ["formal_a", "formal_b", "anchor"]
    assert selected("collect", "anchor") == ["anchor"]
    with pytest.raises(ValueError, match="roles"):
        selected("collect", "typo")
    for role, limit in (("formal", 24), ("pilot", 4), ("noop", 2)):
        units = [{"source_id": str(i), "role": role} for i in range(limit)]
        units += [{"source_id": "anchor", "role": "anchor"}]
        stage = "pilot" if role == "pilot" else "collect"
        assert len(driver.select_sources({"sources": units}, stage, roles=[role, "anchor"])[0]) == limit + 1
        units.append({"source_id": "over_budget", "role": role})
        with pytest.raises(ValueError, match="P2 source budget"):
            driver.select_sources({"sources": units}, stage, roles=[role, "anchor"])
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    monkeypatch.setattr(driver, "load_student", lambda _: pytest.fail("dry-run built student"))
    assert driver.main(["pilot", "--config", str(config_file), "--roles", "pilot", "noop", "anchor", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["sources"] == 3
    # Execute explicitly selected anchors through training and collection.
    monkeypatch.setattr(driver, "load_student", lambda _: FakeStudent())
    out = tmp_path / "roles_pilot"
    driver.run("pilot", config, protocol, data, out, student_factory=lambda _: FakeStudent())
    config["pilot_pass"] = str(out / "pilot_pass.json")
    config_file.write_text(json.dumps(config))
    collected = tmp_path / "roles_collect"
    assert driver.main(["collect", "--config", str(config_file), "--output-dir", str(collected),
                        "--roles", "formal,anchor"]) == 0
    result = json.loads((collected / "result.json").read_text())
    assert [u["source_id"] for u in result["updates"]] == ["formal_a", "formal_b", "anchor"]


@pytest.mark.parametrize("cap", [8, 3])
def test_thread_cap_precedes_model_load(tmp_path, monkeypatch, cap):
    config, protocol, data, _ = fixture_files(tmp_path)
    threads, calls = [640], []
    monkeypatch.setattr(torch, "get_num_threads", lambda: threads[0])
    def set_threads(count):
        calls.append(count)
        threads[0] = count
    monkeypatch.setattr(torch, "set_num_threads", set_threads)
    def factory(_):
        assert threads[0] == cap
        return FakeStudent()
    driver.run("pilot", config, protocol, data, tmp_path / "capped", cpu_threads=cap, student_factory=factory)
    assert calls == [cap]
    assert micro.cap_cpu_threads(cap + 1) == cap  # do not expand a smaller pool
    with pytest.raises(ValueError, match="positive integer"):
        micro.cap_cpu_threads(0)


@pytest.mark.parametrize("bad", ["parameter", "input_ids", "labels", "cuda_index"])
def test_update_device_assertion_synthetic(monkeypatch, bad):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    # Meta and device-only stand-ins exercise CUDA placement without a GPU.
    if bad in {"parameter", "cuda_index"}:
        model = SimpleNamespace(named_parameters=lambda: iter([
            ("adapter", SimpleNamespace(requires_grad=True,
                                         device=torch.device("cpu" if bad == "parameter" else "cuda:1")))]))
        batch = []
        expected = "cuda"
    else:
        model = TinyModel()
        batch = {"input_ids": torch.ones(1, 2, dtype=torch.long), "labels": torch.ones(1, 2, dtype=torch.long)}
        batch[bad] = torch.empty(1, 2, device="meta", dtype=torch.long)
        expected = "cpu"
    with pytest.raises(ValueError, match="micro_update device assertion"):
        micro.assert_update_device(model, batch, expected)
    micro.assert_update_device(TinyModel(), {"pairs": [(torch.ones(1), torch.ones(1))]}, "cpu")


def test_bad_batch_fails_before_forward_or_optimizer(tmp_path, monkeypatch):
    model = TinyModel()
    path = tmp_path / "init.pt"
    micro.save_initial_state(model, path)
    def bad_encode(*args):
        return (torch.empty(1, 2, device="meta", dtype=torch.long), torch.ones(1, 2, dtype=torch.long)), {}
    monkeypatch.setattr(micro, "_encode_row", bad_encode)
    monkeypatch.setattr(model, "forward", lambda *a, **k: pytest.fail("bad batch reached model"))
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **k: pytest.fail("bad batch reached optimizer"))
    cfg = micro.MicroUpdateConfig(model_id="tiny", init_state_path=path, source_id="bad", tokenizer=Tokenizer())
    with pytest.raises(ValueError, match="device assertion.*batch"):
        micro.run_micro_update(lambda: model, [{"prompt": "p", "response": "teacher", "_rejected": "student"}], cfg)


def test_progress_flushes_all_phases_and_profiles_manifest(tmp_path, monkeypatch, capsys):
    import builtins
    import re
    printed = []
    real_print = builtins.print
    def print_spy(*args, **kwargs):
        if args and str(args[0]).startswith("["):
            printed.append(kwargs)
        return real_print(*args, **kwargs)
    monkeypatch.setattr(builtins, "print", print_spy)
    config, protocol, data, _ = fixture_files(tmp_path)
    out = tmp_path / "profile"
    driver.run("pilot", config, protocol, data, out, profile=True, student_factory=lambda _: FakeStudent())
    log = capsys.readouterr().err
    phases = ["model_load", "initial_state_load_verify", "resident_state_cache", "base/evaluation",
              "base/probe_batch/1", "base/probe_batch/2", "base/likelihood_batch/1", "repeat/check", "zero/check",
              "repeat/kl", "zero/kl", "source/pilot/state_restore", "source/pilot/encode",
              "source/pilot/reference_forward", "source/pilot/reference_restore", "source/pilot/optimizer_step/1",
              "source/pilot/optimizer_step/2", "source/pilot/optimizer_step/3", "source/pilot/delta_capture",
              "source/pilot/delta_save", "source/pilot/probe_evaluation", "source/pilot/probe_batch/1",
              "source/pilot/likelihood_batch/1", "source/pilot/policy_kl", "source/pilot/delta_replay_check",
              "source/pilot/final_restore", "source/zero/state_restore", "source/zero/encode", "source/zero/delta_capture"]
    timings = json.loads((out / "manifest.json").read_text())["phase_elapsed_seconds"]
    for phase in phases:
        assert f"] {phase} start" in log
        assert f"] {phase} done seconds=" in log
        assert timings[phase] >= 0
    assert re.search(r"\[\d{4}-\d\d-\d\dT.*\+00:00\] source/pilot/optimizer_step/1 done seconds=\d+\.\d+ loss=", log)
    assert re.search(r"source/pilot/delta_capture done seconds=.* bytes=\d+", log)
    assert re.search(r"source/pilot/delta_save done seconds=.* bytes=\d+", log)
    assert printed and all(k.get("flush") is True for k in printed)
    driver.run("pilot", config, protocol, data, out, profile=True, resume=True,
               student_factory=lambda _: pytest.fail("profile resume loaded model"))
    assert json.loads((out / "manifest.json").read_text())["phase_elapsed_seconds"] == timings


def test_progress_synchronizes_cuda_and_registers_signal(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))
    progress = micro.Progress("cuda:2", profile=True)
    with progress.phase("synthetic"):
        pass
    assert calls == [torch.device("cuda:2")]
    registered = []
    monkeypatch.setattr(micro.faulthandler, "register", lambda *a, **k: registered.append((a, k)))
    micro.install_stack_dump()
    assert registered[0][0] == (micro.signal.SIGUSR1,)
    assert registered[0][1]["all_threads"] is True
    assert registered[0][1]["file"] in (sys.stderr, sys.__stderr__)


def test_resident_sources_never_reload_move_or_rehash_model(tmp_path, monkeypatch):
    config, protocol, data, _ = fixture_files(tmp_path)
    original = micro.run_micro_update
    models = []
    def forbidden(*args, **kwargs):
        pytest.fail("per-source full model/state reload, move or hash")
    def runner(factory, rows, cfg):
        model = factory()
        models.append(model)
        with monkeypatch.context() as patch:
            patch.setattr(torch, "load", forbidden)
            patch.setattr(model, "to", forbidden)
            patch.setattr(micro, "_frozen_hash", forbidden)
            patch.setattr(micro, "tensor_state_hash", forbidden)
            patch.setattr(micro, "file_hash", forbidden)
            return original(factory, rows, cfg)
    monkeypatch.setattr(micro, "run_micro_update", runner)
    driver.run("pilot", config, protocol, data, tmp_path / "resident", student_factory=lambda _: FakeStudent())
    assert len(models) == 2 and models[0] is models[1]


def test_resident_cache_rejects_changed_frozen_weights(tmp_path):
    model = TinyModel()
    path = tmp_path / "init.pt"
    micro.save_initial_state(model, path)
    resident = micro.ResidentState(model, torch.load(path, weights_only=True), path, {})
    with torch.no_grad():
        model.frozen.add_(1)
    with pytest.raises(ValueError, match="resident frozen checkpoint changed"):
        resident.restore()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_device_delta_capture_one_packed_copy_and_exact_replay(tmp_path, monkeypatch, dtype):
    from src.bfas.behavior import deltas
    model = torch.nn.Linear(8, 2).to(dtype)
    with torch.no_grad():
        model.weight.fill_(0.1)
    initial = copy.deepcopy(model)
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        model.weight.fill_(-0.2)  # fp32 needs the roundoff coordinates
        model.bias.add_(0.001)
    calls = []
    cpu = torch.Tensor.cpu
    def cpu_spy(tensor, *a, **k):
        calls.append((tensor.ndim, tensor.dtype))
        return cpu(tensor, *a, **k)
    # Exercise the device implementation with CPU tensors so CI needs no CUDA.
    base_hash = tensor_state_hash(base)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", cpu_spy)
        delta = deltas._capture_device_delta(model, base, deltas.parameter_layout(model), base_hash, None, None)
    assert calls == [(1, torch.float32)]
    expected = deltas.capture_delta(model, base)
    np.testing.assert_array_equal(delta.flat(), expected.flat())
    assert delta.manifest["update_norm"] == pytest.approx(expected.manifest["update_norm"])
    monkeypatch.setattr(deltas, "capture_delta", lambda *a, **k: pytest.fail("save recaptured delta"))
    path = tmp_path / "delta.npz"
    deltas.save_delta(None, None, path, delta=delta)
    loaded = load_delta(path)
    assert driver._verify_delta_replay(model, base, loaded)
    apply_delta(initial, loaded)
    for a, b in zip(initial.parameters(), model.parameters()):
        assert torch.equal(a, b)


@pytest.mark.parametrize("entrypoint", [cli.main, driver.main])
def test_smoke_and_max_sources_flags(tmp_path, monkeypatch, capsys, entrypoint):
    config, _, _, _ = fixture_files(tmp_path)
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    monkeypatch.setattr(driver, "load_student", lambda _: FakeStudent())
    out = tmp_path / "smoke"
    assert entrypoint(["pilot", "--config", str(config_file), "--max-sources", "1", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["sources"] == 1
    assert entrypoint(["pilot", "--config", str(config_file), "--max-sources", "0", "--dry-run"]) == 2
    capsys.readouterr()
    assert entrypoint(["pilot", "--config", str(config_file), "--output-dir", str(out), "--smoke", "--cpu-threads", "2"]) == 0
    result = json.loads((out / "result.json").read_text())
    assert len(result["updates"]) == 1 and result["updates"][0]["steps"] == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["protocol"]["micro_update"]["steps"] == 1
    assert manifest["protocol"]["micro_update"]["cpu_threads"] == 2
    assert manifest["resource_plan"]["probes"] == 1
    assert manifest["protocol"]["evaluation"]["max_new_tokens"] <= 32
    assert manifest["phase_elapsed_seconds"]["source/pilot/optimizer_step/1"] >= 0
    assert "source/pilot/optimizer_step/1 done seconds=" in capsys.readouterr().err
    assert not json.loads((out / "pilot_pass.json").read_text())["passed"]


def test_failed_profile_records_phase_and_resumes(tmp_path, monkeypatch):
    config, protocol, data, _ = fixture_files(tmp_path)
    out = tmp_path / "failed_profile"
    with monkeypatch.context() as patch:
        def fail_step(*args, **kwargs):
            raise RuntimeError("step stalled")
        patch.setattr(torch.optim.AdamW, "step", fail_step)
        with pytest.raises(RuntimeError, match="step stalled"):
            driver.run("pilot", config, protocol, data, out, profile=True, student_factory=lambda _: FakeStudent())
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["phase_elapsed_seconds"]["source/pilot/optimizer_step/1"] >= 0
    assert not (out / "complete.json").exists()
    driver.run("pilot", config, protocol, data, out, profile=True, resume=True, student_factory=lambda _: FakeStudent())
    assert (out / "complete.json").exists()
