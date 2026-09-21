"""CPU-only shard equivalence, concurrent publication, accounting and identity."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from bfas.rtd.benchmarks import alfworld_evaluation as evaluation
from bfas.rtd.benchmarks import alfworld_identity as identity
from bfas.rtd.persistence import ComputeJournal, atomic_json, digest, file_hash, tree_hash
from rtd_alfworld_evaluation_fixtures import ROOT, campaign, put, run
from tools import alf_eval_shard as shards
from tools import rtd_alfworld_evaluate as serial_cli


# Deliberately re-frozen for the launch-bound vLLM backend. Previous projection:
# 8eb084055b8caa8f4b970004b8c2f2521ab0092b956d5b1ebe90beb59ba8c69e
# Covers every projected symbol/import/dispatch and the identity module bytes
# (including SCOPES), independently of git availability or fixture identities.
FROZEN_SCORING_FILES_HASH = "a475f62fd4014f1ec43d1e1ea0b885283a43a2038b396ab9aec7964557f010e9"


def test_frozen_harness_projection_unchanged():
    files = {name: digest(identity.scoring_projection(name, (ROOT / name).read_text()))
             for name in identity.SCOPES}
    name = "src/bfas/rtd/benchmarks/alfworld_identity.py"
    files[name] = file_hash(ROOT / name)
    assert digest(files) == FROZEN_SCORING_FILES_HASH


def shard_run(c, shard=0, of=2, **kwargs):
    return shards.evaluate_shard(c.root, c.manifest,
        **dict(c.kwargs, shard=shard, of=of, device="cpu", **kwargs))


def task_bytes(directory):
    return {p.name: p.read_bytes() for p in (directory / "artifacts/tasks").glob("*.json")}


def forbidden(*args, **kwargs):
    pytest.fail("completed records must not load a backend or environment")


def finalise(c, monkeypatch):
    binding = c.root / "binding.json"
    put(binding, c.manifest)
    monkeypatch.setattr(evaluation, "HFBackend", forbidden)
    monkeypatch.setattr(evaluation, "EvaluationEnvBridge", forbidden)
    assert serial_cli.main(["--root", str(c.root), "run", "--binding", str(binding),
        "--output-root", str(c.output), "--tag", c.tag]) == 0
    return json.loads((c.directory / "campaign.json").read_text())


def test_two_shards_match_serial_bytes_and_normal_cli_finalisation(campaign, monkeypatch):
    c = campaign
    serial = run(c, tag="serial")
    serial_dir = c.output / "serial"
    ids = c.manifest["evaluation_harness"]["expected"]["task_ids"]
    c.backends.clear()
    c.envs.clear()
    for index in range(2):
        result = shard_run(c, shard=index)
        assert result["generated"] == ids[index::2]
        assert result["skipped"] == []
        assert not (c.directory / "campaign.json").exists()
        assert not (c.directory / "artifacts/aggregate.json").exists()
    assert [env.tid for env in c.envs] == ids[::2] + ids[1::2]
    assert len(c.backends) == 2 and all(b.closed for b in c.backends)
    assert all(e.closed for e in c.envs)
    assert task_bytes(c.directory) == task_bytes(serial_dir)
    assert (c.directory / "artifacts/binding.json").read_bytes() == (serial_dir / "artifacts/binding.json").read_bytes()
    audit_before = (c.directory / "audit.jsonl").read_bytes()
    events = ComputeJournal(c.directory / "audit.jsonl").events
    assert [e["shard"] for e in events if e["kind"] == "compute_begin"] == [0, 1]
    ends = [e for e in events if e["kind"] == "compute_end"]
    assert len(ends) == 2 and all(e["status"] == "complete" and e["wall_seconds"] > 0
                                  and e["gpu_seconds"] == 0 for e in ends)
    assert finalise(c, monkeypatch) == serial
    assert (c.directory / "campaign.json").read_bytes() == (serial_dir / "campaign.json").read_bytes()
    assert (c.directory / "audit.jsonl").read_bytes() == audit_before
    assert shard_run(c, backend_factory=forbidden, env_factory=forbidden)["skipped"] == ids[::2]


# Separate interpreters exercise flock, hash-chain refresh, and actual overlap.
# Each child uses the same CPU factories as the serial test suite.
WORKER = r"""
import json, sys, time
from pathlib import Path
import torch
from rtd_alfworld_evaluation_fixtures import FakeBackend, FakeEnv
from tools.alf_eval_shard import evaluate_shard

def forbidden(*args, **kwargs):
    raise AssertionError('shard fixture queried CUDA')
for name in ('is_available', 'device_count', 'get_device_properties', 'init', 'empty_cache', '_lazy_init'):
    setattr(torch.cuda, name, forbidden)
root, output, binding, index, shard, of, overlap = sys.argv[1:]
root = Path(root)
manifest = json.loads(Path(binding).read_text())
calls = []
def env_factory(tid):
    calls.append(tid)
    if len(calls) == 1:
        (root / ('ready-' + index)).touch()
        if overlap == 'yes':
            deadline = time.monotonic() + 20
            while not all((root / ('ready-' + str(i))).exists() for i in range(2)):
                if time.monotonic() > deadline:
                    raise AssertionError('shards did not generate concurrently')
                time.sleep(.02)
        else:
            time.sleep(.2)
    return FakeEnv(tid, won=int(tid.rsplit('_', 1)[1]) < 109)
result = evaluate_shard(root, manifest, output_root=output, tag='round1',
    shard=int(shard), of=int(of), device='cpu', backend_factory=lambda m: FakeBackend(),
    env_factory=env_factory, lock_timeout=30)
(root / ('result-' + index + '.json')).write_text(json.dumps(dict(result=result, calls=calls)))
"""


@pytest.mark.parametrize("duplicate", [False, True])
def test_processes_overlap_or_skip_duplicate_task_and_keep_journal(campaign, monkeypatch, duplicate):
    c = campaign
    binding = c.root / "binding.json"
    put(binding, c.manifest)
    worker_cwd = c.root / "worker-cwd"
    worker_cwd.mkdir()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT), str(ROOT / "tests"))))
    processes = []
    try:
        for index in range(2):
            processes.append(subprocess.Popen([sys.executable, "-B", "-c", WORKER,
                str(c.root), str(c.output), str(binding), str(index),
                "0" if duplicate else str(index), "140" if duplicate else "2",
                "no" if duplicate else "yes"], cwd=worker_cwd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
        for process in processes:
            out, err = process.communicate(timeout=45)
            assert process.returncode == 0, out + err
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate()
    results = [json.loads((c.root / f"result-{i}.json").read_text()) for i in range(2)]
    ids = c.manifest["evaluation_harness"]["expected"]["task_ids"]
    called = [tid for result in results for tid in result["calls"]]
    assert sorted(called) == (ids[:1] if duplicate else ids)
    if duplicate:
        assert sorted(len(r["result"]["generated"]) for r in results) == [0, 1]
        assert sorted(len(r["result"]["skipped"]) for r in results) == [0, 1]
    else:
        assert [r["calls"] for r in results] == [ids[::2], ids[1::2]]
    expected = c.manifest["evaluation_harness"]["expected"]
    assert len(evaluation._records(c.directory / "artifacts", expected,
        identity.campaign_identity(c.manifest), complete=not duplicate)) == len(called)
    events = ComputeJournal(c.directory / "audit.jsonl").events
    assert len(events) == 4
    assert len([e for e in events if e["kind"] == "compute_end"]) == 2
    if not duplicate:
        assert [e["kind"] for e in events[:2]] == ["compute_begin", "compute_begin"]
        serial = run(c, tag="serial")
        before = (c.directory / "audit.jsonl").read_bytes()
        assert finalise(c, monkeypatch) == serial
        assert task_bytes(c.directory) == task_bytes(c.output / "serial")
        assert (c.directory / "audit.jsonl").read_bytes() == before


def test_failed_shard_accounts_work_and_resumes_completed_records(campaign):
    c = campaign
    ids = c.manifest["evaluation_harness"]["expected"]["task_ids"][::2]
    def fail_second(tid):
        if tid == ids[1]:
            raise evaluation.EnvironmentUnavailable("fixture interruption")
        return c.env_factory(tid)
    with pytest.raises(evaluation.EnvironmentUnavailable):
        shard_run(c, env_factory=fail_second)
    assert c.backends[-1].closed and all(e.closed for e in c.envs)
    saved = task_bytes(c.directory)
    assert len(saved) == 1
    events = ComputeJournal(c.directory / "audit.jsonl").events
    assert events[-1]["kind"] == "compute_end" and events[-1]["status"] == "failed"
    result = shard_run(c)
    assert result["generated"] == ids[1:] and result["skipped"] == ids[:1]
    assert all(task_bytes(c.directory)[name] == value for name, value in saved.items())
    assert len(c.envs) == 70


def test_default_device_dispatch_and_gpu_seconds_with_cpu_stubs(campaign, monkeypatch):
    c = campaign
    import torch
    # Exercise the production branch without querying or allocating real CUDA.
    monkeypatch.setattr(shards, "hardware_identity", lambda: c.hardware)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args: 24)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *args: 32)
    def backend(manifest, *, device):
        assert manifest == c.manifest and device == "cuda:0"
        return c.backend_factory(manifest)
    def environment(tid, *, data_root, environment_root):
        assert data_root == c.manifest["paths"]["data_root"]
        assert environment_root == c.manifest["paths"]["environment_root"]
        return c.env_factory(tid)
    monkeypatch.setattr(evaluation, "HFBackend", backend)
    monkeypatch.setattr(evaluation, "EvaluationEnvBridge", environment)
    for index in range(2):
        shards.evaluate_shard(c.root, c.manifest, output_root=c.output, tag=c.tag,
                             shard=index, of=2, device="cuda:0", lock_timeout=0)
    events = ComputeJournal(c.directory / "audit.jsonl").events
    for end in (e for e in events if e["kind"] == "compute_end"):
        begin = events[end["begin_sequence"]]
        assert begin["shard"] in (0, 1) and begin["of"] == 2
        assert end["gpu_seconds"] == end["wall_seconds"] > 0
        assert end["peak_allocated_bytes"] == 24 and end["peak_reserved_bytes"] == 32
    before = (c.directory / "audit.jsonl").read_bytes()
    assert finalise(c, monkeypatch)["aggregate"]["tasks"] == 140
    assert (c.directory / "audit.jsonl").read_bytes() == before


@pytest.mark.parametrize("shard,of", [(-1, 2), (2, 2), (0, 0), (0, -1), (True, 2), (0, 1.5)])
def test_invalid_partition_fails_before_writes(campaign, shard, of):
    with pytest.raises(ValueError, match="zero-based"):
        shard_run(campaign, shard=shard, of=of)
    assert not campaign.output.exists() and not campaign.backends


@pytest.mark.parametrize("fault", ["binding", "record", "unbound", "historical", "input"])
def test_rejects_wrong_binding_corruption_and_unsafe_output(campaign, fault):
    c = campaign
    if fault in ("binding", "record"):
        shard_run(c, of=140)
        path = (c.directory / "artifacts/binding.json" if fault == "binding" else
                next((c.directory / "artifacts/tasks").glob("*.json")))
        value = json.loads(path.read_text())
        if fault == "binding":
            value["identity"]["max_steps"] = 39
        else:
            value["record_hash"] = "wrong"
        atomic_json(path, value)
    elif fault == "unbound":
        put(c.directory / "artifacts/stray.json", {})
    elif fault == "historical":
        put(c.directory / "metrics.json", {})
    output = c.model if fault == "input" else c.output
    before = tree_hash(output)
    with pytest.raises(ValueError):
        shard_run(c, output_root=output, backend_factory=forbidden, env_factory=forbidden)
    assert tree_hash(output) == before


def test_publication_failure_leaves_no_partial_task_and_accounts_failure(campaign, monkeypatch):
    c = campaign
    def failed_link(*args):
        raise OSError("fixture publish failure")
    monkeypatch.setattr(shards.os, "link", failed_link)
    with pytest.raises(OSError, match="publish failure"):
        shard_run(c)
    assert task_bytes(c.directory) == {}
    assert list((c.directory / "artifacts/tasks").iterdir()) == []
    assert not list(c.output.glob(".alfworld-shard-*"))
    assert ComputeJournal(c.directory / "audit.jsonl").events[-1]["status"] == "failed"
    assert c.backends[-1].closed


def test_shard_cli_dispatches_binding_partition_and_device(campaign, monkeypatch, capsys):
    c = campaign
    binding = c.root / "binding.json"
    put(binding, c.manifest)
    original = shards.evaluate_shard
    def injected(root, manifest, **kwargs):
        assert manifest == c.manifest and kwargs["device"] == "cpu"
        return original(root, manifest, backend_factory=c.backend_factory,
                        env_factory=c.env_factory, **kwargs)
    monkeypatch.setattr(shards, "evaluate_shard", injected)
    assert shards.main(["--root", str(c.root), "--binding", str(binding),
        "--output-root", str(c.output), "--tag", c.tag, "--shard", "1", "--of", "2",
        "--device", "cpu"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["generated"] == c.manifest["evaluation_harness"]["expected"]["task_ids"][1::2]
