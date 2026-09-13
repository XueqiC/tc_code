"""CPU-only scratch lifecycle and durable evaluation receipt regressions."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import json
import os
from pathlib import Path
import signal
import socket
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]
from bfas.rtd.baselines import paper_evaluation, paper_scratch
from bfas.rtd.persistence import atomic_json, tree_hash


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in ("SLURM_TMPDIR", "TMPDIR", "SLURM_JOB_ID", "SLURM_JOBID"):
        monkeypatch.delenv(name, raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail("scratch tests must not initialize CUDA or use the network")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


@pytest.mark.parametrize("preferred", ["SLURM_TMPDIR", "TMPDIR", None])
def test_scratch_prefers_first_existing_writable_directory(tmp_path, monkeypatch, preferred):
    for name in ("SLURM_TMPDIR", "TMPDIR"):
        base = tmp_path/name
        base.mkdir()
        if preferred == "SLURM_TMPDIR" or name == preferred:
            monkeypatch.setenv(name, str(base))
    expected = tmp_path/preferred if preferred else Path("/tmp")
    with paper_scratch.export_directory(tmp_path/"run") as export:
        assert export.parent == expected
        assert f"pid-{os.getpid()}-" in export.name
        (export/"weights").write_bytes(b"local weights")
    assert not export.exists()


@pytest.mark.parametrize("failure", ["absent", "file", "unwritable", "mkdir"])
def test_scratch_skips_unusable_preferred_directory(tmp_path, monkeypatch, failure):
    slurm, other = tmp_path/"slurm", tmp_path/"other"
    other.mkdir()
    if failure == "file":
        slurm.write_text("not a directory")
    elif failure != "absent":
        slurm.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(slurm))
    monkeypatch.setenv("TMPDIR", str(other))
    access = os.access
    if failure == "unwritable":
        monkeypatch.setattr(paper_scratch.os, "access",
            lambda path, mode: False if path == slurm else access(path, mode))
    if failure == "mkdir":
        create = paper_scratch.tempfile.TemporaryDirectory
        def temporary(**kwargs):
            if kwargs["dir"] == slurm:
                raise PermissionError("read-only mount despite access check")
            return create(**kwargs)
        monkeypatch.setattr(paper_scratch.tempfile, "TemporaryDirectory", temporary)
    with paper_scratch.export_directory(tmp_path/"run") as export:
        assert export.parent == other
    assert not export.exists()


@pytest.mark.parametrize("failure", ["absent", "unwritable", "mkdir"])
def test_no_usable_scratch_preserves_shared_export_fallback(tmp_path, monkeypatch, failure):
    candidates = {tmp_path/"slurm", tmp_path/"other", Path("/tmp")}
    for name, path in zip(("SLURM_TMPDIR", "TMPDIR"), (tmp_path/"slurm", tmp_path/"other")):
        path.mkdir()
        monkeypatch.setenv(name, str(path))
    if failure == "absent":
        is_dir = Path.is_dir
        monkeypatch.setattr(Path, "is_dir", lambda path: False if path in candidates else is_dir(path))
    elif failure == "unwritable":
        access = os.access
        monkeypatch.setattr(paper_scratch.os, "access",
            lambda path, mode: False if path in candidates else access(path, mode))
    else:
        def denied(**kwargs):
            raise PermissionError("cannot create scratch")
        monkeypatch.setattr(paper_scratch.tempfile, "TemporaryDirectory", denied)
    with paper_scratch.export_directory(tmp_path/"run") as export:
        assert export == tmp_path/"run/export"
        export.mkdir(parents=True)
        (export/"weights").write_bytes(b"shared fallback")
    assert (export/"weights").read_bytes() == b"shared fallback"


def test_concurrent_jobs_and_cells_have_isolated_scratch(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_TMPDIR", str(tmp_path))
    with ExitStack() as contexts:
        exports = []
        for job in ("123", "456", "456"):
            monkeypatch.setenv("SLURM_JOB_ID", job)
            export = contexts.enter_context(paper_scratch.export_directory(tmp_path/"run"))
            assert export.name.startswith(f"baseline-export-{job}-")
            (export/"weights").write_text(str(len(exports)))
            exports.append(export)
        assert len(set(exports)) == 3
        assert [(path/"weights").read_text() for path in exports] == ["0", "1", "2"]
        with paper_scratch.export_directory(tmp_path/"run") as fourth:
            assert fourth not in exports
        assert not fourth.exists() and all(path.exists() for path in exports)
    assert all(not path.exists() for path in exports)


@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("failure", [None, "flatten", "merge", "digest", "official", "kang_sag", "metrics", "term"])
def test_evaluation_scratch_lifetime_and_archived_artifacts(tmp_path, monkeypatch, local, failure):
    from bfas.rtd import evaluation, hardware
    from tools.baseline_run import terminate_worker

    directory, scratch, model = tmp_path/"run", tmp_path/"scratch", tmp_path/"base"
    checkpoint = directory/"checkpoint/lora"
    checkpoint.mkdir(parents=True)
    scratch.mkdir()
    model.mkdir()
    (checkpoint/"adapter_config.json").write_text('{"r": 16}')
    (checkpoint/"adapter_model.safetensors").write_bytes(b"unchanged checkpoint")
    (model/"config.json").write_text("{}")
    monkeypatch.setenv("SLURM_TMPDIR", str(scratch))
    monkeypatch.setenv("SLURM_JOB_ID", "789")
    if not local:
        monkeypatch.setattr(paper_scratch.os, "access", lambda *args: False)
    manifest = dict(config={}, benchmark="bfcl", method="kang", smoke=False,
        checkpoint_sha256=tree_hash(checkpoint.parent), base_checkpoint_hash=tree_hash(model),
        model_path=str(model), hardware={"hard": "cpu-stub"}, teacher_tokens_charged=33, B=40, port=8930)
    atomic_json(directory/"manifest.json", manifest)
    atomic_json(directory/"purchased_rows.json", [dict(package_id="original", tokens=33)])
    atomic_json(directory/"exposure_schedule.json", dict(batches=[[0, 0]]))
    (directory/"train.log").write_text("original training log\n")
    preserved = {p: p.read_bytes() for p in directory.rglob("*") if p.is_file() and p.name != "manifest.json"}
    monkeypatch.setattr(hardware, "hardware_identity", lambda: manifest["hardware"])

    # A path-independent copy of exactly the files hashed by the old export.
    payload = {"config.json": b"{}", "model.safetensors": b"\x80?\x00@bfloat16 weights",
               "tokenizer.json": b'{"vocab": {}}', "templates/chat.jinja": b"{{ messages }}"}
    reference = tmp_path/"legacy_export"
    for name, content in payload.items():
        path = reference/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    export_hash = tree_hash(reference)
    paths, campaigns = [], []

    def fail(where):
        if failure == where:
            raise RuntimeError(f"injected {where} failure")

    def flatten(given, source, out):
        assert source == directory/"checkpoint" and given == manifest
        if local:
            assert out.parent.parent == scratch and out.parent.name.startswith("baseline-export-789-")
        else:
            assert out == directory/"export/adapter"
        paths.append(out.parent)
        out.mkdir(parents=True)
        (out/"model.safetensors").write_bytes(b"flattened weights")
        print("flatten log")
        fail("flatten")

    def merge(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "" and "--verify" in command
        assert Path(command[command.index("--adapter")+1]) == paths[0]/"adapter"
        out = Path(command[command.index("--out")+1])
        assert out == paths[0]/"hub_merged"
        for name, content in payload.items():
            path = out/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        print("merge log")
        fail("merge")
        if failure == "term":
            terminate_worker(signal.SIGTERM, None)

    def hash_policy(path):
        if Path(path).name == "hub_merged":
            fail("digest")
        return tree_hash(path)

    def campaign(root, policy, out, **kwargs):
        assert policy == paths[0]/"hub_merged" and tree_hash(policy) == export_hash
        assert (paths[0]/"adapter/model.safetensors").exists()
        assert out.parent == directory and out.name == ("kang_sag" if kwargs["kang"] else "official")
        campaigns.append(out.name)
        out.mkdir()
        (out/"resultdir").mkdir()
        (out/"resultdir/results.json").write_text('{"official": true}')
        (out/"scoredir").mkdir()
        (out/"scoredir/data_overall.csv").write_text("Overall Acc\n25\n")
        for log in ("vllm.log", "generate.log", "evaluate.log"):
            (out/log).write_text("campaign log\n")
        fail(out.name)
        return dict(tasks=5217, overall_accuracy_percent=25.)

    def write_json(path, value):
        if path == directory/"metrics.json":
            fail("metrics")
        atomic_json(path, value)

    monkeypatch.setattr(evaluation, "_flatten_adapter", flatten)
    monkeypatch.setattr(paper_evaluation.subprocess, "run", merge)
    monkeypatch.setattr(paper_evaluation, "tree_hash", hash_policy)
    monkeypatch.setattr(paper_evaluation, "run_bfcl", campaign)
    monkeypatch.setattr(paper_evaluation, "atomic_json", write_json)
    with (directory/"evaluate.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        if failure:
            with pytest.raises(SystemExit if failure == "term" else RuntimeError):
                paper_evaluation.evaluate_run(ROOT, directory, manifest)
        else:
            result = paper_evaluation.evaluate_run(ROOT, directory, manifest)
    assert len(paths) == 1 and paths[0].exists() is (not local)
    assert list(scratch.iterdir()) == []
    assert all(path.read_bytes() == original for path, original in preserved.items())
    assert json.loads((directory/"manifest.json").read_text()) == manifest
    assert "flatten log" in (directory/"evaluate.log").read_text()
    if local:
        assert not (directory/"export").exists()
    if failure is None:
        assert campaigns == ["official", "kang_sag"]
        assert result["checkpoint_sha256"] == manifest["checkpoint_sha256"]
        assert result["export_sha256"] == result["kang_self_consistency"]["export_sha256"] == export_hash
        assert result["teacher_tokens_charged"] == 33 and result["B"] == 40
        assert result["overall_accuracy_percent"] == 25. and result["tasks"] == 5217
        for name in ("metrics.json", "official_metrics.json", "kang_metrics.json"):
            assert json.loads((directory/name).read_text())["export_sha256"] == export_hash
        assert json.loads((directory/"metrics.json").read_text()) == result
        assert (directory/"official/scoredir/data_overall.csv").read_text() == "Overall Acc\n25\n"
