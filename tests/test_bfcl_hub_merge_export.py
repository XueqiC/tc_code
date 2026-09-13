"""CPU-only export regressions using tiny, entirely local safetensors snapshots."""

import json
from pathlib import Path
import socket
import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]

from tools import bfcl_hub_merge_export as exporter


Q_PROJ = "model.language_model.layers.0.self_attn.q_proj.weight"
K_PROJ = "model.language_model.layers.0.self_attn.k_proj.weight"
NORM = "model.language_model.norm.weight"
VISION = "model.vision_tower.proj.weight"
TRAINED_Q = "model.layers.0.self_attn.q_proj.weight"
TRAINED_K = "model.layers.0.self_attn.k_proj.weight"


@pytest.fixture(autouse=True)
def cpu_only_offline(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))

    def forbidden(*args, **kwargs):
        pytest.fail("export tests must not initialize CUDA or access the network")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


def make_snapshot(tmp_path, layout):
    model = f"test/tiny-{layout}"
    cache = tmp_path / "hub" / ("models--" + model.replace("/", "--"))
    snapshot = cache / "snapshots" / "tiny-revision"
    snapshot.mkdir(parents=True)
    (cache / "refs").mkdir()
    (cache / "refs/main").write_text("tiny-revision\n")
    (snapshot / "config.json").write_text('{"architectures": ["TinyModel"]}\n')
    (snapshot / "preprocessor_config.json").write_text('{"processor": "tiny"}\n')
    (snapshot / "video_preprocessor_config.json").write_text("{}\n")
    (snapshot / "templates").mkdir()
    (snapshot / "templates/chat.jinja").write_text("{{ messages }}\n")
    # Hub snapshots normally link to blobs; exported ancillary files must stand alone.
    (cache / "blobs").mkdir()
    tokenizer = cache / "blobs/tokenizer"
    tokenizer.write_text('{"tokenizer": "tiny"}\n')
    (snapshot / "tokenizer.json").symlink_to(tokenizer)

    base = {
        Q_PROJ: torch.arange(6, dtype=torch.float32).reshape(2, 3),
        K_PROJ: torch.arange(10, 16, dtype=torch.bfloat16).reshape(2, 3),
        NORM: torch.tensor([-0.0, float("nan"), float("inf")]),
        VISION: torch.arange(4, dtype=torch.float16).reshape(2, 2),
    }
    if layout == "single":
        save_file(base, str(snapshot / "model.safetensors"))
    else:
        groups = [(Q_PROJ, NORM), (K_PROJ, VISION)]
        weight_map = {}
        for number, keys in enumerate(groups, 1):
            name = f"model-{number:05d}-of-00002.safetensors"
            save_file({key: base[key] for key in keys}, str(snapshot / name))
            weight_map.update({key: name for key in keys})
        index = {
            "metadata": {"total_size": sum(t.numel() * t.element_size() for t in base.values())},
            "weight_map": weight_map,
        }
        (snapshot / "model.safetensors.index.json").write_text(json.dumps(index))
    return model, snapshot, base


def make_adapter(tmp_path, state=None):
    adapter = tmp_path / "adapter"
    adapter.mkdir(exist_ok=True)
    if state is None:
        state = {
            TRAINED_Q: torch.full((2, 3), 7, dtype=torch.bfloat16),
            TRAINED_K: torch.full((2, 3), 11, dtype=torch.float32),
        }
    save_file(state, str(adapter / "model.safetensors"))
    return adapter, state


def run_export(monkeypatch, model, adapter, out, *, verify=True):
    argv = ["bfcl_hub_merge_export.py", "--model", str(model),
            "--adapter", str(adapter), "--out", str(out)]
    if verify:
        argv.append("--verify")
    monkeypatch.setattr(sys, "argv", argv)
    exporter.main()


def read_weights(directory):
    state = {}
    for path in sorted(directory.glob("*.safetensors")):
        shard = load_file(str(path), device="cpu")
        assert not set(state).intersection(shard)
        state.update(shard)
    return state


def assert_same_tensors(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        left, right = actual[key], expected[key]
        assert left.device.type == right.device.type == "cpu"
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        # Compare bytes to cover untouched NaNs, signed zero, and mixed dtypes.
        assert torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)), key


@pytest.mark.parametrize("layout", ["single", "sharded"])
def test_local_scratch_export_preserves_all_bytes_and_tree_digest(tmp_path, monkeypatch, layout):
    import shutil
    from bfas.rtd.baselines.paper_scratch import export_directory
    from bfas.rtd.persistence import tree_hash
    model, _, _ = make_snapshot(tmp_path, layout)
    adapter, _ = make_adapter(tmp_path)
    shared = tmp_path/"run/export/hub_merged"
    run_export(monkeypatch, model, adapter, shared)
    original = {p.relative_to(shared): p.read_bytes() for p in shared.rglob("*") if p.is_file()}
    scratch = tmp_path/"scratch"
    scratch.mkdir()
    monkeypatch.setenv("SLURM_TMPDIR", str(scratch))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    with export_directory(tmp_path/"other_run") as work:
        assert work.parent == scratch
        local_adapter = work/"adapter"
        shutil.copytree(adapter, local_adapter)
        local_merged = work/"hub_merged"
        run_export(monkeypatch, model, local_adapter, local_merged)
        assert {p.relative_to(local_merged): p.read_bytes() for p in local_merged.rglob("*") if p.is_file()} == original
        assert tree_hash(local_merged) == tree_hash(shared)
    assert not work.exists() and shared.is_dir()


@pytest.mark.parametrize("source", ["cache", "local"])
@pytest.mark.parametrize("verify", [False, True])
def test_single_and_sharded_exports_merge_identically(tmp_path, monkeypatch, capsys, source, verify):
    adapter, trained = make_adapter(tmp_path)
    merged_states = []
    for layout, shard_count in (("single", 1), ("sharded", 2)):
        model, snapshot, base = make_snapshot(tmp_path, layout)
        original_files = {p.relative_to(snapshot): p.read_bytes()
                          for p in snapshot.rglob("*") if p.is_file()}
        out = tmp_path / f"merged-{layout}"
        run_export(monkeypatch, model if source == "cache" else snapshot,
                   adapter, out, verify=verify)
        output = capsys.readouterr().out
        assert f"overlaid 2 tensors, left 2 untouched, wrote {shard_count} shards" in output
        assert ("verification passed" in output) == verify

        merged = read_weights(out)
        expected = {**base, Q_PROJ: trained[TRAINED_Q], K_PROJ: trained[TRAINED_K]}
        assert_same_tensors(merged, expected)
        assert_same_tensors(read_weights(snapshot), base)
        assert {p.relative_to(out) for p in out.rglob("*") if p.is_file()} == set(original_files)
        for relative, content in original_files.items():
            assert (snapshot / relative).read_bytes() == content
            assert not (out / relative).is_symlink()
            if relative.suffix != ".safetensors":
                assert (out / relative).read_bytes() == content
        assert (out / "model.safetensors.index.json").exists() == (layout == "sharded")
        # Exported checkpoints remain usable as the local base for another round.
        assert exporter._snapshot_for_model(str(out)) == out
        merged_states.append(merged)
    assert_same_tensors(*merged_states)


def test_missing_checkpoint_fails_before_export(tmp_path, monkeypatch):
    model, snapshot, _ = make_snapshot(tmp_path, "single")
    (snapshot / "model.safetensors").unlink()
    adapter, _ = make_adapter(tmp_path)
    out = tmp_path / "merged"
    with pytest.raises(SystemExit, match="snapshot is missing checkpoint") as error:
        run_export(monkeypatch, model, adapter, out)
    assert str(snapshot / "model.safetensors.index.json") in str(error.value)
    assert str(snapshot / "model.safetensors") in str(error.value)
    assert not out.exists()


@pytest.mark.parametrize("layout", ["single", "sharded"])
@pytest.mark.parametrize("problem, message", [
    ("unmapped", "unmapped trained tensor"),
    ("shape", "shape mismatch"),
    ("ambiguous", "ambiguous mapping"),
    ("tied", "both map to tied Hub target"),
])
def test_adapter_integrity_checks_apply_to_both_layouts(tmp_path, monkeypatch, layout, problem, message):
    model, _, _ = make_snapshot(tmp_path, layout)
    state = {TRAINED_Q: torch.ones((2, 3))}
    if problem == "unmapped":
        # A valid tensor must not hide an additional adapter target absent from the base.
        state["missing_parameter"] = torch.ones(1)
    elif problem == "shape":
        state[TRAINED_Q] = torch.ones((3, 2))
    elif problem == "ambiguous":
        state["weight"] = torch.ones((2, 3))
    else:
        state[Q_PROJ] = torch.zeros((2, 3))
    adapter, _ = make_adapter(tmp_path, state)
    out = tmp_path / "merged"
    with pytest.raises(SystemExit, match=message):
        run_export(monkeypatch, model, adapter, out)
    assert not out.exists()


@pytest.mark.parametrize("layout", ["single", "sharded"])
@pytest.mark.parametrize("target", [Q_PROJ, VISION])
def test_verification_rejects_changed_exported_tensors(tmp_path, monkeypatch, layout, target):
    model, _, _ = make_snapshot(tmp_path, layout)
    adapter, _ = make_adapter(tmp_path)
    write_shards = exporter._write_shards

    def corrupt_output(out, hub_state, weight_map, shard_names):
        write_shards(out, hub_state, weight_map, shard_names)
        path = out / weight_map[target]
        state = load_file(str(path), device="cpu")
        state[target] = state[target] + 1
        save_file(state, str(path))

    monkeypatch.setattr(exporter, "_write_shards", corrupt_output)
    with pytest.raises(SystemExit, match="verification failed"):
        run_export(monkeypatch, model, adapter, tmp_path / "merged")


@pytest.mark.parametrize("problem, message", [
    ("corrupt", "failed to read checkpoint header"),
    ("empty", "Hub shard contains no tensors"),
    ("invalid_key", "checkpoint contains an invalid tensor key"),
])
def test_invalid_single_file_checkpoint_fails(tmp_path, monkeypatch, problem, message):
    model, snapshot, _ = make_snapshot(tmp_path, "single")
    checkpoint = snapshot / "model.safetensors"
    if problem == "corrupt":
        checkpoint.write_bytes(b"invalid safetensors")
    elif problem == "invalid_key":
        save_file({"": torch.ones(1)}, str(checkpoint))
    else:
        save_file({}, str(checkpoint))
    adapter, _ = make_adapter(tmp_path)
    out = tmp_path / "merged"
    with pytest.raises(SystemExit, match=message):
        run_export(monkeypatch, model, adapter, out)
    assert not out.exists()


@pytest.mark.parametrize("problem, message", [
    ("corrupt", "could not read shard index"),
    ("missing_shard", "missing shard named by index"),
    ("missing_tensor", "indexed Hub tensors were not found"),
    ("unindexed_tensor", "is absent from the index"),
    ("wrong_shard", "but the index names"),
    ("unsafe_shard", "unsafe or invalid shard name"),
])
def test_existing_index_is_authoritative_even_with_single_file(tmp_path, monkeypatch, problem, message):
    model, snapshot, base = make_snapshot(tmp_path, "sharded")
    save_file(base, str(snapshot / "model.safetensors"))
    index_path = snapshot / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    if problem == "missing_shard":
        (snapshot / weight_map[Q_PROJ]).unlink()
    elif problem == "missing_tensor":
        weight_map["missing.weight"] = weight_map[Q_PROJ]
    elif problem == "unindexed_tensor":
        del weight_map[Q_PROJ]
    elif problem == "wrong_shard":
        weight_map[Q_PROJ] = weight_map[K_PROJ]
    elif problem == "unsafe_shard":
        weight_map[Q_PROJ] = "../model.safetensors"
    index_path.write_text("{" if problem == "corrupt" else json.dumps(index))
    adapter, _ = make_adapter(tmp_path)
    out = tmp_path / "merged"
    with pytest.raises(SystemExit, match=message):
        run_export(monkeypatch, model, adapter, out)
    assert not out.exists()
