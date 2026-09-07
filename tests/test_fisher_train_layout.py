"""CPU-only empirical-Fisher checks; no pretrained model or tokenizer is loaded."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tools.behavior_atom.fisher_train_layout import (VERSION, accumulate_fisher,
                                                    ell_mode, memory_estimate,
                                                    read_pool, save_fisher)
from tools.behavior_atom import fisher_train_layout as fisher_cli
from bfas.behavior.deltas import canonical_hash, file_hash, layout_hash, parameter_layout
from bfas.behavior.whiten import DiagonalMetric, from_fisher


class Quadratic(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        # Deliberately nonalphabetical order, mixed ranks, and an unused factor.
        self.z_factor = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=dtype))
        self.a_factor = torch.nn.Parameter(torch.tensor([[0.5]], dtype=dtype))
        self.unused = torch.nn.Parameter(torch.ones(1, dtype=dtype))
        self.frozen = torch.nn.Parameter(torch.ones(3, dtype=dtype), requires_grad=False)

    def forward(self, row):
        center, length = row
        theta = torch.cat([self.z_factor.flatten(), self.a_factor.flatten()])
        return -0.5 * (theta - theta.new_tensor(center)).square().sum()


ROWS = [([0.0, 1.0, -0.5], 2), ([2.0, -4.0, 1.5], 4), ([-1.0, 0.0, 0.0], 1)]


@pytest.mark.parametrize("batch_rows", [1, 2, 8])
@pytest.mark.parametrize("mode", ["full_sequence", "length_normalized", "length_normalised"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_known_quadratic_mean_squared_per_sequence_gradients(batch_rows, mode, dtype):
    model = Quadratic(dtype)
    before = copy.deepcopy(model.state_dict())
    mode = ell_mode(mode)

    def score(student, row):
        value = student(row)
        return value if mode == "full_sequence" else value / row[1]

    # Existing training gradients must neither leak in nor be changed.
    model.z_factor.grad = torch.full_like(model.z_factor, 123)
    result = accumulate_fisher(model, iter(ROWS), score, batch_rows=batch_rows, max_grad_norm=0.5)
    gradients = np.array([np.array(center) - [1.0, -2.0, 0.5] for center, _ in ROWS])
    if mode == "length_normalized":
        gradients /= np.array([length for _, length in ROWS])[:, None]
    expected = np.r_[np.mean(gradients ** 2, axis=0), 0.0].astype(np.float32)
    np.testing.assert_allclose(result.fisher, expected, rtol=1e-6)
    assert not np.allclose(result.fisher[:3], gradients.mean(axis=0) ** 2)
    assert result.fisher.dtype == np.float32
    assert np.all(result.fisher >= 0)
    assert result.n_rows == len(ROWS)
    assert result.layout == parameter_layout(model)
    assert canonical_hash(result.layout) == layout_hash(model)
    assert [e["name"] for e in result.layout] == ["z_factor", "a_factor", "unused"]
    assert [e["offset"] for e in result.layout] == [0, 2, 3]
    assert result.clipping_stats["applied"] is False
    assert result.clipping_stats["would_clip_count"] == len(ROWS)
    np.testing.assert_allclose(result.clipping_stats["per_row_pre_clip_norm"], np.linalg.norm(gradients, axis=1))
    assert torch.all(model.z_factor.grad == 123)
    assert model.a_factor.grad is None
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_bf16_gradient_is_promoted_before_squaring():
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
    # bf16 cannot represent 258**2 exactly; squaring in native storage is wrong.
    value = torch.tensor(258.0, dtype=torch.bfloat16)
    result = accumulate_fisher(model, [None], lambda m, _: m.weight.sum() * value)
    assert result.fisher[0] == float(value) ** 2
    assert result.fisher[0] != float(value.square())


def test_npz_roundtrip_metric_and_mismatched_layout_rejection(tmp_path):
    model = Quadratic()
    result = accumulate_fisher(model, ROWS, lambda m, row: m(row) / row[1])
    path = tmp_path / "fisher.npz"
    manifest = save_fisher(result, path, model_id="synthetic-student", pool_content_hash="pool-sha256",
                           ell="length_normalised", eps=0.125, provenance={"pool_role": "reference"})
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["fisher"], result.fisher)
        assert data["layout_hash"].item() == layout_hash(model)
        assert data["parameter_names"].tolist() == [e["name"] for e in result.layout]
        assert [json.loads(s) for s in data["parameter_shapes"]] == [e["shape"] for e in result.layout]
        assert json.loads(data["layout"].item()) == result.layout
        assert data["n_rows"].item() == len(ROWS)
        assert data["ell"].item() == "length_normalized"
        assert data["version"].item() == VERSION
        assert data["model_id"].item() == "synthetic-student"
        assert data["pool_content_hash"].item() == "pool-sha256"
        assert data["owner"].item() == "student"
        assert json.loads(data["manifest"].item())["clipping_stats"] == result.clipping_stats
        tampered = {key: data[key] for key in data.files}
    saved = json.loads(path.with_suffix(".npz.manifest.json").read_text())
    assert saved == manifest
    assert saved["payload_hash"] == file_hash(path)
    assert saved["pool_role"] == "reference"
    # The repository exposes the constructor as whiten.from_fisher, which
    # returns and validates a DiagonalMetric (there is no classmethod).
    metric = from_fisher(path, parameter_layout(model), eps=saved["eps"], version=saved["version"])
    assert isinstance(metric, DiagonalMetric)
    assert metric.layout_hash == layout_hash(model)
    assert metric.fisher_hash == saved["fisher_hash"]
    np.testing.assert_allclose(metric.diagonal, result.fisher.astype(np.float64) + 0.125)
    assert np.all(metric.diagonal > 0)
    metric.__post_init__()
    with pytest.raises(ValueError, match="layout hash mismatch"):
        from_fisher(path, list(reversed(result.layout)), eps=0.125)
    with pytest.raises(ValueError, match="layout hash mismatch"):
        from_fisher(path, copy.deepcopy(model).to(torch.bfloat16), eps=0.125)
    tampered["layout_hash"] = "incorrect-layout-hash"
    bad = tmp_path / "wrong-layout.npz"
    np.savez(bad, **tampered)
    with pytest.raises(ValueError, match="layout hash mismatch"):
        from_fisher(bad, model, eps=0.125)
    with pytest.raises(FileExistsError):
        save_fisher(result, path, model_id="synthetic", pool_content_hash="hash")


def test_pool_owner_limit_and_full_content_hash(tmp_path):
    path = tmp_path / "pool.jsonl"
    rows = [{"prompt": "p", "_rejected": "student", "response": "teacher", "split": "train"},
            {"prompt": "q", "_rejected": "s", "response": "t"}]
    path.write_text("\n" + "\n".join(json.dumps(row) for row in rows) + "\n")
    selected, provenance = read_pool(path, owner="student", max_rows=1)
    assert selected == rows[:1]
    assert provenance["n_pool_rows"] == 2
    assert provenance["pool_content_hash"] == file_hash(path)
    original_hash = provenance["pool_content_hash"]
    path.write_text(path.read_text() + "\n")
    assert read_pool(path, owner="student", max_rows=1)[1]["pool_content_hash"] != original_hash
    path.write_text(json.dumps({"prompt": "p", "response": "teacher"}) + "\n")
    assert len(read_pool(path, owner="teacher")[0]) == 1
    with pytest.raises(ValueError, match="_rejected"):
        read_pool(path, owner="student")
    for split in ("test", "validation", "calibration", "development"):
        path.write_text(json.dumps({**rows[0], "split": split}) + "\n")
        with pytest.raises(ValueError, match="non-training/reference"):
            read_pool(path, owner="student")


def test_invalid_inputs_and_memory_estimate():
    model = Quadratic()
    score = lambda m, row: m(row)
    with pytest.raises(ValueError, match="zero rows"):
        accumulate_fisher(model, [], score)
    with pytest.raises(ValueError, match="batch_rows"):
        accumulate_fisher(model, ROWS, score, batch_rows=0)
    with pytest.raises(ValueError, match="max_grad_norm"):
        accumulate_fisher(model, ROWS, score, max_grad_norm=float("nan"))
    with pytest.raises(ValueError, match="finite scalar"):
        accumulate_fisher(model, ROWS, lambda m, row: m(row) * float("nan"))
    with pytest.raises(ValueError, match="overflow"):
        accumulate_fisher(model, [None], lambda m, _: m.z_factor[0] * 1e30)
    estimate = memory_estimate(model)
    assert estimate["trainable_parameter_count"] == 4
    assert estimate["parameter_count"] == 7
    assert estimate["host_fisher_accumulator_bytes"] == 16
    assert estimate["gpu_storage_lower_bound_bytes"] == 7 * 4 + 4 * 4
    assert estimate["active_sequences"] == 1
    assert estimate["host_largest_gradient_copy_bytes"] == 16


@pytest.mark.parametrize("available,requested,allow_cpu,expected", [
    (True, None, False, "cuda:0"), (True, "cuda", False, "cuda:0"),
    (False, None, True, "cpu"), (False, "cuda", True, "cpu"),
    (True, "cpu", True, "cpu"), (False, "cpu", True, "cpu"),
    (False, None, False, None), (False, "cuda", False, None),
    (False, "cpu", False, None), (True, "cpu", False, None),
])
def test_device_guard(monkeypatch, available, requested, allow_cpu, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    if expected is None:
        with pytest.raises(RuntimeError, match="--allow-cpu"):
            fisher_cli.resolve_device(requested, allow_cpu=allow_cpu)
    else:
        assert str(fisher_cli.resolve_device(requested, allow_cpu=allow_cpu)) == expected


def test_cli_device_guard_precedes_pool_or_model_loading(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(fisher_cli, "read_pool", lambda *a, **k: pytest.fail("read pool before device guard"))
    with pytest.raises(RuntimeError, match="--allow-cpu"):
        fisher_cli.main([])


def test_accumulator_rejects_mixed_model_or_score_devices():
    model = Quadratic()
    model.frozen = torch.nn.Parameter(torch.empty(3, device="meta"), requires_grad=False)
    with pytest.raises(ValueError, match="parameters and buffers must be on cpu"):
        accumulate_fisher(model, ROWS, lambda *a: pytest.fail("ran mismatched model"))
    model = Quadratic()
    model.register_buffer("misplaced", torch.empty(1, device="meta"))
    with pytest.raises(ValueError, match="parameters and buffers must be on cpu"):
        accumulate_fisher(model, ROWS, lambda *a: pytest.fail("ran mismatched model"))
    with pytest.raises(ValueError, match="ell must be on cpu"):
        accumulate_fisher(Quadratic(), ROWS, lambda *a: torch.empty((), device="meta"))


def test_flat_accumulation_computes_layout_once_without_tensor_hashes(monkeypatch):
    model = Quadratic()
    # A noncontiguous factor must retain the original logical flat ordering.
    model.extra = torch.nn.Parameter(torch.arange(6.0).reshape(2, 3).T)
    original_layout = fisher_cli.parameter_layout
    layouts, accumulations = [], []
    def layout_once(student):
        layouts.append(student)
        assert len(layouts) == 1
        return original_layout(student)
    original_addcmul = torch.Tensor.addcmul_
    def addcmul(total, first, second, **kwargs):
        assert total.device.type == first.device.type == second.device.type == "cpu"
        assert total.dtype == first.dtype == second.dtype == torch.float32
        assert total.shape == first.shape == second.shape == (10,)
        accumulations.append(total.numel())
        return original_addcmul(total, first, second, **kwargs)
    monkeypatch.setattr(fisher_cli, "parameter_layout", layout_once)
    monkeypatch.setattr(torch.Tensor, "addcmul_", addcmul)
    monkeypatch.setattr(fisher_cli, "layout_hash", lambda *a: pytest.fail("rehashed layout"))
    monkeypatch.setattr(fisher_cli, "canonical_hash", lambda *a: pytest.fail("hashed row"))
    monkeypatch.setattr(torch.Tensor, "clone", lambda *a, **k: pytest.fail("cloned row tensor"))
    result = accumulate_fisher(model, ROWS, lambda m, row: m(row) + m.extra.square().sum())
    assert len(layouts) == 1 and len(accumulations) == len(ROWS)
    np.testing.assert_allclose(result.fisher[4:], (2 * model.extra.detach().reshape(-1).numpy()) ** 2)


def test_changed_trainable_count_or_shape_is_rejected():
    for change in (lambda m: m.register_parameter("extra", torch.nn.Parameter(torch.ones(1))),
                   lambda m: setattr(m.unused, "data", torch.ones(2))):
        model = Quadratic()
        def score(m, row):
            value = m(row)
            change(m)
            return value
        with pytest.raises(ValueError, match="layout changed"):
            accumulate_fisher(model, ROWS[:1], score)


class TinyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.arange(8, dtype=torch.bfloat16) / 8, requires_grad=False)
        self.adapter = torch.nn.Parameter(torch.zeros(8, dtype=torch.float32))
        self.config = SimpleNamespace(use_cache=True)
        self.forward_devices, self.backward_devices = [], []
        self.adapter.register_hook(lambda g: self.backward_devices.append(g.device))

    def forward(self, input_ids):
        assert input_ids.device == self.base.device == self.adapter.device
        self.forward_devices.append(input_ids.device)
        logits = self.base + self.adapter.to(dtype=self.base.dtype)
        return SimpleNamespace(logits=logits.expand(*input_ids.shape, 8))


class TinyTokenizer:
    eos_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [1 + ord(c) % 7 for c in text]}


def synthetic_cli(tmp_path, monkeypatch, n_rows=3):
    import appworld_train as trainer
    model = TinyCausalModel()
    pool = tmp_path / "pool.jsonl"
    pool.write_text("\n".join(json.dumps(dict(prompt="prompt", response="teacher", _rejected="student"))
                              for _ in range(n_rows)) + "\n")
    def build(model_id, seed):
        assert model_id == "synthetic" and seed == 0
        return model.to(device=trainer.DEVICE)
    monkeypatch.setattr(trainer, "build_lora_model", build)
    monkeypatch.setattr(trainer, "load_tokenizer", lambda _: TinyTokenizer())
    monkeypatch.setenv("AW_MAX_PROMPT_TOKENS", "4096")
    monkeypatch.setenv("AW_TRUNCATE_SIDE", "tail")
    output = tmp_path / "fisher.npz"
    return model, output, ["--pool", str(pool), "--model-id", "synthetic", "--output", str(output)]


@pytest.mark.parametrize("progress_args,expected_rows", [([], [10, 12]), (["--progress-every", "5"], [5, 10, 12])])
def test_cli_progress_and_native_dtypes(tmp_path, monkeypatch, capsys, progress_args, expected_rows):
    import builtins
    import appworld_train as trainer
    model, output, args = synthetic_cli(tmp_path, monkeypatch, n_rows=12)
    before_device = trainer.DEVICE
    def flushed_print(*a, **kwargs):
        if a and str(a[0]).startswith("[fisher"):
            assert kwargs.get("flush") is True
        return builtins.print(*a, **kwargs)
    monkeypatch.setattr(fisher_cli, "print", flushed_print, raising=False)
    fisher_cli.main([*args, "--device", "cpu", "--allow-cpu", *progress_args])
    captured = capsys.readouterr()
    progress = [line for line in captured.err.splitlines() if line.startswith("[fisher] rows=")]
    assert [int(line.split("rows=")[1].split("/")[0]) for line in progress] == expected_rows
    assert all(all(key in line for key in ("rows/s=", "tokens=", "elapsed=", "ETA=")) for line in progress)
    assert "tokens=168" in progress[-1] and "ETA=0.0s" in progress[-1]
    assert len(model.forward_devices) == len(model.backward_devices) == 12
    assert set(model.forward_devices + model.backward_devices) == {torch.device("cpu")}
    assert model.base.dtype == torch.bfloat16 and model.adapter.dtype == torch.float32
    assert trainer.DEVICE == before_device
    summary = json.loads(captured.out)
    assert summary["n_rows"] == 12
    manifest = json.loads(output.with_suffix(".npz.manifest.json").read_text())
    assert manifest["model_parameter_dtypes"] == ["torch.bfloat16", "torch.float32"]
    assert manifest["effective_loss_tokens"] == 12 * 8
    assert len(manifest["preprocessing"]) == 12
    assert from_fisher(output, model, eps=manifest["eps"]).size == 8


@pytest.mark.parametrize("limit,expected", [([], 2), (["--max-rows", "1"], 1)])
def test_cli_benchmark_limits_rows_reports_memory_and_writes_nothing(tmp_path, monkeypatch, capsys, limit, expected):
    model, output, args = synthetic_cli(tmp_path, monkeypatch)
    initial = tmp_path / "initial.pt"
    # A benchmark must also work alongside an existing campaign output.
    output.write_bytes(b"existing Fisher")
    fisher_cli.main([*args, "--device", "cpu", "--allow-cpu", "--benchmark-rows", "2",
                     "--save-initial-state", str(initial), *limit])
    captured = capsys.readouterr()
    timings = [line for line in captured.err.splitlines() if line.startswith("[fisher benchmark] row=")]
    assert len(timings) == expected
    assert all("seconds=" in line and "peak_gpu_memory_mib=0.00" in line for line in timings)
    summary = json.loads(captured.out)
    assert summary["benchmark"] and summary["n_rows"] == expected
    assert summary["seconds_per_row"] > 0 and summary["peak_gpu_memory_bytes"] == 0
    assert summary["device"] == "cpu"
    assert len(model.forward_devices) == len(model.backward_devices) == expected
    assert output.read_bytes() == b"existing Fisher"
    assert not initial.exists() and not output.with_suffix(".npz.manifest.json").exists()


@pytest.mark.parametrize("option", ["--progress-every", "--benchmark-rows"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_invalid_progress_and_benchmark_options(option, value):
    with pytest.raises(SystemExit) as error:
        fisher_cli.main([option, value])
    assert error.value.code == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_cli_forward_backward_and_benchmark(tmp_path, monkeypatch, capsys):
    model, output, args = synthetic_cli(tmp_path, monkeypatch)
    fisher_cli.main([*args, "--benchmark-rows", "2"])
    summary = json.loads(capsys.readouterr().out)
    assert all(device.type == "cuda" for device in model.forward_devices + model.backward_devices)
    assert len(model.forward_devices) == len(model.backward_devices) == 2
    assert model.base.dtype == torch.bfloat16 and model.adapter.dtype == torch.float32
    assert summary["peak_gpu_memory_bytes"] > 0 and not output.exists()
