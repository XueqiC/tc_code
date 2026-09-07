"""CPU tests for exact parameter replay, Fisher geometry and independent updates.

The optional GPU check requires BOTH CUDA and BEHAVIOR_TEST_GPU=1. The requested
default pytest invocation never starts GPU work, even on a CUDA-equipped host.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import warnings

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bfas.behavior.deltas import (apply_delta, capture_delta, content_key, first_order_diff,
                                  first_order_teacher, layout_hash, load_delta,
                                  parameter_layout, save_delta, tensor_state_hash)
from bfas.behavior.whiten import build_basis, from_fisher, whiten
from bfas.behavior.microupdate import (MicroUpdateConfig, evaluation_score, run_micro_update,
                                      save_initial_state, _model_settings)
import appworld_train as trainer


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(previous)


def snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


@pytest.mark.parametrize("extension", [".npz", ".safetensors"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_delta_roundtrip_exact(tmp_path, extension, dtype):
    torch.random.default_generator.manual_seed(13)
    model = torch.nn.Linear(7, 3).to(dtype)
    initial = copy.deepcopy(model)
    base = snapshot(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.02)
    path = tmp_path / ("update" + extension)
    saved = save_delta(model, base, path, clipping_stats={"count": 2, "steps": 3})
    loaded = load_delta(path)
    assert loaded.manifest["layout_hash"] == layout_hash(model)
    assert loaded.manifest["layout"] == parameter_layout(model)
    assert loaded.manifest["value_dtype"] == "float32"
    assert loaded.manifest["clipping_stats"]["count"] == 2
    assert all(t.dtype == torch.float32 for t in loaded.tensors.values())
    expected = torch.cat([(p.detach().double() - base[n].double()).flatten() for n, p in model.named_parameters()])
    np.testing.assert_array_equal(loaded.flat(), expected.detach().numpy())
    assert loaded.manifest["update_norm"] == pytest.approx(float(expected.norm()))
    apply_delta(initial, loaded)
    for actual, updated in zip(initial.parameters(), model.parameters()):
        assert torch.equal(actual, updated)
    np.testing.assert_array_equal(saved.flat(), loaded.flat())
    with pytest.raises(FileExistsError):
        save_delta(model, base, path)
    with pytest.raises(ValueError, match="fresh base"):
        apply_delta(initial, loaded)


def test_subtraction_roundoff_is_retained(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.1)
    base_model, base = copy.deepcopy(model), snapshot(model)
    with torch.no_grad():
        model.weight.fill_(-0.2)
    ordinary = base["weight"] + (model.weight.detach() - base["weight"])
    assert not torch.equal(ordinary, model.weight)
    delta = save_delta(model, base, tmp_path / "rounded.npz")
    assert "weight" in delta.roundoff
    apply_delta(base_model, load_delta(tmp_path / "rounded.npz"))
    assert torch.equal(base_model.weight, model.weight)


def test_zero_delta_identical_fixed_input_and_rng(tmp_path):
    torch.random.default_generator.manual_seed(4)
    model = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.Dropout(0.4))
    replay = copy.deepcopy(model)
    path = tmp_path / "zero.npz"
    delta = save_delta(model, snapshot(model), path)
    apply_delta(replay, load_delta(path))
    x = torch.randn(3, 5)
    rng = torch.get_rng_state()
    expected = model(x)
    torch.set_rng_state(rng)
    assert torch.equal(expected, replay(x))
    assert delta.manifest["update_norm"] == 0
    assert np.count_nonzero(delta.flat()) == 0


def test_scale_layout_base_and_corruption_validation(tmp_path):
    model = torch.nn.Linear(2, 1)
    # Exact scale assertions require an exactly representable initial update.
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.25, -0.5]]))
    base = snapshot(model)
    initial = copy.deepcopy(model)
    with torch.no_grad():
        model.weight.add_(0.125)
    path = tmp_path / "delta.npz"
    delta = save_delta(model, base, path)
    unchanged = copy.deepcopy(initial)
    apply_delta(unchanged, delta, scale=0)
    assert tensor_state_hash(snapshot(unchanged)) == tensor_state_hash(base)
    apply_delta(initial, delta, scale=-0.5)
    assert torch.equal(initial.weight, base["weight"] - 0.0625)
    with pytest.raises(ValueError, match="layout"):
        apply_delta(torch.nn.Linear(3, 1), delta)
    with pytest.raises(ValueError, match="finite"):
        apply_delta(model, delta, float("nan"))
    with open(path, "ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_delta(path)


def test_layout_includes_order_dtype_and_trainable_status():
    model = torch.nn.Linear(2, 2)
    original = layout_hash(model)
    assert original != layout_hash(copy.deepcopy(model).to(torch.bfloat16))
    model.bias.requires_grad_(False)
    assert layout_hash(model) != original
    left, right = torch.nn.Module(), torch.nn.Module()
    for name in ("a", "b"):
        left.register_parameter(name, torch.nn.Parameter(torch.zeros(2)))
    for name in ("b", "a"):
        right.register_parameter(name, torch.nn.Parameter(torch.zeros(2)))
    assert layout_hash(left) != layout_hash(right)


def test_cache_keys_invalidate_every_identity_component():
    model = torch.nn.Linear(2, 1)
    args = dict(model_id="checkpoint-A", content_hashes={"prompt": "abc", "target": "def"},
                preprocessing={"prompt_cap": 4096, "prompt_side": "head"},
                parameter_layout=layout_hash(model), fingerprint_version="gradient-v1",
                basis_version="fold0-v1", eval_config={"ell": "full_sequence", "seed": 4})
    original = content_key(**args)
    changes = dict(model_id="checkpoint-B", content_hashes={"prompt": "new", "target": "def"},
                   parameter_layout=layout_hash(torch.nn.Linear(3, 1)),
                   fingerprint_version="gradient-v2", basis_version="fold1-v1",
                   eval_config={"ell": "length_normalized", "seed": 4})
    for key, value in changes.items():
        assert content_key(**{**args, key: value}) != original
    for key, value in (("prompt_cap", 100), ("prompt_side", "tail")):
        assert content_key(**{**args, "preprocessing": {**args["preprocessing"], key: value}}) != original
    assert content_key(**dict(reversed(list(args.items())))) == original
    with pytest.raises(ValueError):
        content_key(**{**args, "preprocessing": {}})


def metric_for(size, block_size=5):
    model = torch.nn.Linear(size, 1, bias=False)
    fisher = np.linspace(0.0, 2.0, size)
    H = from_fisher(fisher, model, eps=0.2, layout_hash=layout_hash(model), block_size=block_size)
    return model, fisher, H


def test_named_and_saved_flat_fisher_alignment(tmp_path):
    model = torch.nn.Linear(3, 2)
    entries = parameter_layout(model)
    size = sum(e["numel"] for e in entries)
    flat = np.arange(size, dtype=np.float32)
    named = {e["name"]: flat[e["offset"]:e["offset"] + e["numel"]].reshape(e["shape"]) for e in entries}
    path = tmp_path / "fisher.npz"
    np.savez(path, fisher=flat, layout_hash=layout_hash(model))
    actual = from_fisher(path, entries, eps=1e-3, block_size=2)
    expected = from_fisher(named, model, eps=1e-3, block_size=3)
    np.testing.assert_array_equal(actual.diagonal, expected.diagonal)
    assert actual.fisher_hash == expected.fisher_hash
    np.testing.assert_allclose(whiten(np.ones(size), actual), np.sqrt(flat + 1e-3), rtol=1e-7)
    with pytest.raises(ValueError, match="saved layout hash"):
        from_fisher(flat, model, eps=0.1)
    with pytest.raises(ValueError, match="hash mismatch"):
        from_fisher(flat, model, eps=0.1, layout_hash="old")
    with pytest.raises(ValueError, match="names"):
        from_fisher({"bias": named["bias"]}, model, eps=0.1)
    for invalid in (-1.0, np.nan, np.inf):
        bad = flat.copy()
        bad[0] = invalid
        with pytest.raises(ValueError, match="nonnegative"):
            from_fisher(bad, model, eps=0.1, layout_hash=layout_hash(model))
    with pytest.raises(ValueError, match="positive"):
        from_fisher(named, model, eps=0)
    with pytest.raises(ValueError, match="shape"):
        from_fisher({**named, "weight": np.ones((3, 2))}, model, eps=0.1)


def test_basis_metric_orthogonality_roundtrip_and_residual(tmp_path):
    rng = np.random.default_rng(21)
    _, _, H = metric_for(47)
    deltas = [rng.normal(size=H.size) for _ in range(3)]
    deltas += [deltas[0] + 2 * deltas[1], np.zeros(H.size)]
    basis = build_basis(deltas, H)
    assert basis.rank == 3
    C = np.column_stack([basis.decode(x) for x in np.eye(basis.rank)])
    np.testing.assert_allclose(C.T @ (H.diagonal[:, None] * C), np.eye(3), atol=2e-14)
    Q = np.sqrt(H.diagonal[:, None]) * C
    np.testing.assert_allclose(Q.T @ Q, np.eye(3), atol=2e-14)
    for delta in deltas:
        x, norm = basis.encode(delta)
        assert norm < 1e-13
        np.testing.assert_allclose(basis.decode(x), delta, atol=1e-13)
    held_out = rng.normal(size=H.size)
    x, norm = basis.encode(held_out)
    residual, norm2 = basis.residual(held_out)
    np.testing.assert_allclose(C.T @ (H.diagonal * residual), np.zeros(3), atol=2e-14)
    assert norm == pytest.approx(norm2)
    assert norm == pytest.approx(np.sqrt(residual @ (H.diagonal * residual)))
    assert H.norm(held_out) ** 2 == pytest.approx(float(x @ x) + norm ** 2)
    np.testing.assert_allclose(basis.decode(x, residual), held_out, atol=1e-14)
    doubled_x, doubled_norm = basis.encode(2 * held_out)
    np.testing.assert_allclose(doubled_x, 2 * x)
    assert doubled_norm == pytest.approx(2 * norm)
    basis.save(tmp_path / "basis.json", [{"path": str(i), "content_hash": "hash"} for i in range(len(deltas))])
    saved = json.loads((tmp_path / "basis.json").read_text())
    np.testing.assert_allclose(np.column_stack(deltas) @ saved["coefficients"], C)


@pytest.mark.parametrize("empty", [False, True])
def test_rank_zero_basis_and_held_out_does_not_fit(empty):
    _, _, H = metric_for(9)
    basis = build_basis([] if empty else [np.zeros(9), np.zeros(9)], H)
    assert basis.rank == 0
    x, norm = basis.encode(np.ones(9))
    assert len(x) == 0
    assert norm == pytest.approx(H.norm(np.ones(9)))
    np.testing.assert_array_equal(basis.decode(x), np.zeros(9))
    assert basis.rank == 0


def test_blocked_memmap_basis_matches_dense(tmp_path):
    n, block_size = 200_003, 997
    model = torch.nn.Linear(n, 1, bias=False)
    metric_out = np.memmap(tmp_path / "h.bin", mode="w+", shape=(n,), dtype=np.float64)
    H = from_fisher(np.ones(n, dtype=np.float32), model, eps=0.1,
                    layout_hash=layout_hash(model), block_size=block_size, out=metric_out)
    deltas = []
    for index in range(2):
        vector = np.memmap(tmp_path / f"d{index}.bin", mode="w+", shape=(n,), dtype=np.float32)
        for sl in H.blocks():
            vector[sl] = np.sin(np.arange(sl.start, sl.stop) * (index + 1) * 0.001)
        deltas.append(vector)
    basis = build_basis(deltas, H)
    out = np.memmap(tmp_path / "out.bin", mode="w+", shape=(n,), dtype=np.float64)
    x, residual_norm = basis.encode(deltas[1])
    assert residual_norm < 1e-10
    assert basis.decode(x, out=out) is out
    np.testing.assert_allclose(out, deltas[1], atol=1e-13)
    whiten(deltas[0], H, out=out)
    np.testing.assert_allclose(out, np.sqrt(1.1) * deltas[0].astype(np.float64))
    with pytest.raises(ValueError, match="overwrite"):
        # A float64 training vector would otherwise be overwritten during decode.
        b = build_basis([out], H)
        b.decode([1.0], out=out)


def test_first_order_teacher_and_difference_sign_on_quadratic():
    theta = np.array([0.25, -0.5])
    target_T, target_S = np.array([1.0, 0.0]), np.array([-1.0, -1.0])
    ell = lambda t, target: -0.5 * np.sum((t - target) ** 2)
    gT, gS = target_T - theta, target_S - theta
    delta = 1e-4 * (gT - gS)
    predicted_T = first_order_teacher(gT, delta)
    predicted_diff = first_order_diff(gT, gS, delta)
    actual_T = ell(theta + delta, target_T) - ell(theta, target_T)
    actual_diff = actual_T - (ell(theta + delta, target_S) - ell(theta, target_S))
    assert predicted_T > 0 and predicted_diff > 0
    assert predicted_T == pytest.approx(actual_T, abs=float(delta @ delta))
    assert predicted_diff == pytest.approx(actual_diff)
    assert first_order_teacher(gT, -delta) == -predicted_T
    assert first_order_diff(gT, gS, -delta) == -predicted_diff
    with pytest.raises(ValueError, match="aligned"):
        first_order_diff(gT, gS[:1], delta)


def test_first_order_subtracts_gradients_before_dot_product():
    teacher, student = np.array([1e16, 1]), np.array([1e16, 0])
    assert first_order_diff(teacher, student, np.ones(2)) == 1
    gradient = torch.tensor([1.0, 2.0], requires_grad=True)
    assert first_order_teacher(gradient, torch.ones(2)) == 3


def test_bf16_storage_can_erase_a_small_update():
    norms = []
    for dtype in (torch.float32, torch.bfloat16):
        model = torch.nn.Linear(1, 1, bias=False).to(dtype)
        with torch.no_grad():
            model.weight.fill_(1.0)
        base = snapshot(model)
        with torch.no_grad():
            model.weight.add_(1e-4)
        norms.append(capture_delta(model, base).manifest["update_norm"])
    assert norms[0] > 0
    assert norms[1] == 0


class IntegerTokenizer:
    eos_token_id = 0

    def __call__(self, text, *, add_special_tokens=False):
        return {"input_ids": [int(token) for token in text.split()]}


class TinyLoRALinear(torch.nn.Module):
    """Small causal scorer: frozen linear map plus two trainable LoRA factors."""

    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(6, 6, bias=False)
        self.base.weight.requires_grad_(False)
        self.lora_A = torch.nn.Linear(6, 3, bias=False)
        self.lora_B = torch.nn.Linear(3, 6, bias=False)
        torch.nn.init.zeros_(self.lora_B.weight)
        self.dropout = torch.nn.Dropout(0.35)
        self.register_buffer("offset", torch.zeros(6))

    def forward(self, input_ids):
        x = torch.nn.functional.one_hot(input_ids, 6).to(self.base.weight.dtype)
        logits = self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) + self.offset
        return SimpleNamespace(logits=logits)


@pytest.fixture
def micro_setup(tmp_path):
    torch.random.default_generator.manual_seed(12)
    template = TinyLoRALinear()
    path = tmp_path / "init.pt"
    save_initial_state(template, path)

    def factory():
        model = copy.deepcopy(template)
        # Intentionally wrong fresh init and buffer; the runner must restore both.
        with torch.no_grad():
            model.lora_A.weight.normal_()
            model.lora_B.weight.normal_()
            model.offset.fill_(123)
        return model

    cfg = MicroUpdateConfig(model_id="tiny-linear-v1", init_state_path=path, source_id="A",
                            steps=4, lr=0.03, beta=0.7, seed=19, tokenizer=IntegerTokenizer(),
                            batch_size=2, accumulation=2, scheduler="linear", max_grad_norm=0.01)
    a = [{"event_id": "a1", "prompt": "1 2", "response": "3 4", "_rejected": "5"},
         {"event_id": "a2", "prompt": "2 3", "response": "4", "_rejected": "1 5"}]
    b = [{"event_id": "b1", "prompt": "3 4", "response": "5", "_rejected": "1 2"}]
    return template, factory, cfg, a, b


def test_micro_update_independent_of_source_order_and_rng(micro_setup):
    template, factory, cfg, a, b = micro_setup
    init_hash = tensor_state_hash(snapshot(template))
    torch_state = torch.get_rng_state().clone()
    py_state, np_state = random.getstate(), np.random.get_state()
    a1 = run_micro_update(factory, a, cfg)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == py_state
    np.testing.assert_array_equal(np.random.get_state()[1], np_state[1])
    b1 = run_micro_update(factory, b, replace(cfg, source_id="B"))
    # Perturb ambient RNG and reverse source order.
    torch.rand(23)
    random.random()
    np.random.rand(17)
    b2 = run_micro_update(factory, b, replace(cfg, source_id="B"))
    a2 = run_micro_update(factory, a, cfg)
    np.testing.assert_array_equal(a1.delta.flat(), a2.delta.flat())
    np.testing.assert_array_equal(b1.delta.flat(), b2.delta.flat())
    assert not np.array_equal(a1.delta.flat(), b1.delta.flat())
    assert a1.manifest["loss_trace"] == a2.manifest["loss_trace"]
    assert a1.manifest["checkpoint_hash"] == b1.manifest["checkpoint_hash"]
    assert a1.delta.manifest["base_hash"] == init_hash
    assert a1.manifest["effective_loss_tokens"] == 4 * 2 * 2 * 5
    assert a1.manifest["steps"] == 4
    assert a1.manifest["clipping_stats"]["count"] > 0
    assert a1.manifest["learning_rates"] == pytest.approx([0.03, 0.0225, 0.015, 0.0075])
    assert a1.manifest["optimizer_config"]["initial_state"] == "empty"
    assert tensor_state_hash(snapshot(template)) == init_hash
    with pytest.raises(FileExistsError):
        save_initial_state(template, cfg.init_state_path)


def test_shared_ell_and_gradients_match_imported_trainer(micro_setup):
    template, _, _, a, _ = micro_setup
    template.eval()
    ids, labels = trainer.encode(IntegerTokenizer(), a[0], device="cpu")
    summed = evaluation_score(template, ids, labels, ell="full_sequence")
    n = int((labels[:, 1:] != -100).sum())
    averaged = evaluation_score(template, ids, labels, ell="length_normalized")
    assert torch.equal(summed, trainer.completion_log_prob(template, ids, labels))
    assert torch.equal(averaged, summed / n)
    params = list(snapshot(template))
    trainables = dict(template.named_parameters())
    sum_grad = torch.autograd.grad(summed, [trainables[k] for k in params])
    mean_grad = torch.autograd.grad(averaged, [trainables[k] for k in params])
    for sg, mg in zip(sum_grad, mean_grad):
        torch.testing.assert_close(sg / n, mg)
    assert torch.equal(summed, evaluation_score(template, ids, labels))
    with pytest.raises(ValueError, match="effective loss tokens"):
        evaluation_score(template, ids, torch.full_like(labels, -100))


def test_micro_loss_matches_manual_base_centred_adam_step(micro_setup):
    template, factory, cfg, a, _ = micro_setup
    cfg = replace(cfg, steps=1, accumulation=1, batch_size=1, scheduler="constant",
                  max_grad_norm=None, ell="length_normalized")
    expected = copy.deepcopy(template)
    encoded = [trainer.encode(cfg.tokenizer, row, device="cpu")
               for row in (a[0], trainer.rejected_row(a[0]))]
    expected.eval()
    with torch.no_grad():
        reference = evaluation_score(expected, *encoded[0], ell=cfg.ell) - evaluation_score(expected, *encoded[1], ell=cfg.ell)
    expected.train()
    torch.random.default_generator.manual_seed(cfg.seed)
    opt = torch.optim.AdamW([p for p in expected.parameters() if p.requires_grad], lr=cfg.lr)
    margin = evaluation_score(expected, *encoded[0], ell=cfg.ell) - evaluation_score(expected, *encoded[1], ell=cfg.ell)
    loss = -torch.nn.functional.logsigmoid(cfg.beta * (margin - reference))
    loss.backward()
    opt.step()
    result = run_micro_update(factory, a[:1], cfg)
    assert result.manifest["loss_trace"][0] == pytest.approx(np.log(2))
    replay = copy.deepcopy(template)
    apply_delta(replay, result.delta)
    for p, q in zip(replay.parameters(), expected.parameters()):
        assert torch.equal(p, q)


def test_micro_noop_probe_kl_saved_manifest_and_resume(micro_setup, tmp_path):
    template, factory, cfg, a, _ = micro_setup
    cfg = replace(cfg, steps=0, probe_rows=a[:1], run_id="noop", output_dir=tmp_path / "runs")
    result = run_micro_update(factory, a, cfg)
    assert result.delta.manifest["update_norm"] == 0
    assert result.manifest["policy_kl"] == 0
    assert result.manifest["effective_loss_tokens"] == 0
    assert result.manifest["probe_loss_tokens"] == 3
    state = json.loads((Path(cfg.output_dir) / "noop" / "state.manifest.json").read_text())
    for key in ("commit", "config_hash", "data_manifest_hash", "checkpoint_hash", "layout_hash",
                "optimizer_config", "seed", "precision", "batch_size", "accumulation", "steps"):
        assert state[key] is not None
    resumed = run_micro_update(lambda: pytest.fail("resume must not load a model"), a, replace(cfg, resume=True))
    np.testing.assert_array_equal(resumed.delta.flat(), result.delta.flat())
    with pytest.raises(FileExistsError):
        run_micro_update(factory, a, cfg)
    with pytest.raises(ValueError, match="resume identity"):
        run_micro_update(factory, a, replace(cfg, resume=True, ell="length_normalized"))
    ids, labels = trainer.encode(cfg.tokenizer, a[0], device="cpu")
    replay = copy.deepcopy(template)
    apply_delta(replay, resumed.delta)
    template.eval()
    replay.eval()
    assert torch.equal(evaluation_score(template, ids, labels), evaluation_score(replay, ids, labels))


def test_micro_probe_kl_is_full_vocabulary_theta_to_base(micro_setup):
    template, factory, cfg, a, _ = micro_setup
    result = run_micro_update(factory, a, replace(cfg, probe_rows=a[:1], probe_block_size=1))
    updated = copy.deepcopy(template)
    apply_delta(updated, result.delta)
    template.eval()
    updated.eval()
    ids, labels = trainer.encode(cfg.tokenizer, a[0], device="cpu")
    mask = labels[0, 1:] != -100
    with torch.no_grad():
        base = template(ids).logits[0, :-1][mask].log_softmax(-1)
        current = updated(ids).logits[0, :-1][mask].log_softmax(-1)
        expected = (current.exp() * (current - base)).sum(-1).mean()
    assert result.manifest["policy_kl"] == pytest.approx(float(expected), abs=1e-9)
    assert result.manifest["policy_kl"] > 0


def test_micro_validates_checkpoint_truncation_units_and_interrupted_run(micro_setup, tmp_path, monkeypatch):
    template, factory, cfg, a, _ = micro_setup
    monkeypatch.setenv("AW_MAX_PROMPT_TOKENS", "17")
    with pytest.raises(ValueError, match="prompt exceeds cap"):
        run_micro_update(factory, a, replace(cfg, prompt_cap=1))
    assert os.environ["AW_MAX_PROMPT_TOKENS"] == "17"
    allowed = run_micro_update(factory, a, replace(cfg, steps=0, prompt_cap=1, prompt_side="tail",
                                                 allow_prompt_truncation=True))
    assert allowed.manifest["preprocessing"]["source_counts"][0]["chosen"]["prompt_tokens_dropped"] == 1
    with pytest.raises(ValueError, match="ell must match"):
        run_micro_update(factory, a, replace(cfg, eval_config={"ell": "length_normalized"}))
    with pytest.raises(ValueError, match="mixes source"):
        run_micro_update(factory, [{**a[0], "source_id": "other"}], cfg)

    def wrong_factory():
        model = factory()
        with torch.no_grad():
            model.base.weight.add_(1)
        return model

    with pytest.raises(ValueError, match="frozen checkpoint"):
        run_micro_update(wrong_factory, a, cfg)

    def wrong_dropout():
        model = factory()
        model.dropout.p = 0.9
        return model

    with pytest.raises(ValueError, match="model settings"):
        run_micro_update(wrong_dropout, a, cfg)
    run_root = tmp_path / "partial"
    (run_root / "failed").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="interrupted"):
        run_micro_update(factory, a, replace(cfg, output_dir=run_root, run_id="failed", resume=True))


def test_micro_dry_run_does_not_build_model(micro_setup, capsys):
    _, _, cfg, a, _ = micro_setup
    result = run_micro_update(lambda: pytest.fail("dry run constructed a model"), a, replace(cfg, dry_run=True))
    assert result.delta is None
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["new_teacher_tokens"] == manifest["max_rollouts"] == 0
    assert manifest["source_units"] == manifest["models"] == 1


def test_micro_rejects_mutating_buffers(micro_setup):
    template, _, cfg, a, _ = micro_setup

    def factory():
        model = copy.deepcopy(template)

        def increment_buffer(module, inputs, output):
            if module.training:
                module.offset.add_(1)

        model.register_forward_hook(increment_buffer)
        return model

    with pytest.raises(ValueError, match="mutated buffers"):
        run_micro_update(factory, a, cfg)


def test_peft_initialization_is_weights_only_loadable(tmp_path):
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(torch.nn.Sequential(torch.nn.Linear(3, 2)),
                           LoraConfig(r=2, lora_alpha=4, target_modules=["0"]))
    path = tmp_path / "peft-init.pt"
    save_initial_state(model, path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["model_settings"]["peft_config"]["default"]["peft_type"] == "LORA"
    assert state["layout_hash"] == layout_hash(model)
    assert len(state["trainables"]) == 2  # A and B, not just the initially nonzero factor.
    assert state["trainables_hash"] == tensor_state_hash(state["trainables"])


def test_initial_settings_exclude_runtime_modes_and_placement(tmp_path):
    class Config:
        def to_dict(self):
            return vars(self)
    model = TinyLoRALinear()
    model.config = Config()
    model.config.lora_alpha = 16
    model.config.text_config = {"hidden_size": 6, "use_cache": True, "torch_dtype": torch.float32}
    model.config.gradient_checkpointing = False
    model.config.gradient_checkpointing_kwargs = {"use_reentrant": True}
    model.config.use_cache = True
    model.config.is_training = True
    model.config.device = "cpu"
    model.config.device_map = {"": "cpu"}
    model.config.dtype = torch.float32
    path = tmp_path / "runtime.pt"
    save_initial_state(model, path)
    saved = torch.load(path, weights_only=True)
    assert saved["model_settings"]["model_config"] == {"lora_alpha": 16, "text_config": {"hidden_size": 6}}
    model.eval()
    model.config.gradient_checkpointing = True
    model.config.gradient_checkpointing_kwargs = {"use_reentrant": False}
    model.config.use_cache = False
    model.config.is_training = False
    model.config.device = "cuda:0"
    model.config.device_map = {"": "cuda:0"}
    model.config.dtype = torch.bfloat16
    model.config.text_config.update(use_cache=False, torch_dtype=torch.bfloat16)
    assert _model_settings(model) == saved["model_settings"]
    # Actual storage dtypes are still part of the tensor/layout identity.
    assert layout_hash(model.to(torch.bfloat16)) != saved["layout_hash"]
    model.dropout.p = 0.9
    assert _model_settings(model) != saved["model_settings"]


@pytest.mark.parametrize("strict", [False, True])
def test_micro_settings_diff_warning_or_strict_failure(micro_setup, strict):
    _, factory, cfg, rows, _ = micro_setup
    def changed_settings():
        model = factory()
        model.dropout.p = 0.5
        return model
    cfg = replace(cfg, strict_init_state=strict)
    if strict:
        with pytest.raises(ValueError, match="model settings differ.*strict-init-state"):
            run_micro_update(changed_settings, rows, cfg)
    else:
        with pytest.warns(UserWarning, match="model settings differ"):
            result = run_micro_update(changed_settings, rows, cfg)
        assert result.manifest["init_state_settings_diff"] == ["modules[4].settings"]
        assert result.manifest["model_settings"]["modules"][4]["settings"] == "p=0.5, inplace=False"


def test_initial_trainables_digest_detects_corruption(micro_setup):
    _, factory, cfg, rows, _ = micro_setup
    saved = torch.load(cfg.init_state_path, weights_only=True)
    saved["trainables"]["lora_A.weight"].add_(1)
    torch.save(saved, cfg.init_state_path)
    with pytest.raises(ValueError, match="trainables hash mismatch"):
        run_micro_update(factory, rows, cfg)


def test_default_builder_cannot_start_cuda_for_cpu_config(micro_setup):
    _, _, cfg, a, _ = micro_setup
    with pytest.raises(ValueError, match="supply a factory"):
        run_micro_update(None, a, cfg)


# This standalone test file owns its optional marker; no repository-wide pytest
# configuration is modified. Suppress only its registration warning at collection.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Unknown pytest.mark.gpu", category=pytest.PytestUnknownMarkWarning)
    gpu = pytest.mark.gpu


@gpu
@pytest.mark.skipif(os.environ.get("BEHAVIOR_TEST_GPU") != "1" or not torch.cuda.is_available(),
                    reason="optional GPU test; requires CUDA and BEHAVIOR_TEST_GPU=1")
def test_optional_cuda_delta_replay(tmp_path):
    model = torch.nn.Linear(8, 4, device="cuda", dtype=torch.bfloat16)
    initial, base = copy.deepcopy(model), snapshot(model)
    with torch.no_grad():
        model.weight.add_(0.01)
    save_delta(model, base, tmp_path / "cuda.safetensors")
    apply_delta(initial, load_delta(tmp_path / "cuda.safetensors"))
    assert torch.equal(model.weight, initial.weight)
