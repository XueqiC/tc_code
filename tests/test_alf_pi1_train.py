"""CPU oracles for the offline pi1 CLI and the actual shared trainer path."""
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]

from bfas.rtd.baselines.paper_data import TeacherRow
from bfas.rtd.baselines.paper_seeds import seed_training
from bfas.rtd.baselines.paper_train import PaperTrainer, hyperparameters
from bfas.rtd.baselines.pi1 import (PlainCEState, encode_teacher_turn, exposure_plan,
    boundary_contract, check_optimizer_step, load_bank, load_config, preflight_row, prepare_run)
from bfas.rtd.persistence import ComputeJournal
from test_baseline_run_training import Backend as BaseBackend, Tokenizer as BaseTokenizer
from tools import alf_pi1_train as cli


class Tokenizer(BaseTokenizer):
    all_special_tokens = ["<turn|>"]
    all_special_ids = [106]

    def convert_tokens_to_ids(self, token):
        return 106 if token == "<turn|>" else self.unk_token_id

    def encode(self, text, **kwargs):
        return [106] if text == "<turn|>" else super().encode(text, **kwargs)


class Backend(BaseBackend):
    def __init__(self):
        super().__init__()
        self.tokenizer = Tokenizer()


def config():
    return dict(load_config(ROOT/"configs/rtd/pi1_alfworld_k32.yaml"),
                training_seed=0, training_device="cpu")


def row(index=0, target="THOUGHT: find it.\nACTION: look"):
    return TeacherRow("package", "task", "parent", index,
        "OBSERVATION: a room.\nHistory: observation and previous command.\nAssistant:",
        target, "alfworld", False)


def make_trainer(directory, *, resume=False, backend=None, cfg=None):
    cfg = cfg or config()
    rows = [row(i, text) for i, text in enumerate(("A", "BCDEF", "GHI", "JKLMNO"))]
    backend = backend or Backend()
    hp = hyperparameters(cfg, "pi1_ce")
    costs = [len(encode_teacher_turn(backend.tokenizer, r, cfg["max_context_tokens"])["target_ids"])
             for r in rows]
    plan = exposure_plan(costs, cfg, cfg["training_seed"])
    identity = dict(config=cfg, seed=cfg["training_seed"], hyperparameters=hp,
                    bank=dict(sealed_manifest_sha256="synthetic"),
                    target_boundary=boundary_contract(backend.tokenizer))
    manifest = prepare_run(directory, identity, rows, plan, resume=resume)
    return PaperTrainer(backend, rows, cfg, "pi1_ce", directory,
                        ComputeJournal(directory/"compute.jsonl", cuda=False), manifest=manifest)


def test_observations_have_zero_labels_complete_teacher_turn_has_full_weight(tmp_path):
    t = make_trainer(tmp_path/"run")
    # The teacher's own prose is still supervised even if it mentions observations.
    r = row(target="THOUGHT: inspect.\nObservation: mentioned by teacher.\nACTION: look")
    encoded = encode_teacher_turn(t.backend.tokenizer, r, 2048)
    n = len(encoded["prompt_ids"])
    assert encoded["labels"][:n] == (-100,)*n
    assert encoded["labels"][n:] == tuple(r.target.encode()) + (106,)
    calls = []
    score = t.backend.score_tokens
    def recording(prompt, target, parameters, **kwargs):
        calls.append((prompt, target, kwargs))
        return score(prompt, target, parameters, **kwargs)
    t.backend.score_tokens = recording
    values, kinds = t.logprobs(r)
    assert calls[0][0] == tuple(r.prompt.encode())
    assert calls[0][1] == tuple(r.target.encode()) + (106,)  # full reply plus supervised boundary
    assert calls[0][2]["eos_token_id"] == 106
    assert not calls[0][2].get("truncated", False)
    assert kinds == ["teacher"]*(len(r.target)+1)
    assert t.scored_tokens == len(r.target)+1
    from bfas.rtd.baselines.paper_losses import span_ce
    assert torch.equal(span_ce(values, kinds, "pi1_ce"), -values.mean())


@pytest.mark.parametrize("tokens_per_update", [37, 512, 4096])
def test_both_exposure_endpoints_exact_in_supervised_tokens(tokens_per_update):
    cfg = dict(config(), supervised_tokens_per_update=tokens_per_update)
    costs = [21]*423+[342]  # 424 variable-length turns, 9,225 authored target tokens
    plan = exposure_plan(costs, cfg, 0)
    endpoints = {b["endpoint"]: b for b in plan["batches"] if b["endpoint"]}
    assert endpoints[3]["cumulative_tokens"] == 27675
    assert endpoints[10]["cumulative_tokens"] == 92250
    fixed = exposure_plan([c+1 for c in costs], cfg, 0)
    assert fixed["bank_supervised_tokens"] - plan["bank_supervised_tokens"] == len(costs) == 424
    assert fixed["target_tokens"] == [p * (sum(costs)+len(costs)) for p in cfg["exposure_passes"]]
    assert fixed["target_tokens"] == [28947, 96490]
    assert [b["cumulative_tokens"] for b in fixed["batches"] if b["endpoint"]] == fixed["target_tokens"]
    indices = [i for b in plan["batches"] for i in b["indices"]]
    for start in range(0, len(indices), len(costs)):
        assert sorted(indices[start:start+len(costs)]) == list(range(len(costs)))
    for b in plan["batches"]:
        assert b["supervised_tokens"] == sum(costs[i] for i in b["indices"])
        assert b["endpoint"] or tokens_per_update <= b["supervised_tokens"] < tokens_per_update+max(costs)


def test_seed_controls_order_reproducibly():
    def order(seed):
        return [i for b in exposure_plan([7, 31, 17]*20, config(), seed)["batches"] for i in b["indices"]]
    assert order(0) == order(0)
    assert order(1) == order(1)
    assert order(0) != order(1)


def test_preflight_and_context_limit_keep_unmasked_special_boundary():
    tokenizer, r = Tokenizer(), row()
    encoded = encode_teacher_turn(tokenizer, r, 2048)
    assert encoded["labels"][-1] == encoded["input_ids"][-1] == 106
    assert encoded["labels"][-1] != -100
    limit = len(encoded["input_ids"])
    assert preflight_row(tokenizer, r, limit)["last_8_label_ids"][-1] == 106
    with pytest.raises(ValueError, match="no truncation"):
        preflight_row(tokenizer, r, limit-1)
    tokenizer.all_special_ids = []
    with pytest.raises(ValueError, match="single special-token"):
        encode_teacher_turn(tokenizer, r, 2048)


def test_four_row_execution_check_uses_real_optimizer_loop(tmp_path):
    t = make_trainer(tmp_path/"check", cfg=dict(config(), exposure_passes=[1]))
    result = check_optimizer_step(t)
    assert result["student_commits"] == 1 and result["checked_rows"] == 4
    assert all(c["positions"] == 1 and c["loss_derivative"][0] < 0
               for c in result["boundary_loss_contributions"])
    assert t.backend.model.lora_logits[106] > 0
    assert t.manifest["target_boundary"]["token_id"] == 106


def test_other_paper_training_path_appends_native_boundary(tmp_path):
    t = make_trainer(tmp_path/"paper")
    t.method = "sad"
    prompt, target, kinds, eos = t.encode(row())
    assert target[-1] == eos == 106 and kinds[-1] != "observation"


def test_cli_refuses_existing_output_before_gpu_or_model(tmp_path, monkeypatch):
    out = tmp_path/"already-exists"
    out.mkdir()
    (out/"keep").write_text("unchanged")
    monkeypatch.setattr(cli, "select_device", lambda *a: pytest.fail("must refuse before GPU access"))
    with pytest.raises(FileExistsError, match="refusing existing output"):
        cli.main(["--seed", "0", "--output", str(out)])
    assert list(out.iterdir()) == [out/"keep"]


def test_actual_adamw_update_matches_token_mean_not_sequence_mean(tmp_path, monkeypatch):
    t = make_trainer(tmp_path/"run")
    with torch.no_grad():
        # P(A) lies between token mass 1/15 and sequence mass 1/4, so an
        # incorrect sequence mean even reverses Adam's update direction.
        t.backend.model.lora_logits[ord("A")] = 3.
    expected = t.backend.model.lora_logits.detach().clone().requires_grad_()
    hp = t.hp
    optimizer = torch.optim.AdamW([expected], lr=hp["learning_rate"], betas=tuple(hp["adam_betas"]),
        eps=hp["adam_epsilon"], weight_decay=hp["weight_decay"], foreach=False, fused=False)
    plan = exposure_plan([len(r.target)+1 for r in t.rows], t.config, 0)
    ids = [token for i in plan["batches"][0]["indices"] for token in (*t.rows[i].target.encode(), 106)]
    loss = -expected.log_softmax(0)[ids].mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([expected], hp["gradient_clip"])
    optimizer.step()
    original = PlainCEState.commit
    def stop(self, *args):
        original(self, *args)
        raise InterruptedError("test interruption")
    monkeypatch.setattr(PlainCEState, "commit", stop)
    with pytest.raises(InterruptedError):
        t.train()
    assert torch.allclose(t.backend.model.lora_logits, expected, rtol=1e-6, atol=1e-10)
    trace = json.loads((t.directory/"loss_trace.json").read_text())
    assert trace[0]["loss"] == pytest.approx(float(loss.detach()), rel=1e-6)
    assert t.backend.samples == 0  # no source generation, KL anchor, or teacher calls


def test_resume_after_durable_commit_recovers_endpoint_and_matches_uninterrupted(tmp_path, monkeypatch):
    cfg = dict(config(), supervised_tokens_per_update=17)
    seed_training(0)
    full = make_trainer(tmp_path/"full", cfg=cfg)
    expected_result = full.train()
    expected_random = (random.random(), np.random.random(), torch.rand(1))
    seed_training(0)
    interrupted = make_trainer(tmp_path/"interrupted", cfg=cfg)
    publish = PlainCEState.publish_endpoint
    def crash(self, passes):
        if passes == 3:
            raise InterruptedError("after optimizer state saved, before endpoint publication")
        publish(self, passes)
    monkeypatch.setattr(PlainCEState, "publish_endpoint", crash)
    with pytest.raises(InterruptedError):
        interrupted.train()
    assert not (interrupted.directory/"pass-3").exists()
    monkeypatch.setattr(PlainCEState, "publish_endpoint", publish)
    seed_training(91)  # recovery must restore the saved random states
    resumed = make_trainer(interrupted.directory, resume=True, cfg=cfg)
    assert resumed.train(resume=True) == expected_result
    assert torch.equal(resumed.backend.model.lora_logits, full.backend.model.lora_logits)
    actual_random = (random.random(), np.random.random(), torch.rand(1))
    assert actual_random[:2] == expected_random[:2]
    assert torch.equal(actual_random[2], expected_random[2])
    for passes in (3, 10):
        saved = json.loads((resumed.directory/f"pass-{passes}/manifest.json").read_text())
        assert saved["supervised_token_count"] == passes*19
        assert saved["exposure_check"]["matched"]
        assert saved["hyperparameters"]["optimizer"] == "AdamW"
        assert (resumed.directory/f"pass-{passes}/lora/tiny.pt").is_file()
        assert (resumed.directory/f"pass-{passes}/loss_trace.json").read_bytes() == (
            full.directory/f"pass-{passes}/loss_trace.json").read_bytes()
    # A completed resume is idempotent, with no further optimizer update.
    assert make_trainer(interrupted.directory, resume=True, cfg=cfg).train(resume=True) == expected_result


def test_resume_rejects_different_seed_and_damaged_state(tmp_path):
    t = make_trainer(tmp_path/"run")
    t.train()
    with pytest.raises(ValueError, match="resume identity differs"):
        make_trainer(t.directory, resume=True, cfg=dict(config(), training_seed=1))
    pointer = json.loads((t.directory/"resume/latest.json").read_text())
    (t.directory/"resume"/pointer["file"]).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="binding mismatch"):
        make_trainer(t.directory, resume=True).train(resume=True)


def test_full_uuid_visibility_order_and_verification(monkeypatch):
    uuid = "GPU-00000000-1111-2222-3333-444444444444"
    def smi(command, **kwargs):
        assert cli.os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert cli.os.environ["CUDA_VISIBLE_DEVICES"] == uuid
        assert "--id="+uuid in command
        return SimpleNamespace(stdout=uuid+", 00000000:01:00.0\n")
    monkeypatch.setattr(cli.subprocess, "run", smi)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    assert cli.select_device(uuid)["uuid"] == uuid
    with pytest.raises(ValueError, match="full nvidia-smi"):
        cli.select_device("0")
    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="wrong, pci\n"))
    with pytest.raises(ValueError, match="did not verify"):
        cli.select_device(uuid)


def test_cuda_uuid_mismatch_rejected_before_allocator_or_model(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid="wrong"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: pytest.fail("UUID must be checked first"))
    with pytest.raises(ValueError, match="UUID differs"):
        cli.verify_cuda_device(dict(uuid="GPU-expected"), config())


def test_sealed_bank_loader_uses_full_replies_and_keeps_bank_unchanged():
    cfg = config()
    bank = Path(os.environ.get("ALF_PI1_BANK", ROOT/cfg["bank"]))
    if not bank.is_dir():
        pytest.skip("real bank unavailable; set ALF_PI1_BANK for read-only validation")
    from bfas.rtd.persistence import file_hash
    before = file_hash(bank/"sealed/manifest.json")
    rows, identity = load_bank(bank, cfg, lambda request, history: "rendered context")
    assert len(rows) == 424
    assert len({r.package_id for r in rows}) == 30
    assert all("ACTION:" in r.target for r in rows)
    assert any("THOUGHT:" in r.target for r in rows)
    assert identity["sealed_manifest_sha256"] == before == file_hash(bank/"sealed/manifest.json")


def test_config_resolves_historical_recipe_and_rejects_drift(tmp_path):
    cfg = config()
    hp = hyperparameters(cfg, "pi1_ce")
    assert hp["learning_rate"] == 1e-5 and hp["gradient_clip"] == 1
    assert hp["weight_decay"] == .01 and hp["lora_dropout"] == 0
    path = tmp_path/"changed.yaml"
    path.write_text((ROOT/"configs/rtd/pi1_alfworld_k32.yaml").read_text().replace("1.0e-5", "1.0e-4"))
    with pytest.raises(ValueError, match="recipe differs: learning_rate"):
        load_config(path)


def test_real_peft_exports_load_through_evaluation_adapter_identity(tmp_path):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    from bfas.rtd.benchmarks.alfworld_identity import model_identity
    from bfas.rtd.checkpointing import enable_gradient_checkpointing
    from bfas.rtd.functional_step import lora_parameters
    from bfas.rtd.runtime import HFGenerateBackend
    torch.set_num_threads(1)
    seed_training(0)
    cfg = config()
    model_config = LlamaConfig(vocab_size=256, hidden_size=8, intermediate_size=16,
                              num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2)
    model_config._attn_implementation = "eager"
    base = LlamaForCausalLM(model_config)
    base.save_pretrained(tmp_path/"base")
    base.config._name_or_path = str(tmp_path/"base")
    model = get_peft_model(base, LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"], target_modules=cfg["lora_target_modules"], task_type="CAUSAL_LM"))
    model.eval()
    enable_gradient_checkpointing(model)
    backend = HFGenerateBackend(model, Tokenizer(), base_checkpoint_hash="tiny-base",
        harness_hash="tiny-harness", tokenizer_hash="tiny-tokenizer", max_context_tokens=32768)
    backend.score_position_chunk_size = 2
    t = make_trainer(tmp_path/"run", backend=backend)
    t.train()
    for passes in (3, 10):
        checkpoint = t.directory/f"pass-{passes}"
        receipt = json.loads((checkpoint/"manifest.json").read_text())
        identity = model_identity(tmp_path/"base", checkpoint)
        assert identity["loading"] == "adapter_overlay"
        assert identity["adapter_hash"] == receipt["adapter_hash"]
        loaded = PeftModel.from_pretrained(LlamaForCausalLM.from_pretrained(tmp_path/"base",
            local_files_only=True), checkpoint/"lora", local_files_only=True, is_trainable=True)
        if passes == 10:
            for name, parameter in lora_parameters(loaded).items():
                assert torch.equal(parameter, t.parameters[name])
    assert not torch.cuda.is_initialized()
