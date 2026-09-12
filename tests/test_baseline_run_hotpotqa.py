"""HotpotQA paper integration: native ReAct spans, CPU audits and campaigns."""
from contextlib import contextmanager
from dataclasses import asdict
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]

from bfas import hotpotqa as hp
from bfas.rtd.bank_build import seal_v11, state_record
from bfas.rtd.baselines import paper_evaluation
from bfas.rtd.baselines.paper_data import EMPTY_THOUGHT, STUDENT, TeacherRow
from bfas.rtd.baselines.paper_losses import segment_spans, span_ce, token_kinds
from bfas.rtd.persistence import digest, file_hash, tree_hash
from bfas.rtd.transport import FullState
from tools import baseline_run


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    def forbidden(*args, **kwargs):
        pytest.fail("HotpotQA CPU checks must not use network, CUDA, or model weights")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    from bfas.rtd import runtime
    monkeypatch.setattr(runtime, "load_backend", forbidden)


def native_prompt(prefill):
    return "<bos><|turn>user\nQuestion: Synthetic?\n"+prefill+"<turn|>\n<|turn>model\n"+EMPTY_THOUGHT


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return dict(input_ids=list(text.encode()), offset_mapping=[(i, i+1) for i in range(len(text))])


def test_react_spans_and_losses_mask_wikipedia_and_weight_finish():
    text = ("I need a page.\nAction 1: search[Alpha]\nObservation 1: Wikipedia text.\n"
            "lookup[quoted example]\nThought 2: Find the detail.\nAction 2: lookup[Beta]\n"
            "Observation 2: More Wikipedia.\nThought 3: I can answer.\nAction 3: finish[Gamma]")
    spans = segment_spans(text, benchmark="hotpotqa", prompt=native_prompt("Thought 1:"), final_step=True)
    assert spans[0].start == 0 and spans[-1].end == len(text)
    assert all(a.end == b.start for a, b in zip(spans, spans[1:]))
    def at(word):
        return next(s.kind for s in spans if s.start <= text.index(word) < s.end)
    assert [at(word) for word in ("I need", "Action 1:", "search[", "Wikipedia text",
        "lookup[quoted", "Thought 2:", "Action 2:", "More Wikipedia", "Thought 3:", "Action 3:", "Gamma")] == [
        "reason", "action", "action", "observation", "observation", "reason", "action",
        "observation", "reason", "final", "final"]
    ids, kinds = token_kinds(CharacterTokenizer(), text, benchmark="hotpotqa", prompt=native_prompt("Thought 1:"))
    assert ids == tuple(text.encode())
    values = torch.full((len(ids),), -2., requires_grad=True)
    smartad = span_ce(values, kinds, "smartad")
    smartad.backward()
    counts = {kind: kinds.count(kind) for kind in set(kinds)}
    generated = len(kinds)-counts["observation"]
    assert smartad.item() == pytest.approx(2*(counts["reason"]+1.5*counts["action"]+2*counts["final"])/generated)
    assert values.grad[kinds.index("observation")].item() == 0
    assert values.grad[kinds.index("final")].item() == pytest.approx(-2/generated)
    scores = torch.tensor([-2. if k == "reason" else -6. if k != "observation" else -999. for k in kinds])
    assert span_ce(scores, kinds, "sad").item() == pytest.approx(4.)


@pytest.mark.parametrize("prefill,text,expected", [
    ("Thought 4:", "The page suggests Alpha.", "reason"),
    ("Thought 4: The page suggests Alpha.\nAction 4:", "Alpha", "action"),
    ("Thought 4:", "search[Alpha]", "action"),
    ("Thought 4:", "lookup[Beta]", "action"),
    ("Thought 4:", "finish[Gamma]", "final"),
    ("Action 4:", "finish[Gamma]<eos>", "final"),
    ("Action 4:", "  FiNiSh[Gamma]<turn|>\n", "final"),
    ("Thought 4:", "Action 4: finish[Gamma]", "final"),
    ("Action 4:", "Thought 4: Still thinking.", "reason"),
    ("Thought 4:", "I should finish[Gamma] after checking.", "reason"),
    ("Thought 4:", "Observation 4: finish[quoted text]", "observation"),
])
@pytest.mark.parametrize("final_step", [False, True])
def test_hotpotqa_prefill_standalone_actions_and_terminal_row(prefill, text, expected, final_step):
    _, kinds = token_kinds(CharacterTokenizer(), text, benchmark="hotpotqa", final_step=final_step,
                          prompt=native_prompt(prefill))
    assert set(kinds) == {expected}


def test_hotpotqa_token_crossing_observation_boundary_is_masked():
    text = "Action 1: search[Alpha]\nObservation 1: Wikipedia"
    boundary = text.index("Observation")
    tokenizer = lambda *a, **kw: dict(input_ids=[1, 2], offset_mapping=[(0, boundary-1), (boundary-1, len(text))])
    assert token_kinds(tokenizer, text, benchmark="hotpotqa")[1] == ["action", "observation"]


def synthetic_bank(tmp_path):
    records, payloads = [], {}
    for name, cost, usable in [("good", 20, True), ("bad", 3, False), ("good2", 10, True)]:
        qid = digest(name)
        state = FullState.create(dict(task_id=name), [{"role": "user", "content": "Thought 1:"}],
                                 native_prompt("Thought 1:"), digest(name))
        records.append(state_record(qid, state, "hotpotqa_demo_episode", cost, "exact",
                                    unavailable=None if usable else "failed"))
        payloads[qid] = dict(cost=cost, cost_confidence="exact",
            historical_response=dict(teacher="openai/gpt-5.6-luna", verified=usable, tokens_spent=cost),
            provenance=dict(task_id=name, rendering_student=STUDENT),
            behaviors=[dict(state=asdict(state), text="I know it.\nAction 1: finish[Alpha]")] if usable else [])
    bank = tmp_path/"bank"
    seal_v11(bank, records, payloads, benchmark="hotpotqa", student=STUDENT, public={}, audit={}, inputs={})
    return bank


def test_prepare_only_hotpotqa_synthetic_bank_preserves_prompts_and_charges_failures(tmp_path, monkeypatch):
    bank = synthetic_bank(tmp_path)
    before = tree_hash(bank)
    def forbidden(*args, **kwargs):
        pytest.fail("prepare-only must not start workers or evaluation")
    monkeypatch.setattr(baseline_run, "run_worker", forbidden)
    monkeypatch.setattr(paper_evaluation, "evaluate_run", forbidden)
    monkeypatch.setattr(baseline_run.subprocess, "Popen", forbidden)
    out = tmp_path/"audit"
    assert baseline_run.main(["--method", "smartad", "--benchmark", "hotpotqa", "--bank", str(bank),
        "--budget-tokens", "0,33", "--run-dir", str(out), "--prepare-only"]) == 0
    empty = json.loads((tmp_path/"audit_B0/manifest.json").read_text())
    assert empty["positive_rows"] == empty["teacher_tokens_charged"] == 0
    out = tmp_path/"audit_B33"
    manifest = json.loads((out/"manifest.json").read_text())
    assert manifest["status"] == "prepared"
    assert manifest["B"] == manifest["teacher_tokens_charged"] == 33
    assert manifest["usable_cost_basis"] == 30
    assert len(manifest["charges"]) == 3
    assert manifest["positive_rows"] == manifest["purchased_usable_packages"] == 2
    assert sum(c["tokens"] for c in manifest["charges"] if not c["usable"]) == 3
    assert manifest["gpu_hours"] == manifest["new_teacher_calls"] == manifest["new_teacher_tokens"] == 0
    assert manifest["config"]["max_action_tokens_by_benchmark"] == {"hotpotqa": {"agent_action": 100}}
    assert manifest["evaluation_protocol"]["tasks"] == 500
    for name in ("prompts/hotpotqa_react_6shot.txt", "configs/hotpotqa_support_split.json",
                 "configs/hotpotqa_eval_split.json", "src/bfas/rtd/benchmarks/hotpotqa_evaluation.py",
                 "src/bfas/rtd/benchmarks/hotpotqa_identity.py", "tools/hotpotqa_eval.py", "src/bfas/hotpotqa.py"):
        assert manifest["source_hashes"][name] == file_hash(ROOT/name)
    assert manifest["hyperparameters"]["lora_rank"] == 16
    assert manifest["hyperparameters"]["kang"]["n"] is None
    rows = json.loads((out/"purchased_rows.json").read_text())
    assert all(r["prompt"] == native_prompt("Thought 1:") for r in rows)
    assert all(r["target"] == "I know it.\nAction 1: finish[Alpha]" for r in rows)
    assert tree_hash(bank) == before
    assert not (out/"metrics.json").exists()


def test_hotpotqa_registry_config_and_training_cap(tmp_path):
    from bfas.rtd.benchmarks.config import benchmark_protocol
    from bfas.rtd.benchmarks.registry import get_benchmark
    from bfas.rtd.baselines.paper_train import PaperTrainer
    from bfas.rtd.baselines.paper_losses import hotpotqa_prefill_kind
    config = yaml.safe_load((ROOT/"configs/rtd/v1_1_hotpotqa_luna.yaml").read_text())
    assert all(config[k] == v for k, v in benchmark_protocol("hotpotqa").items())
    providers = get_benchmark(config)
    backend = SimpleNamespace(action_caps={"agent_action": 100}, max_action_tokens=512,
        tokenizer=SimpleNamespace(encode=lambda text, **kw: tuple(text.encode())))
    backend.action_limit = lambda category: providers.action_limit(backend, category)
    prompts = []
    def sample(prompt, *args):
        assert backend.max_action_tokens == 100
        prompts.append(prompt)
        return "sample"
    backend.sample_action = sample
    backend.checked_score_action = lambda *a, **kw: None
    trainer = object.__new__(PaperTrainer)
    trainer.backend, trainer.parameters, trainer.rng = backend, {}, None
    trainer.journal = SimpleNamespace(append=lambda *a, **kw: None)
    row = TeacherRow("pkg", "5a7f9abcdef", "parent", 0, native_prompt("Action 3:"), "finish[Alpha]", "hotpotqa", True)
    assert trainer.sample(row) == "sample"
    assert prompts == [row.prompt] and backend.max_action_tokens == 512
    # PaperTrainer.encode must forward the row prompt, not just its final flag.
    backend.tokenizer = CharacterTokenizer()
    backend.tokenizer.encode = lambda text, **kw: tuple(text.encode())
    backend.tokenizer.eos_token_id = 255
    backend.student_config = {}
    trainer.encoded_rows, trainer.encoded_prompts = {}, {}
    trainer.config = dict(max_context_tokens=1000)
    retry = TeacherRow("pkg", "id", "parent", 1, row.prompt, "Alpha", "hotpotqa", True)
    assert set(trainer.encode(retry)[2]) == {"action"}
    assert hotpotqa_prefill_kind(native_prompt("Thought 3:")) == "reason"


@pytest.mark.parametrize("failure", [None, "renderer", "episode", "gold", "f1", "inventory"])
def test_hotpotqa_adapter_campaign_em_f1_and_cleanup(tmp_path, monkeypatch, failure):
    from bfas import run
    from bfas.adapters import hotpotqa
    from bfas.rtd.benchmarks import hotpotqa_identity
    questions = [dict(_id=f"dev-{i}", question=f"Synthetic {i}?", answer="alpha beta", type="bridge") for i in range(500)]
    expected = dict(task_ids=[q["_id"] for q in questions], questions_hash=digest(questions))
    harness = dict(expected=expected, environment=dict(offline=True, snapshots={"cache.json": "hash"}))
    events = []
    config = {"benchmark": "hotpotqa"}
    def identity(root, given):
        assert root == ROOT and given == config
        return harness
    monkeypatch.setattr(hotpotqa_identity, "evaluation_harness_identity", identity)
    monkeypatch.setenv("BFAS_HOTPOTQA_TEACHER_POOL", "/must/not/import")
    class Adapter:
        def __init__(self, seed, port, offline):
            assert (seed, port, offline) == (0, 8930, True)
        def prepare_renderer(self, model):
            assert model == str(tmp_path/"merged")
            assert os.environ["BFAS_HOTPOTQA_TEACHER_POOL"] == ""
            events.append("renderer")
            if failure == "renderer":
                raise RuntimeError("renderer failed")
        def evaluate(self, model, out):
            events.append("evaluate")
            if failure == "episode":
                raise hp.OfflineCacheMiss("missing snapshot")
            rows = [dict(task_id=q["_id"], question=q["question"], gold=q["answer"], category=q["type"],
                prediction="alpha", em=0., f1=2/3, steps=1, model_calls=1, finished=True,
                prompt_version=hp.PROMPT_VERSION, error=None) for q in questions]
            cfg = dict(temperature=0., max_steps=7, max_tokens=100, task_ids=expected["task_ids"],
                       prompt_version=hp.PROMPT_VERSION, wiki_version=hp.WIKI_VERSION)
            metrics = hp.compute_metrics(rows, cfg) | dict(complete=True, requested_n=500, offline=True)
            if failure == "gold":
                rows[0]["gold"] = "changed"
            elif failure == "f1":
                metrics["f1"] = 1.
            elif failure == "inventory":
                rows.pop()
            (out/"records.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            return metrics
        def release_policy(self):
            events.append("release")
    @contextmanager
    def serve(adapter, model, gpu, port, log):
        assert gpu == "" and port == 8930
        events.append("serve")
        try:
            yield
        finally:
            events.append("close")
    monkeypatch.setattr(hotpotqa, "HotpotQAAdapter", Adapter)
    monkeypatch.setattr(run, "serving_lane", serve)
    def campaign():
        return paper_evaluation.run_hotpotqa(ROOT, tmp_path/"merged", tmp_path/"eval",
                                            kang=False, port=8930, config=config)
    if failure:
        with pytest.raises((ValueError, RuntimeError, hp.OfflineCacheMiss)):
            campaign()
    else:
        result = campaign()
        assert result["tasks"] == result["validation"]["n"] == 500
        assert result["overall_accuracy_percent"] == result["em"] == 0
        assert result["overall_metric"] == "em" and result["f1"] == pytest.approx(2/3)
        assert result["expected"] == expected and result["harness_identity"] == harness
    assert events[-1] == "release"
    assert os.environ["BFAS_HOTPOTQA_TEACHER_POOL"] == "/must/not/import"
    if failure != "renderer":
        assert events == ["renderer", "serve", "evaluate", "close", "release"]
        assert json.loads((tmp_path/"eval/gpu_usage.json").read_text())["gpu_seconds"] >= 0


@pytest.mark.parametrize("method", ["smartad", "kang"])
def test_hotpotqa_evaluate_dispatch_keeps_checkpoint_receipts_and_never_mislabels_sag(tmp_path, monkeypatch, method):
    from bfas.rtd import evaluation, hardware
    model, checkpoint = tmp_path/"base", tmp_path/"checkpoint"
    model.mkdir()
    checkpoint.mkdir()
    (model/"config.json").write_text("{}")
    (checkpoint/"adapter_config.json").write_text("{}")
    manifest = dict(config=dict(benchmark="hotpotqa"), benchmark="hotpotqa", method=method,
        checkpoint_sha256=tree_hash(checkpoint), base_checkpoint_hash=tree_hash(model),
        model_path=str(model), hardware={"hard": "cpu-stub"}, teacher_tokens_charged=33, B=40, port=8930)
    monkeypatch.setattr(hardware, "hardware_identity", lambda: manifest["hardware"])
    monkeypatch.setattr(evaluation, "_flatten_adapter", lambda *a: None)
    def export(command, **kwargs):
        assert command[command.index("--model")+1] == "google/gemma-4-12B-it"
        assert command[command.index("--adapter")+1] == str(tmp_path/"export/adapter")
        assert Path(command[command.index("--adapter")+1]).is_absolute()
        assert kwargs["env"]["HF_HUB_OFFLINE"] == kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        merged = tmp_path/"export/hub_merged"
        merged.mkdir(parents=True)
        (merged/"config.json").write_text("{}")
    monkeypatch.setattr(paper_evaluation.subprocess, "run", export)
    calls = []
    def campaign(root, merged, out, **kwargs):
        calls.append(kwargs)
        assert out.name == "official" and kwargs["kang"] is False and kwargs["config"] == manifest["config"]
        return dict(tasks=500, em=0., f1=.5, overall_accuracy_percent=0., overall_metric="em")
    monkeypatch.setattr(paper_evaluation, "run_hotpotqa", campaign)
    result = paper_evaluation.evaluate_run(ROOT, tmp_path, manifest)
    assert len(calls) == 1
    assert result["teacher_tokens_charged"] == 33 and result["B"] == 40
    assert result["checkpoint_sha256"] == manifest["checkpoint_sha256"]
    assert result["export_sha256"] == tree_hash(tmp_path/"export/hub_merged")
    assert result["evaluation_mode"] == "official_single_sample"
    assert not (tmp_path/"kang_metrics.json").exists()
    if method == "kang":
        assert result["kang_self_consistency"]["status"] == "unsupported"


def test_hotpotqa_rejects_partial_smoke_and_sag(tmp_path):
    with pytest.raises(ValueError, match="500-question"):
        paper_evaluation.protocol("hotpotqa", smoke=True)
    with pytest.raises(NotImplementedError, match="complete sampled ReAct episodes"):
        paper_evaluation.run_hotpotqa(ROOT, tmp_path/"merged", tmp_path/"out", kang=True, port=8930, config={})
