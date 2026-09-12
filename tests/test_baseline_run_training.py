"""Tiny local backend exercises real trainer, no downloads, API, or CUDA."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"src"), str(ROOT)]
from bfas.rtd.baselines.paper_data import TeacherRow
from bfas.rtd.baselines.paper_train import PaperTrainer
from bfas.rtd.baselines.paper_losses import SmallDiscriminator
from bfas.rtd.persistence import ComputeJournal


class Tokenizer:
    eos_token_id = 255
    unk_token_id = 254
    def encode(self, text, **kwargs):
        return list(text.encode())
    def __call__(self, text, **kwargs):
        return dict(input_ids=self.encode(text), offset_mapping=[(i, i+1) for i in range(len(text))])
    def decode(self, ids, **kwargs):
        return "".join(chr(i) if i != 255 else "!" for i in ids)
    def save_pretrained(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory)/"tokenizer.json").write_text("{}")


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_logits = torch.nn.Parameter(torch.zeros(256))
    def save_pretrained(self, directory, **kwargs):
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(directory)/"tiny.pt")


class Backend:
    def __init__(self):
        self.model, self.tokenizer = Model(), Tokenizer()
        self.student_config = {}  # tiny tokenizer, ordinary EOS for this fake
        self.samples = 0
    def identity(self, parameters):
        return "tiny"
    def action_limit(self, category):
        return nullcontext()
    def sample_action(self, prompt, parameters, generator):
        self.samples += 1
        return SimpleNamespace(action_ids=(97+self.samples % 2, 255), prompt_ids=self.tokenizer.encode(prompt))
    def score_tokens(self, prompt, ids, parameters, *, return_details=False, **kwargs):
        values = parameters["lora_logits"].log_softmax(0)[torch.tensor(ids)]
        return (values.sum(), values, {}) if return_details else values.sum()
    def score_action(self, action, parameters, **kwargs):
        return self.score_tokens(action.prompt_ids, action.action_ids, parameters, **kwargs)
    def checked_score_action(self, action, parameters, **kwargs):
        return self.score_action(action, parameters), {}


def trainer(tmp_path, method):
    config = dict(student="tiny", lora_rank=16, lora_alpha=32,
                  lora_target_modules=["q_proj"], max_context_tokens=256)
    rows = [TeacherRow("package", "task", "parent", 0, "prompt", "T", "bfcl", True)]
    return PaperTrainer(Backend(), rows, config, method, tmp_path, ComputeJournal(tmp_path/"compute.jsonl", cuda=False))


@pytest.mark.parametrize("method", ["sad", "smartad", "kang"])
def test_full_24_commit_training_matches_step_rule_and_saves(tmp_path, method):
    t = trainer(tmp_path, method)
    before = t.backend.model.lora_logits.detach().clone()
    result = t.train()
    assert result["student_commits"] == 24
    assert t.backend.model.lora_logits[ord("T")] > before[ord("T")]
    events = [json.loads(line) for line in (tmp_path/"compute.jsonl").read_text().splitlines()]
    assert sum(e["kind"] == "preconditioner" for e in events) == 2
    assert sum(e["kind"] == "student_commit" for e in events) == 24
    assert (tmp_path/"checkpoint/lora/tiny.pt").exists()


def test_gad_missing_teacher_skips_before_sampling(tmp_path):
    t = trainer(tmp_path, "gad")
    assert t.gad_gradient("unknown on-policy prompt", {}) is None
    assert t.backend.samples == 0


def test_gad_real_discriminator_and_advantage_reach_student_only(tmp_path):
    t = trainer(tmp_path, "gad")
    torch.manual_seed(0)
    t.discriminator = SmallDiscriminator(width=4, hidden=6)
    t.discriminator_optimizer = torch.optim.AdamW(t.discriminator.parameters(), lr=.01)
    before = [p.detach().clone() for p in t.discriminator.parameters()]
    gradient, loss = t.gad_gradient("prompt", {"prompt": t.rows[0]})
    assert t.backend.samples == 4
    assert torch.isfinite(gradient["lora_logits"]).all()
    assert gradient["lora_logits"].abs().sum() > 0
    assert any(not torch.equal(old, now) for old, now in zip(before, t.discriminator.parameters()))
    assert all(p.grad is None for p in t.discriminator.parameters())
    assert t.backend.model.lora_logits.grad is None
    # Computing the policy gradient never commits a second hidden student step.
    assert torch.equal(t.backend.model.lora_logits, torch.zeros(256))


def test_gad_24_student_commits_include_four_warmup_and_four_pg_rounds(tmp_path, monkeypatch):
    t = trainer(tmp_path, "gad")
    seen = []
    def gradient(prompt, teachers, *, warmup=False):
        seen.append(warmup)
        return {n: torch.ones_like(p) for n, p in t.parameters.items()}, 1.
    monkeypatch.setattr(t, "gad_gradient", gradient)
    result = t.train()
    assert result["student_commits"] == 24
    assert seen == [True]*(4*40)+[False]*(20*40)
    events = [json.loads(line) for line in (tmp_path/"compute.jsonl").read_text().splitlines()]
    rounds = [e["gad_round"] for e in events if e["kind"] == "student_commit"]
    assert rounds == [None]*4+[1]*5+[2]*5+[3]*5+[4]*5


def test_smoke_steps_cache_rendering_and_log_progress(tmp_path, monkeypatch, capsys):
    from bfas.rtd.baselines.paper_train import hyperparameters
    t = trainer(tmp_path, "smartad")
    t.config["smoke"] = True
    t.hp = hyperparameters(t.config, "smartad")
    encode = t.backend.tokenizer.__class__.__call__
    rendered = []
    def record(self, text, **kwargs):
        rendered.append(text)
        return encode(self, text, **kwargs)
    monkeypatch.setattr(t.backend.tokenizer.__class__, "__call__", record)
    result = t.train()
    assert result["student_commits"] == 2 and len(result["losses"]) == 2
    assert rendered == ["T"]  # Shared by selection and every repeated training slot.
    batches = json.loads((tmp_path/"exposure_schedule.json").read_text())["batches"]
    assert list(map(len, batches)) == [2, 2]
    events = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert {"selection", "rendering", "training", "preconditioner"} <= {e["stage"] for e in events}
    assert all(e["timestamp"].endswith("+00:00") for e in events)
    commits = [e for e in events if e["event"] == "commit"]
    assert len(commits) == 2
    assert all(e["tokens_per_second"] > 0 and e["tokens"] == 4 for e in commits)


def test_trainer_rejects_cpu_model_when_cuda_requested(tmp_path):
    t = trainer(tmp_path, "smartad")
    config = dict(t.config, training_device="cuda:0")
    with pytest.raises(ValueError, match="device cpu differs from scoring device cuda:0"):
        PaperTrainer(t.backend, t.rows, config, "smartad", tmp_path, t.journal)


def test_baseline_journal_does_not_collect_heap_per_forward(tmp_path, monkeypatch):
    from bfas.rtd import memory
    def forbidden(*args):
        pytest.fail("per-forward gc/cache clearing causes the CPU stall")
    monkeypatch.setattr(memory, "release_device_cache", forbidden)
    journal = ComputeJournal(tmp_path/"no_gc.jsonl", cuda=False, release_phase_cache=False)
    with journal.measure_phase("teacher_forced_forward"):
        pass
    assert journal.events[-1]["kind"] == "subphase_memory"


class TinyCausalModel(torch.nn.Module):
    """Exercise actual functional head capture/CE/checkpointing without weights."""
    def __init__(self, *, softcap=None):
        super().__init__()
        self.embedding = torch.nn.Embedding(17, 5, dtype=torch.float64)
        self.embedding.weight.requires_grad_(False)
        self.lora_body = torch.nn.Linear(5, 5, bias=False, dtype=torch.float64)
        self.lora_head = torch.nn.Linear(5, 17, bias=False, dtype=torch.float64)
        self.config = SimpleNamespace(final_logit_softcapping=softcap)

    def get_output_embeddings(self):
        return self.lora_head

    def forward(self, input_ids, **kwargs):
        from bfas.rtd.source_scoring import cap_logits
        h = self.lora_body(self.embedding(input_ids)).tanh()
        return SimpleNamespace(logits=cap_logits(self.lora_head(h), self.config.final_logit_softcapping))


@pytest.mark.parametrize("softcap", [None, 2.])
@pytest.mark.parametrize("truncated", [False, True])
def test_chunked_reference_and_training_match_dense_without_cpu_transfer(monkeypatch, softcap, truncated):
    from bfas.rtd.runtime import HFGenerateBackend
    from bfas.rtd.functional_step import gradients, lora_parameters, snapshot
    model = TinyCausalModel(softcap=softcap).eval()
    backend = HFGenerateBackend(model, Tokenizer(), base_checkpoint_hash="base",
        harness_hash="harness", tokenizer_hash="tokenizer", max_context_tokens=128)
    params = snapshot(lora_parameters(model))
    with torch.no_grad():
        for parameter in params.values():
            parameter.add_(.1)  # Checkpoint replay must use the bound scoring snapshot.
    prompt, action = (1, 2, 3), (4, 5, 6, 7, 8, 9, 10, 16)
    eos = 15 if truncated else 16
    dense, dense_values, _ = backend.score_tokens(prompt, action, params, eos_token_id=eos,
                                                 truncated=truncated, return_details=True)
    dense_grad = gradients(dense, params)
    backend.score_position_chunk_size = 3
    sizes = []
    def inspect_head(module, args, output):
        assert args[0].device == output.device == next(model.parameters()).device
        sizes.append(output.shape[-2])
    hook = model.lora_head.register_forward_hook(inspect_head)
    def forbidden(*a, **kw):
        pytest.fail("reference/scoring tensors must never be materialized on CPU")
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    try:
        # Selection/reference forward has no graph and only bounded logits.
        with torch.no_grad():
            _, reference, metadata = backend.score_tokens(prompt, action, params, eos_token_id=eos,
                truncated=truncated, return_details=True)
        assert sizes == [3, 3, 2]
        assert metadata["logical_device"] == "cpu"  # Explicit CPU oracle; production requires cuda:0.
        sizes.clear()
        score, values, _ = backend.score_tokens(prompt, action, params, eos_token_id=eos,
            truncated=truncated, return_details=True)
        actual_grad = gradients(score, params)
        assert max(sizes) <= 3
        assert torch.allclose(reference, dense_values, atol=1e-12)
        assert torch.allclose(values, dense_values, atol=1e-12)
        for name in params:
            assert actual_grad[name].device == params[name].device
            assert torch.allclose(actual_grad[name], dense_grad[name], atol=1e-12)
    finally:
        hook.remove()


def test_reference_device_mismatch_fails_before_softmax(monkeypatch):
    from bfas.rtd.source_scoring import _position_vjps, require_device
    head = torch.nn.Linear(5, 17, bias=False)
    current = dict(head.named_parameters())
    frozen = {n: p.detach().clone() for n, p in current.items()}
    hidden = torch.zeros(2, 5)
    labels = torch.zeros(2, dtype=torch.long)
    with pytest.raises(ValueError, match="reference_hidden device meta"):
        _position_vjps(head, current, frozen, hidden, hidden.to("meta"), labels)
    with pytest.raises(ValueError, match="reference_logits device cpu.*cuda:0"):
        require_device(torch.device("cuda:0"), reference_logits=torch.zeros(2, 17))
    # A misbehaving head is rejected immediately after projection, before CE.
    original = head.forward
    monkeypatch.setattr(head, "forward", lambda h: original(h).to("meta"))
    with pytest.raises(ValueError, match="reference_logits device meta"):
        _position_vjps(head, current, frozen, hidden, hidden, labels)


def test_frozen_soft_reference_vjp_stays_on_device(monkeypatch):
    from bfas.rtd.source_scoring import _position_vjps
    head = torch.nn.Linear(5, 17, bias=False, dtype=torch.float64)
    current = dict(head.named_parameters())
    frozen = {n: p.detach().clone()+.1 for n, p in current.items()}
    h = torch.randn(3, 5, dtype=torch.float64)
    ref = h+.1
    labels = torch.tensor([1, 2, 3])
    def forbidden(*a, **kw):
        pytest.fail("frozen-reference vocabulary tensors must remain on the scoring device")
    monkeypatch.setattr(torch.Tensor, "cpu", forbidden)
    devices = []
    hook = head.register_forward_hook(lambda module, args, output: devices.append(output.device))
    try:
        hard, soft, kl = _position_vjps(head, current, frozen, h, ref, labels)
    finally:
        hook.remove()
    assert devices == [h.device, h.device]
    assert kl >= 0
    assert all(g.device == h.device and torch.isfinite(g).all() for g in (*hard, *soft) if g is not None)


def test_generation_score_reduction_is_chunked_and_matches_native(monkeypatch):
    from bfas.rtd.source_scoring import generated_token_scores
    scores = tuple(torch.randn(1, 17, dtype=torch.float64) for _ in range(8))
    ids = (1, 2, 3, 4, 5, 6, 7, 8)
    expected = [float(score.log_softmax(-1)[0, token]) for score, token in zip(scores, ids)]
    original = torch.nn.functional.cross_entropy
    sizes = []
    def ce(logits, labels, **kwargs):
        assert logits.device == labels.device == torch.device("cpu")
        sizes.append(len(logits))
        return original(logits, labels, **kwargs)
    monkeypatch.setattr(torch.nn.functional, "cross_entropy", ce)
    actual = generated_token_scores(scores, ids, device=torch.device("cpu"), position_chunk_size=3)
    assert actual == pytest.approx(expected, abs=1e-12)
    assert sizes == [3, 3, 2]


def test_chunk_checkpoints_do_not_save_vocabulary_sized_tensors():
    from bfas.rtd.runtime import HFGenerateBackend
    from bfas.rtd.functional_step import lora_parameters
    model = TinyCausalModel().eval()
    backend = HFGenerateBackend(model, Tokenizer(), base_checkpoint_hash="base",
        harness_hash="harness", tokenizer_hash="tokenizer", max_context_tokens=128)
    backend.score_position_chunk_size = 3
    saved = []
    def pack(tensor):
        saved.append(tuple(tensor.shape))
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        score = backend.score_tokens((1, 2), tuple(range(3, 17)), lora_parameters(model), eos_token_id=16)
    assert score.requires_grad
    assert not any(len(shape) >= 2 and shape[-1] == 17 for shape in saved)
