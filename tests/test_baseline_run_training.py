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
