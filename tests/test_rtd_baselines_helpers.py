"""Synthetic C25-format fixture factory; only tests/fixtures and tmp_path IO."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from bfas.rtd.baselines.config import RUNTIME_KEYS
from bfas.rtd.baselines.evidence import prepare
from bfas.rtd.broker import RequestRecord, seal_bank
from bfas.rtd.functional_step import lora_parameters
from bfas.rtd.ledger import Ledger
from bfas.rtd.persistence import ComputeJournal, atomic_json, digest, file_hash, tree_hash
from bfas.rtd.return_gradient import TaskRollout, TorchPolicyBackend
from bfas.rtd.selector import PublicFeatures, PublicQuerySpec
from bfas.rtd.transport import Behavior, FullState

FIXTURE = Path(__file__).parent/"fixtures/rtd_baselines/fixture.json"


def h(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


class TinyTokenizer:
    eos_token_id = 3
    def encode(self, text, add_special_tokens=False):
        return [0, 1] if text.startswith("prompt") else [{"A": 0, "B": 1}.get(c, 2) for c in text]
    def decode(self, ids, skip_special_tokens=False):
        return "".join("ABC!"[i] for i in ids)
    def save_pretrained(self, directory):
        atomic_json(Path(directory)/"tokenizer_config.json", dict(eos_token_id=3, synthetic=True))


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_transition = torch.nn.Parameter(torch.tensor([
            [.1, -.1, .2, 1.], [.3, .4, .1, 1.1], [-.2, .3, .1, 1.2], [.1, .2, .1, 1.3]], dtype=torch.float64))
    def forward(self, input_ids, attention_mask, use_cache=False):
        return SimpleNamespace(logits=self.lora_transition[input_ids])
    def save_pretrained(self, directory, safe_serialization=True):
        from safetensors.torch import save_file
        directory = Path(directory)
        directory.mkdir(parents=True)
        save_file({"lora_transition": self.lora_transition.detach()}, directory/"adapter_model.safetensors")
        atomic_json(directory/"adapter_config.json", dict(synthetic=True, peft_type="LORA"))


def tiny_backend():
    return TorchPolicyBackend(TinyLM().eval(), TinyTokenizer(), base_checkpoint_hash="base", harness_hash="harness",
        tokenizer_hash="tokenizer", max_action_tokens=3, max_context_tokens=100,
        action_caps={"single_turn": 3, "multi_turn": 3})


class TinySupport:
    def __init__(self):
        self.parents, self.categories, self.states, self.calls = {}, {}, {}, []
        self.zero_rewards = False
        spec = json.loads(FIXTURE.read_text())
        for p in spec["parents"]:
            parent = f"{p['number']:064x}"
            tid = f"{p['category']}_{p['number']}"
            self.parents[parent], self.categories[tid] = tid, p["category"]
            self.states[parent] = FullState.create({"question": f"synthetic {p['number']}"},
                [{"role": "user", "content": f"synthetic {p['number']}"}], f"prompt{p['number']}", parent)
    def feedback(self, parent, backend, parameters, generator, checker):
        self.calls.append((parent, backend.identity(parameters)))
        action = backend.sample_action(self.states[parent].prompt, parameters, generator)
        # A deterministic terminal reward depending only on this sampled action.
        reward = 0. if self.zero_rewards else float(action.action_ids[0] == 3)
        return TaskRollout(self.parents[parent], (action,), reward, backend.identity(parameters))


def runtime_config():
    # Test configuration is declared locally, never read from a live RTD YAML.
    return dict(student="synthetic", model_local_files_only=True, training_seed=0, benchmark="bfcl",
        source_backend="hf_generate", support_manifest="support.json", replay_bank_path="bank",
        lora_rank=16, lora_alpha=32, lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        source_temperature=1., source_top_p=1., source_samples_per_state=2,
        evaluation_temperature=0., max_context_tokens=32768,
        max_action_tokens_by_benchmark={"bfcl": {"single_turn": 512, "multi_turn": 1024}},
        pilot_eta_candidates=[1e-4, 1e-3, 1e-2])


def make_fixture(tmp_path):
    spec = json.loads(FIXTURE.read_text())
    support = TinySupport()
    source, bank = tmp_path/"source", tmp_path/"bank"
    source.mkdir()
    records, payloads = [], {}
    for p in spec["packages"]:
        parent, q = f"{p['parent']:064x}", h(p["name"])
        state = support.states[parent]
        records.append(RequestRecord(PublicQuerySpec(q, state.state_hash, PublicFeatures(), 1, 100,
            p["confidence"], "synthetic declared public cap"), parent, tuple(h(d) for d in p["dependencies"])))
        payloads[q] = dict(cost=p["cost"], cost_confidence=p["confidence"],
            usage=dict(output_tokens=p["cost"], input_tokens=None, reasoning_tokens=None, money=None),
            provenance=dict(synthetic=True), historical_response={},
            behaviors=[asdict(Behavior(state, text)) for text in p["texts"]])
    # An unavailable, unpurchased package: its payload must never be read by prepare.
    q = h("unavailable")
    records.append(RequestRecord(PublicQuerySpec(q, h("hidden"), PublicFeatures(), 1, 100, "exact", "synthetic"),
                                 f"{0:064x}", unavailable_reason="protected"))
    payloads[q] = {"do_not_read": True}
    seal_bank(bank, records, payloads)
    ledger = Ledger(1000, source/"teacher.jsonl")
    ledger.authorize(3000)
    for p in spec["packages"]:
        q = h(p["name"])
        ledger.reserve(q, 100)
        ledger.settle(q, p["cost"], confidence=p["confidence"], usage=payloads[q]["usage"],
                      dependencies=[h(d) for d in p["dependencies"]])
    hard = dict(gpu="synthetic", capability=[0, 0], memory=1, cuda="none", driver="synthetic",
        versions={k: "synthetic" for k in ("torch", "transformers", "peft", "numpy")},
        python="synthetic", machine="synthetic", host_class="synthetic")
    hardware = dict(version="rtd-hardware-class-v1", hard=hard,
        metadata=dict(hostname="synthetic", uuid="synthetic", pci_bus_id=None, cuda_device_order="PCI_BUS_ID"))
    config = dict(runtime_config(), meta_tasks_per_feedback=8, meta_tasks_multi_turn=4,
                  rollouts_per_meta_task=4, rollouts_multi_turn=2)
    atomic_json(tmp_path/"support.json", dict(parents=[dict(parent_hash=p, official_id=t, fold=int(p, 16)%2)
                for p, t in support.parents.items()]))
    atomic_json(tmp_path/"envs/bfcl/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/synthetic.json", dict(synthetic=True))
    from bfas.rtd.cli import data_identity
    manifest = dict(version="rtd-v1.0.1-run", config=config, config_hash=digest(config), arm="R1", smoke=False,
        budget_ceilings=[1000, 2000, 3000], hardware=hardware, hardware_hash=digest(hard),
        base_checkpoint_hash="base", tokenizer_hash="tokenizer", model_path="synthetic",
        evaluation_harness={}, harness_hash=digest({}), data_hash=data_identity(tmp_path, config, bank),
        bank_path=str(bank), bank_public_cap_sum=300, recorded_bank_usage={}, available_packages=3, m=4)
    atomic_json(source/"manifest.json", manifest)
    journal = ComputeJournal(source/"compute.jsonl")
    steps, checkpoints = [], []
    for r in (1, 2, 3):
        for step in range(1, 13):
            actual = h(f"r{r}s{step}")
            row = dict(round=r, step=step, decision=step in (1, 4, 7, 10), inner_fold=(r-1)%2,
                audit_passed=True, actual_hash=actual, actual_spend=31,
                old_exposure=dict(source_action_tokens=16, teacher_action_tokens=8, prompt_tokens=32, teacher_slots=8),
                new_exposure=dict(source_action_tokens=0, teacher_action_tokens=0, prompt_tokens=0, teacher_slots=0))
            if row["decision"]:
                reused = step in (1, 7)
                for role in ("reference_feedback", "actual_feedback"):
                    if role == "actual_feedback" and reused:
                        journal.append("feedback_reused", round=r, step=step, parameter_hash=actual)
                        continue
                    with journal.measure(role, round=r, step=step):
                        for parent, state in support.states.items():
                            if int(parent, 16)%2 == (r-1)%2:
                                continue
                            count = 2 if support.categories[support.parents[parent]].startswith("multi_turn") else 4
                            for _ in range(count):
                                journal.append("feedback_rollout", round=r, step=step, role=role, parent_hash=parent,
                                    rollout=dict(policy_id=actual+role, actions=[dict(prompt_ids=[0, 1], action_ids=[0, 3])]))
                    journal.append("return_gradient", round=r, step=step, role=role,
                        parameter_hash=actual if reused or role == "actual_feedback" else h(actual+role), metadata=dict(rollouts=6))
                row["truncation"] = dict(actual_feedback_reused=reused, reference=dict(rollouts=6), actual=dict(rollouts=6))
            steps.append(row)
        d = source/f"round-{r}"
        (d/"lora").mkdir(parents=True)
        atomic_json(d/"lora/adapter_config.json", dict(synthetic=True))
        (d/"round_state.pt").write_bytes(b"synthetic source checkpoint, never deserialized")
        meta = dict(round=r, parameter_hash=steps[-1]["actual_hash"], owned=sorted(ledger.owned_ids), actual_spend=31,
            authorized_budget=3000, manifest_hash=digest(manifest), config_hash=manifest["config_hash"],
            adapter_hash=tree_hash(d/"lora"), round_state_hash=file_hash(d/"round_state.pt"))
        atomic_json(d/"checkpoint.json", meta)
        checkpoints.append(meta)
    atomic_json(source/"trajectory.json", dict(steps=steps, checkpoints=checkpoints))
    atomic_json(source/"audit.json", dict(passed=True, steps=36, decision_windows=12,
        packages=3, actual_spend=31, authorized_budget=3000))
    return source, bank, support, manifest


def prepared_fixture(tmp_path):
    source, bank, support, manifest = make_fixture(tmp_path)
    out = tmp_path/"evidence.json"
    evidence = prepare(source, bank, out, root=tmp_path, expected_hardware_hash=manifest["hardware_hash"],
                       tokenizer=TinyTokenizer(), support=support)
    return evidence, out, support
