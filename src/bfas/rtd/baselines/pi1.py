"""Offline pi1 inputs, token exposure, and durable state for PaperTrainer.

The optimizer loop and PEFT export stay in PaperTrainer. No environment replay,
teacher request, generation, reference model, or auxiliary objective is used.
"""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from ..persistence import StateStore, atomic_json, digest, file_hash, fsync_directory, tree_hash
from .exposure import ExposureDistribution, budget_match
from .paper_data import TeacherRow


VERSION = "pi1-alfworld-k32-plain-ce-v2"
RECIPE = dict(method="pi1_ce", benchmark="alfworld", student="google/gemma-4-12B-it",
    optimizer="AdamW", learning_rate=1e-5, gradient_clip=1., lora_rank=16,
    lora_alpha=32, lora_dropout=0., supervised_tokens_per_update=512,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    exposure_passes=[3, 10], training_seeds=[0, 1], model_local_files_only=True)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    required = set(RECIPE) | {"bank", "sealed_manifest_sha256", "support_size", "demonstrations",
        "supervised_turns", "bank_supervised_tokens", "exposure_relative_tolerance", "adam_betas",
        "adam_epsilon", "weight_decay", "max_context_tokens", "score_position_chunk_size",
        "cpu_threads", "memory_peak_budget_gb", "memory_reserve_gb"}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError("pi1 config requires exactly the declared configuration keys")
    for key, expected in RECIPE.items():
        if config[key] != expected or type(config[key]) is not type(expected):
            raise ValueError(f"pre-registered pi1 recipe differs: {key}")
    for key in ("support_size", "demonstrations", "supervised_turns", "bank_supervised_tokens",
                "max_context_tokens", "score_position_chunk_size", "cpu_threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"positive integer required: {key}")
    for key in ("adam_epsilon", "weight_decay", "exposure_relative_tolerance",
                "memory_peak_budget_gb", "memory_reserve_gb"):
        if (type(config[key]) not in (int, float) or not math.isfinite(config[key])
                or config[key] < 0):
            raise ValueError(f"finite nonnegative setting required: {key}")
    if (config["adam_epsilon"] <= 0 or config["memory_peak_budget_gb"] <= 0
            or config["exposure_relative_tolerance"] != 0
            or len(config["adam_betas"]) != 2
            or any(type(v) is not float or not 0 <= v < 1 for v in config["adam_betas"])):
        raise ValueError("invalid AdamW, memory, or exact-exposure settings")
    return config


def plain_ce_hyperparameters(config, seed):
    keys = ("student", "optimizer", "learning_rate", "adam_betas", "adam_epsilon",
        "weight_decay", "gradient_clip", "lora_rank", "lora_alpha", "lora_target_modules",
        "lora_dropout", "supervised_tokens_per_update", "max_context_tokens",
        "score_position_chunk_size", "memory_peak_budget_gb", "memory_reserve_gb", "cpu_threads")
    return dict({k: config[k] for k in keys}, seed=seed, method="pi1_ce",
        loss="cross_entropy", loss_normalization="supervised_token_mean_per_update",
        observations_masked=True, target="complete_teacher_reply_plus_native_turn_boundary",
        kl_coefficient=0., auxiliary_losses=[], scheduler="constant", warmup_steps=0,
        adam_amsgrad=False, adam_foreach=False, adam_fused=False, lora_bias="none",
        base_dtype="bfloat16", lora_dtype="float32", attention="eager",
        gradient_checkpointing="non_reentrant_eval_functional_lora", max_live_turn_graphs=1,
        deterministic_algorithms=True, tf32=False)


def native_turn_id(tokenizer):
    """Resolve a single native special token; never tokenize boundary prose."""
    token = "<turn|>"
    if token not in tokenizer.all_special_tokens:
        raise ValueError("student tokenizer lacks the native turn special token")
    token_id = tokenizer.convert_tokens_to_ids(token)
    if (type(token_id) is not int or token_id == tokenizer.unk_token_id or
            token_id not in tokenizer.all_special_ids or
            tokenizer.encode(token, add_special_tokens=False) != [token_id]):
        raise ValueError("native turn boundary must be a single special-token ID")
    return token_id


def boundary_contract(tokenizer):
    return dict(appended=True, token="<turn|>", token_id=native_turn_id(tokenizer),
                supervised=True, tokens_per_turn=1)


def encode_teacher_turn(tokenizer, row, max_context_tokens):
    """Supervise the full authored reply and native end of turn; mask the prompt.

    Prior observations, commands, worked examples, and template prefill are all
    context. Do not infer a mask from words such as 'Observation:' in a reply.
    The boundary is part of the supervised-token exposure denominator.
    """
    prompt = tuple(tokenizer.encode(row.prompt, add_special_tokens=False))
    target = tuple(tokenizer.encode(row.target, add_special_tokens=False))
    if not target:
        raise ValueError("empty complete training target")
    target += (native_turn_id(tokenizer),)
    if not prompt or not target or len(prompt) + len(target) > max_context_tokens:
        raise ValueError("empty or overlong complete training row; no truncation")
    return dict(prompt_ids=prompt, target_ids=target, input_ids=prompt+target,
                labels=(-100,)*len(prompt)+target)


def preflight_row(tokenizer, row, max_context_tokens):
    """Exercise PaperTrainer.encode and the scorer's actual tensor preparation."""
    from .paper_train import PaperTrainer
    from ..source_scoring import teacher_forcing_inputs
    context = SimpleNamespace(method="pi1_ce", backend=SimpleNamespace(tokenizer=tokenizer),
        config=dict(max_context_tokens=max_context_tokens), encoded_rows={})
    prompt, target, kinds, eos = PaperTrainer.encode(context, row)
    inputs = teacher_forcing_inputs(prompt, target, device="cpu")
    ids, labels = inputs["input_ids"][0].tolist(), inputs["labels"][0].tolist()
    boundary = native_turn_id(tokenizer)
    assert boundary == 106, "this preflight is for the registered Gemma student"
    assert labels[-1] == boundary and labels[-1] != -100
    assert ids == list(prompt + target) and labels == [-100]*len(prompt) + list(target)
    assert len(ids) <= max_context_tokens and ids[-1] == boundary == eos
    assert kinds[-1] == "teacher" and labels[:len(prompt)] == [-100]*len(prompt)
    return dict(package_id=row.package_id, turn_index=row.index,
        last_8_input_ids=ids[-8:], last_8_label_ids=labels[-8:],
        decoded_last_tokens=tokenizer.decode(ids[-8:], skip_special_tokens=False),
        boundary=boundary_contract(tokenizer), supervised_tokens=len(target),
        tensor_shape=list(inputs["input_ids"].shape), padding=False, truncation=False,
        assertions_passed=True)


def check_optimizer_step(trainer):
    """Run the production token-mean optimizer loop once on four complete rows."""
    from .paper_losses import span_ce
    if len(trainer.rows) != 4 or trainer.config["exposure_passes"] != [1]:
        raise ValueError("execution check requires four rows and one pass")
    boundary = native_turn_id(trainer.backend.tokenizer)
    assert boundary == 106
    original = trainer.logprobs
    contributions = []

    def checked(row):
        values, kinds = original(row)
        ids = trainer.encode(row)[1]
        assert ids[-1] == boundary and kinds[-1] == "teacher"
        loss = span_ce(values, kinds, "pi1_ce")
        assert torch.isfinite(loss).item()
        derivative, = torch.autograd.grad(loss, values, retain_graph=True)
        positions = torch.tensor([i == boundary for i in ids], device=values.device)
        assert positions[-1].item() and torch.all(derivative[positions] != 0).item()
        assert torch.isfinite(values[positions]).all().item()
        contributions.append(dict(positions=int(positions.sum()),
            boundary_nll=float(-values[positions].detach().sum()),
            loss_derivative=derivative[positions].detach().tolist()))
        return values, kinds

    trainer.logprobs = checked
    try:
        result = trainer.train()
    finally:
        trainer.logprobs = original
    assert result["student_commits"] == 1 and len(contributions) == 4
    assert all(math.isfinite(v) for v in result["losses"])
    return dict(**result, checked_rows=4, boundary_token_id=boundary,
                boundary_loss_contributions=contributions, assertions_passed=True)


def load_bank(bank, config, renderer):
    from ..benchmarks.alfworld_support import audit_verified_bank
    bank = Path(bank).resolve()
    audit_verified_bank(bank, expected_manifest_sha256=config["sealed_manifest_sha256"])
    support = json.loads((bank/"public/support.json").read_text())
    if support["m"] != config["support_size"]:
        raise ValueError("bank differs from registered K=32 support")
    rows, packages = [], 0
    records = json.loads((bank/"public/requests.json").read_text())
    for record in sorted(records, key=lambda r: r["spec"]["query_id"]):
        if record["unavailable_reason"] is not None:
            continue
        qid = record["spec"]["query_id"]
        payload = json.loads((bank/"sealed"/f"{qid}.json").read_text())
        if payload.get("payload_kind") != "teacher_react_turns":
            raise ValueError("full teacher ReAct replies required; command-only supervision forbidden")
        turns, behaviors = payload["teacher_react_turns"], payload["behaviors"]
        if not turns or len(turns) != len(behaviors):
            raise ValueError("teacher turns and verified states must align")
        packages += 1
        for index, (target, behavior) in enumerate(zip(turns, behaviors)):
            state = behavior["state"]
            request, history = json.loads(state["task_json"]), json.loads(state["history_json"])
            # Same frozen renderer as evaluation, from the verified full state.
            prompt = renderer(request, history)
            rows.append(TeacherRow(qid, request["task_id"], state["parent_hash"],
                index, prompt, target, "alfworld", index == len(turns)-1))
    if packages != config["demonstrations"] or len(rows) != config["supervised_turns"]:
        raise ValueError("sealed bank differs from registered demonstration/turn counts")
    return rows, dict(path=str(bank), sealed_manifest_sha256=file_hash(bank/"sealed/manifest.json"),
        support_manifest_hash=support["manifest_hash"], demonstrations=packages, supervised_turns=len(rows),
        target_source="teacher_react_turns", context_source="verified states through FrozenRenderer")


def exposure_plan(costs, config, seed):
    total = sum(costs)
    distribution = ExposureDistribution(tuple(c/total for c in costs), tuple(costs), "tokens")
    batches = distribution.pass_schedule(endpoints=config["exposure_passes"], seed=seed,
        tokens_per_update=config["supervised_tokens_per_update"])
    return dict(rule="seeded exhaustive row permutation per pass; complete turns; flush at endpoints",
        costs=list(costs), bank_supervised_tokens=total, row_probabilities=list(distribution.q),
        target_tokens=[p*total for p in config["exposure_passes"]], batches=batches)


def check_output(directory, *, resume=False):
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("output directory must not be a symlink")
    if not resume and directory.exists():
        raise FileExistsError(f"refusing existing output directory: {directory}; use --resume to continue")
    if resume and not (directory/"manifest.json").is_file():
        raise ValueError("--resume requires an existing pi1 manifest")


def prepare_run(directory, identity, rows, plan, *, resume=False):
    directory = Path(directory)
    check_output(directory, resume=resume)
    row_values = [asdict(r) for r in rows]
    identity = dict(identity, rows_hash=digest(row_values), exposure_schedule_hash=digest(plan))
    if resume:
        manifest = json.loads((directory/"manifest.json").read_text())
        if manifest.get("identity") != identity or manifest.get("identity_hash") != digest(identity):
            raise ValueError("resume identity differs: config/bank/seed/model/tokenizer/code/device")
        for name, expected in (("training_rows.json", row_values), ("exposure_schedule.json", plan)):
            if json.loads((directory/name).read_text()) != expected:
                raise ValueError(f"resume artifact differs: {name}")
        return manifest
    directory.mkdir(parents=True, exist_ok=False)
    manifest = dict(version=VERSION, identity=identity, identity_hash=digest(identity), status="prepared",
        seed=identity["seed"], hyperparameters=identity["hyperparameters"], bank=identity["bank"],
        exposure_passes=identity["config"]["exposure_passes"], exposure_endpoint=None,
        supervised_token_count=0, optimizer_step_count=0, loss_trace="loss_trace.json", endpoints=[])
    manifest["target_boundary"] = identity["target_boundary"]
    atomic_json(directory/"training_rows.json", row_values)
    atomic_json(directory/"exposure_schedule.json", plan)
    atomic_json(directory/"manifest.json", manifest)
    return manifest


class PlainCEState:
    """AdamW and atomic progress around the shared PaperTrainer loop."""
    def __init__(self, trainer, *, resume):
        self.t = trainer
        self.manifest = trainer.manifest
        self.identity = self.manifest["identity"]
        costs = [len(trainer.encode(r)[1]) for r in trainer.rows]
        plan = exposure_plan(costs, trainer.config, trainer.seed)
        if digest(plan) != self.identity["exposure_schedule_hash"]:
            raise ValueError("encoded supervision differs from frozen exposure schedule")
        self.batches, self.bank_tokens = plan["batches"], sum(costs)
        hp = trainer.hp
        self.optimizer = torch.optim.AdamW(list(trainer.parameters.values()),
            lr=hp["learning_rate"], betas=tuple(hp["adam_betas"]), eps=hp["adam_epsilon"],
            weight_decay=hp["weight_decay"], amsgrad=False, foreach=False, fused=False)
        self.losses, self.trace = [], []
        self.store = StateStore(trainer.directory/"resume", self.identity)
        # No teacher ledger: all authorized supervision is already sealed.
        self.ledger = SimpleNamespace(events=[])
        if resume and self.store.pointer.exists():
            saved = self.store.load(self.ledger, device="cpu")
            if set(saved["parameters"]) != set(trainer.parameters):
                raise ValueError("resume LoRA parameter layout differs")
            with torch.no_grad():
                for name, p in trainer.parameters.items():
                    p.copy_(saved["parameters"][name].to(p.device))
            self.optimizer.load_state_dict(saved["optimizer"])
            self.losses, self.trace = saved["losses"], saved["trace"]
            trainer.scored_tokens = saved["supervised_tokens"]
            random.setstate(saved["python_rng"])
            np.random.set_state(saved["numpy_rng"])
            torch.set_rng_state(saved["torch_rng"])
            trainer.rng.set_state(saved["trainer_rng"])
            if trainer.device.type == "cuda":
                torch.cuda.set_rng_state(saved["cuda_rng"], trainer.device)
        elif resume and self.manifest["optimizer_step_count"]:
            raise ValueError("resume state missing for a committed run")
        else:
            self.save()
        # Recover an endpoint export interrupted after its durable optimizer commit.
        if self.trace and self.batches[len(self.trace)-1]["endpoint"] is not None:
            self.publish_endpoint(self.batches[len(self.trace)-1]["endpoint"])
        self.publish_progress()

    def update(self, gradient):
        self.optimizer.zero_grad(set_to_none=True)
        for name, p in self.t.parameters.items():
            p.grad = gradient[name]
        self.grad_norm = float(torch.nn.utils.clip_grad_norm_(list(self.t.parameters.values()),
            self.t.hp["gradient_clip"], error_if_nonfinite=True))
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def save(self):
        t = self.t
        self.store.save(dict(phase="committed", parameters={n: p.detach().cpu() for n, p in t.parameters.items()},
            optimizer=self.optimizer.state_dict(), losses=self.losses, trace=self.trace,
            supervised_tokens=t.scored_tokens, python_rng=random.getstate(), numpy_rng=np.random.get_state(),
            torch_rng=torch.get_rng_state(), trainer_rng=t.rng.get_state(),
            cuda_rng=torch.cuda.get_rng_state(t.device) if t.device.type == "cuda" else None), self.ledger)

    def commit(self, step, loss, tokens):
        batch = self.batches[step-1]
        if tokens != batch["supervised_tokens"] or self.t.scored_tokens != batch["cumulative_tokens"]:
            raise ValueError("realized supervised token exposure differs from schedule")
        self.trace.append(dict(optimizer_step=step, loss=loss, supervised_tokens=tokens,
            cumulative_supervised_tokens=self.t.scored_tokens, gradient_norm_before_clip=self.grad_norm))
        self.save()
        if batch["endpoint"] is not None:
            self.publish_endpoint(batch["endpoint"])
        self.publish_progress()

    def publish_endpoint(self, passes):
        directory = self.t.directory/f"pass-{passes}"
        receipt = dict(version=VERSION, identity=self.identity, identity_hash=digest(self.identity),
            seed=self.t.seed, bank=self.identity["bank"], hyperparameters=self.t.hp,
            exposure_endpoint=passes, target_supervised_tokens=passes*self.bank_tokens,
            supervised_token_count=self.t.scored_tokens, optimizer_step_count=len(self.trace),
            loss_trace="loss_trace.json", adapter="lora", loss_trace_hash=digest(self.trace))
        match = budget_match([passes*self.bank_tokens], [self.t.scored_tokens], [0],
                             relative_tolerance=self.t.config["exposure_relative_tolerance"])
        if not match["matched"]:
            raise ValueError("exposure endpoint missed")
        receipt["exposure_check"] = match
        if directory.exists():
            previous = json.loads((directory/"manifest.json").read_text())
            if (previous != dict(receipt, adapter_hash=tree_hash(directory/"lora"))
                    or json.loads((directory/"loss_trace.json").read_text()) != self.trace):
                raise ValueError("existing endpoint differs; refusing overwrite")
            return
        with tempfile.TemporaryDirectory(prefix=f".pass-{passes}-", dir=self.t.directory) as temporary:
            staged = Path(temporary)/"checkpoint"
            staged.mkdir()
            self.t.save_adapter(staged/"lora")
            atomic_json(staged/"loss_trace.json", self.trace)
            atomic_json(staged/"manifest.json", dict(receipt, adapter_hash=tree_hash(staged/"lora")))
            os.rename(staged, directory)
            fsync_directory(directory.parent)

    def publish_progress(self):
        endpoints = [b["endpoint"] for b in self.batches[:len(self.trace)] if b["endpoint"] is not None]
        self.manifest.update(status="trained" if len(self.trace) == len(self.batches) else "training",
            exposure_endpoint=endpoints[-1] if endpoints else None, endpoints=endpoints,
            supervised_token_count=self.t.scored_tokens, optimizer_step_count=len(self.trace))
        atomic_json(self.t.directory/"loss_trace.json", self.trace)
        atomic_json(self.t.directory/"manifest.json", self.manifest)

    def finish(self):
        self.publish_progress()
        return dict(student_commits=len(self.trace), supervised_tokens=self.t.scored_tokens,
                    endpoints=self.manifest["endpoints"], losses=self.losses)
