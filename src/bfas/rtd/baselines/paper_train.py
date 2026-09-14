"""Shared 24-commit LoRA trainer. Model loading stays behind the worker entry."""
from dataclasses import asdict
import hashlib
from pathlib import Path
import random
import time

import torch

from ..functional_step import FrozenStep, gradients, lora_parameters, rms_diagonal
from ..persistence import atomic_json
from ..student import termination_ids
from .paper_data import kang_action_list_summary, select_smartad
from .paper_fidelity import (KANG_PREFIX, KANG_SUMMARY, method_name,
                             fidelity_metadata, require_unsealed_output)
from .paper_losses import (SmallDiscriminator, discriminator_loss, group_advantages,
                           grpo_loss, span_ce, token_kinds)
from .paper_progress import progress, stage
from .paper_seeds import training_seed, verify_seed_zero
from ..source_scoring import require_device


def hyperparameters(config, method):
    method = method_name(method, config)
    seed = training_seed(config)
    settings = dict(student=config["student"], seed=seed, lora_rank=config["lora_rank"],
        lora_alpha=config["lora_alpha"], lora_target_modules=config["lora_target_modules"],
        lora_dropout=0., optimizer="fixed_preconditioned_single_step",
        preconditioner="train_only_rms_diagonal", preconditioner_refresh_steps=12,
        preconditioner_source_samples=2, preconditioner_damping_relative=.01,
        learning_rate=1e-5, student_steps=2 if config.get("smoke") else 24,
        slots_per_step=2 if config.get("smoke") else 40,
        smoke=bool(config.get("smoke")), smoke_preconditioner_prompts=2 if config.get("smoke") else None,
        loss_normalization="per_sequence_mean", observations_masked=True,
        smartad=dict(weights=dict(reason=1., action=1.5, final=2.), selection="base_student_macro_mean_turn_nll",
                     normalization="sum_of_weights"),
        sad=dict(reason_coefficient=.5, act_coefficient=.5, absent_span="renormalize present groups"),
        kang=dict(prefix=(KANG_SUMMARY if method == KANG_SUMMARY else KANG_PREFIX),
                  n=3, sampling_temperature=.7, official_single_sample_also_reported=True,
                  tie_break="first valid sample; first raw sample when all invalid"),
        gad=dict(discriminator="separate byte GRU", embedding_width=32, hidden_width=64,
                 discriminator_optimizer="AdamW", discriminator_lr=1e-4, discriminator_seed=seed,
                 discriminator_weight_decay=0., discriminator_loss="Bradley-Terry",
                 group_size=4, rounds=4, pg_steps_per_round=5, warmup_student_steps=4,
                 discriminator_steps_per_prompt=1, advantage_epsilon=1e-6, clip=.2,
                 kl_coefficient=0., reward="raw discriminator score",
                 prompts="purchased prompt strings only; fresh current-student responses",
                 missing_teacher="skip exact prompt without purchased teacher response"),
        method=method, fidelity=fidelity_metadata(method))
    if config.get("benchmark") == "hotpotqa":
        settings["kang"].update(n=None, sampling_temperature=None, tie_break=None,
            sag_status="unsupported; requires voting over complete sampled ReAct episodes")
    return settings


class PaperTrainer:
    def __init__(self, backend, rows, config, method, directory, journal, *, manifest=None):
        require_unsealed_output(directory)
        method = method_name(method, config)
        self.backend, self.rows, self.config = backend, list(rows), config
        if method == KANG_PREFIX and any(r.acquisition_method != KANG_PREFIX for r in self.rows):
            raise ValueError("kang_first_thought_prefix requires newly acquired prefixed trajectories; use kang_action_list_summary for legacy data")
        self.method, self.directory, self.journal = method, Path(directory), journal
        self.seed, self.manifest = training_seed(config), manifest
        self.parameters = lora_parameters(backend.model)
        self.device = next(iter(self.parameters.values())).device
        expected = torch.device(config.get("training_device", str(self.device)))
        require_device(expected, **dict(backend.model.named_parameters()))
        require_device(expected, **dict(backend.model.named_buffers()))
        self.encoded_rows, self.encoded_prompts = {}, {}
        self.scored_tokens = 0
        self.hp = hyperparameters(config, method)
        self.rng = torch.Generator(device=self.device).manual_seed(self.seed % (2**64))
        self.discriminator = None
        if method == "gad":
            # A separate RNG scope prevents discriminator initialization from
            # changing student/source sampling or LoRA initialization.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed % (2**64))
                self.discriminator = SmallDiscriminator().to(self.device)
            self.discriminator_optimizer = torch.optim.AdamW(
                self.discriminator.parameters(), lr=1e-4, weight_decay=0.)

    def encode(self, row):
        if row in self.encoded_rows:
            return self.encoded_rows[row]
        if row.prompt not in self.encoded_prompts:
            self.encoded_prompts[row.prompt] = tuple(self.backend.tokenizer.encode(row.prompt, add_special_tokens=False))
        prompt = self.encoded_prompts[row.prompt]
        ids, kinds = token_kinds(self.backend.tokenizer, row.target,
                                benchmark=row.benchmark, final_step=row.final_step, prompt=row.prompt)
        stops = termination_ids(self.backend)
        # Match native termination semantics, including the bank's newline
        # after a final turn marker. No extra EOS after an authored terminator.
        stop_positions = [i for i, token in enumerate(ids) if token in stops]
        if stop_positions:
            end = stop_positions[0] + 1
            tail = self.backend.tokenizer.decode(ids[end:], skip_special_tokens=False)
            if tail.strip():
                raise ValueError("teacher target contains text after a native termination")
            ids, kinds = ids[:end], kinds[:end]
            eos = ids[-1]
        else:
            eos = self.backend.tokenizer.eos_token_id
            ids += (eos,)
            kinds.append(next((k for k in reversed(kinds) if k != "observation"), "final"))
        if not prompt or len(prompt)+len(ids) > self.config["max_context_tokens"]:
            raise ValueError("complete training row exceeds context; no truncation")
        result = prompt, ids, kinds, eos
        self.encoded_rows[row] = result
        return result

    def logprobs(self, row):
        prompt, ids, kinds, eos = self.encode(row)
        _, values, _ = self.backend.score_tokens(prompt, ids, self.parameters,
                                                eos_token_id=eos, return_details=True)
        require_device(self.device, token_logprobs=values)
        self.scored_tokens += sum(k != "observation" for k in kinds)
        return values, kinds

    def base_nll(self, row):
        with torch.no_grad():
            values, kinds = self.logprobs(row)
            mask = values.new_tensor([k != "observation" for k in kinds], dtype=torch.bool)
            return float(-values[mask].sum()), int(mask.sum())

    def sample(self, row):
        category = "agent_action" if row.benchmark in {"alfworld", "hotpotqa"} else row.task_id.rsplit("_", 1)[0]
        with self.backend.action_limit(category):
            action = self.backend.sample_action(row.prompt, self.parameters, self.rng)
        with torch.no_grad():
            # Keep the existing Gemma score consistency guard and diagnostics.
            self.backend.checked_score_action(action, self.parameters,
                expected_prompt_ids=self.backend.tokenizer.encode(row.prompt, add_special_tokens=False),
                record=lambda diagnostic: self.journal.append("score_consistency", diagnostic=diagnostic))
        return action

    def step_rule(self, step, source_rows):
        unique = {row.prompt: row for row in source_rows}
        if self.config.get("smoke"):
            unique = dict(list(unique.items())[:2])
        source_id = self.backend.identity(self.parameters)

        def stream():
            completed = 0
            for row in unique.values():
                for _ in range(self.hp["preconditioner_source_samples"]):
                    action = self.sample(row)
                    loss = -self.backend.score_action(action, self.parameters) / len(action.action_ids)
                    yield row.parent_hash, gradients(loss, self.parameters)
                    completed += 1
                    progress("preconditioner", "rollout", step=step+1, completed=completed,
                             total=2*len(unique), action_tokens=len(action.action_ids))
        with stage("preconditioner", step=step+1, prompts=len(unique), rollouts=2*len(unique)):
            diagonal, metadata = rms_diagonal(self.parameters, stream(),
                inner_parent_hashes={r.parent_hash for r in source_rows}, source_snapshot_id=source_id)
        self.journal.append("preconditioner", step=step, **metadata)
        return FrozenStep(diagonal, 1e-5, str(step//12), metadata)

    def gad_gradient(self, prompt, teacher_by_prompt, *, warmup=False):
        """Sample only after exact prompt lookup; missing supervision is skipped."""
        if prompt not in teacher_by_prompt:
            return None
        row = teacher_by_prompt[prompt]
        actions = [self.sample(row) for _ in range(self.hp["gad"]["group_size"])]
        texts = [self.backend.tokenizer.decode(a.action_ids, skip_special_tokens=False) for a in actions]
        _, teacher_ids, _, _ = self.encode(row)
        teacher_text = self.backend.tokenizer.decode(teacher_ids, skip_special_tokens=False)
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        teacher_scores = self.discriminator([prompt]*len(actions), [teacher_text]*len(actions))
        student_scores = self.discriminator([prompt]*len(actions), texts)
        d_loss = discriminator_loss(teacher_scores, student_scores)
        d_loss.backward()
        self.discriminator_optimizer.step()
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            rewards = self.discriminator([prompt]*len(actions), texts)
            advantages = group_advantages(rewards)
        self.journal.append("gad_discriminator", prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
            package_id=row.package_id, loss=float(d_loss.detach()), rewards=rewards.tolist(),
            advantages=advantages.tolist(), warmup=warmup, on_policy=True)
        if warmup:
            values, kinds = self.logprobs(row)
            loss = span_ce(values, kinds, "sft")
            return gradients(loss, self.parameters), float(loss.detach())
        gradient = {n: torch.zeros_like(p) for n, p in self.parameters.items()}
        loss_value = 0.
        for action, advantage in zip(actions, advantages):
            # One update per sampled group: the teacher-forced old policy is
            # exactly theta before this commit (no stale response reuse).
            _, values, _ = self.backend.score_action(action, self.parameters, return_details=True)
            require_device(self.device, token_logprobs=values)
            self.scored_tokens += len(action.action_ids)
            loss = grpo_loss(values, values.detach(), advantage) / len(actions)
            g = gradients(loss, self.parameters)
            for n in gradient:
                gradient[n].add_(g[n])
            loss_value += float(loss.detach())
        return gradient, loss_value

    def train(self):
        if not self.rows:
            raise ValueError("no usable purchased trajectories; cannot train a baseline")
        source_rows = self.rows
        with stage("rendering", rows=len(self.rows)):
            for index, row in enumerate(self.rows, 1):
                self.encode(row)
                if index % 25 == 0 or index == len(self.rows):
                    progress("rendering", "rows", completed=index, total=len(self.rows))
        if self.method == "smartad":
            with stage("selection", rows=len(self.rows), device=str(self.device)):
                self.rows, selection = select_smartad(self.rows, self.base_nll,
                    on_progress=lambda **fields: progress("selection", "row", **fields))
            atomic_json(self.directory/"smartad_selection.json", selection)
        elif self.method == KANG_SUMMARY:
            with stage("rendering_kang", rows=len(self.rows)):
                self.rows = kang_action_list_summary(self.rows)
                for row in self.rows:
                    self.encode(row)
        atomic_json(self.directory/"training_rows.json", [asdict(row) for row in self.rows])
        if self.manifest is not None and self.seed:
            verify_seed_zero(self.directory, self.manifest, selection=True)
            atomic_json(self.directory/"manifest.json", self.manifest)
        teacher_by_prompt = {}
        for row in self.rows:
            teacher_by_prompt.setdefault(row.prompt, row)
        schedule_rng = random.Random(self.seed)
        schedule = [[schedule_rng.randrange(len(self.rows)) for _ in range(self.hp["slots_per_step"])]
                    for _ in range(self.hp["student_steps"])]
        atomic_json(self.directory/"exposure_schedule.json", dict(
            rows=[dict(package_id=r.package_id, index=r.index) for r in self.rows], batches=schedule))
        losses = []
        for step, indices in enumerate(schedule):
            with stage("training", step=step+1, total_steps=len(schedule)), self.journal.measure("student_step", step=step+1):
                if step % 12 == 0:
                    rule = self.step_rule(step, source_rows)
                started, before_tokens = time.monotonic(), self.scored_tokens
                gradient = {n: torch.zeros_like(p) for n, p in self.parameters.items()}
                total = 0.
                for slot, index in enumerate(indices, 1):
                    row = self.rows[index]
                    if self.method == "gad":
                        # Several purchased teachers may share a prompt. Train
                        # against the actual scheduled response, not always the
                        # first version stored at that prompt.
                        teacher_by_prompt[row.prompt] = row
                        g, loss_value = self.gad_gradient(row.prompt, teacher_by_prompt, warmup=step < 4)
                    else:
                        values, kinds = self.logprobs(row)
                        loss = span_ce(values, kinds, self.method)
                        require_device(self.device, loss=loss)
                        g, loss_value = gradients(loss, self.parameters), float(loss.detach())
                    for n in gradient:
                        gradient[n].add_(g[n], alpha=1/len(indices))
                    total += loss_value/len(indices)
                    if slot % 5 == 0 or slot == len(indices):
                        progress("training", "slots", step=step+1, completed=slot, total=len(indices),
                                 loss=total*len(indices)/slot,
                                 tokens_per_second=(self.scored_tokens-before_tokens)/(time.monotonic()-started))
                updated = rule.update(self.parameters, gradient)
                with torch.no_grad():
                    for n, parameter in self.parameters.items():
                        parameter.copy_(updated[n])
                losses.append(total)
                tokens = self.scored_tokens-before_tokens
                seconds = time.monotonic()-started
                progress("training", "commit", step=step+1, loss=total, tokens=tokens,
                         tokens_per_second=tokens/seconds, elapsed_seconds=seconds)
                self.journal.append("student_commit", step=step+1, loss=total,
                    tokens=tokens, tokens_per_second=tokens/seconds, wall_seconds=seconds,
                    gad_round=(None if step < 4 or self.method != "gad" else (step-4)//5+1))
        self.backend.model.save_pretrained(self.directory/"checkpoint/lora", safe_serialization=True)
        self.backend.tokenizer.save_pretrained(self.directory/"checkpoint/lora")
        if self.discriminator is not None:
            torch.save(self.discriminator.state_dict(), self.directory/"checkpoint/discriminator.pt")
        return dict(method=self.method,
                    student_commits=len(schedule), losses=losses, trained_rows=len(self.rows),
                    discriminator_prompt_updates=sum(map(len, schedule)) if self.method == "gad" else 0)
