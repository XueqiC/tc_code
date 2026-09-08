"""B1--B4 runner; all mutable state lives exclusively in a new output directory."""
from dataclasses import asdict
import json
from pathlib import Path

import torch

from ...behavior.deltas import tensor_state_hash
from ..functional_step import FrozenStep, commit_step, gradients, kl_pilot, lora_parameters, rms_diagonal, snapshot
from ..persistence import atomic_json, digest, file_hash, tree_hash
from ..return_gradient import ActionTrace, GateController
from ..transport import Behavior, FullState, SourceSample
from .exposure import ExposureDistribution, budget_match
from .feedback import collect_groups, combine_pg
from .losses import supervised_gradient, teacher_weight_vjp, teacher_weights
from .pool import build_slots


class BaselineRunner:
    def __init__(self, config, evidence, manifest, directory, backend, support, journal, *, checker=None,
                 development_schedule=None):
        self.config, self.evidence, self.manifest = config, evidence, manifest
        self.directory, self.backend, self.support, self.journal = Path(directory), backend, support, journal
        self.checker, self.development_schedule = checker, development_schedule
        self.parameters = lora_parameters(backend.model)
        self.initial = snapshot(self.parameters)
        self.device = next(iter(self.parameters.values())).device
        self.source_rng = torch.Generator(device=self.device).manual_seed(config.seed)
        self.feedback_rng = torch.Generator(device=self.device).manual_seed(config.seed+10000)
        self.development_rng = torch.Generator(device=self.device).manual_seed(config.seed+20000)
        self.query_ids = sorted(evidence["owned"])
        self.u = next(iter(self.parameters.values())).new_zeros(len(self.query_ids), requires_grad=True)
        self.controller = GateController(learning_rate=config.meta_lr, ridge=config.ridge)
        self.optimizer = (torch.optim.AdamW(list(self.parameters.values()), lr=config.lr, weight_decay=.01)
                          if config.optimizer == "adamw" else None)
        self.steps, self.checkpoints, self.development_scores = [], [], []
        self.actual_tokens, self.max_batch_tokens = [], []
        self.references = {}

    def refresh_sources(self, r):
        frozen = snapshot(self.parameters)
        source_id = self.backend.identity(frozen)
        cache = {}
        with self.journal.phase("baseline_source_refresh", round=r):
            for h, item in sorted(self.evidence["states"].items()):
                state = FullState(**item["state"])
                if int(state.parent_hash, 16) % 2 != (r-1) % 2:
                    continue
                samples = []
                for index in range(2):
                    category = self.support.categories[self.support.parents[state.parent_hash]]
                    with self.backend.action_limit(category):
                        action = self.backend.sample_action(state.prompt, frozen, self.source_rng)
                    with torch.no_grad():
                        self.backend.checked_score_action(action, frozen,
                            expected_prompt_ids=self.backend.tokenizer.encode(state.prompt, add_special_tokens=False),
                            record=lambda diagnostic: self.journal.append("score_consistency", diagnostic=diagnostic,
                                round=r, role="source", state_hash=h, sample_index=index))
                    samples.append(SourceSample(Behavior(state, action.text), source_id, action.action_ids,
                        action.eos_token_id, action.generation_logprob, truncated=action.truncated))
                    self.journal.append("source_sample", round=r, state_hash=h, parent_hash=state.parent_hash,
                                        source_id=source_id, sample_index=index, action=asdict(action))
                    tokens = len(action.prompt_ids)+len(action.action_ids)
                    self.journal.usage("generation", input_tokens=tokens, action_tokens=len(action.action_ids),
                                       backward=False, round=r, operation="source_generation")
                    self.journal.usage("aux", input_tokens=tokens, action_tokens=len(action.action_ids),
                                       backward=False, round=r, operation="source_score_check")
                cache[h] = tuple(samples)
        return frozen, cache

    def frozen_step(self, r, frozen, cache, slots):
        inner = {s.source.behavior.state.parent_hash for s in slots}
        sources = [source for samples in cache.values() for source in samples]
        def source_gradients():
            for source in sources:
                g = gradients(-self.backend.score_source(source, frozen), frozen)
                prompt = len(self.backend.tokenizer.encode(source.behavior.state.prompt, add_special_tokens=False))
                self.journal.usage("aux", input_tokens=prompt+source.length, action_tokens=source.length,
                                   round=r, operation="preconditioner")
                yield source.behavior.state.parent_hash, g
        with self.journal.phase("baseline_fixed_step_setup", round=r):
            # P sees ALL legal inner sources, including teacher-unavailable states.
            legal = {s.behavior.state.parent_hash for s in sources}
            diagonal, meta = rms_diagonal(self.parameters, source_gradients(), inner_parent_hashes=legal,
                                         source_snapshot_id=self.backend.identity(frozen))
            teacher_slots = [s for s in slots if s.teacher is not None]
            if not teacher_slots:
                raise ValueError("fixed-step KL pilot requires legal teacher evidence")
            rng = __import__("numpy").random.default_rng(r+30000)
            probabilities = [s.p for s in teacher_slots]
            probabilities = [p/sum(probabilities) for p in probabilities]
            pilot_slots = [teacher_slots[i] for i in rng.choice(len(teacher_slots), 8, p=probabilities)]
            g, _ = supervised_gradient(pilot_slots, [1.]*8, self.backend, self.parameters, "b3_mix",
                                       journal=self.journal, role="aux", context=dict(round=r, operation="pilot_gradient"))
            actions = [ActionTrace(tuple(self.backend.tokenizer.encode(s.source.behavior.state.prompt, add_special_tokens=False)),
                s.source.token_ids, s.source.eos_token_id, s.source.behavior.text, s.source.logprob,
                self.backend.backend_id, self.backend.identity(frozen), truncated=s.source.truncated) for s in pilot_slots]
            def source_kl(updated):
                value = self.backend.source_kl(actions, frozen, updated)
                self.journal.usage("aux", input_tokens=2*sum(len(a.prompt_ids)+len(a.action_ids) for a in actions),
                    action_tokens=2*sum(len(a.action_ids) for a in actions), backward=False, round=r, operation="pilot_kl")
                return value
            eta, pilot = kl_pilot(self.parameters, diagonal,
                lambda alpha: sum((self.parameters[n]*g[n].detach()).sum() for n in self.parameters), source_kl,
                candidates=self.config.runtime["pilot_eta_candidates"], owned_ids=self.evidence["owned"],
                evidence_ids={s.query_id for s in pilot_slots}, inner_parent_hashes=legal,
                evidence_parents={s.teacher.state.parent_hash for s in pilot_slots},
                source_parents={s.source.behavior.state.parent_hash for s in pilot_slots})
            self.journal.append("kl_pilot", round=r, metadata=pilot, multiplier=self.config.eta_multiplier)
        return FrozenStep(diagonal, eta*self.config.eta_multiplier, f"r{r}", meta)

    def reference_scores(self, slots, r, step):
        scores = []
        with self.journal.phase("baseline_reference", round=r, step=step):
            for slot in slots:
                if slot.teacher is None:
                    scores.append(None)
                    continue
                key = digest(slot.identity())
                if key not in self.references:
                    with torch.no_grad():
                        pair = (self.backend.score_behavior(slot.teacher, self.initial).detach(),
                                self.backend.score_source(slot.source, self.initial).detach())
                    self.references[key] = pair
                    costs = slot.costs("b2")
                    self.journal.usage("reference", input_tokens=costs["input_tokens"],
                                       action_tokens=costs["action_tokens"], backward=False, round=r, step=step)
                scores.append(self.references[key])
        return scores

    def checkpoint(self, r, source, cache, rule):
        directory = self.directory/f"round-{r}"
        directory.mkdir(exist_ok=False)
        with self.journal.phase("baseline_checkpoint", round=r):
            self.backend.model.save_pretrained(directory/"lora", safe_serialization=True)
            self.backend.tokenizer.save_pretrained(directory/"lora")
            torch.save(dict(parameters=snapshot(self.parameters), initial=self.initial, source=source,
                source_cache=cache, step=rule, u=self.u.detach(), query_ids=self.query_ids,
                controller=vars(self.controller), optimizer=self.optimizer.state_dict() if self.optimizer else None,
                source_rng=self.source_rng.get_state(), feedback_rng=self.feedback_rng.get_state(),
                development_rng=self.development_rng.get_state(), round=r,
                optimizer_commits=len(self.steps), references=self.references), directory/"round_state.pt")
            meta = dict(round=r, parameter_hash=tensor_state_hash(self.parameters),
                adapter_hash=tree_hash(directory/"lora"), round_state_hash=file_hash(directory/"round_state.pt"),
                manifest_hash=digest(self.manifest), config_hash=self.manifest["config_hash"],
                source_id=self.backend.identity(source), owned=self.evidence["owned"],
                actual_spend=self.evidence["budgets"]["teacher_tokens"],
                authorized_budget=self.evidence["budgets"]["authorized_cap"],
                optimizer_commits=len(self.steps), baseline=self.config.recipe,
                checkpoint_meaning=self.evidence["checkpoint_meaning"])
            atomic_json(directory/"checkpoint.json", meta)
            self.checkpoints.append(meta)
            atomic_json(self.directory/"trajectory.json", dict(steps=self.steps, checkpoints=self.checkpoints))

    def run(self):
        self.journal.append("teacher_budget", teacher_tokens=self.evidence["budgets"]["teacher_tokens"],
            new_teacher_tokens=0, charges=self.evidence["charges"], repeated_cache_reads_charge_again=False)
        for r in (1, 2, 3):
            frozen, cache = self.refresh_sources(r)
            slots, pool_audit = build_slots(self.evidence, cache, self.backend.tokenizer, round_number=r,
                recipe=self.config.recipe, distribution=self.config.distribution)
            dist = ExposureDistribution(tuple(s.p for s in slots), tuple(s.costs(self.config.recipe)["input_tokens"] for s in slots), self.config.view)
            commits = self.config.commits // 3
            target = self.config.update_tokens[r-1] if self.config.update_tokens else None
            reserve = self.config.pg_reserved_tokens[r-1] if self.config.recipe.startswith("b3") and self.config.pg_reserved_tokens else 0
            batches = dist.schedule(commits=commits, seed=40000+r, target_tokens=target, pg_reserved_tokens=reserve)
            schedule = dict(round=r, **dist.journal(), pool=pool_audit, batches=batches,
                slots=[s.identity() | s.costs(self.config.recipe) for s in slots],
                target_tokens=target, pg_reserved_tokens=reserve)
            self.journal.append("exposure_schedule", **schedule)
            atomic_json(self.directory/f"exposure-{r}.json", schedule)
            rule = self.frozen_step(r, frozen, cache, slots) if self.config.optimizer == "fixed" else None
            round_tokens = max_batch = 0
            for index, batch in enumerate(batches):
                step = index + 1
                selected, rhos = [slots[i] for i in batch], [dist.rho[i] for i in batch]
                decision = index % (commits//4) == 0
                # 36 and 72 commits map to exactly the same four external windows.
                reference_step = 1+3*(index//(commits//4))
                groups = [g for g in self.evidence["feedback"] if g["round"] == r and g["step"] == reference_step]
                start_hash = tensor_state_hash(self.parameters)
                feedback, vjp = None, None
                context = dict(round=r, step=step)
                refs = self.reference_scores(selected, r, step) if self.config.recipe == "b2" else None
                weights = teacher_weights(self.u, self.query_ids, pool_audit["query_mass"]) if self.config.recipe == "b4" else None
                with self.journal.phase("baseline_student_step", **context):
                    g, loss = supervised_gradient(selected, rhos, self.backend, self.parameters, self.config.recipe,
                        references=refs, weights=weights, journal=self.journal, context=context)
                    if self.config.recipe == "b4":
                        updated = rule.update(self.parameters, g)
                        if decision:
                            feedback = collect_groups(groups, self.support, self.backend, updated, self.feedback_rng,
                                self.checker, self.journal, role="aux")
                            if feedback.parameter_hash != tensor_state_hash(updated):
                                raise ValueError("B4 feedback not at the actual fixed-step theta+")
                            vjp = teacher_weight_vjp(selected, rhos, self.backend, self.parameters, self.u,
                                self.query_ids, pool_audit["query_mass"], rule, feedback, journal=self.journal, context=context)
                            active = [i for i, q in enumerate(self.query_ids) if q in pool_audit["query_mass"]]
                            new_u, meta = self.controller.update(self.u[active], vjp[active])
                            next_u = self.u.detach().clone()
                            next_u[active] = new_u.detach()
                            self.u = next_u.requires_grad_(True)
                            self.journal.append("teacher_weight_update", **context, vjp=vjp.tolist(), u=self.u.tolist(),
                                features_off=True, transport_term=False, identifiable=feedback.metadata["identifiable"], **meta)
                        commit_step(self.backend.model, updated, expected_start_hash=start_hash)
                    else:
                        if self.config.recipe.startswith("b3") and decision:
                            feedback = collect_groups(groups, self.support, self.backend, self.parameters,
                                self.feedback_rng, self.checker, self.journal)
                            g = combine_pg(g, feedback, parameter_hash=start_hash, coefficient=self.config.lambda_pg)
                        if self.optimizer:
                            self.optimizer.zero_grad(set_to_none=True)
                            for n, p in self.parameters.items():
                                p.grad = g[n].detach()
                            self.optimizer.step()
                        else:
                            commit_step(self.backend.model, rule.update(self.parameters, g), expected_start_hash=start_hash)
                costs = [s.costs(self.config.recipe) for s in selected]
                batch_tokens = sum(c["input_tokens"] for c in costs)
                round_tokens += batch_tokens + (feedback.metadata["score_input_tokens"]
                    if feedback and self.config.recipe.startswith("b3") else 0)
                max_batch = max(max_batch, batch_tokens)
                row = dict(round=r, step=step, decision=decision, reference_step=reference_step if decision else None,
                    inner_fold=(r-1)%2, optimizer_commits=1, raw_slots=len(selected), weighted_slots=len(selected),
                    source_slots=sum(c["source_slots"] for c in costs), teacher_slots=sum(c["teacher_slots"] for c in costs),
                    importance_weight_sum=sum(rhos),
                    source_exposure_action_tokens=sum(s.source.length for s in selected),
                    teacher_exposure_action_tokens=sum(s.teacher_tokens for s in selected),
                    loss=loss, start_hash=start_hash, actual_hash=tensor_state_hash(self.parameters),
                    pg_gradient_norm=feedback.metadata["gradient_norm"] if feedback and self.config.recipe.startswith("b3") else 0.,
                    feedback=feedback.metadata if feedback else None,
                    sampling_parameter_hash=feedback.parameter_hash if feedback else None,
                    supervised_tokens=batch_tokens, draws=[s.identity() | dict(p=s.p, q=dist.q[i], rho=dist.rho[i])
                        for s, i in zip(selected, batch)])
                self.journal.append("baseline_step", **row)
                self.steps.append(row)
            self.actual_tokens.append(round_tokens)
            self.max_batch_tokens.append(max_batch)
            if self.development_schedule:
                groups = [g for g in self.development_schedule if g["round"] == r]
                scores = collect_groups(groups, self.support, self.backend, self.parameters, self.development_rng,
                    self.checker, self.journal, tuning=True)
                self.development_scores.append(dict(round=r, parent_scores=scores,
                    score=sum(scores.values())/len(scores)))
            self.checkpoint(r, frozen, cache, rule)
        match = (budget_match(self.config.update_tokens, self.actual_tokens, self.max_batch_tokens)
                 if self.config.view == "tokens" else dict(matched=sum(len(e["draws"]) for e in self.steps) == 288))
        audit = dict(complete=True, passed=len(self.steps) == self.config.commits,
                     baseline=self.config.recipe, optimizer_commits=len(self.steps), exposure=match,
                     feedback_rollouts=sum((e["feedback"] or {}).get("rollouts", 0) for e in self.steps),
                     pg_nonzero_commits=sum(e["pg_gradient_norm"] > 0 and e["start_hash"] != e["actual_hash"] for e in self.steps),
                     resources=self.journal.summary(), development_scores=self.development_scores)
        atomic_json(self.directory/"audit.json", audit)
        return audit
