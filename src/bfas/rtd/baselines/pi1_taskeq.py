"""Additive pi1 task-equal CE registration and supervised-token endpoints.

The legacy row CE, encoder, AdamW state and exhaustive pass schedule are shared.
Only task-equal runs multiply the existing row accumulation coefficient by w_t.
"""
from collections import Counter
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from ..persistence import atomic_json, digest, fsync_directory, tree_hash
from .exposure import ExposureDistribution
from .pi1 import PlainCEState, RECIPE, VERSION, plain_ce_hyperparameters


TASK_EQUAL_FORMULA = (
    "T_t = sum_{r in update: r.task_id = t} supervised_tokens(r); "
    "N_update = sum_t T_t; n_tasks_update = number of distinct row.task_id in update; "
    "w_t = (N_update / n_tasks_update) / T_t; "
    "loss = sum_{supervised tokens i in update}(w_{task(i)} * nll_i) / N_update"
)


def _positive_endpoints(endpoints):
    if (not isinstance(endpoints, list) or not endpoints
            or any(type(t) is not int or t < 1 for t in endpoints)
            or endpoints != sorted(set(endpoints))):
        raise ValueError("exposure_tokens requires increasing positive integer endpoints")


def registered_recipe(config):
    """Keep legacy pins; the new registrations allow seeds 0, 1 and 2."""
    method = config.get("method")
    if method not in {"pi1_ce", "pi1_ce_taskeq"}:
        raise ValueError("pre-registered pi1 recipe differs: method")
    recipe = dict(RECIPE, method=method, training_seeds=[0, 1, 2])
    if "exposure_tokens" in config:
        if "exposure_passes" in config:
            raise ValueError("specify exactly one of exposure_tokens and exposure_passes")
        _positive_endpoints(config["exposure_tokens"])
        recipe.pop("exposure_passes")
        recipe["exposure_tokens"] = config["exposure_tokens"]
    return recipe


def task_equal_hyperparameters(config, seed):
    return dict(plain_ce_hyperparameters(config, seed), method="pi1_ce_taskeq",
        loss_normalization="task_equal_weight_per_update", loss_formula=TASK_EQUAL_FORMULA,
        task_definition="support row.task_id; pool all packages of that task within each update")


def task_equal_token_weights(task_ids, costs):
    """Return w_t, counting every occurrence of a complete row in this update."""
    if (not costs or len(task_ids) != len(costs)
            or any(type(c) is not int or c < 1 for c in costs)):
        raise ValueError("aligned task IDs and positive supervised-token counts required")
    totals = Counter()
    for task, count in zip(task_ids, costs):
        totals[task] += count
    mass_per_task = sum(costs) / len(totals)
    return {task: mass_per_task / count for task, count in totals.items()}


def token_exposure_plan(costs, config, seed):
    """Freeze the first update reaching each threshold, with legacy pass flushes.

    A target equal to an integer number of bank passes keeps the existing short
    endpoint update at that pass boundary. All other saves use the first ordinary
    >= tokens_per_update batch reaching the target. Multiple targets crossed in
    one update share its checkpoint; no turn is split and the RNG stream is the
    same np.random.default_rng(seed).permutation stream as pass_schedule.
    """
    if "exposure_passes" in config:
        raise ValueError("specify exactly one of exposure_tokens and exposure_passes")
    targets = config["exposure_tokens"]
    _positive_endpoints(targets)
    budget = config["supervised_tokens_per_update"]
    if (not costs or any(type(c) is not int or c < 1 for c in costs)
            or type(budget) is not int or budget < 1):
        raise ValueError("positive integer costs and token budget required")
    bank_tokens = sum(costs)
    distribution = ExposureDistribution(tuple(c/bank_tokens for c in costs), tuple(costs), "tokens")
    rng = np.random.default_rng(seed)
    batches, saves, pending = [], [], []
    tokens, total, next_target = 0, 0, 0
    pass_targets = {t for t in targets if t % bank_tokens == 0}
    while next_target < len(targets):
        order = rng.permutation(len(costs)).tolist()
        for offset, index in enumerate(order):
            pending.append(index)
            tokens += costs[index]
            pass_flush = offset == len(order)-1 and total+tokens in pass_targets
            if tokens < budget and not pass_flush:
                continue
            total += tokens
            reached = []
            while next_target < len(targets) and total >= targets[next_target]:
                target = targets[next_target]
                reached.append(target)
                saves.append(dict(target_supervised_tokens=target, supervised_token_count=total,
                                  optimizer_step_count=len(batches)+1))
                next_target += 1
            batches.append(dict(indices=pending, supervised_tokens=tokens,
                                cumulative_tokens=total, endpoint=reached or None))
            pending, tokens = [], 0
            if next_target == len(targets):
                break
    return dict(rule="seeded exhaustive row permutation per pass; complete turns; "
        "first update >= token target; retain endpoint flush at exact bank-pass targets",
        endpoint_mode="tokens", costs=list(costs), bank_supervised_tokens=bank_tokens,
        row_probabilities=list(distribution.q), target_tokens=list(targets),
        endpoint_saves=saves, batches=batches)


class TokenCEState(PlainCEState):
    """Reuse durable optimizer/resume commits; publish token threshold receipts."""

    def publish_endpoint(self, targets):
        batch = self.batches[len(self.trace)-1]
        previous = self.batches[len(self.trace)-2]["cumulative_tokens"] if len(self.trace) > 1 else 0
        for target in targets:
            if (target not in (batch["endpoint"] or []) or not previous < target <= self.t.scored_tokens
                    or self.t.scored_tokens != batch["cumulative_tokens"]):
                raise ValueError("exposure endpoint missed its first update")
            directory = self.t.directory/f"tokens-{target}"
            receipt = dict(version=VERSION, identity=self.identity, identity_hash=digest(self.identity),
                method=self.t.method, endpoint_mode="tokens", seed=self.t.seed, bank=self.identity["bank"],
                hyperparameters=self.t.hp, exposure_endpoint=target, target_supervised_tokens=target,
                supervised_token_count=self.t.scored_tokens, optimizer_step_count=len(self.trace),
                loss_trace="loss_trace.json", adapter="lora", loss_trace_hash=digest(self.trace),
                exposure_check=dict(matched=True, rule="first_update_at_or_above_target",
                    previous_cumulative_tokens=previous, target_tokens=target,
                    actual_tokens=self.t.scored_tokens, overshoot_tokens=self.t.scored_tokens-target,
                    exact=self.t.scored_tokens == target))
            if directory.exists():
                prior = json.loads((directory/"manifest.json").read_text())
                if (prior != dict(receipt, adapter_hash=tree_hash(directory/"lora"))
                        or json.loads((directory/"loss_trace.json").read_text()) != self.trace):
                    raise ValueError("existing endpoint differs; refusing overwrite")
                continue
            with tempfile.TemporaryDirectory(prefix=f".tokens-{target}-", dir=self.t.directory) as temporary:
                staged = Path(temporary)/"checkpoint"
                staged.mkdir()
                self.t.save_adapter(staged/"lora")
                atomic_json(staged/"loss_trace.json", self.trace)
                atomic_json(staged/"manifest.json", dict(receipt, adapter_hash=tree_hash(staged/"lora")))
                os.rename(staged, directory)
                fsync_directory(directory.parent)

    def publish_progress(self):
        saves = [dict(target_supervised_tokens=target, supervised_token_count=b["cumulative_tokens"],
                      optimizer_step_count=step)
                 for step, b in enumerate(self.batches[:len(self.trace)], 1)
                 for target in (b["endpoint"] or [])]
        endpoints = [s["target_supervised_tokens"] for s in saves]
        self.manifest.update(status="trained" if len(self.trace) == len(self.batches) else "training",
            exposure_endpoint=endpoints[-1] if endpoints else None, endpoints=endpoints,
            endpoint_saves=saves, supervised_token_count=self.t.scored_tokens,
            optimizer_step_count=len(self.trace))
        atomic_json(self.t.directory/"loss_trace.json", self.trace)
        atomic_json(self.t.directory/"manifest.json", self.manifest)
