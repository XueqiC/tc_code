"""Explicit external quotas, fresh on-policy BFCL rollouts, public REINFORCE."""
from collections import defaultdict
from dataclasses import asdict

import torch

from ...behavior.deltas import tensor_state_hash
from ..return_gradient import ReturnGradient, reinforce_gradient


def collect_groups(groups, support, backend, parameters, generator, checker, journal, *, role="update", tuning=False):
    """Groups independently compute same-task LOO, then average by rollout count.

    reference/actual labels are resource provenance only. Every non-reused group
    is freshly sampled at the same supplied baseline parameters. Reused actual
    consumes zero new rollouts; cached RTD answers/rewards are never accepted.
    BFCLSupport.feedback calls the existing public bfcl_task_rollout function.
    """
    if not groups:
        raise ValueError("missing explicit feedback quota")
    results, counts, rewards, score_tokens = [], [], defaultdict(list), 0
    sampling_hash = tensor_state_hash(parameters)
    for group in groups:
        context = dict(round=group["round"], step=group["step"], role=group["role"], tuning=tuning)
        if group["reused"]:
            journal.append("baseline_feedback_reused", sampling_parameter_hash=sampling_hash, **context)
            continue
        rollouts = []
        seen = set()
        with journal.phase("baseline_feedback", **context):
            for task in group["tasks"]:
                parent, count = task["parent_hash"], task["rollout_count"]
                if (parent in seen or parent not in support.states or type(count) is not int or count < 2
                        or int(parent, 16) % 2 == (group["round"]-1) % 2):
                    raise ValueError("duplicate/unclean feedback parent or invalid LOO quota")
                seen.add(parent)
                for _ in range(count):
                    rollout = support.feedback(parent, backend, parameters, generator, checker)
                    if rollout.task_id != support.parents[parent] or rollout.policy_id != backend.identity(parameters):
                        raise ValueError("on-policy feedback parent/sampling parameter mismatch")
                    rollouts.append(rollout)
                    rewards[parent].append(rollout.reward)
                    journal.append("feedback_rollout", parent_hash=parent, rollout=asdict(rollout),
                                   sampling_parameter_hash=sampling_hash, **context)
                    for action in rollout.actions:
                        count_tokens = len(action.prompt_ids)+len(action.action_ids)
                        journal.usage("tuning" if tuning else "generation", input_tokens=count_tokens,
                                      action_tokens=len(action.action_ids), backward=False, operation="feedback_generation",
                                      round=group["round"], step=group["step"])
                        score_tokens += 0 if tuning else count_tokens
            if not tuning:
                result = reinforce_gradient(rollouts, backend, parameters,
                    diagnostic_record=lambda rollout, index, diagnostic: journal.append("score_consistency",
                        diagnostic=diagnostic, task_id=rollout.task_id, action_index=index, **context))
                for rollout in rollouts:
                    for action in rollout.actions:
                        journal.usage(role, input_tokens=len(action.prompt_ids)+len(action.action_ids),
                                      action_tokens=len(action.action_ids), operation="feedback_score_backward",
                                      round=group["round"], step=group["step"])
                journal.append("return_gradient", parameter_hash=result.parameter_hash, metadata=result.metadata, **context)
                results.append(result)
                counts.append(len(rollouts))
            else:
                # Tuning observes terminal support returns only, with no score/backprop.
                journal.append("baseline_tuning_block", rewards={p: list(v) for p, v in rewards.items()}, **context)
    if tensor_state_hash(parameters) != sampling_hash:
        raise ValueError("student moved during feedback sampling")
    if tuning:
        return {p: sum(values)/len(values) for p, values in rewards.items()}
    if not results:
        raise ValueError("feedback schedule contains no physical group")
    total = sum(counts)
    gradient = {n: sum(result.gradient[n] * count / total for result, count in zip(results, counts))
                for n in parameters}
    return ReturnGradient(gradient, sampling_hash, dict(rollouts=total, groups=len(results),
        identifiable=any(r.metadata["identifiable"] for r in results), score_input_tokens=score_tokens,
        group_metadata=[r.metadata for r in results],
        gradient_norm=float(torch.sqrt(sum(g.double().square().sum() for g in gradient.values())))))


def combine_pg(supervised, feedback, *, parameter_hash, coefficient=1.):
    if feedback.parameter_hash != parameter_hash:
        raise ValueError("PG must be sampled at the same pre-commit supervised parameters")
    if tuple(supervised) != tuple(feedback.gradient):
        raise ValueError("PG parameter layout differs")
    return {n: g - coefficient*feedback.gradient[n].detach() for n, g in supervised.items()}
