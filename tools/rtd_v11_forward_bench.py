#!/usr/bin/env python3
"""D13 GPU comparison on the SAME 80 real teacher-forced actions.

Use --actions D12_OUTPUT/compute.jsonl or a previous forward actions.json;
omit it to generate fresh samples using the unchanged D12 sampler. The common
D12 CLI loads only local weights, optional authenticated --window, and BFCL.
"""
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch

from bfas.behavior.deltas import tensor_state_hash
from bfas.rtd.forward_batch import plan_forwards
from bfas.rtd.generation_batch import action_cap
from bfas.rtd.persistence import atomic_json, digest
from bfas.rtd.return_gradient import ActionTrace
from bfas.rtd.scoring import score_diagnostic


def load_actions(path):
    """Read D12 per-action score records or D13's exact ActionTrace export."""
    path = Path(path)
    if path.suffix == '.jsonl':
        with path.open() as stream:
            records = [r for line in stream if (r := json.loads(line))['kind'] == 'score_consistency'
                       and r.get('context') == 'benchmark_batched']
        actions = [ActionTrace(tuple(r['prompt_ids']), tuple(r['action_ids']), r['eos']['token_id'],
            r['generated_text'], r['generation_logprob'], r['backend_id'], r['policy_id'],
            tuple(r['generation_token_logprobs']), r['generation_backend'], truncated=r['truncated'])
            for r in records]
    else:
        records = json.loads(path.read_text())['actions']
        actions = [ActionTrace(**(r | {k: tuple(r[k]) for k in
                    ('prompt_ids', 'action_ids', 'generation_token_logprobs')})) for r in records]
    if len(actions) != 80:
        raise ValueError(f'expected exactly 80 real actions, found {len(actions)}')
    return tuple(actions)


def compare_logprobs(reference, batched, *, atol):
    """Every position of every action; nonfinite/missing values always fail."""
    if not torch.isfinite(torch.tensor(atol)) or atol < 0:
        raise ValueError('finite nonnegative tolerance required')
    if len(reference) != len(batched) or not reference:
        raise ValueError('aligned nonempty action scores required')
    errors, failures = [], []
    for i, (old, new) in enumerate(zip(reference, batched)):
        old, new = torch.as_tensor(old, dtype=torch.float64), torch.as_tensor(new, dtype=torch.float64)
        if old.shape != new.shape or old.ndim != 1 or not old.numel():
            raise ValueError('per-token coverage changed')
        finite = bool(torch.isfinite(old).all() and torch.isfinite(new).all())
        error = float((old-new).abs().max()) if finite else None
        errors.append(error)
        if error is None or error > atol:
            failures.append(i)
    return dict(passed=not failures, atol=atol, failed_action_indices=failures,
                max_abs_logprob_diff=max(errors) if all(e is not None for e in errors) else None,
                per_action_max_abs_diff=errors, tokens=sum(len(v) for v in reference))


def score_case(backend, actions, parameters, *, batched):
    original = backend.generation_batch
    backend.generation_batch = replace(original, forward_prompts_per_batch=
                                       original.forward_prompts_per_batch if batched else 0)
    backend.context = 'forward_benchmark_batched' if batched else 'forward_benchmark_unbatched'
    journal = backend.journal
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    started, begin = time.perf_counter(), len(journal.events)
    try:
        with journal.measure_phase('benchmark_forward', case=backend.context), torch.no_grad():
            details = backend.score_actions_batch(actions, parameters, return_details=True)
            # Save/compare CPU per-token values without retaining GPU tensors.
            values = [r[1].cpu().tolist() for r in details]
        torch.cuda.synchronize()
        seconds = time.perf_counter()-started
        memory = next(r for r in reversed(journal.events)
                      if r['kind'] == 'subphase_memory' and r['operation'] == 'benchmark_forward')
        checks = [score_diagnostic(a, v, s, m, backend.score_tolerance, expected_prompt_ids=a.prompt_ids)
                  for a, (s, v, m) in zip(actions, details)]
        for row in checks:
            journal.append('score_consistency', context=backend.context, **row)
        return dict(wall_seconds=seconds, actions=len(actions), action_tokens=sum(map(len, values)),
            peak_allocated_gb=memory['peak_allocated_bytes']/1e9,
            peak_reserved_gb=memory['peak_reserved_bytes']/1e9,
            forward_calls=sum(r['kind'] == 'compute_begin' and r.get('operation') == 'teacher_forced_forward'
                              for r in journal.events[begin:]),
            generation_score_guard_passed=all(r['passed'] for r in checks)), values
    finally:
        backend.generation_batch = original


def run_forward_benchmark(backend, rows, support, parameters, args, report):
    if args.actions:
        actions = load_actions(args.actions)
    else:
        requests = [(support.states[r['parent_hash']].prompt, 2,
            action_cap(backend, support.categories[support.parents[r['parent_hash']]])) for r in rows]
        backend.context = 'forward_benchmark_fresh_samples'
        actions = backend._sample_requests(requests, parameters, torch.Generator(device='cuda:0').manual_seed(args.seed))
    if len(actions) != 80:
        raise ValueError('expected exactly 80 real actions')
    # No rebinding saved actions to a different student/backend to pass checks.
    identity = backend.identity(parameters)
    if any(a.backend_id != backend.backend_id or a.policy_id != identity for a in actions):
        raise ValueError('saved actions differ from the loaded backend/source; use their original --window/config/model')
    atomic_json(args.out/'actions.json', dict(version='rtd-v11-d13-forward-actions',
        parameter_hash=tensor_state_hash(parameters), backend_id=backend.backend_id, actions=[asdict(a) for a in actions]))
    groups = plan_forwards(actions, backend.generation_batch, backend.max_context_tokens)
    report.update(version='rtd-v11-d13-forward-benchmark', action_input=str(args.actions) if args.actions else 'fresh',
        parameter_hash=tensor_state_hash(parameters), action_hashes=[digest(asdict(a)) for a in actions],
        forward_plan=[dict(indices=[r['index'] for r in g], lengths=[len(r['ids']) for r in g],
                           padded_tokens=len(g)*max(len(r['ids']) for r in g)) for g in groups],
        tolerance=args.atol, gradient_batching=False)
    # Warm both scoring paths on the same real short actions, outside the report.
    warm = sorted(actions, key=lambda a: len(a.prompt_ids)+len(a.action_ids))[:2]
    for mode in (False, True):
        score_case(backend, warm, parameters, batched=mode)
    order = ('unbatched', 'batched') if args.order == 'unbatched-first' else ('batched', 'unbatched')
    scores = {}
    for name in order:
        print(f'Scoring {name}: the same 80 actions', flush=True)
        report[name], scores[name] = score_case(backend, actions, parameters, batched=name == 'batched')
        atomic_json(args.out/'report.json', report)
    report['agreement'] = compare_logprobs(scores['unbatched'], scores['batched'], atol=args.atol)
    report['wall_speedup'] = report['unbatched']['wall_seconds']/report['batched']['wall_seconds']
    report['batched_forward_under_40gb'] = report['batched']['peak_allocated_gb'] < 40
    report['passed'] = (report['agreement']['passed'] and report['batched_forward_under_40gb']
                       and all(report[n]['generation_score_guard_passed'] for n in ('unbatched', 'batched')))
    atomic_json(args.out/'scores.json', scores)
    atomic_json(args.out/'report.json', report)
    print(json.dumps({k: report[k] for k in ('unbatched', 'batched', 'agreement', 'wall_speedup', 'passed')}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    from tools.rtd_v11_generation_bench import main
    raise SystemExit(main(forward_only=True))
