#!/usr/bin/env python3
"""D12 GPU benchmark: 40 real support slots x 2 actions, original HF vs batching.

Run from this checkout with PYTHONPATH=src:. and exactly one visible GPU.
Uses local weights only. --window selects an authenticated archived source
student; otherwise use the real configured base student with initial LoRA.
The fixed D5/D6 task set explicitly repeats available states when a fold has
fewer than 20 runnable parents. The report includes every slot and unique count.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from bfas.rtd.functional_step import lora_parameters, snapshot
from bfas.rtd.generation_batch import GenerationBatch, action_cap
from bfas.rtd.forward_batch import source_score_scope
from bfas.rtd.metrics_v11 import freeze_tasks, task_rows
from bfas.rtd.persistence import ComputeJournal, atomic_json, digest, file_hash, tree_hash
from bfas.rtd.runtime import HFGenerateBackend, load_backend
from bfas.rtd.scoring import ScoreTolerance, score_diagnostic


def agreement(backend, actions, parameters):
    """Unbatched teacher forcing; enforce the existing guard per ACTION."""
    diagnostics = []
    with torch.no_grad(), source_score_scope(backend, actions, parameters) as draws:
        for action in draws:
            score, values, metadata = backend.score_action(action, parameters, return_details=True)
            row = score_diagnostic(action, values, score, metadata, backend.score_tolerance,
                                   expected_prompt_ids=action.prompt_ids)
            if backend.journal:
                backend.journal.append('score_consistency', context=backend.context, **row)
            diagnostics.append(row)
    tokens = sum(d['n_tokens'] for d in diagnostics)
    outliers = sum(abs(float(g)-float(t)) > 1. for d in diagnostics
                   for g, t in zip(d['generation_token_logprobs'], d['teacher_forced_token_logprobs']))
    return dict(passed=all(d['passed'] for d in diagnostics), tokens=tokens,
        max_abs_diff=max(d['max_abs_difference'] for d in diagnostics),
        mean_abs_diff=sum(d['mean_abs_difference']*d['n_tokens'] for d in diagnostics)/tokens,
        fraction_gt_1_nat=outliers/tokens, tokens_gt_1_nat=outliers,
        max_outliers_per_action=max(d['outlier_token_count'] for d in diagnostics),
        failed_action_indices=[i for i, d in enumerate(diagnostics) if not d['passed']],
        tolerance=asdict(backend.score_tolerance))


def run_case(backend, rows, support, parameters, *, seed, batched):
    generator = torch.Generator(device='cuda:0').manual_seed(seed)
    backend.context = 'benchmark_batched' if batched else 'benchmark_unbatched'
    requests = [(support.states[r['parent_hash']].prompt, 2,
        action_cap(backend, support.categories[support.parents[r['parent_hash']]])) for r in rows]
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    begin = len(backend.journal.events)
    with backend.journal.measure_phase('benchmark_generation', case=backend.context):
        if batched:
            # This is also the mixed-cap planner used by source-pair/parse prefetch.
            actions = backend._sample_requests(requests, parameters, generator)
        else:
            actions = []
            for row in rows:
                parent = row['parent_hash']
                with backend.action_limit(support.categories[support.parents[parent]]):
                    for _ in range(2):
                        actions.append(backend.sample_action(support.states[parent].prompt, parameters, generator))
    torch.cuda.synchronize()
    seconds = time.perf_counter()-started
    # Nested generation/scoring phases reset CUDA's counters. Use the journal's
    # ancestor accumulator, not a final read of the last call's reset counter.
    memory = next(r for r in reversed(backend.journal.events)
                  if r['kind'] == 'subphase_memory' and r['operation'] == 'benchmark_generation')
    peak, reserved = memory['peak_allocated_bytes'], memory['peak_reserved_bytes']
    calls = sum(r['kind'] == 'compute_begin' and r.get('operation') == 'generation'
                for r in backend.journal.events[begin:])
    tokens = sum(len(a.action_ids) for a in actions)
    # Rescoring has its own timer and memory report; never included in tok/s.
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with backend.journal.measure_phase('benchmark_rescoring', case=backend.context):
        scores = agreement(backend, actions, parameters)
    torch.cuda.synchronize()
    scoring_memory = next(r for r in reversed(backend.journal.events)
                         if r['kind'] == 'subphase_memory' and r['operation'] == 'benchmark_rescoring')
    return dict(actions=len(actions), generation_calls=calls, wall_seconds=seconds,
        action_tokens=tokens, tokens_per_second=tokens/seconds,
        peak_allocated_gb=peak/1e9, peak_reserved_gb=reserved/1e9,
        rescoring_wall_seconds=time.perf_counter()-started,
        rescoring_peak_allocated_gb=scoring_memory['peak_allocated_bytes']/1e9,
        complete=sum(not a.truncated for a in actions), truncated=sum(a.truncated for a in actions),
        agreement=scores, action_hashes=[digest(a.action_ids) for a in actions],
        rng_after=digest(generator.get_state().tolist()), backend_id=backend.backend_id)


def main(argv=None, *, forward_only=False):
    from bfas.rtd.cli import ROOT, load_config
    parser = argparse.ArgumentParser(description='D13: same-action teacher-forced forward comparison' if forward_only else __doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/rtd/v1_1_bfcl.yaml')
    parser.add_argument('--window', type=Path, help='optional authenticated controls/windows/rN-sNN.pt')
    parser.add_argument('--model', type=Path, help='local base checkpoint directory (no downloads)')
    parser.add_argument('--out', type=Path, default=ROOT/'results/rtd_v1_1'/('forward_bench' if forward_only else 'generation_bench'))
    parser.add_argument('--prompts-per-batch', type=int, default=8)
    parser.add_argument('--max-batch-tokens', type=int, default=16384)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--kernels', choices=('auto', 'torch'), default='auto')
    parser.add_argument('--order', choices=('unbatched-first', 'batched-first'), default='unbatched-first')
    if forward_only:
        parser.add_argument('--actions', type=Path, help='D12 compute.jsonl or D13 actions.json; omitted = fresh 80 samples')
        parser.add_argument('--forward-prompts-per-batch', type=int, default=8)
        parser.add_argument('--atol', type=float, default=1e-4, help='maximum absolute per-token difference; default 1e-4')
    args = parser.parse_args(argv)
    policy = GenerationBatch(args.prompts_per_batch, args.max_batch_tokens,
                             args.forward_prompts_per_batch if forward_only else 0)
    if forward_only and (not torch.isfinite(torch.tensor(args.atol)) or args.atol < 0 or not args.forward_prompts_per_batch):
        parser.error('finite nonnegative --atol and positive --forward-prompts-per-batch required')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error('exactly one visible GPU required; this script is not a CPU performance proxy')
    if args.out.exists():
        parser.error('choose a new --out directory')
    config = load_config(args.config, arm='V0')
    source = None
    if args.window:
        from bfas.rtd.controls_v11 import load_window
        payload = load_window(args.window)
        manifest = payload['manifest']
        config = dict(manifest['config'])
        source = payload['source']
        del payload
        if args.model and args.model.resolve() != Path(manifest['model_path']).resolve():
            parser.error('--window and --model identify different base checkpoints')
        if tree_hash(Path(manifest['model_path'])) != manifest['base_checkpoint_hash']:
            parser.error('archived base checkpoint content changed')
    else:
        from tools.bfcl_hub_merge_export import _snapshot_for_model
        model = args.model or Path(config['student'])
        if not model.is_dir():
            if args.model:
                parser.error('--model must be a local checkpoint directory')
            model = _snapshot_for_model(config['student'])
        manifest = dict(model_path=str(model.resolve()), base_checkpoint_hash=tree_hash(model),
            harness_hash=digest(['d12-generation-benchmark', config['support_manifest']]),
            tokenizer_hash=digest([(p.name, file_hash(p)) for p in sorted(model.glob('*'))
                if p.is_file() and any(k in p.name for k in ('token', 'vocab', 'merges', 'chat_template'))]))
    if config['benchmark'] != 'bfcl':
        parser.error('this benchmark requires real BFCL support states')
    config = config | dict(generation_batch=asdict(policy))
    # The benchmark cannot weaken a saved run's score guard.
    saved_tolerance, guard = asdict(ScoreTolerance.from_config(config)), asdict(ScoreTolerance())
    config['score_consistency_tolerance'] = {k: min(v, saved_tolerance[k]) for k, v in guard.items()}
    from bfas.rtd.experiment import BFCLSupport
    support = BFCLSupport(ROOT, config)
    fixed = freeze_tasks([dict(parent_hash=h, official_id=t) for h, t in support.parents.items()],
                         short_fold='repeat', states=support.states)
    rows = task_rows(fixed, support)
    assert len(rows) == 40
    args.out.mkdir(parents=True)
    journal = ComputeJournal(args.out/'compute.jsonl', cuda=True)
    batched = load_backend(config, manifest, journal)
    if args.kernels == 'torch':
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        for module in batched.model.modules():
            if isinstance(module, qwen.Qwen3_5GatedDeltaNet):
                module.causal_conv1d_fn = None
                module.causal_conv1d_update = qwen.torch_causal_conv1d_update
                module.chunk_gated_delta_rule = qwen.torch_chunk_gated_delta_rule
                module.recurrent_gated_delta_rule = qwen.torch_recurrent_gated_delta_rule
                norm = qwen.Qwen3_5RMSNormGated(module.head_v_dim, eps=module.layer_norm_epsilon).to(
                    device=module.norm.weight.device, dtype=module.norm.weight.dtype)
                with torch.no_grad():
                    norm.weight.copy_(module.norm.weight)
                module.norm = norm.requires_grad_(False).eval()
    parameters = (snapshot(lora_parameters(batched.model)) if source is None else
                  {n: p.to(device='cuda:0') for n, p in source.items()})
    legacy = HFGenerateBackend(batched.model, batched.tokenizer,
        base_checkpoint_hash=batched.base_checkpoint_hash, harness_hash=batched.harness_hash,
        tokenizer_hash=manifest['tokenizer_hash'], max_action_tokens=batched.max_action_tokens,
        max_context_tokens=batched.max_context_tokens, action_caps=batched.action_caps,
        journal=journal, score_tolerance=batched.score_tolerance)
    # Warm both paths with a real short prompt and a separate, discarded RNG.
    prompt = min((s.prompt for s in support.states.values()), key=lambda p: len(batched._prompt_ids(p)))
    for b in (legacy, batched):
        previous = b.max_action_tokens
        b.max_action_tokens = 8
        b.context = 'benchmark_warmup'
        try:
            b.sample_action(prompt, parameters, torch.Generator(device='cuda:0').manual_seed(args.seed+1))
        finally:
            b.max_action_tokens = previous
    report = dict(version='rtd-v11-d12-generation-benchmark', seed=args.seed, kernels=args.kernels,
        gpu=torch.cuda.get_device_name(), generation_batch=asdict(policy), model_path=manifest['model_path'],
        source='archived frozen source' if source is not None else 'configured base with initial LoRA',
        window=str(args.window) if args.window else None, fixed_task_set=fixed, slots=rows,
        unique_states=len({r['state_hash'] for r in rows}), samples_per_state=2,
        sampling='temperature=1 top_p=1 top_k=0; independent realizations, identical categorical law',
        torch_version=torch.__version__, order=args.order)
    if forward_only:
        from tools.rtd_v11_forward_bench import run_forward_benchmark
        return run_forward_benchmark(batched, rows, support, parameters, args, report)
    order = [('unbatched', legacy), ('batched', batched)]
    if args.order == 'batched-first':
        order.reverse()
    for name, b in order:
        print(f'Running {name}: 40 support slots, 80 actions', flush=True)
        report[name] = run_case(b, rows, support, parameters, seed=args.seed, batched=name == 'batched')
        print(json.dumps({name: report[name]}, indent=2), flush=True)
        atomic_json(args.out/'report.json', report)
    report['wall_speedup'] = report['unbatched']['wall_seconds']/report['batched']['wall_seconds']
    report['token_throughput_speedup'] = report['batched']['tokens_per_second']/report['unbatched']['tokens_per_second']
    report['score_agreement_passed'] = all(report[k]['agreement']['passed'] for k in ('unbatched', 'batched'))
    report['batched_generation_under_40gb'] = report['batched']['peak_allocated_gb'] < 40
    atomic_json(args.out/'report.json', report)
    print(json.dumps({k: report[k] for k in ('wall_speedup', 'token_throughput_speedup',
        'score_agreement_passed', 'batched_generation_under_40gb')}, indent=2), flush=True)
    return 0 if report['score_agreement_passed'] and report['batched_generation_under_40gb'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
