#!/usr/bin/env python3
"""CPU-only C26-E preparation. This entrypoint contains no RTD training loop."""
import argparse
import json
from pathlib import Path

from bfas.rtd.benchmarks.alfworld_config import load_config, bank_audit, manifest_section
from bfas.rtd.benchmarks.registry import get_benchmark

ROOT = Path(__file__).resolve().parents[1]
NOT_INTEGRATED = 'not yet integrated: run/resume require the C26-F edits'


def smoke_plan(root, config):
    audit = bank_audit(root, config)
    support = get_benchmark(config).support_protocol(root, config)
    bank = Path(audit['bank_path'])
    records = json.loads((bank / 'public/requests.json').read_text())
    selected, replay = {}, []
    for fold in (0, 1):
        parents = sorted(h for h in support.states if int(h, 16) % 2 == fold)
        if len(parents) < 4:
            raise ValueError('smoke retains four feedback parents per fold and K=2')
        selected[str(fold)] = parents[:4]
        # Replay two usable train demos, one per fold, independent of smoke selection.
        rows = sorted((r for r in records if r['unavailable_reason'] is None
                       and int(r['parent_hash'], 16) % 2 == fold), key=lambda r: r['spec']['query_id'])
        if not rows:
            raise ValueError('smoke needs a usable replay in each fold')
        q = rows[0]['spec']['query_id']
        payload = json.loads((bank / f'sealed/{q}.json').read_text())
        replay.append(dict(query_id=q, task_id=payload['provenance']['task_id'],
            successful_demo_commands=len(payload['commands']), reset_repetitions=2,
            failure_path='from a separate reset: fixed invalid raw action through existing parser; then 40 look actions or terminal',
            writes='new smoke directory only; never reseal the bank'))
    return dict(status='plan only; ' + NOT_INTEGRATED, gpu_used=False, host='rai',
        inputs=dict(config_hash=manifest_config_hash(config), bank=bank.as_posix(),
                    sealed_manifest_sha256=audit['verification']['manifest_sha256'],
                    support_hash=audit['support_manifest_hash']),
        resources=dict(student=config['student'], local_weights_only=True,
            gpu_devices_for_future_window=1, new_teacher_calls=0, new_teacher_tokens=0,
            cpu_environment_workers_concurrent=1, max_episode_steps=40, max_action_tokens=256,
            max_context_tokens=config['max_context_tokens'],
            memory={k: config[k] for k in ('max_state_batch_size', 'memory_peak_budget_gb',
                'memory_reserve_gb', 'memory_state_estimate_gb')},
            rounds=1, committed_steps=1, decision_steps=[1], slots=8,
            source_samples_per_state=2, meta_tasks_per_feedback=4, rollouts_per_meta_task=2,
            maximum_feedback_branches=2, maximum_feedback_episodes=16,
            maximum_feedback_environment_actions=640, maximum_feedback_action_tokens=163840,
            reset_source_actions=8, maximum_reset_source_action_tokens=2048,
            additional_source_actions='2 per distinct owned teacher state; depends on explicit acquisition draw',
            checkpoint='one committed step; partial smoke, never a full 12-step round',
            official_evaluation='separate 140-task greedy campaign after releasing training model; up to 5600 actions / 1433600 action tokens'),
        parent_hashes_by_fold=selected, task_ids_by_fold={f: [support.parents[h] for h in hs]
            for f, hs in selected.items()}, train_replays=replay,
        steps=[
            'Apply reviewed C26-F edits in an isolated checkout; run CPU acceptance tests and audit.',
            'On rai, replay both listed train demos twice; compare every reset/prefix/state hash, then exercise both failure paths.',
            'Bind one free rai GPU UUID and a fresh output directory; initialize Qwen3.5-4B LoRA and freeze source/P/train-only eta.',
            'Run existing RTDExperiment round 1 step 1: reference -> selected -> revealed -> actual -> feedback -> committed.',
            'Retain M=4/K=2, eight slots, positive mixture and LOO; reference/actual reuse follows the existing no-op rule.',
            'Check score tolerance, owned/fold/ledger, RNG checkpoint/resume and identities; release model before greedy campaign.',
            'Run/reuse only a complete 140-task valid_seen campaign; zero rewards or unchanged gate are valid outcomes.'])


def manifest_config_hash(config):
    from bfas.rtd.persistence import digest
    return digest(config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('audit', 'smoke-plan', 'run', 'resume'):
        sub = commands.add_parser(name)
        sub.add_argument('--root', type=Path, default=ROOT)
        sub.add_argument('--config', type=Path, default=Path('configs/rtd/v1_alfworld_c26.yaml'))
    args = parser.parse_args(argv)
    if args.command in ('run', 'resume'):
        print(NOT_INTEGRATED)
        return 2
    root = args.root.resolve()
    try:
        config = load_config(root / args.config)
        if args.command == 'audit':
            section = manifest_section(root, config)
            result = dict(status='passed', stage='C26-E preparation only', gpu_used=False,
                environment_started=False, training_integrated=False, manifest_section=section)
        else:
            result = smoke_plan(root, config)
    except (ValueError, OSError) as exc:
        parser.exit(1, f'C26-E {args.command} failed: {exc}\n')
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
