#!/usr/bin/env python3
"""Purchase REPAIR-CE teacher suffixes after a preregistered student takeover."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]

from bfas.adapter import Demo, TeacherEpisode, Turn
from bfas.adapters.alfworld import ALFWorldAdapter
from bfas.ledger import AcquisitionStopped, _minimum_interval, _purchase_lock, append_episode
from bfas.rtd.benchmarks.alfworld_state import _observed, canonical_hash, replay_commands
from bfas.rtd.transport import FullState
from tools import alfworld_teacher_pool as pool
from tools.alf_bank_subset import read_json, register_config, sweep_statistics, SWEEP_PATTERN
from tools.alf_pi1_support_rollouts import frozen_inputs

RULE = dict(version='repair-ce-takeover-v1', index='zero-based; prefix commands[0:k]',
            trigger='first repeated command OR post-command observation casefold startswith Nothing happens.',
            tie_break='repeated_command', fallback='floor(steps/2)', successful_episodes='no request',
            max_steps=40, sweep_pattern=SWEEP_PATTERN, sweep_filter='teacher suffix share > 0.5')


def takeover(record):
    if type(record.get('success')) is not bool or record.get('status') != 'completed':
        raise ValueError('only completed episodes with a boolean environment outcome are eligible')
    commands, observations, steps = record['commands'], record['observations'], record['steps']
    if (type(steps) is not int or not 1 <= steps <= 40 or
            len(commands) != steps or len(observations) != steps or
            any(not isinstance(c, str) or not c.strip() for c in commands) or
            any(not isinstance(o, str) for o in observations)):
        raise ValueError('aligned executed commands and post-step observations required')
    if record['success']:
        return None
    seen = set()
    for k, (command, observation) in enumerate(zip(commands, observations)):
        repeat, nothing = command in seen, observation.casefold().startswith('nothing happens.')
        if repeat or nothing:
            return dict(k=k, rule='repeated_command' if repeat else 'nothing_happens',
                        triggers=[name for name, yes in [('repeated_command', repeat), ('nothing_happens', nothing)] if yes])
        seen.add(command)
    return dict(k=steps // 2, rule='midpoint', triggers=[])


def read_rollouts(directory, source, support, requests, resets):
    directory = Path(directory)
    index = read_json(directory / 'index.json')
    identity = index['identity']
    if (index.get('complete') is not True or identity['split'] != 'train' or
            identity['source_manifest_sha256'] != pool.bank.file_hash(source / 'sealed/manifest.json') or
            identity['support_manifest_hash'] != support['manifest_hash']):
        raise ValueError('rollouts must be complete and bound to this frozen source')
    entries = index['tasks']
    if [e['task_id'] for e in entries] != sorted(requests):
        raise ValueError('rollouts must cover each support task exactly once, in sorted order')
    records = {}
    for entry in entries:
        tid = entry['task_id']
        path = directory / entry['file']
        if entry['file'] != canonical_hash(tid) + '.json' or pool.bank.file_hash(path) != entry['sha256']:
            raise ValueError('rollout file identity mismatch')
        record = read_json(path)
        if (record['task_id'] != tid or record['split'] != 'train' or
                record['identity_hash'] != canonical_hash(identity) or
                record['request_state_hash'] != canonical_hash(requests[tid]) or
                record['reset_state_hash'] != FullState(**resets[tid]).state_hash or
                record['commands'] != [t['command'] for t in record['turns']] or
                record['observations'] != [t['observation'] for t in record['turns']]):
            raise ValueError('rollout record differs from the frozen task or executed turns')
        takeover(record)
        records[tid] = record
    return records


class CursorStepper:
    """Expose replay's reached cursor so the teacher continues in the same worker."""
    def __init__(self, stepper):
        self.stepper = stepper

    def reset(self, request):
        self.cursor, self.observation = self.stepper.reset(request)
        return self.cursor, self.observation

    def step(self, cursor, command):
        self.cursor, self.observation = self.stepper.step(cursor, command)
        return self.cursor, self.observation

    def close(self):
        self.stepper.close()


class RepairAdapter(pool.PoolAdapter):
    def __init__(self, *args, records, resets, request_directory, **kwargs):
        super().__init__(*args, **kwargs)
        self.records, self.resets, self.request_directory = records, resets, request_directory

    def teacher_episode(self, task_id, attempt_index, temperature):
        record = self.records[task_id]
        selection = takeover(record)
        if selection is None:
            raise ValueError('successful student episodes cannot be purchased')
        self.budget.ensure_available()
        request = self.support['tasks'][task_id]['request']
        prefix = record['commands'][:selection['k']]
        path = self.request_directory / (canonical_hash(task_id) + '.json')
        material = read_json(path)
        turns, commands, responses, observations = [], [], [], []
        won, stepper = False, None
        try:
            config = self.config_loader(self.teacher)
            stepper = CursorStepper(self.stepper_factory(request))
            replay = replay_commands(request, prefix, stepper, pool.render,
                                     student_prefix_steps=len(prefix), allow_empty=True)
            history = json.loads(replay.states[-1].history_json)
            expected = json.loads(self.resets[task_id]['history_json'])
            for index, turn in enumerate(record['turns'][:len(prefix)], 1):
                expected.extend([dict(role='assistant', index=index, content=turn['command']),
                                 dict(role='tool', index=index, content=turn['observation'],
                                      admissible=turn['admissible'], done=turn['done'])])
            if history != expected or replay.done:
                raise ValueError('student prefix replay differs from the recorded reached state')
            material.update(takeover_state=asdict(replay.states[-1]),
                            takeover_state_hash=replay.states[-1].state_hash, status='purchasing')
            pool.write_json(path, material)  # bind actual context before the first paid call
            for _ in range(40 - len(prefix)):
                messages = pool.prompt_messages(request, history)
                reply, _ = self.purchase(config, task_id, attempt_index, messages, temperature)
                responses.append(reply)
                command = ALFWorldAdapter._teacher_command(pool.appworld_teacher.strip_think(reply),
                                                          stepper.observation.admissible)
                context = pool.prompt_messages(request, history, react=False)
                turns.append(Turn(context[0]['content'], reply, context))
                commands.append(command)
                _, observation = stepper.step(stepper.cursor, command)
                observations.append(asdict(observation))
                history.extend([dict(role='assistant', index=observation.index, content=command), _observed(observation)])
                material.update(teacher_commands=list(commands), teacher_responses=list(responses),
                                teacher_observations=list(observations), **sweep_statistics(commands))
                pool.write_json(path, material)
                if observation.done:
                    won = observation.won
                    break
        except Exception as exc:
            material['error'] = dict(type=type(exc).__name__, message=str(exc))
            if isinstance(exc, AcquisitionStopped):
                self.budget.cancel(str(exc))
        finally:
            if stepper is not None:
                try:
                    stepper.close()
                except Exception as exc:
                    won = False
                    material['cleanup_error'] = dict(type=type(exc).__name__, message=str(exc))
            usage = self.budget.usage(task_id, attempt_index)
            material.update(status='collected' if won else 'failed', teacher_commands=commands,
                            teacher_responses=responses, teacher_observations=observations,
                            environment_success=won, usage=usage, **sweep_statistics(commands))
            pool.write_json(path, material)
        demo = Demo(task_id, tuple(turns), '\n'.join(commands)[-4000:],
                    raw=dict(teacher_commands=commands, student_prefix_commands=prefix)) if won and turns else None
        return TeacherEpisode(task_id, demo is not None, demo, tuple(responses),
                              usage['completion_tokens'], teacher=self.teacher, usage=usage)


def collect_repair(rollouts, source, source_collection, collection, output, *,
                   max_tokens=400000, max_usd=Decimal('5'), attempt_index=0, temperature=0.0,
                   rate_limit_retries=pool.appworld_teacher.RATE_LIMIT_RETRIES,
                   stepper_factory=pool.real_stepper, generate_reply=None, config_loader=None,
                   support_size=32, config_path=None, model_path=None):
    import math
    import os
    if (type(attempt_index) is not int or attempt_index < 0 or
            not math.isfinite(temperature) or temperature < 0 or
            type(rate_limit_retries) is not int or rate_limit_retries < 0):
        raise ValueError('invalid attempt index, temperature, or retry count')
    source, source_collection = Path(source).resolve(), Path(source_collection).resolve()
    collection, output = Path(collection).absolute(), Path(output).absolute()
    destinations = [collection, output] + ([Path(config_path).absolute()] if config_path else [])
    if len({p.resolve() for p in destinations}) != len(destinations):
        raise ValueError('collection, bank, and config must be separate outputs')
    for destination in destinations:
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        if any(destination.resolve().is_relative_to(p.resolve()) for p in (source, source_collection, Path(rollouts))):
            raise ValueError('outputs must be outside input directories')
    for left in destinations:
        for right in destinations:
            if left != right and left.resolve().is_relative_to(right.resolve()):
                raise ValueError('collection, bank, and config must be separate outputs')
    if collection == output:
        raise ValueError('collection and bank must be separate outputs')
    support, requests, resets = frozen_inputs(source, support_size=support_size)
    records = read_rollouts(rollouts, source, support, requests, resets)
    original = read_json(source_collection / 'identity.json')
    if original['support_manifest'] != support['manifest_hash']:
        raise ValueError('source pricing identity is not bound to the frozen D0 support')
    # Bind the supplied pricing identity to the D0 ledger actually sealed in the source.
    source_ledger = source_collection / 'teacher_ledger.jsonl'
    inventory = read_json(source / 'sealed/audit.json')['historical_inventory']['source_files']
    if pool.bank.file_hash(source_ledger) not in {f['sha256'] for f in inventory}:
        raise ValueError('source collection ledger is not the D0 bank inventory')
    limits = pool.Limits(max_tokens, max_usd, *map(Decimal, original['prices']))
    teacher = original['teacher']
    os.environ['BFAS_OPENAI_SERVICE_TIER'] = original.get('service_tier', 'flex')
    collection.mkdir(parents=True, exist_ok=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    request_directory = collection / 'requests'
    request_directory.mkdir()
    identity = dict(version='repair-ce-v1', source_bank=str(source),
                    source_manifest_sha256=pool.bank.file_hash(source / 'sealed/manifest.json'),
                    source_collection=str(source_collection), source_identity_sha256=pool.bank.file_hash(source_collection / 'identity.json'),
                    rollouts_index_sha256=pool.bank.file_hash(Path(rollouts) / 'index.json'),
                    rule=RULE, teacher=teacher, prices=original['prices'],
                    attempt_index=attempt_index, temperature=temperature,
                    max_tokens=max_tokens, max_usd=str(max_usd), rate_limit_retries=rate_limit_retries,
                    source_files={name: pool.bank.file_hash(ROOT / name) for name in (
                        'tools/alf_repair_collect.py', 'tools/alfworld_teacher_pool.py',
                        'src/bfas/rtd/benchmarks/alfworld_state.py', 'src/bfas/rtd/benchmarks/alfworld_support.py')})
    pool.write_json(collection / 'identity.json', identity)
    selected = {}
    task_inventory = []
    for tid, record in records.items():
        selection = takeover(record)
        task_inventory.append(dict(task_id=tid, success=record['success'], takeover=selection))
        if selection is not None:
            material = dict(task_id=tid, request_state_hash=canonical_hash(requests[tid]),
                            reset_state_hash=FullState(**resets[tid]).state_hash, **selection,
                            student_prefix_commands=record['commands'][:selection['k']],
                            remaining_horizon=40-selection['k'], status='pending',
                            **sweep_statistics([]))
            selected[tid] = material
            pool.write_json(request_directory / (canonical_hash(tid) + '.json'), material)
    pool.write_json(collection / 'tasks.json', task_inventory)
    ledger = collection / 'teacher_ledger.jsonl'
    ledger.touch()
    budget = pool.Budget(collection / 'usage.jsonl', limits)
    budget.path.touch()
    adapter = RepairAdapter(support, budget, teacher, records=records, resets=resets,
                            request_directory=request_directory, stepper_factory=stepper_factory,
                            generate_reply=generate_reply, config_loader=config_loader,
                            rate_limit_retries=rate_limit_retries)
    stop_reason = 'complete'
    next_start = 0.0
    with pool.collection_lock(collection):
        try:
            for tid in selected:
                delay = next_start - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                next_start = time.monotonic() + _minimum_interval()
                episode = adapter.teacher_episode(tid, attempt_index, temperature)
                usage = budget.usage(tid, attempt_index)
                with _purchase_lock(ledger):
                    append_episode(ledger, task_id=tid, teacher=teacher, attempt_index=attempt_index,
                                   temperature=temperature, verified=episode.verified, demo=episode.demo,
                                   tokens_spent=usage['completion_tokens'], usage=usage, purpose='repair-ce')
                if budget.stopped:
                    break
        except AcquisitionStopped as exc:
            stop_reason = str(exc)
        finally:
            pool.recover_attempts(ledger, budget, teacher)
            pool.audit_verified_bank(source, expected_manifest_sha256=identity['source_manifest_sha256'])
            metadata = {tid: read_json(request_directory / (canonical_hash(tid) + '.json')) for tid in selected}
            usable = pool.export_pool(source, output, ledger, support, stepper_factory=stepper_factory,
                                      budget=budget, method='repair-ce', sweep_filter=True,
                                      material_metadata=metadata, usable_only=True, new_output=True)
            candidates = read_json(collection / 'candidate_sets.json')['tasks']
            for tid, material in metadata.items():
                attempts = candidates[tid]['attempts']
                material.update(training_usable=bool(attempts and attempts[0]['status'] == 'usable'),
                                unavailable_reason=attempts[0]['unavailable_reason'] if attempts else 'not purchased: budget stopped')
                if material['status'] == 'pending':
                    material['status'] = 'not_purchased'
                pool.write_json(request_directory / (canonical_hash(tid) + '.json'), material)
    summary = dict(tasks=len(records), requests=len(selected), usable_packages=usable,
                   stop_reason=budget.stopped or stop_reason, purchase_cost=pool.purchase_cost(list(budget.calls.values()), limits),
                   collection=str(collection), bank=str(output), registration=None,
                   registration_skipped_reason='no usable packages' if config_path and not usable else None)
    pool.write_json(collection / 'summary.json', summary)
    if config_path and usable:
        summary['registration'] = register_config(output, config_path, model_path=model_path)
        pool.write_json(collection / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollouts', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--source-collection', type=Path, required=True)
    parser.add_argument('--collection', type=Path, required=True, help='new; existing collections are refused')
    parser.add_argument('--output', type=Path, required=True, help='new frozen bank')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, help='cached student tokenizer for registration')
    parser.add_argument('--max-tokens', type=int, default=400000)
    parser.add_argument('--max-usd', type=Decimal, default=Decimal('5'))
    parser.add_argument('--attempt-index', type=int, default=0)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--rate-limit-retries', type=int, default=pool.appworld_teacher.RATE_LIMIT_RETRIES)
    args = parser.parse_args(argv)
    summary = collect_repair(args.rollouts, args.source, args.source_collection, args.collection, args.output,
                             max_tokens=args.max_tokens, max_usd=args.max_usd, attempt_index=args.attempt_index,
                             temperature=args.temperature, rate_limit_retries=args.rate_limit_retries,
                             config_path=args.config, model_path=args.model_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
