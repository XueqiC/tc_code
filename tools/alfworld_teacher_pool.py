#!/usr/bin/env python3
"""Collect a capped, ledger-backed ALFWorld pool on the frozen C26 task IDs.

Only this command's collection phase contacts a teacher. No student model or
GPU is loaded. The sibling <out>.collection directory holds durable accounting;
<out> contains only the standard public/ and sealed/ bank artifacts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_FLOOR
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import appworld_teacher
from bfas.adapter import Demo, TeacherEpisode, Turn
from bfas.adapters.alfworld import ALFWorldAdapter
from bfas.ledger import AcquisitionStopped, acquire_demos, append_record, read_records
from bfas.rtd.benchmarks import alfworld_bank as bank
from bfas.rtd.benchmarks.alfworld_state import _observed, canonical_hash, parent_hash
from bfas.rtd.benchmarks.alfworld_support import (
    ALFWorldSupport, RealStepper, _signed, audit_verified_bank, prompt_messages,
    seal_verified_bank, validate_support, verify_package,
)
from bfas.rtd.transport import FullState

ATTEMPTS = 3
DEFAULT_TEACHER = "openai/gpt-5.6-luna"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def render(request, history):
    # Portable archive rendering. Conversion later applies the student template.
    return '\n\n'.join(f"{m['role']}: {m['content']}" for m in prompt_messages(request, history))


def prompt_bound(messages):
    # Text-only BPE input: at most one token per UTF-8 byte, plus generous
    # message/framing overhead. Never assume a cache hit before it is reported.
    return 256 + sum(64 + len(m['role'].encode('utf-8')) + len(m['content'].encode('utf-8'))
                     for m in messages)


@dataclass(frozen=True)
class Limits:
    max_tokens: int = 400000  # prompt + completion, including hidden reasoning
    max_usd: Decimal = Decimal('5.0')
    usd_in: Decimal = Decimal('0.10')
    usd_out: Decimal = Decimal('0.60')
    usd_cached: Decimal = Decimal('0.01')

    def __post_init__(self):
        if type(self.max_tokens) is not int or self.max_tokens < 0:
            raise ValueError('max_tokens must be a non-negative integer')
        for field in ('max_usd', 'usd_in', 'usd_out', 'usd_cached'):
            value = Decimal(str(getattr(self, field)))
            if not value.is_finite() or value < 0:
                raise ValueError(f'{field} must be finite and non-negative')
            object.__setattr__(self, field, value)
        if self.usd_cached > self.usd_in:
            raise ValueError('cached input rate must not exceed input rate')

    def cost(self, usage):
        return ((usage['prompt_tokens'] - usage['cached_tokens']) * self.usd_in
                + usage['cached_tokens'] * self.usd_cached
                + usage['completion_tokens'] * self.usd_out) / Decimal(1000000)


class Budget:
    """Write-ahead reservations; uncertain/unfinished calls keep their full bill."""

    def __init__(self, path, limits):
        self.path, self.limits = Path(path), limits
        self.calls = {}
        self.stopped = None
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                record = json.loads(line)
                self.calls[record['id']] = record

    def usage(self, task_id=None, attempt_index=None):
        result = dict(prompt_tokens=0, cached_tokens=0, completion_tokens=0)
        for call in self.calls.values():
            if task_id is not None and (call['task_id'], call['attempt_index']) != (task_id, attempt_index):
                continue
            for key in result:
                result[key] += call['usage'][key]
        return result

    def has_calls(self, task_id, attempt_index):
        return any((c['task_id'], c['attempt_index']) == (task_id, attempt_index) for c in self.calls.values())

    def stop(self, reason):
        self.stopped = reason
        raise AcquisitionStopped(reason)

    def ensure_available(self):
        usage = self.usage()
        if self.stopped:
            raise AcquisitionStopped(self.stopped)
        if usage['prompt_tokens'] + usage['completion_tokens'] >= self.limits.max_tokens:
            self.stop('max-tokens reached')
        if self.limits.cost(usage) >= self.limits.max_usd:
            self.stop('max-usd reached')

    def reserve(self, task_id, attempt_index, messages):
        self.ensure_available()
        used, prompt = self.usage(), prompt_bound(messages)
        remaining = self.limits.max_tokens - used['prompt_tokens'] - used['completion_tokens']
        output = min(appworld_teacher.MAX_COMPLETION_TOKENS, remaining - prompt)
        money = self.limits.max_usd - self.limits.cost(used) - self.limits.usd_in * prompt / Decimal(1000000)
        if money < 0:
            self.stop('max-usd cannot cover next prompt')
        if self.limits.usd_out:
            output = min(output, int((money * Decimal(1000000) / self.limits.usd_out).to_integral_value(rounding=ROUND_FLOOR)))
        if output < 1:
            self.stop('budget cannot reserve next prompt and completion')
        call = dict(id=len(self.calls), task_id=task_id, attempt_index=attempt_index,
                    status='reserved', usage=dict(prompt_tokens=prompt, cached_tokens=0, completion_tokens=output))
        # fsync happens before the external request; crashes cannot erase spend.
        append_record(self.path, call)
        self.calls[call['id']] = call
        return call

    def settle(self, call, reported):
        def count(key):
            value = reported.get(key)
            return value if type(value) is int and value >= 0 else None
        prompt, output = count('prompt_tokens'), count('completion_tokens')
        details = reported.get('prompt_tokens_details') or {}
        cached = details.get('cached_tokens', 0) if isinstance(details, dict) else 0
        if type(cached) is not int or cached < 0:
            cached = 0
        if prompt is None or output is None:
            return  # retain reservation for absent/incomplete usage
        usage = dict(prompt_tokens=prompt, cached_tokens=min(prompt, cached), completion_tokens=output)
        value = dict(call, status='reported', usage=usage)
        append_record(self.path, value)
        self.calls[call['id']] = value
        if prompt > call['usage']['prompt_tokens'] or output > call['usage']['completion_tokens']:
            self.stop('provider usage exceeded the reserved request envelope')


def real_stepper(request):
    # Includes remote response time; worker cleanup remains bounded.
    return RealStepper(environment_hash=request['environment_hash'], episode_timeout=14400)


class PoolAdapter:
    """BFAS episode interface using the ALFWorld scaffold and official CPU metric."""

    def __init__(self, support, budget, teacher, *, stepper_factory=real_stepper,
                 generate_reply=None, config_loader=None):
        self.support, self.budget, self.teacher = support, budget, teacher
        self.stepper_factory = stepper_factory
        self.generate_reply = generate_reply or appworld_teacher.generate_reply
        self.config_loader = config_loader or appworld_teacher.load_teacher_config

    def teacher_name(self):
        return self.teacher

    def teacher_episode(self, task_id, attempt_index, temperature):
        # A crash after a paid call but before ledger append must not repurchase
        # the same task/attempt. Recover a conservatively charged failed episode.
        recovered = self.budget.has_calls(task_id, attempt_index)
        if not recovered:
            self.budget.ensure_available()
        turns, responses, won = [], [], False
        if not recovered:
            try:
                config = self.config_loader(self.teacher)
            except (RuntimeError, ValueError, KeyError) as exc:
                raise AcquisitionStopped(f'teacher configuration unavailable: {exc}') from exc
            request = self.support['tasks'][task_id]['request']
            stepper = self.stepper_factory(request)
            try:
                cursor, observation = stepper.reset(request)
                history = [_observed(observation)]
                for _ in range(request['max_episode_steps']):
                    if observation.done:
                        won = observation.won
                        break
                    messages = prompt_messages(request, history)
                    call = self.budget.reserve(task_id, attempt_index, messages)
                    reply = self.generate_reply(
                        config, messages, temperature=temperature, retries=0,
                        max_completion_tokens=call['usage']['completion_tokens'],
                        usage_callback=lambda usage: self.budget.settle(call, usage),
                    )
                    responses.append(reply)
                    command = ALFWorldAdapter._teacher_command(appworld_teacher.strip_think(reply), observation.admissible)
                    context = prompt_messages(request, history, react=False)
                    turns.append(Turn(context[0]['content'], command, context))
                    cursor, observation = stepper.step(cursor, command)
                    history.extend([dict(role='assistant', index=observation.index, content=command), _observed(observation)])
                    if observation.done:
                        won = observation.won
                        break
            except AcquisitionStopped:
                if not self.budget.has_calls(task_id, attempt_index):
                    raise  # no attempted purchase, so do not consume an attempt
            except KeyboardInterrupt:
                self.budget.stopped = 'interrupted; resume with the same --out'
                if not self.budget.has_calls(task_id, attempt_index):
                    raise AcquisitionStopped(self.budget.stopped) from None
            except Exception as exc:
                print(f'[alfworld-pool] task={task_id} attempt={attempt_index} failed: {type(exc).__name__}', flush=True)
            finally:
                stepper.close()
        usage = self.budget.usage(task_id, attempt_index)
        demo = Demo(task_id, tuple(turns), '\n'.join(t.target for t in turns)[-4000:]) if won and turns else None
        return TeacherEpisode(task_id, demo is not None, demo, tuple(responses),
                              usage['completion_tokens'], teacher=self.teacher, usage=usage)


def load_source(source):
    source = Path(source)
    audit_verified_bank(source)
    support = validate_support(json.loads((source / 'public/support.json').read_text()))
    requests = json.loads((source / 'public/reset_requests.json').read_text())
    rows = json.loads((source / 'public/requests.json').read_text())
    public_ids = {requests[r['spec']['query_id']]['task_id'] for r in rows}
    sealed_ids = {json.loads((source / 'sealed' / f"{r['spec']['query_id']}.json").read_text())['provenance']['task_id'] for r in rows}
    if public_ids != sealed_ids or public_ids != set(support['historical_task_ids']):
        raise ValueError('source public/sealed/support task inventories differ')
    return support


def collection_support(source_support):
    support = deepcopy(source_support)
    # Retain exactly the frozen tasks/worlds/folds, but identify this collector's
    # current code and portable renderer instead of claiming the old tokenizer.
    environment = dict(
        historical_environment_hash=support['environment']['environment_hash'],
        collection='alfworld_teacher_pool', rendering='portable ReAct messages; student rendered at conversion',
        source_files={p: bank.file_hash(ROOT / p) for p in (
            'tools/alfworld_teacher_pool.py', 'src/appworld_teacher.py',
            'src/bfas/adapters/alfworld.py', 'src/bfas/rtd/benchmarks/alfworld_support.py')},
        max_episode_steps=40,
    )
    environment['environment_hash'] = canonical_hash(environment)
    support['environment'] = environment
    for task in support['tasks'].values():
        task['request']['environment_hash'] = environment['environment_hash']
        task['request_hash'] = canonical_hash(task['request'])
    return validate_support(_signed(support))


def export_pool(source, out, ledger, support, *, stepper_factory=real_stepper):
    """Rebuild solely from paid ledger rows; verification makes no teacher calls."""
    payloads, records, resets = {}, [], {}
    old_resets = json.loads((Path(source) / 'public/reset_states.json').read_text())
    for tid, raw in old_resets.items():
        request = support['tasks'][tid]['request']
        history = json.loads(raw['history_json'])
        resets[tid] = asdict(FullState.create(request, history, render(request, history), parent_hash(tid)))
    ledger_hash = bank.file_hash(ledger)
    for line, raw in enumerate(ledger.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        tid = row['task_id']
        request = support['tasks'][tid]['request']
        q = bank.query_id(ledger_hash, row, line)
        commands = [t['target'] for t in row.get('demo', {}).get('turns', [])]
        excluded = ['protected probe/calibration parent'] if support['tasks'][tid]['excluded'] else []
        candidate = bool(row['verified'] and commands and not excluded)
        payload = dict(
            query_id=q, request_state_hash=canonical_hash(request), dependencies=[],
            status='candidate' if candidate else 'unavailable',
            unavailable_reason=None if candidate else 'failed, interrupted, or protected attempt',
            exclusion_reasons=excluded, success=row['verified'],
            payload_kind='extracted_teacher_commands' if commands else None,
            commands=commands, behaviors=[], cost=row['tokens_spent'], cost_basis='reported_or_reserved',
            cost_confidence='estimated', usage=row.get('usage', {}),
            provenance=dict(kind='alf_demo_episode', ledger_path=str(ledger), ledger_sha256=ledger_hash,
                            line=line, task_id=tid, attempt_index=row['attempt_index'], timestamp=row['timestamp'],
                            teacher=row['teacher'], state_validation='pending CPU replay'),
            historical_response=row, raw_ledger_line=raw, historical_events=[],
        )
        if candidate:
            payload = verify_package(payload, request, stepper_factory(request), render, support=ALFWorldSupport(support))
            if payload['status'] == 'usable':
                resets[tid] = payload['verification']['states'][0]
        payloads[q] = payload
        records.append(bank.public_record(q, request))
    archive = bank.Archive(records, payloads, {}, [], dict(source_files=[dict(path=str(ledger), sha256=ledger_hash)]))
    out = Path(out)
    with tempfile.TemporaryDirectory(prefix=f'.{out.name}-', dir=out.parent) as temporary:
        target = Path(temporary) / 'pool'
        seal_verified_bank(target, archive, support, payloads, resets)
        audit_verified_bank(target)
        previous = out.with_name(out.name + '.previous')
        if previous.exists():
            shutil.rmtree(previous)
        if out.exists():
            out.rename(previous)
        try:
            target.rename(out)
        except BaseException:
            if previous.exists():
                previous.rename(out)
            raise
        if previous.exists():
            shutil.rmtree(previous)
    return sum(p['status'] == 'usable' for p in payloads.values())


@contextmanager
def collection_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'collection.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def collect(source, out, *, limits=None, stepper_factory=real_stepper, adapter_factory=PoolAdapter):
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError('source and output must be separate directories')
    limits = limits or Limits()
    teacher = os.environ.get('BFAS_TEACHER', DEFAULT_TEACHER)
    os.environ['BFAS_OPENAI_SERVICE_TIER'] = 'flex'
    source_support = load_source(source)
    support = collection_support(source_support)
    directory = out.with_name(out.name + '.collection')
    with collection_lock(directory):
        identity = dict(source_manifest=bank.file_hash(source / 'sealed/manifest.json'),
                        support_manifest=support['manifest_hash'], teacher=teacher, service_tier='flex',
                        attempts=ATTEMPTS, prices=[str(limits.usd_in), str(limits.usd_out), str(limits.usd_cached)])
        identity_path = directory / 'identity.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise ValueError('resume source, collector, teacher, or prices changed; choose a new --out')
        else:
            if out.exists() or any(directory.glob('*.jsonl')):
                raise ValueError('refusing to adopt an existing output or unbound accounting')
            write_json(identity_path, identity)
        ledger = directory / 'teacher_ledger.jsonl'
        ledger.touch(exist_ok=True)
        rows = read_records(ledger)
        seen = set()
        for row in rows:
            key = row['task_id'], row['attempt_index']
            if (key in seen or key[0] not in support['historical_task_ids'] or key[1] >= ATTEMPTS
                    or row['teacher'] != teacher):
                raise ValueError('ledger contains duplicate/outside task attempts or a different teacher')
            seen.add(key)
        budget = Budget(directory / 'usage.jsonl', limits)
        for row in rows:
            usage = budget.usage(row['task_id'], row['attempt_index'])
            if row['tokens_spent'] != usage['completion_tokens'] or row.get('usage', {}) != usage:
                raise ValueError('ledger and durable request accounting disagree')
        adapter = adapter_factory(support, budget, teacher, stepper_factory=stepper_factory)
        reason = 'complete'
        try:
            acquire_demos(ledger, adapter, support['historical_task_ids'], attempts=ATTEMPTS)
        except AcquisitionStopped as exc:
            reason = str(exc)
        except KeyboardInterrupt:
            reason = 'interrupted; resume with the same --out'
        finally:
            verified = export_pool(source, out, ledger, support, stepper_factory=stepper_factory)
        usage = budget.usage()
        summary = dict(tasks=len(support['historical_task_ids']),
                       tasks_attempted=len({r['task_id'] for r in read_records(ledger)}),
                       verified=verified, tokens=usage['prompt_tokens'] + usage['completion_tokens'],
                       **usage, estimated_usd=float(limits.cost(usage)),
                       uncertain_calls=sum(c['status'] == 'reserved' for c in budget.calls.values()),
                       stop_reason=budget.stopped or reason, teacher=teacher, service_tier='flex',
                       ledger=str(ledger), out=str(out))
        write_json(directory / 'summary.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/rtd/v1_alfworld_c26')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/rtd/v1_alfworld_luna')
    parser.add_argument('--max-tokens', type=int, default=400000)
    parser.add_argument('--max-usd', type=Decimal, default=Decimal('5.0'))
    parser.add_argument('--usd-per-mtok-in', type=Decimal, default=Decimal('0.10'))
    parser.add_argument('--usd-per-mtok-out', type=Decimal, default=Decimal('0.60'))
    parser.add_argument('--usd-per-mtok-cached', type=Decimal, default=Decimal('0.01'))
    args = parser.parse_args(argv)
    try:
        limits = Limits(args.max_tokens, args.max_usd, args.usd_per_mtok_in,
                        args.usd_per_mtok_out, args.usd_per_mtok_cached)
    except ValueError as exc:
        parser.error(str(exc))
    collect(args.source, args.out, limits=limits)


if __name__ == '__main__':
    main()
