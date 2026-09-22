#!/usr/bin/env python3
"""Collect a capped, ledger-backed ALFWorld pool on the frozen C26 task IDs.

Only this command's collection phase contacts a teacher. No student model or
GPU is loaded. The sibling <out>.collection directory holds durable accounting;
<out> contains only the standard public/ and sealed/ bank artifacts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_FLOOR
import fcntl
import json
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import appworld_teacher
from bfas.adapter import Demo, TeacherEpisode, Turn
from bfas.adapters.alfworld import ALFWorldAdapter
from bfas.ledger import AcquisitionStopped, _minimum_interval, _purchase_lock, append_episode, append_record, read_records
from bfas.protocol import SAMPLING_TEMPERATURE
from bfas.rtd.benchmarks import alfworld_bank as bank
from bfas.rtd.benchmarks.alfworld_state import _observed, canonical_hash, parent_hash
from bfas.rtd.benchmarks.alfworld_support import (
    ALFWorldSupport, RealStepper, _signed, audit_verified_bank, prompt_messages,
    seal_verified_bank, teacher_payload_fields, validate_support, verify_package,
)
from bfas.rtd.transport import FullState

ATTEMPTS = 3
DEFAULT_TEACHER = "openai/gpt-5.6-luna"
METHODS = ('plain', 'smartad', 'kang-ftp')


def reasoning_messages(request, observation):
    """FTP's plain reasoning purchase sees only the goal and reset observation."""
    return [dict(role='system', content=(
        'Explain how you would solve this ALFWorld household task. Give clear reasoning '
        'in paragraphs, beginning with your initial plan. This is a planning question; '
        'do not issue an ACTION or interact with the environment.')),
        dict(role='user', content=f"Goal: {request['goal']}\n\nInitial observation:\n{observation.observation}")]


def first_thought_prefix(response):
    # Exact vendored build_prefix_memory.py rule, with ALFWorld's marker casing.
    return 'THOUGHT: ' + response.split('\n\n')[0] + '\n\n'


class PrefixMemory:
    """One durable reasoning response per task, shared by trajectory retries."""

    def __init__(self, directory):
        self.path = directory / 'prefix_records.json'
        self.memory_path = directory / 'prefix_memory.json'
        self.lock = threading.Lock()
        self.records = json.loads(self.path.read_text()) if self.path.exists() else {}
        for record in self.records.values():
            if record['prefix'] != first_thought_prefix(record['response']):
                raise ValueError('prefix memory differs from its reasoning response')

    def save(self, task_id, call, messages, response):
        with self.lock:
            record = dict(call_id=call['id'], attempt_index=call['attempt_index'],
                          messages=messages, response=response, prefix=first_thought_prefix(response))
            self.records[task_id] = record
            write_json(self.path, self.records)
            self.export()
            return record['prefix']

    def export(self):
        write_json(self.memory_path, {tid: r['prefix'] for tid, r in self.records.items()})

    def missing_paid_response(self, task_id, calls):
        return task_id not in self.records and any(
            c['task_id'] == task_id and c.get('phase') == 'cot' for c in calls.values())


def sampling_parameters(config, temperature):
    # Archive only sampling fields, never configuration/headers/credentials.
    if not hasattr(config, 'backend'):  # injected test transport
        return dict(requested_temperature=temperature, temperature=temperature,
                    temperature_source='injected transport')
    body = appworld_teacher._build_request_body(config, [], temperature)
    return dict(requested_temperature=temperature, temperature=body.get('temperature'),
                temperature_source='explicit' if 'temperature' in body else 'provider default',
                seed=body.get('seed'), top_p=body.get('top_p'))


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
        self._condition = threading.Condition()
        self._pending = {}
        self._calls = {}
        self.stopped = None

    @contextmanager
    def _locked(self):
        # Separate opens of the existing ledger lock also serialize threads.
        # Reload under flock so another Budget instance cannot admit stale spend.
        with self._condition, _purchase_lock(self.path):
            self._calls = {}
            if self.path.exists():
                for line in self.path.read_text().splitlines():
                    record = json.loads(line)
                    self._calls[record['id']] = record
            yield

    @property
    def calls(self):
        with self._locked():
            return dict(self._calls)

    def _usage(self, task_id=None, attempt_index=None):
        result = dict(prompt_tokens=0, cached_tokens=0, completion_tokens=0)
        for call in self._calls.values():
            if task_id is not None and (call['task_id'], call['attempt_index']) != (task_id, attempt_index):
                continue
            for key in result:
                result[key] += call['usage'][key]
        return result

    def usage(self, task_id=None, attempt_index=None):
        with self._locked():
            return self._usage(task_id, attempt_index)

    def has_calls(self, task_id, attempt_index):
        return any((c['task_id'], c['attempt_index']) == (task_id, attempt_index) for c in self.calls.values())

    def cancel(self, reason):
        with self._condition:
            self.stopped = self.stopped or reason
            self._condition.notify_all()

    def stop(self, reason):
        self.cancel(reason)
        raise AcquisitionStopped(self.stopped)

    def ensure_available(self):
        self._admit()

    def _envelope(self, messages):
        usage = self._usage()
        if usage['prompt_tokens'] + usage['completion_tokens'] >= self.limits.max_tokens:
            return None, 'max-tokens reached'
        if self.limits.cost(usage) >= self.limits.max_usd:
            return None, 'max-usd reached'
        if messages is None:
            return None, None
        prompt = prompt_bound(messages)
        remaining = self.limits.max_tokens - usage['prompt_tokens'] - usage['completion_tokens']
        output = min(appworld_teacher.MAX_COMPLETION_TOKENS, remaining - prompt)
        money = self.limits.max_usd - self.limits.cost(usage) - self.limits.usd_in * prompt / Decimal(1000000)
        if money < 0:
            return None, 'max-usd cannot cover next prompt'
        if self.limits.usd_out:
            output = min(output, int((money * Decimal(1000000) / self.limits.usd_out).to_integral_value(rounding=ROUND_FLOOR)))
        if output < 1:
            return None, 'budget cannot reserve next prompt and completion'
        return dict(prompt_tokens=prompt, cached_tokens=0, completion_tokens=output), None

    def _admit(self, task_id=None, attempt_index=None, messages=None, *, phase='trajectory', sampling=None):
        with self._condition:
            while True:
                with self._locked():
                    if self.stopped:
                        raise AcquisitionStopped(self.stopped)
                    usage, reason = self._envelope(messages)
                    if reason is None:
                        if messages is None:
                            return
                        call = dict(id=max(self._calls, default=-1) + 1, task_id=task_id,
                                    attempt_index=attempt_index, status='reserved', usage=usage,
                                    phase=phase, sampling=dict(sampling or {},
                                        max_completion_tokens=usage['completion_tokens']))
                        # fsync precedes the external call, while admission is locked.
                        append_record(self.path, call)
                        self._pending[call['id']] = threading.get_ident()
                        return call
                    if not any(owner != threading.get_ident() for owner in self._pending.values()):
                        self.stop(reason)
                # Live requests may release headroom. Uncertain calls left by a
                # crash are charged but never waited on. No file lock spans I/O.
                self._condition.wait()

    def reserve(self, task_id, attempt_index, messages, *, phase='trajectory', sampling=None):
        return self._admit(task_id, attempt_index, messages, phase=phase, sampling=sampling)

    def finish(self, call):
        """A request ended, possibly without usage; retain any uncertain bill."""
        with self._condition:
            self._pending.pop(call['id'], None)
            self._condition.notify_all()

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
        with self._locked():
            current = self._calls[call['id']]
            value = dict(call, status='reported', usage=usage)
            if current['status'] == 'reported':
                if current != value:
                    raise ValueError('conflicting usage for an already settled request')
                return
            append_record(self.path, value)
            self.finish(call)
            if prompt > call['usage']['prompt_tokens'] or output > call['usage']['completion_tokens']:
                self.stop('provider usage exceeded the reserved request envelope')


def real_stepper(request):
    # Includes remote response time; worker cleanup remains bounded.
    return RealStepper(environment_hash=request['environment_hash'], episode_timeout=14400)


class PoolAdapter:
    """BFAS episode interface using the ALFWorld scaffold and official CPU metric."""

    def __init__(self, support, budget, teacher, *, stepper_factory=real_stepper,
                 generate_reply=None, config_loader=None,
                 rate_limit_retries=appworld_teacher.RATE_LIMIT_RETRIES, prefix_memory=None):
        self.support, self.budget, self.teacher = support, budget, teacher
        self.rate_limit_retries = rate_limit_retries
        self.stepper_factory = stepper_factory
        self.generate_reply = generate_reply or appworld_teacher.generate_reply
        self.config_loader = config_loader or appworld_teacher.load_teacher_config
        self.prefix_memory = prefix_memory

    def purchase(self, config, task_id, attempt_index, messages, temperature, *, phase='trajectory'):
        call = self.budget.reserve(task_id, attempt_index, messages, phase=phase,
                                   sampling=sampling_parameters(config, temperature))
        raw_response = []
        def capture_response(event):
            # The generic transport strips content whitespace. FTP must split
            # the original response, including leading blank lines, verbatim.
            data = event.get('data', {})
            choices = data.get('choices') if isinstance(data, dict) else None
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                message = choices[0].get('message')
                if isinstance(message, dict) and isinstance(message.get('content'), str):
                    raw_response.append(message['content'])
        try:
            reply = self.generate_reply(
                config, messages, temperature=temperature, retries=0,
                rate_limit_retries=self.rate_limit_retries,
                max_completion_tokens=call['usage']['completion_tokens'],
                usage_callback=lambda usage: self.budget.settle(call, usage),
                **(dict(response_callback=capture_response) if phase == 'cot' else {}),
            )
            return (raw_response[-1] if raw_response else reply), call
        finally:
            self.budget.finish(call)

    def teacher_name(self):
        return self.teacher

    def teacher_episode(self, task_id, attempt_index, temperature):
        # A crash after a paid call but before ledger append must not repurchase
        # the same task/attempt. Recover a conservatively charged failed episode.
        recovered = self.budget.has_calls(task_id, attempt_index)
        if not recovered:
            self.budget.ensure_available()
        turns, responses, commands, won = [], [], [], False
        if not recovered:
            try:
                config = self.config_loader(self.teacher)
            except (RuntimeError, ValueError, KeyError) as exc:
                raise AcquisitionStopped(f'teacher configuration unavailable: {exc}') from exc
            request = self.support['tasks'][task_id]['request']
            stepper = None
            try:
                stepper = self.stepper_factory(request)
                cursor, observation = stepper.reset(request)
                history = [_observed(observation)]
                prefix = None
                if self.prefix_memory is not None:
                    record = self.prefix_memory.records.get(task_id)
                    if record is None:
                        if self.prefix_memory.missing_paid_response(task_id, self.budget.calls):
                            raise ValueError('paid reasoning response missing; refusing to repurchase')
                        messages = reasoning_messages(request, observation)
                        response, call = self.purchase(config, task_id, attempt_index, messages, 0.0, phase='cot')
                        prefix = self.prefix_memory.save(task_id, call, messages, response)
                    else:
                        prefix = record['prefix']
                for step_index in range(request['max_episode_steps']):
                    if observation.done:
                        won = observation.won
                        break
                    messages = prompt_messages(request, history)
                    if prefix is not None and step_index == 0:
                        messages[0]['content'] += (
                            '\nContinue the supplied assistant first-thought prefix. Return only '
                            'the continuation, ending with the required ACTION. Do not repeat the prefix.')
                        messages.append(dict(role='assistant', content=prefix))
                    reply, _ = self.purchase(config, task_id, attempt_index, messages, temperature)
                    responses.append(reply)
                    if prefix is not None and step_index == 0 and not reply.startswith(prefix):
                        reply = prefix + reply
                    command = ALFWorldAdapter._teacher_command(appworld_teacher.strip_think(reply), observation.admissible)
                    context = prompt_messages(request, history, react=False)
                    # Preserve the generated turn verbatim; only commands drive the environment.
                    turns.append(Turn(context[0]['content'], reply, context))
                    commands.append(command)
                    cursor, observation = stepper.step(cursor, command)
                    history.extend([dict(role='assistant', index=observation.index, content=command), _observed(observation)])
                    if observation.done:
                        won = observation.won
                        break
            except AcquisitionStopped:
                if not self.budget.has_calls(task_id, attempt_index):
                    raise  # no attempted purchase, so do not consume an attempt
            except KeyboardInterrupt:
                self.budget.cancel('interrupted; resume with the same --out')
                if not self.budget.has_calls(task_id, attempt_index):
                    raise AcquisitionStopped(self.budget.stopped) from None
            except Exception as exc:
                # Carry the cause: a bare type name hides why a whole run bought nothing.
                print(f'[alfworld-pool] task={task_id} attempt={attempt_index} failed: '
                      f'{type(exc).__name__}: {exc}', flush=True)
            finally:
                if stepper is not None:
                    stepper.close()
        usage = self.budget.usage(task_id, attempt_index)
        demo = Demo(task_id, tuple(turns), '\n'.join(commands)[-4000:],
                    raw=dict(teacher_commands=commands)) if won and turns else None
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


def purchase_cost(calls, limits):
    usage = {key: sum(c['usage'][key] for c in calls)
             for key in ('prompt_tokens', 'cached_tokens', 'completion_tokens')}
    return dict(**usage, tokens=usage['prompt_tokens'] + usage['completion_tokens'],
                estimated_usd=float(limits.cost(usage)), estimated_usd_decimal=str(limits.cost(usage)),
                calls=len(calls), uncertain_calls=sum(c['status'] == 'reserved' for c in calls))


def attempt_material(row, calls, limits, method, prefix_memory):
    purchases = [c for c in calls.values()
                 if (c['task_id'], c['attempt_index']) == (row['task_id'], row['attempt_index'])]
    phases = {phase: purchase_cost([c for c in purchases if c.get('phase', 'trajectory') == phase], limits)
              for phase in ('cot', 'trajectory')}
    fields = teacher_payload_fields(row)
    result = dict(method=method, candidate_id=canonical_hash([row['task_id'], row['attempt_index']]),
                  attempt_index=row['attempt_index'], requested_temperature=row['temperature'],
                  purchase_cost=purchase_cost(purchases, limits), phase_costs=phases,
                  sampling=[dict(call_id=c['id'], phase=c.get('phase', 'trajectory'), **c['sampling'])
                            for c in purchases],
                  trajectory_hash=canonical_hash(fields) if row['verified'] else None,
                  commands_hash=canonical_hash(fields['commands']) if row['verified'] else None)
    if prefix_memory is not None and phases['trajectory']['calls']:
        record = prefix_memory.records[row['task_id']]
        result['first_thought_prefix'] = record['prefix']
        result['prefix_cot_call_id'] = record['call_id']
    return result


def kang_first_attempt_ids(payloads):
    """Usable attempt zero only; a later first success is never a first draw."""
    return sorted(qid for qid, payload in payloads.items()
                  if payload['status'] == 'usable' and payload['provenance']['attempt_index'] == 0)


def export_pool(source, out, ledger, support, *, stepper_factory=real_stepper,
                budget=None, method='plain', candidates_per_task=1, prefix_memory=None):
    """Rebuild solely from paid ledger rows; verification makes no teacher calls."""
    payloads, records, resets = {}, [], {}
    calls = budget.calls if budget is not None else {}
    candidate_sets = {tid: dict(attempts=[], verified_candidates=0, distinct_verified_candidates=0,
                               distinct_verified_command_trajectories=0)
                      for tid in support['historical_task_ids']}
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
        teacher_fields = teacher_payload_fields(row)
        commands = teacher_fields['commands']
        excluded = ['protected probe/calibration parent'] if support['tasks'][tid]['excluded'] else []
        candidate = bool(row['verified'] and commands and not excluded)
        payload = dict(
            query_id=q, request_state_hash=canonical_hash(request), dependencies=[],
            status='candidate' if candidate else 'unavailable',
            unavailable_reason=None if candidate else 'failed, interrupted, or protected attempt',
            exclusion_reasons=excluded, success=row['verified'],
            **teacher_fields, behaviors=[], cost=row['tokens_spent'], cost_basis='reported_or_reserved',
            cost_confidence='estimated', usage=row.get('usage', {}),
            provenance=dict(kind='alf_demo_episode', ledger_path=str(ledger), ledger_sha256=ledger_hash,
                            line=line, task_id=tid, attempt_index=row['attempt_index'], timestamp=row['timestamp'],
                            teacher=row['teacher'], state_validation='pending CPU replay'),
            historical_response=row, raw_ledger_line=raw, historical_events=[],
        )
        if budget is not None:
            payload['collection'] = attempt_material(row, calls, budget.limits, method, prefix_memory)
        if candidate:
            payload = verify_package(payload, request, stepper_factory(request), render, support=ALFWorldSupport(support))
            if payload['status'] == 'usable':
                resets[tid] = payload['verification']['states'][0]
        payloads[q] = payload
        records.append(bank.public_record(q, request))
        if budget is not None:
            candidate_sets[tid]['attempts'].append(dict(query_id=q, status=payload['status'],
                collection_verified=row['verified'], **payload['collection']))
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
    if budget is not None:
        for tid, group in candidate_sets.items():
            candidates = [a for a in group['attempts'] if a['status'] == 'usable']
            group.update(attempt_count=len(group['attempts']), verified_candidates=len(candidates),
                         distinct_verified_candidates=len({c['trajectory_hash'] for c in candidates}),
                         distinct_verified_command_trajectories=len({c['commands_hash'] for c in candidates}),
                         requested_candidates=candidates_per_task,
                         shortfall=max(0, candidates_per_task - len(candidates)),
                         purchase_cost=purchase_cost([c for c in calls.values() if c['task_id'] == tid], budget.limits),
                         phase_costs={phase: purchase_cost([c for c in calls.values()
                             if c['task_id'] == tid and c.get('phase', 'trajectory') == phase], budget.limits)
                             for phase in ('cot', 'trajectory')})
        export_metadata = {}
        if method == 'kang-ftp':
            ids = kang_first_attempt_ids(payloads)
            export_metadata = dict(adaptation_label='prompt-continuation FTP adaptation',
                first_attempt_package_ids=ids,
                first_attempt_rule='usable attempt_index == 0; exclude failed first attempts, never substitute retries',
                cost_policy='retain full collection ledger costs, including planning, retries and failures')
            write_json(ledger.parent / 'first_attempt_package_ids.json', ids)
        write_json(ledger.parent / 'candidate_sets.json', dict(
            method=method, bank_manifest_sha256=bank.file_hash(out / 'sealed/manifest.json'),
            ledger_sha256=ledger_hash, usage_sha256=bank.file_hash(budget.path) if budget.path.exists() else None,
            distinctness='exact teacher payload (full ReAct targets and executed commands)',
            tasks=candidate_sets, **export_metadata))
    return sum(p['status'] == 'usable' for p in payloads.values())


@contextmanager
def collection_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'collection.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def recover_attempts(ledger, budget, teacher):
    """Reconcile orphan reservations only while no episode workers are running."""
    with _purchase_lock(ledger):
        seen = {(r['task_id'], r['attempt_index']) for r in read_records(ledger)}
        attempts = dict.fromkeys((c['task_id'], c['attempt_index']) for c in budget.calls.values())
        for task_id, attempt_index in attempts:
            if (task_id, attempt_index) in seen:
                continue
            usage = budget.usage(task_id, attempt_index)
            append_episode(ledger, task_id=task_id, teacher=teacher, attempt_index=attempt_index,
                           temperature=0.0 if attempt_index == 0 else SAMPLING_TEMPERATURE,
                           verified=False, tokens_spent=usage['completion_tokens'], usage=usage)


def acquire_pool(ledger, support, budget, teacher, *, workers, adapter_factory, stepper_factory,
                 rate_limit_retries, candidates_per_task=1, attempts_per_task=ATTEMPTS, prefix_memory=None):
    # The collection lock excludes other collectors, including the old sequential
    # command. Queue ownership excludes duplicate tasks within this collector.
    # Only accounting holds the ledger lock; each worker owns its adapter/stepper.
    tasks = Queue()
    interval = _minimum_interval()
    start_gate, next_start = threading.Lock(), 0.0

    def pace():
        nonlocal next_start
        if interval:
            with start_gate, budget._condition:
                while not budget.stopped:
                    delay = next_start - time.monotonic()
                    if delay <= 0:
                        next_start = time.monotonic() + interval
                        return
                    budget._condition.wait(delay)
                raise AcquisitionStopped(budget.stopped)

    with _purchase_lock(ledger):
        rows = read_records(ledger)
    for task_id in support['historical_task_ids']:
        previous = [r for r in rows if r['task_id'] == task_id]
        verified = sum(r['verified'] for r in previous)
        first_attempt = max((r['attempt_index'] for r in previous), default=-1) + 1
        if (first_attempt < attempts_per_task and verified < candidates_per_task
                and not (prefix_memory is not None and prefix_memory.missing_paid_response(task_id, budget.calls))):
            tasks.put((task_id, first_attempt, verified))

    def work():
        try:
            adapter = adapter_factory(support, budget, teacher, stepper_factory=stepper_factory,
                                      rate_limit_retries=rate_limit_retries, prefix_memory=prefix_memory)
            while True:
                try:
                    task_id, first_attempt, verified = tasks.get_nowait()
                except Empty:
                    return
                try:
                    for attempt_index in range(first_attempt, attempts_per_task):
                        pace()
                        temperature = 0.0 if attempt_index == 0 else SAMPLING_TEMPERATURE
                        episode = adapter.teacher_episode(task_id, attempt_index, temperature)
                        if (not isinstance(episode, TeacherEpisode) or episode.task_id != task_id
                                or bool(episode.verified) != (episode.demo is not None)):
                            raise ValueError('invalid teacher episode')
                        # The request journal is authoritative, even if a worker
                        # fails between its final response and this ledger append.
                        usage = budget.usage(task_id, attempt_index)
                        with _purchase_lock(ledger):
                            append_episode(ledger, task_id=task_id, teacher=teacher,
                                           attempt_index=attempt_index, temperature=temperature,
                                           verified=episode.verified, demo=episode.demo,
                                           tokens_spent=usage['completion_tokens'], usage=usage)
                        verified += episode.verified
                        if (verified >= candidates_per_task or
                                prefix_memory is not None and prefix_memory.missing_paid_response(task_id, budget.calls)):
                            break
                finally:
                    tasks.task_done()
        except AcquisitionStopped as exc:
            budget.cancel(str(exc))
        except BaseException as exc:
            budget.cancel('interrupted; resume with the same --out' if isinstance(exc, KeyboardInterrupt)
                          else 'worker failed; resume with the same --out')
            raise

    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='alfworld-pool')
    try:
        futures = [executor.submit(work) for _ in range(workers)]
        for future in as_completed(futures):
            future.result()
    except BaseException:
        budget.cancel('interrupted; resume with the same --out')
        raise
    finally:
        # In-flight requests must settle and workers must close their environments
        # before recovery/export can take a stable accounting snapshot.
        executor.shutdown(wait=True, cancel_futures=True)


def collection_options(method, candidates_per_task, attempts_per_task):
    if method not in METHODS:
        raise ValueError(f'method must be one of {METHODS}')
    if type(candidates_per_task) is not int or candidates_per_task < 1:
        raise ValueError('candidates_per_task must be a positive integer')
    if method != 'smartad' and candidates_per_task != 1:
        raise ValueError('multiple candidates require --method smartad')
    if attempts_per_task is None:
        attempts_per_task = ATTEMPTS * candidates_per_task
    if type(attempts_per_task) is not int or attempts_per_task < candidates_per_task:
        raise ValueError('attempts_per_task must be an integer >= candidates_per_task')
    return attempts_per_task


def collect(source, out, *, limits=None, workers=1, stepper_factory=real_stepper, adapter_factory=PoolAdapter,
            rate_limit_retries=appworld_teacher.RATE_LIMIT_RETRIES, method='plain',
            candidates_per_task=1, attempts_per_task=None):
    attempts_per_task = collection_options(method, candidates_per_task, attempts_per_task)
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    if type(rate_limit_retries) is not int or rate_limit_retries < 0:
        raise ValueError('rate_limit_retries must be a non-negative integer')
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
                        attempts=attempts_per_task, method=method, candidates_per_task=candidates_per_task,
                        prices=[str(limits.usd_in), str(limits.usd_out), str(limits.usd_cached)])
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
        with _purchase_lock(ledger):
            rows = read_records(ledger)
        seen = set()
        for row in rows:
            key = row['task_id'], row['attempt_index']
            if (key in seen or key[0] not in support['historical_task_ids'] or not 0 <= key[1] < attempts_per_task
                    or row['teacher'] != teacher):
                raise ValueError('ledger contains duplicate/outside task attempts or a different teacher')
            seen.add(key)
        budget = Budget(directory / 'usage.jsonl', limits)
        for call in budget.calls.values():
            if call['task_id'] not in support['historical_task_ids'] or not 0 <= call['attempt_index'] < attempts_per_task:
                raise ValueError('request accounting contains an outside task attempt')
        prefix_memory = PrefixMemory(directory) if method == 'kang-ftp' else None
        if prefix_memory is not None:
            calls = budget.calls
            for tid, record in prefix_memory.records.items():
                call = calls.get(record['call_id'], {})
                if (call.get('task_id') != tid or call.get('phase') != 'cot'
                        or call.get('attempt_index') != record['attempt_index']):
                    raise ValueError('prefix memory lacks its paid reasoning call')
            prefix_memory.export()
        for row in rows:
            usage = budget.usage(row['task_id'], row['attempt_index'])
            if row['tokens_spent'] != usage['completion_tokens'] or row.get('usage', {}) != usage:
                raise ValueError('ledger and durable request accounting disagree')
        reason = 'complete'
        recover_attempts(ledger, budget, teacher)
        try:
            acquire_pool(ledger, support, budget, teacher, workers=workers,
                         adapter_factory=adapter_factory, stepper_factory=stepper_factory,
                         rate_limit_retries=rate_limit_retries, candidates_per_task=candidates_per_task,
                         attempts_per_task=attempts_per_task, prefix_memory=prefix_memory)
        except AcquisitionStopped as exc:
            reason = str(exc)
        except KeyboardInterrupt:
            reason = 'interrupted; resume with the same --out'
        finally:
            recover_attempts(ledger, budget, teacher)
            with _purchase_lock(ledger):
                verified = export_pool(source, out, ledger, support, stepper_factory=stepper_factory,
                    budget=budget, method=method, candidates_per_task=candidates_per_task, prefix_memory=prefix_memory)
        usage = budget.usage()
        groups = json.loads((directory / 'candidate_sets.json').read_text())['tasks']
        per_task = {tid: {k: v for k, v in group.items() if k != 'attempts'} for tid, group in groups.items()}
        if prefix_memory is not None:
            for tid, task in per_task.items():
                if prefix_memory.missing_paid_response(tid, budget.calls):
                    task['blocked_reason'] = 'paid reasoning response missing; not repurchased'
        summary = dict(tasks=len(support['historical_task_ids']),
                       tasks_attempted=len({r['task_id'] for r in read_records(ledger)}),
                       verified=verified, tokens=usage['prompt_tokens'] + usage['completion_tokens'],
                       **usage, estimated_usd=float(limits.cost(usage)),
                       uncertain_calls=sum(c['status'] == 'reserved' for c in budget.calls.values()),
                       stop_reason=budget.stopped or reason, teacher=teacher, service_tier='flex',
                       ledger=str(ledger), out=str(out))
        summary.update(method=method, candidates_per_task=candidates_per_task, attempts_per_task=attempts_per_task,
                       attempts=sum(g['attempt_count'] for g in groups.values()), verified_candidates=verified,
                       distinct_verified_candidates=sum(g['distinct_verified_candidates'] for g in groups.values()),
                       tasks_at_target=sum(g['shortfall'] == 0 for g in groups.values()),
                       shortfall=sum(g['shortfall'] for g in groups.values()), per_task=per_task,
                       candidate_sets=str(directory / 'candidate_sets.json'),
                       phase_costs={phase: purchase_cost([c for c in budget.calls.values()
                           if c.get('phase', 'trajectory') == phase], limits) for phase in ('cot', 'trajectory')})
        if summary['shortfall'] and summary['stop_reason'] == 'complete':
            summary['stop_reason'] = 'attempts exhausted or unavailable; candidate shortfall'
        if prefix_memory is not None:
            summary.update(prefix_memory=str(prefix_memory.memory_path), prefix_records=str(prefix_memory.path),
                           prefix_memory_sha256=bank.file_hash(prefix_memory.memory_path),
                           prefix_records_sha256=bank.file_hash(prefix_memory.path) if prefix_memory.path.exists() else None)
        write_json(directory / 'summary.json', summary)
        print(json.dumps(summary, indent=2), flush=True)
        return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'data/rtd/v1_alfworld_c26')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/rtd/v1_alfworld_luna')
    parser.add_argument('--max-tokens', type=int, default=400000)
    parser.add_argument('--method', choices=METHODS, default='plain')
    parser.add_argument('--candidates-per-task', type=int, default=1,
                        help='SmartAD verified trajectory target; duplicates are retained (default: 1)')
    parser.add_argument('--attempts-per-task', type=int,
                        help='total trajectory attempt cap per task (default: 3 * candidates-per-task)')
    parser.add_argument('--workers', type=int, default=1,
                        help='concurrent episodes, each with its own environment process (default: 1)')
    parser.add_argument('--rate-limit-retries', type=int, default=appworld_teacher.RATE_LIMIT_RETRIES,
                        help='retries per request for HTTP 429 only (default: %(default)s; 0 disables)')
    parser.add_argument('--max-usd', type=Decimal, default=Decimal('5.0'))
    parser.add_argument('--usd-per-mtok-in', type=Decimal, default=Decimal('0.10'))
    parser.add_argument('--usd-per-mtok-out', type=Decimal, default=Decimal('0.60'))
    parser.add_argument('--usd-per-mtok-cached', type=Decimal, default=Decimal('0.01'))
    args = parser.parse_args(argv)
    try:
        if args.workers < 1:
            raise ValueError('workers must be a positive integer')
        if args.rate_limit_retries < 0:
            raise ValueError('rate_limit_retries must be a non-negative integer')
        collection_options(args.method, args.candidates_per_task, args.attempts_per_task)
        limits = Limits(args.max_tokens, args.max_usd, args.usd_per_mtok_in,
                        args.usd_per_mtok_out, args.usd_per_mtok_cached)
    except ValueError as exc:
        parser.error(str(exc))
    collect(args.source, args.out, limits=limits, workers=args.workers,
            rate_limit_retries=args.rate_limit_retries, method=args.method,
            candidates_per_task=args.candidates_per_task, attempts_per_task=args.attempts_per_task)


if __name__ == '__main__':
    main()
