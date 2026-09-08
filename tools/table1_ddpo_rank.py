#!/usr/bin/env python3
"""Sample the initial BFCL student, then buy one ranking per purchased task.

Run from this worktree with PYTHONPATH=src:. and PY=.venv/bin/python.
See docs/table1_budget_ledger_zh.md §5.1 for commands and accounting semantics.
Importing this module does not load a model, read credentials, or call an API.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

from tools import table1_common as io
import appworld_teacher as at
from tools.bfcl_pair_pools import RANK_PROMPT, last_two_numbers

ROOT = io.ROOT
BFCL = ROOT / 'envs/bfcl/gorilla/berkeley-function-call-leaderboard'
MODEL = 'Qwen/Qwen3.5-4B'
TEACHER = 'deepseek-v4-pro'
MAX_RATE_RETRIES = 2


def rows(path):
    return [r for _, r in io.read_rows(path)] if path.exists() else []


def atomic_write(path, value, *, jsonl=False):
    path = io.output_path(path)
    temp = io.output_path(path.with_name(path.name + '.tmp'))
    if jsonl:
        io.write_rows(temp, value)
    else:
        io.write_json(temp, value)
    os.replace(temp, path)


def append(path, row):
    with io.output_path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def locked(directory):
    with io.output_path(directory / '.ddpo.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another dDPO sample/rank command is running') from None
        yield


def purchased(audit, cap):
    """Use official parent IDs in tasks_covered, never gen_/oos_ suffix guesses."""
    pools = []
    for seed in range(3):
        directory = audit / 'pools/bfcl' / f'B{cap}_seed{seed}'
        m = io.read_json(directory / 'manifest.json')
        base = rows(directory / 'pool_sft.jsonl')
        if (m['benchmark'] != 'bfcl' or m['cap'] != cap or
                m['initial_checkpoint'] != io.INITIAL_CHECKPOINT or
                io.digest(base) != m['arms']['sft']['pool_sha256'] or
                m['sealed_replay_spend'] != m['arms']['sft']['C_m']):
            raise ValueError(f'invalid audited SFT pool: {directory}')
        pools.append((directory, m, base))
    _, first, base = pools[0]
    for _, m, other in pools[1:]:
        if (other != base or any(m[k] != first[k] for k in
                ('tasks_covered', 'purchased_ids', 'sealed_replay_spend')) or
                m['arms']['sft']['teacher_call_ids'] != first['arms']['sft']['teacher_call_ids']):
            raise ValueError('the three training seeds do not share the purchased pool')
    return pools, sorted(set(first['tasks_covered']))


def task_entries(task_ids):
    """Read public single-turn questions/schemas; no possible answers for ranking."""
    entries, selection = {}, {}
    for tid in task_ids:
        category = tid.rsplit('_', 1)[0]
        if category.startswith(('multi_turn', 'memory', 'web_search')):
            raise ValueError(f'only single-turn BFCL supported: {tid}')
        selection.setdefault(category, []).append(tid)
    for category, wanted in selection.items():
        path = BFCL / 'bfcl_eval/data' / f'BFCL_v4_{category}.json'
        for entry in rows(path):
            if entry.get('id') in wanted:
                if len(entry.get('question', [])) != 1 or 'function' not in entry:
                    raise ValueError(f'not a single-turn function task: {entry.get("id")}')
                entries[entry['id']] = entry
    missing = set(task_ids) - entries.keys()
    if missing:
        raise ValueError(f'official BFCL tasks missing: {sorted(missing)}')
    return entries, selection


def sample_rows(path, task_ids):
    out = {}
    for row in rows(path):
        tid, index = row['task_id'], row['sample_index']
        if (tid not in task_ids or type(index) is not int or index < 0 or
                not isinstance(row['response'], str) or type(row['verified']) is not bool):
            raise ValueError('invalid or unpurchased student sample')
        key = (tid, index)
        if key in out and out[key] != row:
            raise ValueError(f'conflicting sample: {key}')
        out[key] = row
    return out


def collect_repeat(run_dir, index, selection):
    from bfas.adapters.bfcl import extract_verdicts, read_score_summaries

    result_dir, score_dir = run_dir / f'result_r{index}', run_dir / f'score_r{index}'
    wanted = {tid for ids in selection.values() for tid in ids}
    results = {}
    for path in result_dir.rglob('*_result.json'):
        for row in rows(path):
            tid = row['id']
            if tid in results or tid not in wanted:
                raise ValueError(f'duplicate/unselected official result: {tid}')
            if 'traceback' in row or str(row['result']).startswith('Error during inference:'):
                raise ValueError(f'official generation failed for {tid}; inspect {path}')
            results[tid] = row
    if results.keys() != wanted:
        raise ValueError(f'incomplete repeat {index}: missing {sorted(wanted-results.keys())}')
    if set(read_score_summaries(score_dir)) != set(selection):
        raise ValueError(f'incomplete official scores for repeat {index}')
    verdicts = extract_verdicts(score_dir, selection)
    output = []
    for tid in sorted(wanted):
        row = results[tid]
        prompts = [log.get('content', {}).get('formatted_prompt')
                   for log in row.get('inference_log', [])
                   if isinstance(log, dict) and isinstance(log.get('content'), dict)]
        prompt = next((p for p in prompts if isinstance(p, str) and p), None)
        if prompt is None:
            raise ValueError(f'missing official --include-input-log prompt for {tid}')
        response = row['result']
        if not isinstance(response, str):
            response = json.dumps(response, ensure_ascii=False)
        output.append(dict(task_id=tid, sample_index=index, response=response,
                           verified=verdicts[tid], prompt=prompt, messages=[]))
    return output


def sample_environment(run_dir, gpu, port):
    env = os.environ.copy()
    # The official CLI loads PROJECT_ROOT/.env; use an isolated root without one.
    for key in list(env):
        if key.startswith(('BFCL_', 'REMOTE_OPENAI_', 'OPENAI_', 'OLLAMA_', 'AZURE_')):
            env.pop(key)
    env.update(BFCL_PROJECT_ROOT=str(run_dir), PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=f'{ROOT}/src:{ROOT}', CUDA_DEVICE_ORDER='PCI_BUS_ID',
               CUDA_VISIBLE_DEVICES=gpu, LOCAL_SERVER_ENDPOINT='127.0.0.1',
               LOCAL_SERVER_PORT=str(port), VLLM_USE_FLASHINFER_SAMPLER='0',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_HUB_DISABLE_TELEMETRY='1', VLLM_NO_USAGE_STATS='1',
               VLLM_CACHE_ROOT=str(run_dir / 'cache/vllm'),
               HF_MODULES_CACHE=str(run_dir / 'cache/hf_modules'),
               XDG_CACHE_HOME=str(run_dir / 'cache'),
               TRITON_CACHE_DIR=str(run_dir / 'cache/triton'),
               TORCHINDUCTOR_CACHE_DIR=str(run_dir / 'cache/torchinductor'),
               CUDA_CACHE_PATH=str(run_dir / 'cache/cuda'),
               TMPDIR=str(run_dir / 'tmp'))
    # Keep model-cache lookup at its original location even after relocating XDG.
    env['HF_HOME'] = os.environ.get('HF_HOME', str(Path(
        os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'huggingface'))
    return env


def sample_commands(run_dir, port, index):
    server = [str(ROOT / 'envs/vllm-serve/.venv/bin/vllm'), 'serve', MODEL,
              '--served-model-name', MODEL, '--host', '127.0.0.1', '--port', str(port),
              '--gpu-memory-utilization', '0.85', '--max-model-len', '32768']
    bfcl = str(ROOT / 'envs/bfcl/.venv/bin/bfcl')
    generate = [bfcl, 'generate', '--model', MODEL + '-FC', '--run-ids',
                '--skip-server-setup', '--temperature', '1.0', '--num-threads', '4',
                '--include-input-log', '--result-dir', str(run_dir / f'result_r{index}')]
    evaluate = [bfcl, 'evaluate', '--model', MODEL + '-FC', '--result-dir',
                str(run_dir / f'result_r{index}'), '--score-dir',
                str(run_dir / f'score_r{index}'), '--partial-eval']
    return server, generate, evaluate


def sample(audit, cap, repeats, gpu, port):
    _, tids = purchased(audit, cap)
    _, selection = task_entries(tids)
    directory = audit / 'bfcl'
    path = directory / f'ddpo_samples_B{cap}.jsonl'
    run_dir = io.output_path(directory / f'ddpo_sample_B{cap}' / 'run.json').parent
    with locked(directory):
        identity = dict(model=MODEL, adapter=None, temperature=1.0, task_ids=tids)
        identity_path = run_dir / 'identity.json'
        if identity_path.exists() and io.read_json(identity_path) != identity:
            raise ValueError('sampling identity changed; refuse to mix runs')
        if path.exists() and not identity_path.exists():
            raise ValueError('existing samples lack initial-student provenance')
        atomic_write(identity_path, identity)
        have = sample_rows(path, tids)
        todo = [i for i in range(repeats) if any((tid, i) not in have for tid in tids)]
        if not todo:
            print(f'Already sampled {len(tids)} tasks × {repeats} repeats: {path}')
            return
        if (run_dir / '.env').exists():
            raise ValueError('isolated BFCL root must not contain .env overrides')
        atomic_write(run_dir / 'test_case_ids_to_generate.json', selection)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0 if port == 'auto' else int(port)))
            chosen_port = sock.getsockname()[1]
        env = sample_environment(run_dir, gpu, chosen_port)
        io.output_path(run_dir / 'tmp/placeholder').parent.mkdir(exist_ok=True)
        server_cmd, _, _ = sample_commands(run_dir, chosen_port, 0)
        atomic_write(run_dir / 'run.json', dict(**identity, gpu=gpu, port=chosen_port,
                     repeats=repeats, commands=[sample_commands(run_dir, chosen_port, i) for i in todo]))
        print('CUDA_VISIBLE_DEVICES=' + shlex.quote(gpu) + ' ' + shlex.join(server_cmd), flush=True)
        with io.output_path(run_dir / 'vllm.log').open('a') as log:
            server = subprocess.Popen(server_cmd, cwd=run_dir, env=env,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + 1500
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                while True:
                    if server.poll() is not None:
                        raise RuntimeError(f'vLLM exited; see {run_dir}/vllm.log')
                    try:
                        with opener.open(f'http://127.0.0.1:{chosen_port}/v1/models', timeout=3) as r:
                            if MODEL in [m['id'] for m in json.load(r)['data']]:
                                break
                    except (OSError, ValueError, urllib.error.URLError):
                        pass
                    if time.monotonic() >= deadline:
                        raise RuntimeError('vLLM startup timed out')
                    time.sleep(5)
                for index in todo:
                    _, generate, evaluate = sample_commands(run_dir, chosen_port, index)
                    for name, command in [('generate', generate), ('evaluate', evaluate)]:
                        print(shlex.join(command), flush=True)
                        with io.output_path(run_dir / f'{name}_r{index}.log').open('a') as log:
                            subprocess.run(command, cwd=run_dir, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, check=True)
                    for row in collect_repeat(run_dir, index, selection):
                        key = (row['task_id'], index)
                        if key in have and have[key] != row:
                            raise ValueError(f'official sample changed on resume: {key}')
                        have[key] = row
                    atomic_write(path, [have[k] for k in sorted(have)], jsonl=True)
                    print(f'Saved repeat {index}: {len(tids)} official checked samples', flush=True)
            finally:
                if server.poll() is None:
                    os.killpg(server.pid, signal.SIGTERM)
                    try:
                        server.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(server.pid, signal.SIGKILL)
                        server.wait()


class TeacherClient:
    def __init__(self, key_file, max_output_tokens):
        self.key_file = key_file
        self.max_output_tokens = max_output_tokens
        self.config = None

    def prepare(self):
        # Lazy credentials: a no-op resume performs no key-file read or API call.
        if self.config is None:
            key = Path(self.key_file).expanduser().read_text().strip()
            if not key:
                raise ValueError('empty teacher key file')
            os.environ['OLLAMA_API_KEY'] = key
            os.environ.setdefault('OLLAMA_BASE_URL', 'https://ollama.com')
            self.config = at.load_teacher_config(TEACHER)

    def __call__(self, messages, usage):
        self.prepare()
        old_limit, old_think = at.MAX_COMPLETION_TOKENS, os.environ.get('TEACHER_THINK')
        at.MAX_COMPLETION_TOKENS = self.max_output_tokens
        os.environ['TEACHER_THINK'] = '0'
        try:
            return at.generate_reply(self.config, messages, temperature=0.0, usage_out=usage,
                                     max_retries=0, rotate_keys=False)
        finally:
            at.MAX_COMPLETION_TOKENS = old_limit
            if old_think is None:
                os.environ.pop('TEACHER_THINK', None)
            else:
                os.environ['TEACHER_THINK'] = old_think


@lru_cache(maxsize=1)
def response_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL, local_files_only=True)


def response_token_count(text):
    """Offline CPU tokenizer count; proxy tokenizer, hence always estimated."""
    return len(response_tokenizer().encode(text, add_special_tokens=False))


def charge(usage, reply, token_counter):
    for key in ('completion_tokens', 'output_tokens', 'eval_count'):
        cost = usage.get(key)
        if type(cost) is int and cost >= 0:
            return cost, 'exact', f'provider.{key}'
    if reply is None:
        return None, 'unknown', 'no provider usage or response (requires reconciliation)'
    # Unknown usage plus an empty response is not a provider-confirmed free call.
    return max(1, token_counter(reply)), 'estimated', 'Qwen3.5-4B response tokenizer (minimum 1)'


def rank_context(tid, candidates, base):
    sample_prompt = candidates[0].get('prompt')
    if sample_prompt:
        if any(c.get('prompt') != sample_prompt for c in candidates):
            raise ValueError(f'samples have different task contexts: {tid}')
        return dict(messages=candidates[0].get('messages', []), prompt=sample_prompt)
    # Useful for compatible imported samples. Never use a generated child's prompt.
    original = next((r for r in base if r['task_id'] == tid), None)
    if original is not None and original['prompt']:
        return dict(messages=original['messages'], prompt=original['prompt'])
    raise ValueError(f'{tid}: samples need the official prompt (run sample first)')


def preference(tid, context, candidates, pair):
    best, worst = pair
    answer = candidates[best - 1]
    return dict(task_id=tid, teacher='rank', turn_index=0,
                messages=context['messages'], prompt=context['prompt'],
                response=answer, _rejected=candidates[worst - 1],
                token_hint=max(len(answer) // 4, 1))


def read_calls(path):
    calls = {}
    for row in rows(path):
        call_id = row['call_id']
        if call_id in calls and calls[call_id] != row:
            raise ValueError(f'conflicting ranking call: {call_id}')
        if row.get('purpose') != 'rank' or row.get('teacher') != TEACHER:
            raise ValueError('not a compatible dDPO ranking ledger')
        cost = row['tokens_spent']
        if cost is not None and (type(cost) is not int or cost < 0):
            raise ValueError(f'invalid ranking cost: {call_id}')
        calls[call_id] = row
    return calls


def with_unsettled(calls, intents):
    """Keep the crash window visible without pretending an intent is settled."""
    return dict(calls, **{q: dict(r, status='unsettled', tokens_spent=None,
                                  confidence='unknown')
                          for q, r in intents.items() if q not in calls})


def render_ranked_pools(pools, tids, calls, eligible, abort_reason=None, *, row_renderer=None):
    """Pure replay of paid rankings; optionally adapt recorded preference rows."""
    relevant = [c for c in calls.values() if c['task_id'] in tids]
    ranked = {}
    for call in relevant:
        if call['status'] == 'ranked':
            if call['task_id'] in ranked:
                raise ValueError(f'more than one ranking for {call["task_id"]}')
            ranked[call['task_id']] = call['preference_row']
    cost = sum(c['tokens_spent'] for c in relevant if c['tokens_spent'] is not None)
    unknown = [c['call_id'] for c in relevant if c['tokens_spent'] is None]
    terminal = {c['task_id'] for c in relevant if c['status'] in ('ranked', 'parse_failed')}
    pending = sorted(set(eligible) - terminal)
    status = ('incomplete_ranking_costs' if unknown else 'partial_ranking' if pending or abort_reason
              else 'ready' if ranked else 'no_rankable_preferences')
    rendered = []
    for directory, source_manifest, base in pools:
        manifest = deepcopy(source_manifest)
        preferences = [deepcopy(ranked[t]) for t in sorted(ranked)]
        if row_renderer is not None:
            preferences = [row_renderer(row) for row in preferences]
        output = deepcopy(base) + preferences
        arm = manifest['arms']['ddpo']
        purchased_cost = manifest['sealed_replay_spend']
        total = purchased_cost + cost
        arm.update(status=status, trainer_status=status, rows=len(output),
                   C_m=total, purchased_C_m=purchased_cost, ranking_cost=cost,
                   ranking_call_ids=sorted(c['call_id'] for c in relevant),
                   teacher_call_ids=sorted(set(manifest['arms']['sft']['teacher_call_ids']) |
                                          {c['call_id'] for c in relevant}),
                   remaining_budget=manifest['cap'] - total, exceeds_cap=total > manifest['cap'],
                   over_cap_tokens=max(0, total-manifest['cap']),
                   pool_sha256=io.digest(output), pbsd_pairs=len(ranked), rank_pairs=len(ranked),
                   ranking_cost_complete=not unknown, unknown_ranking_call_ids=unknown,
                   ranking_cost_confidence=('unknown' if unknown else 'estimated' if any(
                       c['confidence'] == 'estimated' for c in relevant) else 'exact'),
                   pending_task_ids=pending,
                   skipped_parse_task_ids=sorted(terminal-set(ranked)),
                   fewer_than_two_distinct_task_ids=sorted(set(tids)-set(eligible)),
                   abort_reason=abort_reason,
                   note='Audited SFT + initial-student rank preferences; all ranking attempts charged. '
                        'Cap limits evidence acquisition; ranking spend is additional. '
                        'C_m excludes unresolved unknown costs when ranking_cost_complete=false.')
        manifest['ranking']['new_ddpo'] = dict(
            ledger='results/table1_audit/bfcl/ddpo_rank_ledger.jsonl',
            call_ids=arm['ranking_call_ids'], cost=cost, status=status, ranked_tasks=sorted(ranked))
        rendered.append((directory, manifest, output))
    summary = dict(status=status, ranked_tasks=len(ranked), ranking_cost=cost,
                C_m=rendered[0][1]['arms']['ddpo']['C_m'],
                exceeds_cap=rendered[0][1]['arms']['ddpo']['exceeds_cap'],
                pending_task_ids=pending, unknown_call_ids=unknown, abort_reason=abort_reason)
    return rendered, summary


def materialize(pools, tids, calls, eligible, abort_reason=None):
    rendered, summary = render_ranked_pools(pools, tids, calls, eligible, abort_reason)
    for directory, manifest, output in rendered:
        atomic_write(directory / 'pool_ddpo.jsonl', output, jsonl=True)
        atomic_write(directory / 'manifest.json', manifest)
    return summary


def rank(audit, cap, max_calls, client, *, token_counter=response_token_count, sleep=time.sleep):
    pools, tids = purchased(audit, cap)
    directory = audit / 'bfcl'
    with locked(directory):
        samples = sample_rows(directory / f'ddpo_samples_B{cap}.jsonl', tids)
        if not samples:
            raise ValueError('no student samples; run sample first')
        grouped = {}
        for key in sorted(samples):
            row = samples[key]
            group = grouped.setdefault(row['task_id'], [])
            if row['response'] not in [r['response'] for r in group]:
                group.append(row)
        eligible = {t: c for t, c in grouped.items() if len(c) >= 2}
        if set(grouped) != set(tids):
            raise ValueError('sample file does not cover all purchased task IDs')
        ledger = directory / 'ddpo_rank_ledger.jsonl'
        journal = directory / 'ddpo_rank_attempts.jsonl'
        calls = read_calls(ledger)
        intents = {r['call_id']: r for r in rows(journal)}
        unresolved = set(intents) - calls.keys()
        if unresolved:
            materialize(pools, tids, with_unsettled(calls, intents), eligible,
                        'unsettled ranking attempts require reconciliation')
            raise RuntimeError(f'unsettled ranking attempts; reconcile ledger before resume: {sorted(unresolved)}')
        # A cumulative hard ceiling on HTTP attempts for this evidence set, including retries.
        used = sum(c['task_id'] in tids for c in calls.values())
        completed = {c['task_id'] for c in calls.values() if c['status'] in ('ranked', 'parse_failed')}
        contexts = {t: rank_context(t, c, pools[0][2]) for t, c in eligible.items()
                    if t not in completed}
        abort = None
        try:
            for tid, candidate_rows in sorted(eligible.items()):
                if tid in completed:
                    continue
                candidates = [r['response'] for r in candidate_rows]
                context = contexts[tid]
                prompt = RANK_PROMPT.format(question=context['prompt'], cands='\n'.join(
                    f'[{i+1}] {response}' for i, response in enumerate(candidates)))
                prior = [c for c in calls.values() if c['task_id'] == tid]
                if any(c['status'] == 'api_error' and c.get('http_status') != 429 for c in prior):
                    abort = f'{tid}: previous API failure requires reconciliation'
                    break
                for retry in range(MAX_RATE_RETRIES + 1):
                    if used >= max_calls:
                        break
                    if hasattr(client, 'prepare'):
                        client.prepare()
                    call_id = 'ddpo-rank-' + uuid.uuid4().hex
                    record = dict(call_id=call_id, task_id=tid, teacher=TEACHER, purpose='rank',
                                  attempt_index=len(prior), temperature=0.0, think=False,
                                  verified=False, timestamp=datetime.now(timezone.utc).isoformat(),
                                  cap=cap, candidates=candidates, context=context,
                                  max_output_tokens=getattr(client, 'max_output_tokens', None),
                                  prompt_sha256=io.digest(prompt))
                    append(journal, record)
                    intents[call_id] = record.copy()
                    used += 1
                    usage, reply, error = {}, None, None
                    try:
                        reply = client([dict(role='user', content=prompt)], usage)
                    except at.TeacherAPIError as exc:
                        error = exc
                    try:
                        cost, confidence, source = charge(usage, reply, token_counter)
                    except Exception as exc:
                        # Preserve paid text even if the local fallback tokenizer fails.
                        cost, confidence, source = None, 'unknown', f'token count failed: {type(exc).__name__}'
                    pair = last_two_numbers(reply) if reply is not None else None
                    valid = pair is not None and pair[0] != pair[1] and all(
                        1 <= n <= len(candidates) for n in pair)
                    record.update(tokens_spent=cost, confidence=confidence, cost_source=source,
                                  usage=usage, response=reply, status='api_error' if error else
                                  'ranked' if valid else 'parse_failed',
                                  http_status=getattr(error, 'status_code', None))
                    if error:
                        record['error'] = str(error)
                    if valid:
                        record['preference_row'] = preference(tid, context, candidates, pair)
                        record['best_worst'] = list(pair)
                    append(ledger, record)
                    calls[call_id] = record
                    prior.append(record)
                    print(f'{tid}: {record["status"]}, tokens={cost} ({confidence}), calls={used}/{max_calls}', flush=True)
                    if error is None:
                        break  # Includes malformed/zero-usage replies: never buy another ranking.
                    if getattr(error, 'status_code', None) != 429 or retry == MAX_RATE_RETRIES:
                        abort = f'{tid}: {error}'
                        break
                    if used < max_calls:
                        sleep(min(2 ** retry, 4))
                if abort or used >= max_calls:
                    break
        except BaseException:
            materialize(pools, tids, with_unsettled(calls, intents), eligible,
                        'interrupted; reconcile any unsettled journal attempts')
            raise
        summary = materialize(pools, tids, calls, eligible, abort)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return summary


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    sampling = commands.add_parser('sample', help='GPU-only official BFCL student sampling; no teacher')
    sampling.add_argument('--cap', type=positive, default=13843)
    sampling.add_argument('--repeats', type=positive, default=4)
    sampling.add_argument('--gpu', required=True, help='CUDA GPU UUID')
    sampling.add_argument('--port', default='auto', help='auto or an unused local TCP port')
    ranking = commands.add_parser('rank', help='CPU + teacher API; cumulative attempt cap, resume safe')
    ranking.add_argument('--cap', type=positive, default=13843)
    ranking.add_argument('--key-file', default='~/.ollama_api_key')
    ranking.add_argument('--max-calls', type=positive, default=40)
    ranking.add_argument('--max-output-tokens', type=positive, default=64)
    args = parser.parse_args(argv)
    try:
        if args.command == 'sample':
            sample(io.DEFAULT_OUT, args.cap, args.repeats, args.gpu, args.port)
            return 0
        result = rank(io.DEFAULT_OUT, args.cap, args.max_calls,
                      TeacherClient(args.key_file, args.max_output_tokens))
        return 0 if result['status'] == 'ready' else 2
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f'dDPO: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.dont_write_bytecode = True
    raise SystemExit(main())
