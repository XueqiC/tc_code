"""Offline Luna task packages; charge every recorded attempt to its task."""
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from ... import hotpotqa as hp
from ...adapters.hotpotqa import HotpotQAAdapter
from ..bank_build import privileged, load_student_tokenizer, ledger_cost, state_record, seal_v11
from ..caps import resolve_budget_checkpoints
from ..persistence import digest
from ..transport import FullState
from .hotpotqa_support import freeze_support, parent_hash, reset_state, task_request


def _snapshot(path, *, lines=False):
    raw = Path(path).read_bytes()
    # A collecting writer may be midway through its final append. Bind only the
    # complete durable prefix, and disclose the ignored suffix in the audit.
    end = raw.rfind(b'\n') + 1 if lines else len(raw)
    content = raw[:end]
    value = [json.loads(line) for line in content.splitlines() if line.strip()] if lines else json.loads(content)
    return value, dict(sha256=hashlib.sha256(content).hexdigest(), bytes=len(content), trailing_bytes=len(raw)-end)


def _validate_usage(usage):
    if (any(type(usage.get(k)) is not int or usage[k] < 0
            for k in ('prompt_tokens', 'completion_tokens', 'cached_tokens'))
            or usage['cached_tokens'] > usage['prompt_tokens']):
        raise ValueError('invalid HotpotQA recorded usage')


def _confidence(row):
    return 'estimated' if (row['usage_status'] == 'estimated'
                          or row.get('cost_confidence') == 'estimated') else 'exact'


def build_hotpotqa_bank(root, directory, *, pool, ledger, config, tokenizer=None, wiki_factory=None):
    privileged()
    if config['benchmark'] != 'hotpotqa':
        raise ValueError('HotpotQA config required')
    pool, ledger = Path(pool), Path(ledger)
    if pool.is_dir():
        pool = pool/'demos.json'
    archive, pool_binding = _snapshot(pool)
    rows, ledger_binding = _snapshot(ledger, lines=True)
    support = freeze_support()
    if (archive.get('schema_version') != 1 or archive.get('prompt_version') != hp.PROMPT_VERSION
            or archive.get('teacher') != 'openai/gpt-5.6-luna'):
        raise ValueError('HotpotQA pool teacher/prompt/schema mismatch')
    questions = {q['_id']: q for q in hp.load_questions('train')}
    if set(questions) != set(support['split']['ids']):
        raise ValueError('HotpotQA support question inventory differs')
    adapter = HotpotQAAdapter(offline=True)
    adapter._tokenizer = tokenizer or load_student_tokenizer(config)
    wiki_factory = wiki_factory or (lambda: hp.Wikipedia(cache_dir=Path(root)/config['hotpotqa_cache_root'], offline=True))
    resets = {t: asdict(reset_state(adapter, q)) for t, q in questions.items()}
    inputs = dict(pool=pool_binding, ledger=ledger_binding)
    optional = {}
    for name in ('attempts.jsonl', 'usage.jsonl', 'identity.json'):
        path = ledger.with_name(name)
        if path.exists():
            optional[name], inputs[name] = _snapshot(path, lines=name.endswith('.jsonl'))
        else:
            inputs[name] = None
    identity = optional.get('identity.json')
    if identity is not None and (identity['teacher'] != archive['teacher']
            or identity['prompt_version'] != hp.PROMPT_VERSION or identity['support'] != support['split']):
        raise ValueError('HotpotQA collector identity differs')
    paid, grouped = {}, defaultdict(list)
    totals = defaultdict(int)
    for index, row in enumerate(rows):
        if row.get('task_id') not in questions:
            raise ValueError('HotpotQA ledger attempt has missing or unknown task')
        key = row['task_id'], row['attempt_index']
        ledger_cost(row, default_confidence='exact' if row.get('usage_status') == 'reported' else 'estimated')
        if (key in paid
                or row['teacher'] != archive['teacher'] or row['prompt_version'] != hp.PROMPT_VERSION
                or row.get('usage_status') not in {'reported', 'estimated'}):
            raise ValueError('duplicate, outside, or incompatible HotpotQA ledger attempt')
        usage = row['usage']
        _validate_usage(usage)
        if usage['completion_tokens'] != row['tokens_spent']:
            raise ValueError('HotpotQA ledger cost differs from recorded usage')
        for k, v in usage.items():
            totals[k] += v
        paid[key] = row
        grouped[key[0]].append((index, row))
    archived_keys = set()
    for row in optional.get('attempts.jsonl', []):
        key = row['task_id'], row['attempt_index']
        # A newer audit suffix may accompany an older ledger prefix snapshot.
        if key not in paid:
            continue
        if key in archived_keys or any(row[k] != paid[key][k] for k in
                ('teacher', 'verified', 'usage', 'tokens_spent', 'usage_status', 'prompt_version')):
            raise ValueError('HotpotQA attempts archive differs from ledger')
        archived_keys.add(key)
    latest = {r['id']: r for r in optional.get('usage.jsonl', [])}
    by_attempt = defaultdict(list)
    for row in latest.values():
        if row.get('task_id') not in questions:
            raise ValueError('HotpotQA request usage has missing or unknown task')
        _validate_usage(row['usage'])
        by_attempt[row['task_id'], row['attempt_index']].append(row)
    if latest:
        for key, row in paid.items():
            calls = by_attempt[key]
            if not calls or any(sum(c['usage'][k] for c in calls) != row['usage'][k]
                                for k in ('prompt_tokens', 'completion_tokens', 'cached_tokens')):
                raise ValueError('HotpotQA request journal differs from episode usage')
            if row['usage_status'] == 'reported' and any(c['status'] != 'reported' for c in calls):
                raise ValueError('exact HotpotQA episode contains uncertain request usage')
    records, payloads, demos = [], {}, defaultdict(list)
    replayed = {}
    for index, row in enumerate(rows):
        tid, attempt = row['task_id'], row['attempt_index']
        behaviors, verification, empty_targets = [], None, []
        if row['verified']:
            demo = row['demo']
            demos[tid].append(demo)
            turns = iter(demo['turns'])
            # Replay cached tools and compare every archived context. No model,
            # token estimation, or regenerated teacher output is involved.
            def replay(messages, stop, temperature):
                turn = next(turns)
                if turn['context'] != messages or not isinstance(turn['target'], str):
                    raise ValueError('HotpotQA paid context differs from replayed ReAct history')
                state = FullState.create(task_request(questions[tid]), messages, adapter._render(messages), parent_hash(tid))
                target = turn['target']
                if not target.strip():
                    # teacher_tokens appends EOS to blank replies. Spell that
                    # same native termination explicitly for paper readers that
                    # require nonempty targets; preserve raw text in the ledger.
                    eos = adapter._tokenizer.eos_token
                    if not isinstance(eos, str) or not eos:
                        raise ValueError('HotpotQA empty reply requires a native EOS token')
                    empty_targets.append(len(behaviors))
                    target += eos
                behaviors.append(dict(state=asdict(state), text=target))
                return turn['target']
            wiki = wiki_factory()
            try:
                verification = hp.run_episode(questions[tid], wiki, replay)
            finally:
                if hasattr(wiki, 'close'):
                    wiki.close()
            if (not verification['verified'] or verification['error'] or next(turns, None) is not None
                    or len(behaviors) != len(demo['turns'])
                    or demo['worked_example'] != f"Question: {questions[tid]['question']}\n" + verification['transcript']):
                raise ValueError('HotpotQA paid demo does not replay to official EM=1')
        replayed[tid, attempt] = behaviors, verification, empty_targets
    if set(archive['demos']) - set(demos):
        raise ValueError('HotpotQA pool includes unpurchased demos; retry a consistent snapshot')
    for tid, demo in archive['demos'].items():
        # Collectors may publish a later success. It must still match a paid,
        # replay-verified attempt, even when the package uses the earlier one.
        if demo not in demos[tid]:
            raise ValueError('HotpotQA pool demo differs from paid ledger')
    for tid, attempts in grouped.items():
        attempts = sorted(attempts, key=lambda item: item[1]['attempt_index'])
        verified = [(i, r) for i, r in attempts if r['verified']]
        index, row = (verified or attempts)[0]
        attempt = row['attempt_index']
        total = sum(r['tokens_spent'] for _, r in attempts)
        confidence = 'estimated' if any(_confidence(r) == 'estimated' for _, r in attempts) else 'exact'
        usage = {k: sum(r['usage'][k] for _, r in attempts) for k in totals}
        qid = digest(['hotpotqa-paid-episode', row['teacher'], tid, attempt, hp.PROMPT_VERSION])
        behaviors, verification, empty_targets = replayed[tid, attempt]
        state = FullState(**behaviors[0]['state']) if behaviors else None
        records.append(state_record(qid, state, 'hotpotqa_demo_episode', total, confidence,
            parent=parent_hash(tid), unavailable=None if verified else 'no verified teacher attempt'))
        payloads[qid] = dict(cost=total, cost_confidence=confidence, usage=usage,
            provenance=dict(task_id=tid, attempt_index=attempt, teacher=row['teacher'], ledger_row=index,
                ledger_rows=[i for i, _ in attempts], charged_attempt_indices=[r['attempt_index'] for _, r in attempts],
                selection='earliest_verified_attempt_index', cost_scope='all_recorded_task_attempts',
                ledger_sha256=ledger_binding['sha256'], verified=row['verified'], rendering_student=config['student'],
                empty_targets_rendered_with_native_eos=empty_targets),
            historical_response=row, historical_attempts=[r for _, r in attempts],
            behaviors=behaviors, verification=verification)
    accounting = dict(historical_output_tokens=sum(r['tokens_spent'] for r in rows),
        historical_attempts=len(rows), verified_episodes=sum(r['verified'] for r in rows),
        historical_tasks=len(grouped), verified_tasks=len(demos),
        later_verified_attempts=sum(len(ds)-1 for ds in demos.values()),
        failed_attempts=sum(not r['verified'] for r in rows),
        failed_output_tokens=sum(r['tokens_spent'] for r in rows if not r['verified']),
        exact_output_tokens=sum(r['tokens_spent'] for r in rows if _confidence(r) == 'exact'),
        estimated_output_tokens=sum(r['tokens_spent'] for r in rows if _confidence(r) == 'estimated'),
        cost_confidence='estimated' if any(_confidence(r) == 'estimated' for r in rows) else 'exact',
        usage=dict(totals), scope='completed ledger prefix; all attempts charged once per task',
        new_teacher_calls=0, new_teacher_tokens=0)
    audit = dict(m=200, **accounting, snapshot=inputs, pool_demos=len(archive['demos']),
        packages=len(records), unavailable_packages=len(grouped)-len(demos),
        available_packages_by_confidence=dict(Counter(p['cost_confidence'] for p in payloads.values()
            if p['provenance']['verified'])),
        budget_checkpoints_tokens=resolve_budget_checkpoints(config,
            sum(p['cost'] for p in payloads.values() if p['provenance']['verified'])),
        ledger_demos_not_yet_in_pool=sorted(set(demos)-set(archive['demos'])),
        attempts_with_trajectory_archive=len(archived_keys),
        request_calls_outside_completed_ledger=sum((r['task_id'], r['attempt_index']) not in paid for r in latest.values()))
    return seal_v11(directory, records, payloads, benchmark='hotpotqa', student=config['student'],
        public={'support.json': support, 'reset_states.json': resets}, audit=audit,
        inputs=inputs, teacher_accounting=accounting)
