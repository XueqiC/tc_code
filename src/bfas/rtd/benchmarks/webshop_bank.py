"""Offline GPT-5.4 demos, live replay verification, and one-state packages."""
from dataclasses import asdict
import json
from types import SimpleNamespace

from ...adapters.webshop import WebShopAdapter, webshop_eval
from ..bank_build import (privileged, read_pool, load_student_tokenizer, allocate_cost,
                          ledger_cost, state_record, seal_v11)
from ..persistence import digest, file_hash
from ..transport import FullState
from .webshop_support import freeze_support, reset_state, parent_hash


def build_webshop_bank(root, directory, *, pool, ledger, config, adapter=None, tokenizer=None):
    privileged()
    adapter = adapter or WebShopAdapter()
    adapter._tokenizer = tokenizer or load_student_tokenizer(config)
    try:
        support = freeze_support(adapter)
        demand = set(support['split']['demand'])
        resets = {t: asdict(reset_state(adapter, t)) for t in sorted(demand)}
        demos = {str(r['task_id']): r for r in read_pool(pool)}
        rows = read_pool(ledger)
        records, payloads, used = [], {}, set()
        historical = 0
        for index, row in enumerate(rows):
            if str(row.get('teacher', '')).removeprefix('azure/') != 'gpt-5.4':
                raise ValueError('WebShop bank requires GPT-5.4 ledger')
            tid = str(row['task_id'])
            # The adapter can estimate missing Azure usage. Older gateway rows
            # do not distinguish that fallback, so never claim exact billing.
            total, episode_confidence = ledger_cost(row, default_confidence='estimated')
            historical += total
            q = digest(['webshop-ledger', file_hash(ledger), index, tid, row['attempt_index']])
            if not row['verified'] or tid not in demand:
                records.append(state_record(q, None, 'webshop_demo_state', total, episode_confidence,
                    unavailable='failed attempt or outside S_d', parent=parent_hash(tid)))
                payloads[q] = dict(cost=total, cost_confidence=episode_confidence, usage={'output_tokens': total},
                    provenance={'ledger_row': index}, historical_response=row, behaviors=[])
                continue
            if tid in used:
                raise ValueError('duplicate successful ledger episode')
            demo = row['demo']; turns = demo['turns']
            pool_demo = demos.get(tid, {}).get('demo', demos.get(tid, {}))
            if pool_demo.get('turns') != turns:
                raise ValueError('pool demo differs from paid ledger payload')
            used.add(tid)
            captured = []
            class ReplayClient:
                def __init__(self):
                    self.chat = SimpleNamespace(completions=self)
                def create(self, **kwargs):
                    i = len(captured)
                    if i >= len(turns):
                        raise ValueError('teacher episode has fewer actions than replay')
                    # The adapter archives action targets, whereas teacher history
                    # can contain thoughts/format failures. Verify the live page,
                    # then render the replayed command history through deployment.
                    source = turns[i]['context']
                    current = kwargs['messages'][1]['content'].rsplit('\nObservation: ', 1)[-1]
                    if (source[0] != kwargs['messages'][0] or
                            not source[1]['content'].endswith('Observation: ' + current)):
                        raise ValueError('teacher episode observation differs from live replay')
                    captured.append(kwargs['messages'])
                    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=turns[i]['target']))])
            verified = adapter._episode(tid, 0., client=ReplayClient())
            if not verified.verified or not adapter._environment().done or len(captured) != len(turns):
                raise ValueError('teacher demo does not replay to success within 15 steps')
            costs = allocate_cost(total, [t['target'] for t in turns], adapter._tokenizer)
            previous = []
            for i, (turn, cost) in enumerate(zip(turns, costs)):
                messages = captured[i]
                state = FullState.create(dict(benchmark='webshop', session=int(tid)), messages,
                    adapter._render(messages), parent_hash(tid))
                if webshop_eval.parse_action(turn['target']) != turn['target']:
                    raise ValueError('teacher target must be the adapter action')
                state_q = digest([q, i])
                confidence = episode_confidence if len(turns) == 1 else 'estimated'
                records.append(state_record(state_q, state, 'webshop_demo_state', cost, confidence, dependencies=previous))
                payloads[state_q] = dict(cost=cost, cost_confidence=confidence,
                    usage=dict(recorded_episode_output_tokens=total, allocation='target-token proportional; integer total preserved'),
                    provenance=dict(task_id=tid, ledger_row=index, ledger_sha256=file_hash(ledger),
                        episode_id=q, state_index=i, verified=True, verified_episode_steps=len(turns),
                        transcript_hash=digest(captured)), historical_response=turn,
                    behaviors=[dict(state=asdict(state), text=turn['target'])])
                previous = [state_q]
        if set(demos)-used:
            raise ValueError('pool contains unpaid, unverified or protected demos')
        return seal_v11(directory, records, payloads, benchmark='webshop', student=config['student'],
            public={'support.json': support, 'reset_states.json': resets},
            audit=dict(m=len(demand), historical_output_tokens=historical, historical_attempts=len(rows),
                       verified_episodes=len(used), failed_attempts=sum(not r['verified'] for r in rows)),
            inputs=dict(pool=file_hash(pool), ledger=file_hash(ledger)))
    finally:
        adapter.close()
