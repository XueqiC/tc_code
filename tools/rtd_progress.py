#!/usr/bin/env python3
"""Read-only feedback progress from a live RTD compute journal (no model imports).

Generation totals are horizon bounds/estimates: terminal episodes can stop at
any step. Count compute_end once per compute_begin, never subphase_memory.
Ignore only an unfinished final JSONL record; do not repair the live journal.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import yaml


def records(path):
    with Path(path).open('rb') as stream:
        remaining = Path(path).stat().st_size  # fixed snapshot of an append-only file
        while remaining:
            line = stream.readline(remaining)
            remaining -= len(line)
            if not line or not line.endswith(b'\n'):
                break
            yield json.loads(line)


def feedback_progress(directory, config):
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    configured = config['meta_tasks_per_feedback'] * config['rollouts_per_meta_task']
    saved = manifest.get('config', config)
    phases, begins, owners, active, ended = [], {}, {}, {}, set()
    available, last_timestamp = {}, None

    def matching(row):
        for phase in reversed(phases):
            if (phase['round'], phase['step'], phase['role']) == (row.get('round'), row.get('step'), row.get('role')):
                return phase

    for row in records(directory / 'compute.jsonl'):
        kind = row['kind']
        last_timestamp = row.get('timestamp', last_timestamp)
        if kind == 'round_frozen':
            available[row['round']] = len(row['feedback'])
        if kind == 'compute_begin':
            sequence = row['sequence']
            begins[sequence] = row
            phase = owners.get(row.get('parent_sequence'))
            if ('round' in row and 'step' in row and
                    ('feedback' in row['operation'] or row['operation'].startswith('validation_'))):
                phase = dict(begin=sequence, round=row['round'], step=row['step'], role=row['operation'],
                    status='running', planned=None, accepted=0, episodes={}, generations=0,
                    sequences=0, samples=0, generation_seconds=0., sampled_steps=0, retries=0,
                    sizes=Counter(), timestamp=row.get('timestamp'))
                phases.append(phase)
                active[sequence] = phase
            owners[sequence] = phase
        elif kind == 'compute_end':
            ended.add(row['begin_sequence'])
            begin = begins.get(row['begin_sequence'], {})
            phase = owners.get(row['begin_sequence'])
            if phase is not None:
                if begin.get('operation') == 'generation' and row['status'] == 'complete':
                    size = begin.get('sequences', 1)
                    phase['generations'] += 1
                    phase['sequences'] += size
                    phase['generation_seconds'] += row['wall_seconds']
                    phase['sizes'][size] += 1
                if row['begin_sequence'] == phase['begin']:
                    phase['status'] = row['status']
                    active.pop(phase['begin'], None)
        elif kind == 'feedback_plan':
            phase = matching(row)
            if phase is not None:
                phase['planned'] = row['episodes']
        elif kind in {'feedback_rollout', 'validation_rollout'}:
            phase = matching(row)
            if phase is not None:
                phase['accepted'] += 1
                phase['sampled_steps'] += len(row['rollout']['actions'])
        elif kind in {'alfworld_episode', 'alfworld_episode_retry', 'generated_tokens'} and active:
            phase = active[max(active)]
            if kind == 'alfworld_episode':
                phase['episodes'][(row['task_id'], row['rollout_index'])] = not row.get('excluded', False)
            elif kind == 'alfworld_episode_retry':
                phase['retries'] += 1
            elif row.get('context') == f"r{phase['round']}/s{phase['step']}/{phase['role']}":
                phase['samples'] += 1
    if not phases:
        raise ValueError('no feedback stage found in compute.jsonl')
    phase = phases[-1]
    prior = next((p for p in reversed(phases[:-1]) if p['status'] == 'complete' and p['accepted'] and
                  (p['round'], p['step']) == (phase['round'], phase['step'])), None)
    if phase['planned'] is not None:
        total, source = phase['planned'], 'feedback_plan'
    elif phase['status'] == 'complete':
        total, source = phase['accepted'], 'completed stage records'
    elif prior is not None:
        total, source = prior['accepted'], 'completed feedback stage in the same window'
    elif saved.get('method') in {'rtd_v1_1', 'rtd_v1'} and saved.get('benchmark') == 'alfworld':
        limit, count = (2, 1) if manifest.get('smoke') else (8, 4)
        total = min(limit, available.get(phase['round'], limit)) * count
        source = 'legacy RTD selector (fixed task/rollout counts)'
    else:
        total, source = configured, 'config (selected task count not yet recorded)'
    horizon = config.get('alfworld_max_episode_steps', 40)
    done = max(phase['accepted'], sum(phase['episodes'].values()))
    # Realized episode length from a completed, same-window measurement is a
    # useful ETA proxy. Never present it as the exact future generation total.
    estimate = (prior['sampled_steps'] / prior['accepted'] * total if prior else None)
    rate = phase['generation_seconds'] / phase['sequences'] if phase['sequences'] else None
    remaining = max(0., estimate - phase['samples']) if estimate is not None else None
    in_flight = sum(1 for seq, begin in begins.items() if begin.get('operation') == 'generation' and
                   owners.get(seq) is phase and seq not in ended)
    finished = done >= total
    estimated_calls = (phase['generations'] * estimate / phase['sequences']
                       if estimate is not None and phase['sequences'] else None)
    return dict(stage=f"r{phase['round']}/s{phase['step']}/{phase['role']}", status=phase['status'],
        timestamp=last_timestamp, episodes_done=done, episodes_total=total, episodes_source=source,
        configured_episodes=configured, accepted_rollouts=phase['accepted'], retries=phase['retries'],
        generations_done=phase['generations'], generations_in_flight=in_flight,
        generations_total=phase['generations'] if finished else None, estimated_generations=estimated_calls,
        generation_batch_sizes=dict(sorted(phase['sizes'].items())), sampled_actions=phase['samples'],
        action_horizon_bound=total*horizon, estimated_actions=estimate,
        generation_eta_seconds=(0. if finished else
            remaining*rate if remaining is not None and rate is not None else None),
        max_generation_seconds_remaining=(0. if finished else
            max(0, total*horizon-phase['samples'])*rate if rate else None))


def duration(seconds):
    if seconds is None:
        return 'unknown'
    minutes = round(seconds / 60)
    return f'{minutes//60}h {minutes%60:02d}m'


def format_progress(report):
    r = report
    lines = [f"Stage: {r['stage']} ({r['status']}; journal through {r['timestamp']})",
        f"Feedback episodes: {r['episodes_done']}/{r['episodes_total']} done; "
        f"{r['accepted_rollouts']} accepted; {r['retries']} RPC retries",
        f"Episode total source: {r['episodes_source']} (config requests {r['configured_episodes']})",
        f"Generated actions: {r['sampled_actions']}/{r['action_horizon_bound']} horizon bound; "
        "exact total depends on episode termination",
        f"Generate calls: {r['generations_done']} complete, {r['generations_in_flight']} in flight; "
        f"batch sizes {r['generation_batch_sizes']}",
        "Exact remaining call count depends on live prompts and token-budget splits."]
    if r['estimated_actions'] is not None:
        lines.append(f"Same-window estimate: ~{r['estimated_actions']:.0f} total actions; "
                     f"generation ETA ~{duration(r['generation_eta_seconds'])}")
    if r['generations_total'] is not None:
        lines.append(f"Generation finished: {r['generations_total']} calls total; scoring may still be running.")
    elif r['estimated_generations'] is not None:
        lines.append(f"Estimated total generate calls at observed batching: ~{r['estimated_generations']:.0f}")
    lines.append(f"Horizon-based generation ETA: {duration(r['max_generation_seconds_remaining'])}; "
                 "excludes environment, retries and scoring time")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--json', action='store_true', help='print machine-readable counters and estimates')
    args = parser.parse_args()
    report = feedback_progress(args.run_dir, yaml.safe_load(args.config.read_text()))
    print(json.dumps(report, indent=2) if args.json else format_progress(report))


if __name__ == '__main__':
    main()
