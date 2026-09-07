"""Official full-campaign export, complete task validation, and budget reports."""
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .persistence import ComputeJournal, atomic_json, digest, file_hash, tree_hash
from .evaluation_lock import evaluation_lock, reserve_port, tag_lock_path
from .identity import evaluation_harness_metadata, guard_harness, record_code_drift, verified_checkpoint


def official_expectations(root):
    leaderboard = Path(root) / 'envs/bfcl/gorilla/berkeley-function-call-leaderboard'
    sys.path.insert(0, str(leaderboard))
    from bfcl_eval.utils import (parse_test_category_argument, is_format_sensitivity,
        load_dataset_entry, extract_test_category_from_id)
    generation, scoring = defaultdict(set), {}
    for category in parse_test_category_argument(['all']):
        if is_format_sensitivity(category):
            continue  # same FC exclusion as the campaign
        for row in load_dataset_entry(category):
            generation[extract_test_category_from_id(row['id'])].add(row['id'])
        scoring[category] = sorted(r['id'] for r in load_dataset_entry(category, include_prereq=False))
    return dict(generation={k: sorted(v) for k, v in generation.items()}, scoring=scoring)


def validate_evaluation(expected, resultdir, scoredir):
    """Missing tasks/categories never become implicit passes or zero scores."""
    from ..adapters.bfcl import extract_verdicts, read_score_summaries
    actual = defaultdict(list)
    for path in Path(resultdir).rglob('*_result.json'):
        category = path.stem.removeprefix('BFCL_v4_').removesuffix('_result')
        for line in path.read_text().splitlines():
            if line.strip():
                actual[category].append(json.loads(line)['id'])
    coverage = {}
    for category in sorted(set(expected['generation']) | set(actual)):
        wanted = set(expected['generation'].get(category, []))
        ids = actual.get(category, [])
        coverage[category] = dict(missing=sorted(wanted-set(ids)), unexpected=sorted(set(ids)-wanted),
                                  duplicates=sorted(k for k, v in Counter(ids).items() if v > 1))
    if any(any(r.values()) for r in coverage.values()):
        raise ValueError('incomplete evaluation generation: ' + json.dumps(coverage))
    summaries = read_score_summaries(Path(scoredir))
    for path in Path(scoredir).rglob('*_score.json'):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ids = [r['id'] for r in rows[1:] if 'id' in r]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate evaluation verdict IDs')
    if set(summaries) != set(expected['scoring']):
        raise ValueError('incomplete evaluation score categories')
    for category, ids in expected['scoring'].items():
        summary = summaries[category]
        if summary['total'] != len(ids) or not 0 <= summary['verified'] <= summary['total']:
            raise ValueError(f'incomplete evaluation summary: {category}')
    # Official score JSON contains failures plus a positive summary, not one
    # line per pass. Reconcile both before inferring any passing task.
    verdicts = extract_verdicts(Path(scoredir), expected['scoring'])
    return dict(complete=True, coverage=coverage, verdicts=verdicts, summaries=summaries)


def _flatten_adapter(manifest, checkpoint, destination):
    from peft import PeftModel
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # CPU export avoids a second student allocation on the evaluation GPU.
    base = AutoModelForCausalLM.from_pretrained(manifest['model_path'], local_files_only=True,
                                               torch_dtype=torch.bfloat16, trust_remote_code=False,
                                               device_map='cpu')
    # PEFT infers the adapter deserialization device independently of
    # device_map; without torch_device it loads safetensors onto any visible
    # CUDA GPU before copying them to the CPU model.
    model = PeftModel.from_pretrained(base, checkpoint / 'lora', local_files_only=True,
                                      device_map='cpu', torch_device='cpu')
    merged = model.merge_and_unload(safe_merge=True)
    merged.save_pretrained(destination, safe_serialization=True, max_shard_size='100GB')
    AutoTokenizer.from_pretrained(manifest['model_path'], local_files_only=True).save_pretrained(destination)
    if not (destination / 'model.safetensors').exists():
        raise ValueError('campaign requires the flattened model.safetensors export')


def evaluate(root, directory, round_number, *, port=None, base_evaluation=None,
             lock_timeout=None, lock_log_interval=None):
    root, directory = Path(root).resolve(), Path(directory).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    checkpoint = directory / f'round-{round_number}'
    meta = verified_checkpoint(directory, manifest, round_number)
    if tree_hash(manifest['model_path']) != manifest['base_checkpoint_hash']:
        raise ValueError('evaluation base checkpoint changed')
    from .cli import data_identity, hardware_identity
    if (digest(hardware_identity()) != manifest['hardware_hash']
            or data_identity(root, manifest['config'], manifest['bank_path']) != manifest['data_hash']):
        raise ValueError('evaluation hardware/data differs from training')
    identities = guard_harness(root, directory, manifest)
    drift = record_code_drift(root, directory, manifest, context=f'evaluation-{round_number}', identities=identities)
    expected = official_expectations(root)
    identity = dict(checkpoint=meta, config_hash=manifest['config_hash'], data_hash=manifest['data_hash'],
                    base_checkpoint_hash=manifest['base_checkpoint_hash'],
                    tokenizer_hash=manifest.get('tokenizer_hash'),
                    hardware_hash=manifest['hardware_hash'], expected_hash=digest(expected),
                    evaluation_harness_hash=identities['harness_hash'],
                    evaluation_temperature=manifest['config']['evaluation_temperature'])
    tag = f"rtd_{manifest['arm']}_{digest(identity)[:16]}_r{round_number}"
    out = root / 'results/bfcl_std' / tag
    completed = directory / f'evaluation-{round_number}.json'
    journal = ComputeJournal(directory / 'compute.jsonl', cuda=False)
    wait_seconds = 0.
    def record_wait(**event):
        nonlocal wait_seconds
        wait_seconds += event['wall_seconds']
        journal.append('evaluation_lock_wait', round=round_number, tag=tag,
                       accounting='idle', idle_seconds=event['wall_seconds'],
                       gpu_seconds=0., gpu_reserved_seconds=0., **event)
    settings = dict(timeout=lock_timeout, log_interval=lock_log_interval, on_wait=record_wait)
    with ExitStack() as locks:
        tag_fd = locks.enter_context(evaluation_lock(tag_lock_path(root, tag), tag=tag, **settings))
        # The cache check and publication use the same tag lease as the shell
        # export/copy path, including invocations outside the RTD coordinator.
        if completed.exists():
            result = json.loads(completed.read_text())
            if result['identity'] != identity or result['artifacts_hash'] != tree_hash(out):
                raise ValueError('evaluation identity/artifacts changed')
            validate_evaluation(expected, out / 'resultdir', out / 'scoredir')
            if result.get('code_drift') != drift:
                result['code_drift'] = drift
                atomic_json(completed, result)
            return result
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not visible or ',' in visible or visible == '-1':
            raise ValueError('evaluate requires one CUDA_VISIBLE_DEVICES GPU')
        port, port_fd = locks.enter_context(reserve_port(root, tag=tag, port=port,
            gpu_uuid=manifest.get('hardware', {}).get('uuid'), **settings))
        print(f'[rtd] evaluation resources tag={tag} port={port}', flush=True)
        stage = root / 'results/appworld_students' / tag
        stage.mkdir(parents=True, exist_ok=True)
        binding = stage / 'rtd_binding.json'
        if binding.exists() and json.loads(binding.read_text()) != identity:
            raise ValueError('campaign stage belongs to another checkpoint')
        atomic_json(binding, identity)
        merged, adapter = stage / 'hub_merged', stage / 'adapter'
        if not (stage / 'export.json').exists():
            # Incomplete local exports are reconstructable from the bound LoRA.
            for p in (merged, adapter):
                if p.exists():
                    shutil.rmtree(p)
            _flatten_adapter(manifest, checkpoint, adapter)
            subprocess.run([sys.executable, str(root / 'tools/bfcl_hub_merge_export.py'), '--adapter', str(adapter),
                '--out', str(merged), '--model', manifest['model_path'], '--verify'], check=True, cwd=root,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
            atomic_json(stage / 'export.json', dict(merged_hash=tree_hash(merged), adapter_hash=tree_hash(adapter)))
        export = json.loads((stage / 'export.json').read_text())
        if export['merged_hash'] != tree_hash(merged) or export['adapter_hash'] != tree_hash(adapter):
            raise ValueError('export changed after checkpoint binding')
        atomic_json(stage / 'expected.json', expected)
        if out.exists():
            # Preserve incomplete campaign evidence; never mix two attempts.
            out.rename(out.with_name(out.name + f'.incomplete-{time.time_ns()}'))
        env = dict(os.environ, BFCLSTD_PREMERGED=str(merged), BFCLSTD_PRESERVE_GENERATION='1',
                   BFCLSTD_BASE_MODEL=manifest['model_path'],
                   BFCLSTD_TAG_LOCK_FD=str(tag_fd), BFCLSTD_PORT_LOCK_FD=str(port_fd),
                   BFCLSTD_TEMPERATURE=str(manifest['config']['evaluation_temperature']))
        env.pop('BFCLSTD_LOCKED', None)  # always enter the validating wrapper
        log = stage / f'campaign-{time.time_ns()}.log'
        start = time.monotonic()
        event = journal.append('evaluation_begin', round=round_number, identity=identity, port=port,
                               tag=tag, merged_hash=export['merged_hash'])
        returncode = None
        try:
            with log.open('w') as stream:
                process = subprocess.run(['bash', str(root / 'tools/bfcl_std_campaign.sh'), visible, str(port), tag],
                    cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, pass_fds=(tag_fd, port_fd))
            returncode = process.returncode
        finally:
            elapsed = time.monotonic()-start
            journal.append('evaluation_end', begin_sequence=event, round=round_number, returncode=returncode,
                           wall_seconds=elapsed, gpu_reserved_seconds=elapsed)
        if process.returncode or f'[bfclstd] {tag} OVERALL=' not in log.read_text():
            raise RuntimeError(f'official campaign failed; see {log}')
        # Catch harness edits during the campaign before admitting its score.
        guard_harness(root, directory, manifest)
        drift = record_code_drift(root, directory, manifest, context=f'evaluation-{round_number}', identities=identities)
        validation = validate_evaluation(expected, out / 'resultdir', out / 'scoredir')
        with (out / 'data_overall.csv').open() as stream:
            score = float(next(csv.DictReader(stream))['Overall Acc'])
        if not math.isfinite(score) or not 0 <= score <= 100:
            raise ValueError('invalid official aggregate')
        result = dict(identity=identity, merged_hash=export['merged_hash'], expected=expected, code_drift=drift,
            evaluation_harness_metadata=evaluation_harness_metadata(root),
            artifacts_hash=tree_hash(out), overall_accuracy_percent=score, validation=validation,
            campaign_log=str(log), campaign_seconds=elapsed, evaluation_lock_idle_seconds=wait_seconds,
            port=port, output_directory=str(out))
        if base_evaluation:
            base = json.loads(Path(base_evaluation).read_text())
            if base['expected'] != expected or not base['validation']['complete']:
                raise ValueError('base comparison needs matching complete evaluation')
            before = base['validation']['verdicts']
            result['repairs_damage'] = {tid: dict(before=before[tid], after=ok,
                repaired=not before[tid] and ok, damaged=before[tid] and not ok)
                for tid, ok in validation['verdicts'].items()}
        atomic_json(completed, result)
        return result


def report(directories, output):
    from .ledger import Ledger
    from .persistence import ComputeJournal
    rows = []
    for directory in map(Path, directories):
        manifest = json.loads((directory / 'manifest.json').read_text())
        ledger = Ledger.resume(manifest['budget_ceilings'][0], directory / 'teacher.jsonl')
        trajectory = json.loads((directory / 'trajectory.json').read_text())
        journal = ComputeJournal(directory / 'compute.jsonl')
        compute = [e for e in journal.events if e['kind'] == 'compute_end']
        # Enclosing phases carry total time; inner per-forward events are
        # retained for diagnostics but must not be double-counted in totals.
        begin = {e['sequence']: e for e in journal.events if e['kind'] == 'compute_begin'}
        outer = [e for e in compute if begin[e['begin_sequence']].get('parent_sequence') is None]
        for checkpoint in trajectory['checkpoints']:
            r = checkpoint['round']
            checkpoint_dir = directory / f'round-{r}'
            if (checkpoint['manifest_hash'] != digest(manifest)
                    or json.loads((checkpoint_dir/'checkpoint.json').read_text()) != checkpoint
                    or tree_hash(checkpoint_dir/'lora') != checkpoint['adapter_hash']
                    or file_hash(checkpoint_dir/'round_state.pt') != checkpoint['round_state_hash']):
                raise ValueError('report checkpoint/manifest artifacts changed')
            p = directory / f'evaluation-{r}.json'
            result = json.loads(p.read_text()) if p.exists() else None
            if result:
                if result['identity']['checkpoint'] != checkpoint:
                    raise ValueError('score belongs to another checkpoint')
                out = Path(result['output_directory'])
                if result['artifacts_hash'] != tree_hash(out):
                    raise ValueError('evaluation artifacts changed')
                validate_evaluation(result['expected'], out / 'resultdir', out / 'scoredir')
            ids = set(checkpoint['owned'])
            charges = [e for e in ledger.events if e['kind'] == 'reveal' and e['query_id'] in ids]
            if not ids <= ledger.owned_ids or sum(e['cost'] for e in charges) != checkpoint['actual_spend']:
                raise ValueError('report checkpoint usage disagrees with reveal ledger')
            steps = [e for e in trajectory['steps'] if e['round'] <= r]
            rollouts = [e for e in journal.events if e['kind'] == 'feedback_rollout' and e['round'] <= r]
            windows = [dict(round=e['round'], step=e['step'], **e['truncation'])
                       for e in steps if e.get('decision') and 'truncation' in e]
            def event_round(event):
                context_round = event.get('context', '').split('/')[0]
                return event.get('round', int(context_round[1:])
                                 if context_round.startswith('r') and context_round[1:].isdigit() else 0)
            phases = [e for e in journal.events if e['kind'] == 'subphase_memory' and event_round(e) <= r]
            phase_peaks = {}
            for phase in phases:
                peak = phase_peaks.setdefault(phase['operation'], dict(peak_allocated_bytes=0, peak_reserved_bytes=0))
                for key in peak:
                    peak[key] = max(peak[key], phase[key])
            rows.append(dict(arm=manifest['arm'], round=r, mode=manifest['config']['mode'],
                actual_spend_x=sum(e['cost'] for e in charges), authorized_cap_budget=checkpoint['authorized_budget'],
                public_bank_cap_sum=manifest['bank_public_cap_sum'], purchased_packages=len(ids),
                actual_exact=sum(e['cost'] for e in charges if e['confidence']=='exact'),
                actual_estimated=sum(e['cost'] for e in charges if e['confidence']=='estimated'),
                historical_demo_output_exact=1233607, historical_generation_output_estimated=142727,
                committed_schedule_steps=len(steps), parameter_updates=sum(not e['exact_noop'] for e in steps),
                raw_slots=sum(e['raw_old_slots']+e['raw_new_slots'] for e in steps),
                weighted_slots=sum(e['weighted_old_slots']+e['weighted_new_slots'] for e in steps),
                feedback_rollouts=len(rollouts),
                # Journal totals include failed/repeated work; windows come
                # from committed trajectory and exclude reused actual feedback.
                sampled_truncated_rollouts=sum(e['rollout'].get('truncated', False) for e in rollouts),
                sampled_truncated_actions=sum(a.get('truncated', False) for e in rollouts for a in e['rollout']['actions']),
                sampled_truncated_sources=sum(e['action'].get('truncated', False) for e in journal.events
                    if e['kind'] == 'source_sample' and e['round'] <= r),
                committed_truncated_rollouts=sum(w['truncated_rollouts'] for w in windows),
                truncation_by_window=windows, subphase_memory_peaks=phase_peaks,
                source_exposure_tokens=sum(e['old_exposure']['source_action_tokens']+e['new_exposure']['source_action_tokens'] for e in steps),
                teacher_exposure_tokens=sum(e['old_exposure']['teacher_action_tokens']+e['new_exposure']['teacher_action_tokens'] for e in steps),
                training_prompt_tokens=sum(e['old_exposure']['prompt_tokens']+e['new_exposure']['prompt_tokens'] for e in steps),
                interrupted_compute_events=len(set(begin)-{e['begin_sequence'] for e in compute}),
                gpu_seconds=sum(e['gpu_seconds'] for e in outer if begin[e['begin_sequence']].get('round', 0) <= r),
                evaluation_lock_idle_seconds=sum(e['idle_seconds'] for e in journal.events
                    if e['kind'] == 'evaluation_lock_wait' and e['round'] <= r),
                evaluation_status='complete' if result else 'missing',
                code_drift=result.get('code_drift', []) if result else [],
                official_accuracy_percent=result['overall_accuracy_percent'] if result else None,
                checkpoint_hash=checkpoint['parameter_hash'], config_hash=manifest['config_hash'],
                hardware_hash=manifest['hardware_hash'], run=str(directory)))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    comparable = len({r['hardware_hash'] for r in rows}) <= 1
    atomic_json(output / 'budget_curve.json', dict(x_axis='actual recorded spend (exact + estimated flagged)',
        same_hardware=comparable, comparison_status='matched' if comparable else 'hardware mismatch; no matched-arm comparison', rows=rows))
    if rows:
        with (output / 'budget_curve.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = ['# RTD budget curves', '', 'The x-axis is actual recorded output-token spend; caps authorize purchases.', '',
             '| Arm | Round | Actual spend | Cap budget | Packages | Official accuracy (%) | Truncated rollouts (sampled / committed) | Evaluation |',
             '|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        score = '—' if r['official_accuracy_percent'] is None else str(r['official_accuracy_percent'])
        lines.append(f"| {r['arm']} | {r['round']} | {r['actual_spend_x']} | {r['authorized_cap_budget']} | "
                     f"{r['purchased_packages']} | {score} | {r['sampled_truncated_rollouts']} / "
                     f"{r['committed_truncated_rollouts']} | {r['evaluation_status']} |")
    lines.extend(['', 'Truncation per committed decision window (actual reuse is counted once):', '',
                  '| Run | Round | Step | Truncated rollouts | Truncated actions | Actual reused |',
                  '|---|---:|---:|---:|---:|---|'])
    for run in sorted({row['run'] for row in rows}):
        latest = max((row for row in rows if row['run'] == run), key=lambda row: row['round'])
        for window in latest['truncation_by_window']:
            lines.append(f"| {run} | {window['round']} | {window['step']} | {window['truncated_rollouts']} | "
                         f"{window['truncated_actions']} | {window['actual_feedback_reused']} |")
    if not comparable:
        lines.extend(['', 'Hardware hashes differ: these runs do not form a matched arm comparison.'])
    (output / 'budget_curve.md').write_text('\n'.join(lines)+'\n')
    return rows
