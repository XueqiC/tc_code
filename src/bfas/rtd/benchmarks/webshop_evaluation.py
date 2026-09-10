"""Official adapter campaigns with the shared RTD checkpoint/hardware contract."""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import time

from ..persistence import atomic_json, digest, tree_hash, ComputeJournal


@contextmanager
def frozen_environment(benchmark):
    settings = dict(BFAS_NO_SERVER='0', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if benchmark == 'alfworld':
        settings.update(BFAS_ALFWORLD_EVAL_SPLIT='valid_seen', BFAS_ALFWORLD_EVAL_GAMES='140',
                        BFAS_ALFWORLD_MAX_STEPS='40', BFAS_ALFWORLD_STUDENT_REACT='1')
    old = {k: os.environ.get(k) for k in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def validate_records(benchmark, out, metrics, expected):
    rows = [json.loads(line) for line in (Path(out)/'records.jsonl').read_text().splitlines() if line.strip()]
    ids = [str(r['session']) if benchmark == 'webshop' else r['task_id'] for r in rows]
    wanted = [str(i) for i in range(500)] if benchmark == 'webshop' else expected['task_ids']
    if sorted(ids) != sorted(wanted) or len(set(ids)) != len(ids):
        raise ValueError('official evaluation must cover every session/task exactly once')
    if any(r.get('error') for r in rows):
        raise ValueError('incomplete official evaluation: environment/backend errors')
    verdicts = {tid: r['reward'] == 1. if benchmark == 'webshop' else r['won'] for tid, r in zip(ids, rows)}
    rate = sum(verdicts.values())/len(rows)
    if any(type(v) is not bool for v in verdicts.values()) or not math.isclose(metrics['success_rate'], rate, abs_tol=1e-12):
        raise ValueError('official metric disagrees with episode records')
    if any(type(r['steps']) is not int or not 0 < r['steps'] <= (15 if benchmark == 'webshop' else 40) for r in rows):
        raise ValueError('official episode horizon differs')
    return dict(complete=True, verdicts=verdicts, n=len(rows))


def evaluate_adapter(root, directory, round_number, *, port=None, base_evaluation=None,
                     lock_timeout=None, lock_log_interval=None):
    from ..cli import data_identity, hardware_identity
    from ..hardware import guard_hardware
    from ..identity import verified_checkpoint, guard_harness, record_code_drift
    from ..evaluation import _flatten_adapter
    from ..evaluation_lock import evaluation_lock, tag_lock_path
    from ...run import serving_lane
    root, directory = Path(root), Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    config = manifest['config']; benchmark = config['benchmark']
    meta = verified_checkpoint(directory, manifest, round_number)
    if tree_hash(manifest['model_path']) != manifest['base_checkpoint_hash']:
        raise ValueError('evaluation base checkpoint changed')
    hardware = guard_hardware(root, directory, manifest, hardware_identity(), context=f'evaluation-{round_number}')
    if data_identity(root, config, manifest['bank_path']) != manifest['data_hash']:
        raise ValueError('evaluation data differs from training')
    identities = guard_harness(root, directory, manifest)
    drift = record_code_drift(root, directory, manifest, context=f'evaluation-{round_number}', identities=identities)
    expected = identities['evaluation_harness']['expected']
    identity = dict(benchmark=benchmark, checkpoint=meta, config_hash=manifest['config_hash'],
        data_hash=manifest['data_hash'], base_checkpoint_hash=manifest['base_checkpoint_hash'],
        tokenizer_hash=manifest['tokenizer_hash'], hardware_hash=digest(hardware['hard']),
        expected_hash=digest(expected), evaluation_harness_hash=identities['harness_hash'], evaluation_temperature=0.)
    tag = f"rtd_{benchmark}_{manifest['arm']}_{digest(identity)[:16]}_r{round_number}"
    out = root/'results'/f'{benchmark}_std'/tag
    completed = directory/f'evaluation-{round_number}.json'
    journal = ComputeJournal(directory/'compute.jsonl', cuda=False)
    with evaluation_lock(tag_lock_path(root, tag), tag=tag, timeout=lock_timeout,
            log_interval=lock_log_interval, on_wait=lambda **event: journal.append('evaluation_lock_wait',
                round=round_number, gpu_seconds=0., gpu_reserved_seconds=0., idle_seconds=event['wall_seconds'], **event)):
        if completed.exists():
            result = json.loads(completed.read_text())
            if result['identity'] != identity or result['artifacts_hash'] != tree_hash(out):
                raise ValueError('evaluation identity/artifacts changed')
            validate_records(benchmark, out, result['metrics'], expected)
            return result
        out.mkdir(parents=True, exist_ok=True)
        merged = directory/f'evaluation-export-{round_number}'
        export_identity = dict(checkpoint=meta, base_checkpoint_hash=manifest['base_checkpoint_hash'])
        export_stamp = directory/f'evaluation-export-{round_number}.json'
        if merged.exists():
            stamp = json.loads(export_stamp.read_text())
            if stamp != dict(identity=export_identity, hash=tree_hash(merged)):
                raise ValueError('evaluation export identity changed')
        else:
            _flatten_adapter(manifest, directory/f'round-{round_number}', merged)
            atomic_json(export_stamp, dict(identity=export_identity, hash=tree_hash(merged)))
        if benchmark == 'webshop':
            from ...adapters.webshop import WebShopAdapter
            adapter = WebShopAdapter(port=port or 8900)
        else:
            from ...adapters.alfworld import ALFWorldAdapter
            adapter = ALFWorldAdapter(port=port or 8900)
        start = time.monotonic()
        try:
            with frozen_environment(benchmark):
                adapter.prepare_renderer(str(merged))
                with serving_lane(adapter, str(merged), os.environ['CUDA_VISIBLE_DEVICES'], port or 8900, out/'vllm.log'):
                    metrics = adapter.evaluate(str(merged), out)
        finally:
            adapter.release_policy()
        elapsed = time.monotonic()-start
        validation = validate_records(benchmark, out, metrics, expected)
        result = dict(identity=identity, campaign_identity=identity, expected=expected,
            hardware_class=hardware['hard'], hardware_class_hash=digest(hardware['hard']), code_drift=drift,
            artifacts_hash=tree_hash(out), merged_hash=tree_hash(merged), validation=validation,
            overall_accuracy_percent=100*metrics['success_rate'], overall_metric='success_rate',
            success_rate=metrics['success_rate'], metrics=metrics,
            checkpoint_spend=meta['actual_spend'], authorized_budget=meta['authorized_budget'],
            campaign_seconds=elapsed, campaign_log=str(out/'vllm.log'), reused_campaign=False,
            output_directory=str(out), port=port or 8900,
            evaluation_label='development evaluation', score_objective='official_greedy_score; separate from stochastic J')
        if base_evaluation:
            base = json.loads(Path(base_evaluation).read_text())
            if (base['hardware_class_hash'] != result['hardware_class_hash'] or base['expected'] != expected
                    or not base['validation']['complete']):
                raise ValueError('base comparison needs the same complete protocol and hardware')
            before = base['validation']['verdicts']
            result['repairs_damage'] = {t: dict(before=before[t], after=v,
                repaired=not before[t] and v, damaged=before[t] and not v) for t, v in validation['verdicts'].items()}
        journal.append('evaluation_end', round=round_number, identity=identity, wall_seconds=elapsed,
                       gpu_reserved_seconds=elapsed, overall_accuracy_percent=result['overall_accuracy_percent'])
        atomic_json(completed, result)
        return result


def evaluate(root, directory, round_number, **kwargs):
    return evaluate_adapter(root, directory, round_number, **kwargs)
