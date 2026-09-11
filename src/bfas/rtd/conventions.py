"""D7 frozen arm identities, exposure schedules and reporting vocabulary.

Schedules contain evidence identities only. Source actions and gate values are
deliberately absent: each student draws fresh actions from its own round model.
"""
from collections import Counter
import json
import math
from pathlib import Path

from .persistence import atomic_json, digest, file_hash

ARMS = {
    'V0': dict(acquisition_value_mode='independent', acquisition='random',
               gate_mode='fixed_alpha', fixed_alpha=.5, d_mode='zero', ledger_replay=False),
    'V1': dict(acquisition_value_mode='joint', acquisition='random',
               gate_mode='learned_alpha', fixed_alpha=.5, d_mode='learned', ledger_replay=True),
    'V2': dict(acquisition_value_mode='joint', acquisition='bayesian_linear_posterior_sampling',
               gate_mode='learned_alpha', fixed_alpha=.5, d_mode='learned', ledger_replay=False),
}
ADAPTIVE_EFFECT = '自适应蒸馏整体效果'
CORE_CONTROL = '同 α 的 learned d vs d=0：隔离核心机制'
BASE_CHECKPOINT = '发行方原始 checkpoint,未进行本项目 bank 适配'
DEVELOPMENT = 'development evaluation'
CERTIFICATION = 'certification: bootstrap bounds on held-out data after freezing'


def schedule_path(path):
    path = Path(path)
    return path/'exposure_schedule.json' if path.is_dir() else path


def replay_config(config, path):
    """Share D14 mode validation and binding across all exposure followers."""
    result = dict(config)
    path = schedule_path(path).resolve()
    mode = result.get('replay_mode', 'complete')
    if mode not in {'complete', 'streaming'}:
        raise ValueError('replay_mode must be complete or streaming')
    if mode == 'streaming':
        for key, default in [('replay_poll_seconds', 60.), ('replay_timeout_seconds', 36*3600.)]:
            value = result.setdefault(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive finite seconds')
        if result.get('replay_schedule_hash') is not None:
            raise ValueError('streaming replay cannot use a complete-mode config hash')
        result.update(replay_schedule=str(path), replay_mode=mode)
    else:
        expected = result.get('replay_schedule_hash')
        actual = file_hash(path)
        if expected is not None and expected != actual:
            raise ValueError('replay schedule changed')
        result.update(replay_schedule=str(path), replay_schedule_hash=actual)
    return result


def arm_config(config, arm=None, replay_schedule=None, **replay_options):
    """A first-class arm selects every coupled setting before validation/hash."""
    if config.get('method') == 'rtd_unified':
        from .unified.config import arm_config as unified_arm_config
        return unified_arm_config(config, arm, replay_schedule, **replay_options)
    arm = arm or config.get('arm') or ('V2' if 'gate_mode' in config else 'R1')
    if arm not in ARMS:
        if arm not in {'R0', 'R1'} or config.get('protocol_version') == '1.1.0' or 'gate_mode' in config:
            raise ValueError('R0/R1 are legacy arms; alpha/d uses V0/V1/V2')
        return dict(config), arm
    if config.get('protocol_version') != '1.1.0':
        raise ValueError('V0/V1/V2 require v1.1')
    result = config | ARMS[arm] | dict(arm=arm, source_estimator='alpha_d', source_samples_per_state=2)
    result.update({k: v for k, v in replay_options.items() if v is not None})
    if arm != 'V0' and 'acquisition_value_mode' in config:
        result['acquisition_value_mode'] = config['acquisition_value_mode']
    path = replay_schedule or result.get('replay_schedule')
    if arm == 'V1':
        if not path:
            raise ValueError('V1 requires --replay-schedule <V0 run dir>')
        result = replay_config(result, path)
    elif path:
        raise ValueError('only V1 may replay an exposure schedule')
    elif any(k in result for k in ('replay_mode', 'replay_poll_seconds', 'replay_timeout_seconds')):
        raise ValueError('only V1 may configure exposure replay')
    return result, arm


def record_identity(record, weight):
    return dict(query_id=record.query_id, record_index=record.record_index,
                state_hash=record.state.state_hash, has_teacher=record.teacher is not None,
                is_new=record.is_new, weight=float(weight))


def repetition_counts(records):
    counts = Counter((r['query_id'], r['record_index'], r['state_hash']) for r in records)
    return [r | dict(repetition_count=counts[r['query_id'], r['record_index'], r['state_hash']]) for r in records]


def step_content(row):
    return {k: v for k, v in row.items() if k != 'content_hash'}


def hashed_step(row):
    content = step_content(row)
    return content | dict(content_hash=digest(content))


def exposure_step(state):
    ref = state['d_reference']
    return dict(round=state['round'], step=state['step'], decision=state['decision'],
        selected=list(state['selected']), window_budget=state.get('window_budget') if state['decision'] else None,
        pool_before=sorted(state['owned_before']), pool_after=sorted(set(state['owned_before']) | set(state['selected'])),
        records=repetition_counts([record_identity(p.record, w) for p, w in zip(state['alpha_pairs'], ref.weights)]),
        reference_records=repetition_counts([record_identity(p.record, w) for p, w in
            zip(state['old_reference_pairs'], state['old_reference'].weights)]))


def schedule_identity(manifest, config, support):
    from .feedback_rng import feedback_rng_identity
    bank = Path(manifest['bank_path'])
    return dict(data_hash=manifest.get('data_hash'), base_checkpoint_hash=manifest.get('base_checkpoint_hash'),
        bank_public_hash=file_hash(bank/'public/requests.json'), bank_integrity_hash=file_hash(bank/'sealed/integrity.json'),
        initial_parameter_hash=manifest.get('initial_parameter_hash'), budget_ceilings=manifest['budget_ceilings'],
        support_hash=digest(support.parents), rounds=config.get('rounds', 2),
        training_seed=config['training_seed'], slots=config.get('slots_per_step', 40),
        K=config.get('max_new_packages_per_window', 20), **feedback_rng_identity(config, manifest))


def load_schedule(config, manifest, support, *, smoke=False, check_initial_parameters=True):
    path = schedule_path(config['replay_schedule'])
    if file_hash(path) != config['replay_schedule_hash']:
        raise ValueError('replay schedule changed')
    data = json.loads(path.read_text())
    if data.get('version') != 'rtd-v11-exposure-v1' or data.get('arm') != 'V0' or not data.get('complete'):
        raise ValueError('V1 requires a completed V0 exposure schedule')
    expected, actual = dict(data['identity']), schedule_identity(manifest, config, support)
    if not check_initial_parameters:
        # CPU preflight has no initialized LoRA tensors. Execution always checks
        # this field with the default above, including on resume.
        expected.pop('initial_parameter_hash', None)
        actual.pop('initial_parameter_hash', None)
    if expected != actual:
        raise ValueError('V0/V1 exposure schedule identity differs')
    if data.get('smoke') != smoke:
        raise ValueError('V0/V1 smoke mode differs')
    expected = [(1, 1)] if smoke else [(r, s) for r in range(1, config['rounds']+1) for s in range(1, 13)]
    if [(r['round'], r['step']) for r in data['steps']] != expected:
        raise ValueError('exposure schedule has missing/duplicate/out-of-order steps')
    # Historical complete schedules without step hashes remain supported.
    for row in data['steps']:
        if 'content_hash' in row and row['content_hash'] != digest(step_content(row)):
            raise ValueError('replay step content hash changed')
    return {(r['round'], r['step']): step_content(r) for r in data['steps']}


def export_schedule(engine):
    s = engine.state
    # Hash only export metadata: durable training rows and RNGs are unchanged.
    rows = [(hashed_step(r['exposure_schedule']) if engine.manifest['arm'] in {'V0', 'V1'} else r['exposure_schedule'])
            for r in s['steps']]
    atomic_json(engine.directory/'exposure_schedule.json', dict(version='rtd-v11-exposure-v1',
        arm=engine.manifest['arm'], smoke=s['smoke'], identity=schedule_identity(engine.manifest, engine.config, engine.support),
        complete=len(rows) == (1 if s['smoke'] else 12*s['rounds']), steps=rows))


PARENT_FOLD_RULE = 'int(parent_hash, 16) % 2'


def resolve_parent_folds(parents):
    """Normalize support parents, preserving explicit folds and auditing fallback.

    BFCL/WebShop use record lists; ALFWorld keys records by parent hash. Older
    callers may supply bare hashes or records without folds. All builders use
    hash parity by default, after excluding protected calibration/probe parents.
    """
    if isinstance(parents, dict):
        parents = [dict(p, parent_hash=h) if isinstance(p, dict)
                   else dict(parent_hash=h, official_id=p) for h, p in parents.items()]
    records, derived = [], []
    for parent in parents:
        if isinstance(parent, str):
            parent = dict(parent_hash=parent)
        if 'selected_task_id' in parent:
            parent = dict(parent, official_id=parent['selected_task_id'])
        parent_hash = parent['parent_hash']
        if not isinstance(parent_hash, str):
            raise ValueError('parent record requires a hash string')
        if 'fold' not in parent:
            fold = int(parent_hash, 16) % 2
            derived.append(parent_hash)
        else:
            fold = parent['fold']
            if type(fold) is not int or fold not in (0, 1):
                raise ValueError('parent record requires explicit fold 0 or 1')
        records.append(dict(parent, fold=fold))
    return records, sorted(derived)


def fold_roles(parents):
    """Rotating training/feedback groups, with hash parity for missing folds."""
    records, _ = resolve_parent_folds(parents)
    groups = {'0': [], '1': []}
    for parent in records:
        groups[str(parent['fold'])].append(parent['parent_hash'])
    groups = {f: sorted(hashes) for f, hashes in groups.items()}
    return {str(f): dict(inner_parent_groups=groups[str(f)], feedback_parent_groups=groups[str(1-f)]) for f in (0, 1)}


def certification_metadata(*, frozen_checkpoint_hash, held_out_data_hash):
    """Plumbing only; callers must supply independent held-out data after freeze."""
    if not frozen_checkpoint_hash or not held_out_data_hash:
        raise ValueError('certification needs frozen checkpoint and held-out data identities')
    return dict(label=CERTIFICATION, method='bootstrap', frozen_checkpoint_hash=frozen_checkpoint_hash,
                held_out_data_hash=held_out_data_hash, bounds=None, status='plumbing_only')
