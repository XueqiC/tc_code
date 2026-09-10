"""P1 config validation and an explicit adapter to the v1.1 scheduling shell."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import yaml

from ..persistence import digest, file_hash
from ..conventions import schedule_path
from .arms import ARMS
from .engine import UnifiedConfig

ROOT = Path(__file__).resolve().parents[4]
P1_DEFAULTS = dict(schedule='recorded_exposure', optimization='shared_fixed_eta',
    feedback='three_roles_rotating_training', sampling='shared_role_seed',
    smoke_slots=2, smoke_packages=1, smoke_feedback_tasks=1, smoke_rollouts=2,
    d0=dict(mode='rtd_sft_kl', teacher_weight=1.0, kl_weight=1.0,
            tuning_split='calibration_only', teacher_weight_grid=[0.5, 1.0], kl_weight_grid=[0.1, 1.0, 10.0]))


def enabled(config):
    return config.get('method') == 'rtd_unified'


def arm_config(config, arm=None, replay_schedule=None, **options):
    result = deepcopy(config)
    arm = arm or result.get('arm') or 'D3'
    if arm not in ARMS:
        raise ValueError('unified configurations require D0/D1/D2/D3 or a D3 attribution arm')
    if options and any(v is not None for v in options.values()):
        if any(k != 'replay_mode' or v != 'complete' for k, v in options.items() if v is not None):
            raise ValueError('P1 requires one completed recorded schedule (D14 content hashes)')
    result.update(arm=arm, arm_preset=asdict(ARMS[arm]))
    result['unified']['estimator'] = ARMS[arm].estimator
    result.setdefault('p1', deepcopy(P1_DEFAULTS))
    path = replay_schedule or result.get('replay_schedule')
    if path:
        path = schedule_path(path).resolve()
        actual = file_hash(path)
        if result.get('replay_schedule_hash', actual) != actual:
            raise ValueError('P1 replay schedule changed')
        result.update(replay_schedule=str(path), replay_schedule_hash=actual, replay_mode='complete')
    return result, arm


def runtime_config(config):
    """Only the executor adapter sees v1.1 flags; manifest keeps unified identity."""
    benchmark = config['benchmark']
    base = ({} if config.get('p1_runtime_defaults_frozen') else
            yaml.safe_load((ROOT/f'configs/rtd/v1_1_{benchmark}.yaml').read_text()))
    result = base | deepcopy(config)
    result.update(method='rtd_v1_1', protocol_version='1.1.0', source_estimator='alpha_d',
        loss='alpha_d_single_step_estimator', gate_mode='learned_alpha', d_mode='learned',
        acquisition='random', acquisition_value_mode='joint', metrics_v11={'enabled': False})
    return result


def validate_config(config, *, arm=None, replay_schedule=None, **options):
    config, _ = arm_config(config, arm, replay_schedule, **options)
    UnifiedConfig.from_config(config)
    p1 = config['p1']
    if set(p1) != set(P1_DEFAULTS) or any(p1[k] != P1_DEFAULTS[k] for k in P1_DEFAULTS if k != 'd0'):
        raise ValueError('P1 matching/smoke protocol changed')
    d0 = p1['d0']
    if (d0.get('mode') != 'rtd_sft_kl' or d0.get('tuning_split') != 'calibration_only'
            or set(d0) != set(P1_DEFAULTS['d0'])):
        raise ValueError('D0 calibration-only SFT/KL settings required')
    import math
    for key in ('teacher_weight', 'kl_weight'):
        grid = d0[key+'_grid']
        if not grid or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in grid) or d0[key] not in grid:
            raise ValueError('D0 coefficients must belong to the preregistered calibration grid')
    if (config['rounds'] != 2 or config['budget_checkpoints_bank_fraction'] != [.1, .25]
            or config['memory_peak_budget_gb'] != 60 or config['mode'] != 'sealed_replay'
            or config['training_seed'] != 0 or not math.isfinite(config['initial_eta']) or config['initial_eta'] <= 0):
        raise ValueError('P1 requires two rounds, 60 GB, seed 0 and positive frozen eta')
    # Validate the shared sampling, ledger, harness, optimizer and LoRA contract
    # using the existing validator, without exposing a D arm to legacy presets.
    from ..cli import validate_config as validate_legacy
    runtime = runtime_config(config)
    for key in ('arm', 'replay_schedule', 'replay_schedule_hash', 'replay_mode'):
        runtime.pop(key, None)
    checked = validate_legacy(runtime)
    # Persist every inherited executor knob. Resume must not consult today's
    # benchmark YAML for missing alpha/d/acquisition defaults.
    config = checked | config | dict(p1_runtime_defaults_frozen=True)
    for key in ('score_consistency_tolerance', 'generation_batch'):
        config[key] = checked[key]
    return config


def manifest_fields(config, manifest):
    if config.get('replay_schedule') and file_hash(config['replay_schedule']) != config['replay_schedule_hash']:
        raise ValueError('P1 recorded schedule changed')
    matched = dict(initialization=manifest['base_checkpoint_hash'], tokenizer=manifest['tokenizer_hash'],
        data=manifest['data_hash'], hardware=manifest['hardware_hash'],
        schedule=config.get('replay_schedule_hash'), seed=config['training_seed'],
        eta=config['initial_eta'], rank=config['lora_rank'], lora_alpha=config['lora_alpha'],
        lora_target_modules=config['lora_target_modules'], steps=1 if manifest['smoke'] else 24,
        smoke=manifest['smoke'],
        protocol=config['p1'])
    return dict(version='rtd-unified-p1-run', distillation_protocol=config['arm'],
        arm_components=config['arm_preset'], p1_matching=matched,
        campaign_identity=digest(dict(matched=matched, arm=config['arm'], config=manifest['config_hash'])),
        replay_schedule_hash=config.get('replay_schedule_hash'),
        loss_normalization='total_nll' if config['arm'] == 'D1' else config['unified']['loss_definition']['normalization'],
        source_sampling=dict(temperature=1., top_p=1., refresh='every_commit', cache_reuse=False,
                             rng_stream='seed/round/step/role; common across P1 arms'),
        baseline_trainer_hash=file_hash(ROOT/'src/appworld_train.py'))
