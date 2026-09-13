"""Command boundary for RTD. CPU commands never load a production model."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

from .bank import build_bfcl_bank
from .broker import SealedReplayBroker
from .caps import PUBLIC_CLASS_CAPS, affordability
from .caps import V11_BUDGET_BASIS, resolve_budget_checkpoints, validate_budget_checkpoints
from .config_v11 import v11_config, V11_DEFAULTS
from .alpha_d import ALPHA_D_DEFAULTS, enabled as alpha_d_enabled, validate_config as validate_alpha_d_config, validate_arm
from .ledger import Ledger
from .persistence import ComputeJournal, atomic_json, digest, exclusive_run, file_hash, tree_hash
from .scoring import ScoreTolerance, tolerance_override
from .memory import MemoryPolicy
from .generation_batch import GenerationBatch
from .source_estimator import validate_source_config
from .identity import (audit_legacy, evaluation_harness_identity, evaluation_harness_metadata,
                       source_identity, validate_resume, verified_checkpoint)
from .hardware import hardware_identity, instance, device_class, comparison_hash
from .conventions import (arm_config, BASE_CHECKPOINT, ADAPTIVE_EFFECT, CORE_CONTROL,
                          DEVELOPMENT, CERTIFICATION, fold_roles, resolve_parent_folds, PARENT_FOLD_RULE)

ROOT = Path(__file__).resolve().parents[3]


def load_config(path, *, arm=None, replay_schedule=None, **replay_options):
    config = yaml.safe_load(Path(path).read_text())
    return validate_config(config, arm=arm, replay_schedule=replay_schedule, **replay_options)


def validate_config(config, *, arm=None, replay_schedule=None, **replay_options):
    config = dict(config) if isinstance(config, dict) else config
    if not isinstance(config, dict):
        raise ValueError('configuration must be a mapping')
    if config.get('method') == 'rtd_unified':
        from .unified.config import validate_config as validate_unified
        return validate_unified(config, arm=arm, replay_schedule=replay_schedule, **replay_options)
    if config.get('arm') or arm or replay_schedule or replay_options:
        config, _ = arm_config(config, arm, replay_schedule, **replay_options)
    v11 = config.get('protocol_version') == '1.1.0'
    if (config.get('protocol_version') not in {'1.0.1', '1.1.0'} or config.get('mode') not in {'sealed_replay', 'fixed_evidence'}
            or config.get('budget_basis') != (V11_BUDGET_BASIS if v11 else 'usable_public_cap_sum')):
        raise ValueError('use the v1.0.1 public-cap configuration')
    canonical = yaml.safe_load((ROOT / 'configs/rtd/v1_bfcl_c25.yaml').read_text())
    benchmark = config.get('benchmark', 'bfcl')
    from .benchmarks.registry import REGISTRY
    if benchmark not in REGISTRY:
        raise ValueError('unknown RTD benchmark: ' + str(benchmark))
    if benchmark != 'bfcl':
        if not v11:
            raise ValueError('frozen protocol: interactive benchmarks require the v1.1 shared runner')
        from .benchmarks.config import benchmark_protocol
        canonical.update(benchmark_protocol(benchmark))
        for key in ('historical_demo_output_tokens_exact', 'historical_generation_output_tokens_estimated'):
            canonical.pop(key, None)
    if v11:
        config, canonical = v11_config(config, canonical)
        if benchmark != 'bfcl':
            canonical.update(benchmark_protocol(benchmark))
    elif ((set(V11_DEFAULTS)-set(canonical)) | {'budget_checkpoints_tokens'}) & config.keys():
        raise ValueError('v1.1 batch settings require protocol_version 1.1.0')
    mutable = {'student', 'output_root', 'replay_bank_path', 'support_manifest', 'max_action_tokens',
               'max_context_tokens', 'pilot_eta_candidates', 'initial_eta', 'model_local_files_only',
               'evaluate_after_round', 'mode', 'gate', 'acquisition', 'preconditioner',
               'score_consistency_tolerance', 'max_action_tokens_by_benchmark', 'max_state_batch_size',
               'memory_peak_budget_gb', 'memory_reserve_gb', 'memory_state_estimate_gb',
               'source_samples_per_state'}
    if v11:
        mutable |= {'rounds', 'budget_checkpoints_bank_fraction', 'budget_checkpoints_tokens', 'exposure_slots_per_window',
                    'max_new_packages_per_window', 'slots_per_step', 'drift_reference_packages', 'value_noise_floor'}
        mutable |= set(ALPHA_D_DEFAULTS)
        from .metrics_v11 import options as metric_options
        metric_options(config)
    validate_alpha_d_config(config)
    for key, expected in canonical.items():
        if key not in mutable and config.get(key) != expected:
            raise ValueError(f'frozen protocol value changed: {key}')
    if config.get('gate') not in {'linear_sigmoid', 'scalar_sigmoid'}:
        raise ValueError('supported gate components: linear_sigmoid/scalar_sigmoid')
    if config.get('acquisition') not in {'bayesian_linear_posterior_sampling', 'random'}:
        raise ValueError('unsupported acquisition component')
    if config.get('preconditioner') not in {'train_only_rms_diagonal', 'identity'}:
        raise ValueError('unsupported preconditioner component')
    if config.get('gate_override') not in {None, 'fixed_half', 'teacher_only'}:
        raise ValueError('unsupported fixed gate endpoint')
    if config['mode'] == 'fixed_evidence' and not config.get('fixed_evidence_ids'):
        # An empty source ledger is legal, and still retrains every cell.
        if 'fixed_evidence_ids' not in config:
            raise ValueError('fixed evidence mode needs an explicit ledger collection')
    if config['mode'] == 'sealed_replay' and 'fixed_evidence_ids' in config:
        raise ValueError('fixed collection must be labelled fixed_evidence')
    if config['source_backend'] != 'hf_generate' or not config['model_local_files_only']:
        raise ValueError('C25 uses cached local HF weights; no remote teacher/model calls')
    config['score_consistency_tolerance'] = vars(ScoreTolerance.from_config(config))
    caps = config.setdefault('max_action_tokens_by_benchmark', {})
    if not isinstance(caps, dict):
        raise ValueError('benchmark action caps must be a mapping')
    if not isinstance(config.get('student'), str) or not config['student'].strip():
        raise ValueError('student must be an explicit nonempty model reference')
    if config.get('student_call_format', 'qwen') not in {'qwen', 'gemma4'}:
        raise ValueError('student_call_format must be gemma4 or qwen')
    if benchmark == 'bfcl':
        caps.setdefault('bfcl', {})
    for cap_benchmark, limits in caps.items():
        allowed = {'agent_action'} if cap_benchmark in {'alfworld', 'webshop', 'hotpotqa'} else {'single_turn', 'multi_turn'}
        if not isinstance(cap_benchmark, str) or not isinstance(limits, dict) or set(limits)-allowed:
            raise ValueError('benchmark action caps require single_turn/multi_turn limits')
        if any(type(v) is not int or v < 1 for v in limits.values()):
            raise ValueError('positive integer action caps required')
    for kind, cap in [('single_turn', 512), ('multi_turn', 1024)]:
        if benchmark == 'bfcl':
            caps['bfcl'].setdefault(kind, cap)
    for key in ('max_action_tokens', 'max_context_tokens'):
        if key in config and (type(config[key]) is not int or config[key] < 1):
            raise ValueError('positive integer token limits required')
    for key, value in [('max_state_batch_size', 2), ('memory_peak_budget_gb', 38),
                       ('memory_reserve_gb', 2), ('memory_state_estimate_gb', 16)]:
        config.setdefault(key, value)
    MemoryPolicy.from_config(config)
    generation_batch = GenerationBatch.from_config(config)
    if generation_batch is not None:
        config['generation_batch'] = generation_batch.config()
    estimator, _, _ = validate_source_config(config)
    # Explicit legacy defaults serialize exactly as an omitted v1.1 option.
    if estimator == 'hard2':
        config.pop('source_estimator', None)
        config.pop('cv_cs_mode', None)
    return config


def harness_hash(root, config):
    return digest(evaluation_harness_identity(root, config))


def data_identity(root, config, bank):
    if config.get('method') == 'rtd_unified':
        from .unified.config import runtime_config
        return data_identity(root, runtime_config(config), bank)
    root, bank = Path(root), Path(bank)
    if config.get('benchmark', 'bfcl') != 'bfcl':
        from .benchmarks.config import data_identity as identity
        return identity(root, config, bank)
    data = root / 'envs/bfcl/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data'
    identity = dict(public=file_hash(bank/'public/requests.json'), integrity=file_hash(bank/'sealed/integrity.json'),
                    support=file_hash(root/config['support_manifest']), official=tree_hash(data))
    if config.get('protocol_version') == '1.1.0':
        from .bank_v11 import CERTIFICATE
        identity['cap_certificate'] = file_hash(bank/CERTIFICATE)
    return digest(identity)


def bank_audit(config, *, build=False):
    if config.get('method') == 'rtd_unified':
        from .unified.config import runtime_config
        return bank_audit(runtime_config(config), build=build)
    bank = ROOT / config['replay_bank_path']
    if config.get('protocol_version') == '1.1.0':
        from .bank_v11 import build_v11_bank, validate_v11_certificate, CERTIFICATE
        if build and (config['benchmark'] != 'bfcl' or config.get('student_call_format') == 'gemma4'):
            raise ValueError('build this bank with tools/rtd_bank_build.py --benchmark --pool --ledger --out')
        if build:
            build_v11_bank(ROOT/'data/rtd/v1_bfcl_c25', bank)
        from .bank_build import validate_state_certificate
        saved_certificate = json.loads((bank/CERTIFICATE).read_text())
        certificate = (validate_state_certificate(bank, benchmark=config['benchmark'], student=config['student'])
                       if saved_certificate['core']['version'] == 'rtd-v1.1.0-state-bank'
                       else validate_v11_certificate(bank))
        summary = json.loads((bank/'sealed/audit_v11.json').read_text())
        original = json.loads((bank/'sealed/audit.json').read_text())
        core = certificate['core']
        if (summary['budget_denominator'] != core['budget_denominator']
                or summary['recorded_bank_usage'] != core['budget_denominator']
                or sum(summary['available_cost_by_confidence'].values()) != core['budget_denominator']
                or summary['available_packages'] != core['available_packages']
                or summary['cap_certificate_sha256'] != file_hash(bank/CERTIFICATE)
                or summary['bank_public_cap_sum'] != sum(core['class_counts'][k]*v for k, v in core['class_caps'].items())):
            raise ValueError('v1.1 bank audit disagrees with certified content budget')
        result = dict(summary, bank_path=str(bank.resolve()), m=original['m'],
            budget_ceilings=resolve_budget_checkpoints(config, core['budget_denominator']),
            budget_denominator=core['budget_denominator'], cap_certificate_sha256=file_hash(bank/CERTIFICATE),
            public_cost_assumption=core['public_cost_assumption'], cost_scope=core['cost_scope'])
        print(json.dumps(result, indent=2), flush=True)
        return result
    if build:
        build_bfcl_bank(ROOT, bank)
    broker = SealedReplayBroker(bank, Ledger(0), inner_parent_hashes=set())
    usable = [r for r in broker._records.values() if r.unavailable_reason is None]
    for record in usable:
        provenance = json.loads(record.spec.cap_provenance)
        if record.spec.cost_upper_bound != PUBLIC_CLASS_CAPS[provenance['request_class']]:
            raise ValueError('bank uses stale/nonuniform C25 public caps; rebuild into a new directory')
    points = affordability(list(broker._records.values()))
    summary = json.loads((bank / 'sealed/audit.json').read_text())
    print('| checkpoint | cap budget | individually affordable demo/item | class capacity demo/item |', flush=True)
    for p in points:
        a, c = p['by_class'], p['capacity_by_class']
        print(f"| {p['percent']}% | {p['budget']} | {a['demo_attempt']}/{a['generator_item']} | "
              f"{c['demo_attempt']}/{c['generator_item']} |", flush=True)
    print('| checkpoint | decision step | remaining windows | b (zero prior spend) | class capacity demo/item |')
    for p in points:
        for w in p['windows']:
            c = w['capacity_by_class']
            print(f"| {p['percent']}% | {w['step']} | {w['remaining_windows']} | {w['b']} | "
                  f"{c['demo_attempt']}/{c['generator_item']} |", flush=True)
    print('Capacities are single-class illustrations, bounded by bank inventory. '
          'One new package at most per window. Live b uses actual remaining budget; '
          'b is an expected-cost constraint, not a per-package hard cap.', flush=True)
    return dict(bank_path=str(bank.resolve()), bank_public_cap_sum=sum(r.spec.cost_upper_bound for r in usable),
        budget_ceilings=[p['budget'] for p in points], affordability=points,
        recorded_bank_usage=summary['bank_cost'], available_cost_by_confidence=summary['available_cost_by_confidence'],
        available_packages=len(usable), m=summary['m'])


def make_manifest(config, arm, audit, *, smoke=False, hardware=None):
    if config.get('method') == 'rtd_unified':
        from .unified.config import runtime_config, manifest_fields
        if arm != config['arm']:
            raise ValueError('manifest arm differs from unified preset')
        manifest = make_manifest(runtime_config(config), arm, audit, smoke=smoke, hardware=hardware)
        manifest.update(config=config, config_hash=digest(config))
        manifest.update(manifest_fields(config, manifest))
        manifest.update(acquisition_protocol='fixed_recorded_pool_and_exposure',
            acquisition_label='disabled_in_P1', paired_validation='disabled_in_P1_all_arms',
            step_size_selection='preregistered_shared_fixed_eta',
            feedback_roles=['acquisition_reference_feedback', 'same_batch_reference_feedback', 'post_commit_feedback'])
        if arm != 'D1':
            for key in ('alpha_features', 'alpha_update', 'd_feedback_reference', 'comparison_labels'):
                manifest.pop(key, None)
        return manifest
    validate_arm(config, arm)
    from tools.bfcl_hub_merge_export import _snapshot_for_model
    model = Path(config['student'])
    if not model.is_dir():
        # Same local refs/main resolution as cc_three_arms' export. A usable
        # weight snapshot need not contain unrelated Hub README/license files.
        model = _snapshot_for_model(config['student'])
    hardware = hardware_identity() if hardware is None else hardware
    harness = evaluation_harness_identity(ROOT, config)
    manifest = dict(version='rtd-v1.0.1-run', arm=arm, smoke=smoke, config=config, config_hash=digest(config),
        model_path=str(model.resolve()), base_checkpoint_hash=tree_hash(model),
        tokenizer_hash=digest([(p.name, file_hash(p)) for p in sorted(model.glob('*'))
                               if p.is_file() and any(k in p.name for k in ('token', 'vocab', 'merges', 'chat_template'))]),
        harness_hash=digest(harness), evaluation_harness=harness, rtd_source=source_identity(ROOT),
        evaluation_harness_metadata=evaluation_harness_metadata(ROOT),
        data_hash=data_identity(ROOT, config, audit['bank_path']),
        hardware=hardware, hardware_hash=digest(hardware['hard']), **{k: audit[k] for k in
            ('bank_path', 'bank_public_cap_sum', 'budget_ceilings', 'recorded_bank_usage', 'available_packages', 'm')},
        backend='HF generate: local KV cache, unwarped categorical; same HF model teacher-forced CE; eager attention',
        backend_rationale='one resident model avoids vLLM reloads at reference/actual/source snapshots; throughput unmeasured',
        # Keep declarations byte-stable on resume; effective defaults are journaled separately.
        score_consistency=dict(tolerance=dict(config.get('score_consistency_tolerance', {})), units='nats/token including EOS',
            records='compute.jsonl: score_consistency, per state/action, including failed checks',
            generation='hf-generate-kv-categorical-v1', scoring='torch-functional-teacher-forced-native-ce-v1',
            reinforce_likelihood='same score_tokens native CE tensor as teacher forcing'),
        resources=dict(new_teacher_calls=0, new_teacher_tokens=0, support_feedback='rotating meta-training',
                       controller_pretrained=False, certification_access='evaluate only',
                       historical_demo_output_exact=1233607, historical_generator_output_estimated=142727),
        checkpoint_schedule='cumulative 10/25/50 percent after rounds 1/2/3; four windows per round')
    if config.get('protocol_version') == '1.1.0':
        budget_form = validate_budget_checkpoints(config)
        manifest.update(version='rtd-v1.1.0-run', trajectory_schema_version=2, ledger_schema_version=2,
            acquisition_protocol='batch_common_reference_v1', budget_basis=V11_BUDGET_BASIS,
            budget_checkpoint_form=budget_form,
            budget_ceilings=resolve_budget_checkpoints(config, audit['budget_denominator']),
            budget_denominator=audit['budget_denominator'], budget_rounding='positive_integer_half_up',
            cost_scope=audit['cost_scope'], cap_certificate_sha256=audit['cap_certificate_sha256'],
            public_cost_assumption=audit['public_cost_assumption'],
            exposure_slots_per_window=config['exposure_slots_per_window'],
            max_new_packages_per_window=config['max_new_packages_per_window'],
            replay_semantics='unfilled slots use old data; no fixed empty prior',
            checkpoint_schedule=('cumulative 10/25 percent after rounds 1/2; four windows per round'
                                 if config['rounds'] == 2 else manifest['checkpoint_schedule']))
        if budget_form == 'tokens':
            manifest.update(budget_rounding='none_absolute_tokens',
                checkpoint_schedule='cumulative recorded-output-token caps after each round; four windows per round')
    override = tolerance_override(config)
    if override is not None:
        manifest['score_consistency_tolerance_override'] = override
    if config['benchmark'] != 'bfcl' or config.get('student_call_format') == 'gemma4':
        manifest['resources'].pop('historical_demo_output_exact', None)
        manifest['resources'].pop('historical_generator_output_estimated', None)
        manifest['resources']['bank_audit'] = audit
    from .feedback_rng import feedback_rng_identity
    manifest.update(feedback_rng_identity(config))
    if config.get('generation_batch') is not None:
        from .generation_batch import RNG_RULE
        manifest['score_consistency']['generation'] = 'hf-generate-kv-batched-categorical-v1'
        manifest['generation_rng_rule'] = RNG_RULE
    if alpha_d_enabled(config):
        manifest.update(trajectory_schema_version=5, distillation_protocol='alpha_d_rev31_same_batch',
            return_objective='temperature_1_stochastic_policy_expected_return',
            exposure_unit='one_state_supervision_record_plus_two_source_actions',
            exposure_mode='full exposure' if config['slots_per_step'] == 40 else 'random exposure',
            planned_exposure_units=config['slots_per_step'], source_action_capacity_per_commit=2*config['slots_per_step'],
            microbatch_states=4, microbatches_per_commit=(config['slots_per_step']+3)//4,
            all_purchased_packages_trained_each_step=False,
            cold_start_missing_teacher='reference-pool state: alpha=0, soft-source term only; paid and reference records form old pool',
            supervision_record_mapping='adapter.supervision_records(package); default ordered package.behaviors',
            alpha_features='initial_model_state_hidden_projection_and_intercept; before_source_sampling',
            alpha_update='blocked_at_theta_S_d; d_fixed; no_differentiation_through_solver',
            acquisition_reference='old-evidence virtual d=0 update; acquisition labels only',
            d_feedback_reference='same-batch theta_S(0); same_batch_reference_feedback',
            feedback_roles=['acquisition_reference_feedback', 'same_batch_reference_feedback', 'post_commit_feedback'],
            acquisition_label=('acquisition_surrogate_full_update_with_teacher_injection'
                if config.get('acquisition_value_mode', 'joint') == 'joint' else 'independent_control'),
            acquisition_context='pre_purchase_round_step_purchased_set_alpha_distribution_feedback_age',
            acquisition_mean_shrinkage='95_percent_until_positive_prequential_explained_variance',
            paired_validation=('disabled_for_V0' if arm == 'V0' else
                'one_predetermined_purchased_package_when_joint_and_controller_pair; new_feedback_batches; never_fit'),
            v1_exposure_replay_implemented=True,
            replay_schedule_hash=config.get('replay_schedule_hash'),
            arm_components={k: config.get(k) for k in ('acquisition_value_mode', 'gate_mode', 'd_mode', 'ledger_replay')},
            base_checkpoint_description=BASE_CHECKPOINT,
            base_checkpoint_bank_adapted=False,
            comparison_labels={'V1−V0': ADAPTIVE_EFFECT, 'same_alpha_d_vs_zero': CORE_CONTROL},
            evaluation_label=DEVELOPMENT, certification_label=CERTIFICATION,
            certification_status='plumbing_only', report_x_axis='actual_spend',
            budget_checkpoints_are='authorization_caps; never force spend',
            source_sampling=dict(temperature=1., top_p=1., refresh='every_commit', cache_reuse=False,
                                 rng_stream='training_seed + arm + source_and_feedback'),
            support_roles=dict(bfcl_m=40, alfworld_parent_groups=135,
                alfworld_folds={'0': dict(inner=79, feedback=56, available_packages=63),
                                '1': dict(inner=56, feedback=79, available_packages=44)},
                alfworld_excluded_probe_groups=4),
            uncertainty_scope='reprojected trajectory contributions; excludes feedback staleness bias')
        from .streaming_replay import enabled as streaming_enabled
        if streaming_enabled(config):
            manifest.update(replay_mode='streaming', replay_consumed_steps=[], replay_schedule_hash=None)
        if 'crcd' in str(config['student']).lower():
            raise ValueError('v1.1 base must be the issuer checkpoint, without project bank adaptation')
        support_path = ROOT/config['support_manifest']
        if support_path.is_file():
            from .benchmarks.registry import get_benchmark
            from .metrics_v11 import options as metric_options, freeze_tasks
            # Folds belong to frozen support parents, not paid request packages.
            # Preserve the ALFWorld/WebShop metadata lost by the hash -> task view.
            parents, derived = resolve_parent_folds(json.loads(support_path.read_text())['parents'])
            manifest['parent_group_roles_by_fold'] = fold_roles(parents)
            if derived:
                manifest['parent_group_fold_derivation'] = dict(rule=PARENT_FOLD_RULE, parent_groups=derived)
            if metric_options(config)['enabled']:
                support = get_benchmark(config).support_protocol(ROOT, config)
                manifest['fixed_task_set_v11'] = freeze_tasks(parents,
                    short_fold=metric_options(config)['short_fold'], states=support.states)
    if config['benchmark'] != 'bfcl' or config.get('student_call_format') == 'gemma4':
        manifest['support_roles'] = dict(benchmark=config['benchmark'], training_parent_groups=audit['m'],
            inner_feedback='rotating parent-hash folds; calibration excluded')
    for key in ('data_hash', 'base_checkpoint_hash', 'hardware_hash'):
        if config.get('fixed_source_' + key, manifest[key]) != manifest[key]:
            raise ValueError('fixed ledger source differs: ' + key)
    return manifest


def startup_config(path, arm=None, replay_schedule=None, *, smoke=True,
                   smoke_deadline_seconds=900, **replay_options):
    """The same validated launch configuration for preflight, smoke and run."""
    if arm in {'V0', 'V1', 'V2'} or replay_schedule or replay_options:
        config = load_config(path, arm=arm, replay_schedule=replay_schedule, **replay_options)
    else:
        config = load_config(path)
    config, arm = arm_config(config, arm, replay_schedule, **replay_options)
    if config.get('method') == 'rtd_unified' and not smoke and not config.get('replay_schedule'):
        raise ValueError('P1 run requires --replay-schedule <V0 run>; '
                         'use --replay-mode streaming for a live V0; standalone smoke is exempt')
    if smoke:
        config = dict(config, smoke_override=dict(parents_per_fold=2, slots=2, rollouts=1, windows=1,
                     baseline='action-independent zero', max_seconds=smoke_deadline_seconds))
        if config.get('method') == 'rtd_unified':
            config['smoke_override'].update(slots=config['p1']['smoke_slots'],
                max_new_packages=config['p1']['smoke_packages'],
                feedback_tasks=config['p1']['smoke_feedback_tasks'], rollouts=config['p1']['smoke_rollouts'])
    return config, arm


def run_command(args):
    from .experiment import RTDExperiment, executor_config
    from .runtime import load_backend
    from .functional_step import lora_parameters
    from ..behavior.deltas import tensor_state_hash
    from tools.behavior_atom.checker_bridge import CheckerBridge
    started = time.monotonic()
    resume = args.command == 'resume'
    smoke = args.command == 'smoke'
    saved = None
    replay_options = {k: getattr(args, k) for k in ('replay_mode', 'replay_poll_seconds', 'replay_timeout_seconds')
                      if getattr(args, k, None) is not None}
    if resume:
        if args.run_dir is None:
            raise ValueError('resume requires --run-dir')
        saved = json.loads((Path(args.run_dir)/'manifest.json').read_text())
        smoke = saved['smoke']
        if args.arm is not None and args.arm != saved['arm']:
            raise ValueError('resume arm changed')
        args.arm = saved['arm']
    if resume:
        config = resume_config(args.config, saved)
    else:
        config, args.arm = startup_config(args.config, args.arm, getattr(args, 'replay_schedule', None),
            smoke=smoke, smoke_deadline_seconds=getattr(args, 'smoke_deadline_seconds', 900), **replay_options)
        from .preflight import preflight_config
        preflight_config(config, args.arm, smoke=smoke)
    if resume:
        if getattr(args, 'replay_schedule', None) or replay_options:
            candidate, _ = arm_config(config, args.arm, getattr(args, 'replay_schedule', None), **replay_options)
            if candidate != config:
                raise ValueError('resume replay schedule changed')
    if config.get('method') == 'rtd_unified':
        from .unified.experiment import P1Experiment
        RTDExperiment = P1Experiment
        if not smoke and not config.get('replay_schedule'):
            raise ValueError('P1 run requires --replay-schedule <V0 run>; '
                             'use --replay-mode streaming for a live V0; standalone smoke is exempt')
    if config['evaluate_after_round'] and not smoke and not args.training_worker:
        return run_campaign(args, config)
    smoke_deadline_seconds = getattr(args, 'smoke_deadline_seconds', 900)
    if smoke and resume:
        config = dict(config, smoke_override=dict(parents_per_fold=2, slots=2, rollouts=1, windows=1,
                     baseline='action-independent zero', max_seconds=smoke_deadline_seconds))
        if config.get('method') == 'rtd_unified':
            config['smoke_override'].update(slots=config['p1']['smoke_slots'],
                max_new_packages=config['p1']['smoke_packages'],
                feedback_tasks=config['p1']['smoke_feedback_tasks'], rollouts=config['p1']['smoke_rollouts'])
    audit = bank_audit(config)
    directory = Path(args.run_dir or ROOT / config['output_root'] / (args.arm + ('_smoke' if smoke else ''))).resolve()
    with exclusive_run(directory):
        if not resume and (directory/'manifest.json').exists():
            raise ValueError('existing run; use resume with the same configuration')
        manifest = make_manifest(config, args.arm, audit, smoke=smoke)
        current_hardware = manifest['hardware']
        expected_gpu = getattr(args, 'expected_gpu_uuid', None)
        if expected_gpu is not None and instance(current_hardware)['uuid'] != expected_gpu:
            raise ValueError('training worker GPU UUID differs from coordinator; refusing model load')
        if saved:
            validate_resume(ROOT, directory, saved, manifest,
                            acknowledge=getattr(args, 'acknowledge_code_drift', False))
            # Preserve the original binding used by StateStore and every round.
            manifest = saved
        deadline = started + smoke_deadline_seconds if smoke else None
        journal = ComputeJournal(directory/'compute.jsonl', cuda=True, deadline=deadline,
                                 deadline_seconds=smoke_deadline_seconds)
        journal.append('device_binding', cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            cuda_device_order=os.environ.get('CUDA_DEVICE_ORDER'), logical_device='cuda:0',
            gpu_uuid=instance(current_hardware)['uuid'], gpu_name=device_class(current_hardware)['gpu'],
            total_memory_bytes=device_class(current_hardware)['memory'], coordinator_gpu_uuid=expected_gpu)
        def timeout(signum, frame):
            raise TimeoutError(f'smoke exceeded {smoke_deadline_seconds} seconds; resume state retained')
        previous = signal.signal(signal.SIGALRM, timeout) if smoke else None
        if smoke:
            signal.setitimer(signal.ITIMER_REAL, max(1., deadline-time.monotonic()))
        try:
            backend = load_backend(config, manifest, journal)
            initial_hash = tensor_state_hash(lora_parameters(backend.model))
            if saved and saved['initial_parameter_hash'] != initial_hash:
                raise ValueError('initial LoRA construction changed on resume')
            if not saved:
                manifest['initial_parameter_hash'] = initial_hash
                atomic_json(directory/'manifest.json', manifest)
            from .experiment import prepare_support
            from .preflight import prepare_renderer
            execution_config = executor_config(config, args.arm)
            support = prepare_support(ROOT, execution_config, manifest)
            prepare_renderer(execution_config, support, backend.tokenizer, journal)
            atomic_json(directory/'support_access.json', dict(parents=support.parents,
                runnable_source_feedback_parents=sorted(support.states), unavailable=support.unavailable,
                labels='official truth accessed only by feedback scorer', calibration_training_access=False))
            from contextlib import nullcontext
            with (_checker_context(config) if config['benchmark'] == 'bfcl' else nullcontext()) as checker:
                experiment = RTDExperiment(config, manifest, directory, backend, support, resume=resume,
                    smoke=smoke, checker=checker, journal=journal)
                result = experiment.run(stop_after_round=args.through_round if args.training_worker else False)
            if smoke:
                elapsed = time.monotonic()-started
                if elapsed >= smoke_deadline_seconds:
                    raise AssertionError(f'smoke runtime exceeded {smoke_deadline_seconds} seconds')
                result['smoke_seconds'] = elapsed
                atomic_json(directory/'audit.json', result)
            print(json.dumps(result, indent=2))
        finally:
            if 'support' in locals() and hasattr(support, 'close'):
                support.close()
            if smoke:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, previous)
    return 0


def resume_config(path, saved):
    """Resume the exact saved config; today's defaults must not enter the run."""
    config = saved['config']
    if digest(config) != saved['config_hash']:
        raise ValueError('saved config hash mismatch')
    if path is not None:
        supplied = yaml.safe_load(Path(path).read_text())
        if supplied != config:
            replay_options = {k: config[k] for k in ('replay_mode', 'replay_poll_seconds', 'replay_timeout_seconds') if k in config}
            loaded = (load_config(path, arm=saved['arm'], replay_schedule=config.get('replay_schedule'), **replay_options)
                      if saved.get('arm') in {'V0', 'V1', 'V2'} or config.get('method') == 'rtd_unified' else load_config(path))
            if loaded != config:
                raise ValueError('resume config changed; omit --config to use the saved manifest')
    return config


def run_campaign(args, config):
    """Release the training process/GPU before each official endpoint campaign.

    Resume admits each completed round/checkpoint/evaluation once. Evaluation
    metrics never feed a training configuration, schedule or selector.
    """
    from .evaluation import evaluate, report
    if config['benchmark'] == 'hotpotqa':
        from ..hotpotqa import preflight_retrieval
        preflight_retrieval()
    # Snapshot the complete launch environment once. In particular, never
    # reinterpret the visible physical ordinal/UUID as a local cuda index.
    worker_env = os.environ.copy()
    coordinator_gpu = hardware_identity()
    print('[rtd] device binding ' + json.dumps(dict(
        cuda_visible_devices=worker_env.get('CUDA_VISIBLE_DEVICES'),
        cuda_device_order=worker_env.get('CUDA_DEVICE_ORDER'), logical_device='cuda:0',
        gpu_uuid=instance(coordinator_gpu)['uuid'], gpu_name=device_class(coordinator_gpu)['gpu'],
        total_memory_bytes=device_class(coordinator_gpu)['memory'])), flush=True)
    directory = Path(args.run_dir or ROOT/config['output_root']/args.arm).resolve()
    if args.command == 'run' and (directory/'manifest.json').exists():
        raise ValueError('existing campaign; use resume')
    with exclusive_run(directory/'coordinator'):
        if args.command == 'resume':
            with exclusive_run(directory):
                saved = json.loads((directory/'manifest.json').read_text())
                current = make_manifest(config, args.arm, bank_audit(config))
                validate_resume(ROOT, directory, saved, current,
                                acknowledge=getattr(args, 'acknowledge_code_drift', False),
                                training=not all((directory/f'round-{r}').exists() for r in range(1, config.get('rounds', 3)+1)))
                from .streaming_replay import enabled as streaming_enabled
                if streaming_enabled(saved):
                    # Completed campaigns skip the training worker entirely.
                    # They must still revalidate the consumed source schedule.
                    from .streaming_replay import validate_saved_replay
                    validate_saved_replay(directory, saved)
        for round_number in range(1, config.get('rounds', 3)+1):
            checkpoint = directory/f'round-{round_number}'
            if checkpoint.exists():
                with exclusive_run(directory):
                    saved = json.loads((directory/'manifest.json').read_text())
                    verified_checkpoint(directory, saved, round_number)
            elif not (directory/f'evaluation-{round_number}.json').exists():
                command = 'resume' if (directory/'manifest.json').exists() else 'run'
                worker = [sys.executable, str(ROOT/'tools/rtd_experiment.py'), command, '--arm', args.arm,
                    '--run-dir', str(directory), '--training-worker', '--through-round', str(round_number),
                    '--expected-gpu-uuid', instance(coordinator_gpu)['uuid']]
                if command == 'run':
                    worker += ['--config', str(Path(args.config).resolve())]
                    if config.get('replay_schedule'):
                        worker += ['--replay-schedule', config['replay_schedule']]
                        if config.get('replay_mode') == 'streaming':
                            worker += ['--replay-mode', 'streaming',
                                       '--replay-poll-seconds', str(config['replay_poll_seconds']),
                                       '--replay-timeout-seconds', str(config['replay_timeout_seconds'])]
                if command == 'resume' and getattr(args, 'acknowledge_code_drift', False):
                    worker += ['--acknowledge-code-drift']
                subprocess.run(worker, check=True, cwd=ROOT, env=worker_env)
            with exclusive_run(directory):
                evaluate(ROOT, directory, round_number, port=args.port,
                         lock_timeout=getattr(args, 'evaluation_lock_timeout', None),
                         lock_log_interval=getattr(args, 'evaluation_lock_log_interval', None))
        report([directory], directory/'report')
    return 0


def _checker_context(config=None):
    from contextlib import contextmanager
    from tools.behavior_atom.checker_bridge import CheckerBridge
    @contextmanager
    def context():
        checker = (CheckerBridge(checker_model=config['student']+'-FC',
                   student_call_format=config.get('student_call_format', 'qwen')) if config else CheckerBridge())
        try:
            yield checker
        finally:
            checker.close()
    return context()


def replay_ledger(args):
    directory = Path(args.run_dir)
    manifest = json.loads((directory/'manifest.json').read_text())
    ledger = Ledger.resume(manifest['budget_ceilings'][0], directory/'teacher.jsonl')
    print(json.dumps(dict(requests=len(ledger.owned_ids), actual_spend=ledger.spent,
        authorized_budget=ledger.budget, remaining=ledger.remaining, reservations=ledger.reservations), indent=2))
    if args.out:
        config = load_config(args.config)
        if ledger.reservations:
            raise ValueError('finish or resume pending transactions before freezing the ledger')
        if config['replay_bank_path'] != manifest['config']['replay_bank_path']:
            raise ValueError('fixed ledger bank differs')
        config.update(mode='fixed_evidence', fixed_evidence_ids=sorted(ledger.owned_ids),
            fixed_ledger_source=str(directory.resolve()), fixed_ledger_hash=digest(ledger.events),
            fixed_initial_parameter_hash=manifest['initial_parameter_hash'])
        config.update({'fixed_source_'+k: manifest[k] for k in ('data_hash','base_checkpoint_hash','hardware_hash')})
        config['fixed_source_hardware_hash'] = comparison_hash(ROOT, manifest)
        # All four cells start from theta0 and retrain the entire final collection.
        # These configs do not treat adaptive R0/R1 as the diagonal cells.
        config['output_root'] = str(Path(config['output_root']) / args.out.stem)
        Path(args.out).write_text(yaml.safe_dump(config, sort_keys=False))
    return 0


def positive_seconds(value):
    seconds = int(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError('deadline must be a positive number of seconds')
    return seconds


def main(argv=None):
    parser = argparse.ArgumentParser(description='RTD protocol v1.0.7 / v1.0.1 sealed BFCL replay. No teacher API path.')
    subs = parser.add_subparsers(dest='command', required=True)
    for name in ('audit', 'audit-legacy', 'update-identity', 'update-hardware-identity', 'smoke', 'run', 'resume', 'replay-ledger', 'swap-component', 'evaluate', 'report'):
        p = subs.add_parser(name)
        if name not in ('report', 'audit-legacy', 'update-identity', 'update-hardware-identity'):
            p.add_argument('--config', default=None if name == 'resume' else
                           os.environ.get('RTD_CONFIG', str(ROOT/'configs/rtd/v1_bfcl_c25.yaml')))
        if name == 'resume':
            p.add_argument('--acknowledge-code-drift', action='store_true',
                           help='acknowledge recorded RTD source changes before continuing training')
        if name in ('smoke', 'run', 'resume'):
            from .unified.arms import ARMS as P1_ARMS
            p.add_argument('--arm', choices=['R0', 'R1', 'V0', 'V1', 'V2', *P1_ARMS], default=os.environ.get('RTD_ARM'))
            p.add_argument('--smoke-deadline-seconds', type=positive_seconds, default=900,
                           help='smoke time limit in seconds (default: 900; resume must match the saved config)')
            p.add_argument('--replay-schedule', type=Path, help='V1/unified arms: V0 run directory or exposure_schedule.json')
            p.add_argument('--replay-mode', choices=['complete', 'streaming'],
                           help='V1/unified schedule mode (default: complete; resume uses saved mode)')
            p.add_argument('--replay-poll-seconds', type=float,
                           help='streaming replay poll interval (default: 60 seconds)')
            p.add_argument('--replay-timeout-seconds', type=float,
                           help='streaming V1 overall timeout per execution attempt (default: 129600 seconds / 36 hours)')
            p.add_argument('--training-worker', action='store_true', help=argparse.SUPPRESS)
            p.add_argument('--expected-gpu-uuid', help=argparse.SUPPRESS)
            p.add_argument('--through-round', type=int, choices=[1,2,3], default=3, help=argparse.SUPPRESS)
        if name in ('smoke', 'run', 'resume', 'evaluate'):
            p.add_argument('--port', type=int, help='campaign port (default: job/GPU-derived with free-port fallback)')
            p.add_argument('--evaluation-lock-timeout', type=float,
                           help='maximum lock wait in seconds (env RTD_EVALUATION_LOCK_TIMEOUT_SECONDS; default 21600)')
            p.add_argument('--evaluation-lock-log-interval', type=float,
                           help='lock wait log interval in seconds (env RTD_EVALUATION_LOCK_LOG_INTERVAL_SECONDS; default 60)')
        if name in ('smoke', 'run', 'resume', 'replay-ledger', 'evaluate', 'audit-legacy', 'update-identity', 'update-hardware-identity'):
            p.add_argument('--run-dir', type=Path, required=name in ('replay-ledger', 'evaluate', 'audit-legacy', 'update-identity', 'update-hardware-identity'))
        if name == 'update-hardware-identity':
            p.add_argument('--host-class', help='assert the saved host class; cannot override it')
            p.add_argument('--driver-version', help='establish the previously unrecorded target NVIDIA driver version')
            p.add_argument('--reference-manifest', type=Path,
                           help='also require equality with this already migrated manifest class')
        if name == 'audit':
            p.add_argument('--build-bank', action='store_true')
        if name in ('replay-ledger', 'swap-component'):
            p.add_argument('--out', type=Path, required=name == 'swap-component')
        if name == 'swap-component':
            p.add_argument('--component', choices=['gate', 'preconditioner', 'acquisition'], required=True)
            p.add_argument('--value', required=True)
        if name == 'evaluate':
            p.add_argument('--round', type=int, choices=[1, 2, 3], required=True)
            p.add_argument('--base-evaluation', type=Path)
        if name == 'report':
            p.add_argument('--run-dir', type=Path, action='append', required=True)
            p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in ('run', 'smoke', 'resume'):
        return run_command(args)
    if args.command == 'audit':
        print(json.dumps(bank_audit(load_config(args.config), build=args.build_bank), indent=2))
    elif args.command == 'audit-legacy':
        print(json.dumps(audit_legacy(ROOT, args.run_dir), indent=2, sort_keys=True))
    elif args.command == 'update-identity':
        from .identity_update import IdentityUpdateRefused, update_identity
        try:
            result = update_identity(ROOT, args.run_dir)
        except IdentityUpdateRefused as exc:
            print(json.dumps(exc.evidence, indent=2, sort_keys=True))
            return 1
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == 'update-hardware-identity':
        from .hardware import bound_hardware, update_hardware_identity
        try:
            reference = (bound_hardware(ROOT, json.loads(args.reference_manifest.read_text()))
                         if args.reference_manifest else None)
            result = update_hardware_identity(ROOT, args.run_dir,
                expected_host_class=args.host_class, driver=args.driver_version, reference=reference)
        except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
            print(json.dumps(dict(updated=False, refusals=[str(exc)]), indent=2, sort_keys=True))
            return 1
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == 'replay-ledger':
        return replay_ledger(args)
    elif args.command == 'swap-component':
        config = load_config(args.config)
        choices = {'gate': {'scalar_sigmoid', 'linear_sigmoid'}, 'preconditioner': {'identity', 'train_only_rms_diagonal'},
                   'acquisition': {'random', 'bayesian_linear_posterior_sampling'}}
        if args.value not in choices[args.component] or config.get('component_swap'):
            raise ValueError('one supported component swap per parent configuration')
        config.update(component_swap=dict(parent_config_hash=digest(config), component=args.component,
                                          before=config[args.component], after=args.value))
        config[args.component] = args.value
        config['output_root'] = str(Path(config['output_root']) / args.out.stem)
        args.out.write_text(yaml.safe_dump(config, sort_keys=False))
    elif args.command == 'evaluate':
        from .evaluation import evaluate
        with exclusive_run(args.run_dir):
            result = evaluate(ROOT, args.run_dir, args.round, port=args.port, base_evaluation=args.base_evaluation,
                              lock_timeout=args.evaluation_lock_timeout,
                              lock_log_interval=args.evaluation_lock_log_interval)
        print(json.dumps({'overall_accuracy_percent': result['overall_accuracy_percent'], 'complete': True}))
    else:
        from .evaluation import report
        report(args.run_dir, args.out)
    return 0
