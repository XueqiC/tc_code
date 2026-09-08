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
from .caps import V11_BUDGET_BASIS, recorded_budget_ceilings
from .config_v11 import v11_config, V11_DEFAULTS
from .alpha_d import ALPHA_D_DEFAULTS, enabled as alpha_d_enabled, validate_config as validate_alpha_d_config, validate_arm
from .ledger import Ledger
from .persistence import ComputeJournal, atomic_json, digest, exclusive_run, file_hash, tree_hash
from .scoring import ScoreTolerance
from .memory import MemoryPolicy
from .source_estimator import validate_source_config
from .identity import (audit_legacy, evaluation_harness_identity, evaluation_harness_metadata,
                       source_identity, validate_resume, verified_checkpoint)
from .hardware import hardware_identity, instance, device_class, comparison_hash

ROOT = Path(__file__).resolve().parents[3]


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError('configuration must be a mapping')
    v11 = config.get('protocol_version') == '1.1.0'
    if (config.get('protocol_version') not in {'1.0.1', '1.1.0'} or config.get('mode') not in {'sealed_replay', 'fixed_evidence'}
            or config.get('budget_basis') != (V11_BUDGET_BASIS if v11 else 'usable_public_cap_sum')):
        raise ValueError('use the v1.0.1 public-cap configuration')
    canonical = yaml.safe_load((ROOT / 'configs/rtd/v1_bfcl_c25.yaml').read_text())
    if v11:
        config, canonical = v11_config(config, canonical)
    elif (set(V11_DEFAULTS)-set(canonical)) & config.keys():
        raise ValueError('v1.1 batch settings require protocol_version 1.1.0')
    mutable = {'student', 'output_root', 'replay_bank_path', 'support_manifest', 'max_action_tokens',
               'max_context_tokens', 'pilot_eta_candidates', 'initial_eta', 'model_local_files_only',
               'evaluate_after_round', 'mode', 'gate', 'acquisition', 'preconditioner',
               'score_consistency_tolerance', 'max_action_tokens_by_benchmark', 'max_state_batch_size',
               'memory_peak_budget_gb', 'memory_reserve_gb', 'memory_state_estimate_gb',
               'source_samples_per_state'}
    if v11:
        mutable |= {'rounds', 'budget_checkpoints_bank_fraction', 'exposure_slots_per_window',
                    'max_new_packages_per_window', 'slots_per_step', 'drift_reference_packages', 'value_noise_floor'}
        mutable |= set(ALPHA_D_DEFAULTS)
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
    caps.setdefault('bfcl', {})
    for benchmark, limits in caps.items():
        if not isinstance(benchmark, str) or not isinstance(limits, dict) or set(limits)-{'single_turn', 'multi_turn'}:
            raise ValueError('benchmark action caps require single_turn/multi_turn limits')
        if any(type(v) is not int or v < 1 for v in limits.values()):
            raise ValueError('positive integer action caps required')
    for kind, cap in [('single_turn', 512), ('multi_turn', 1024)]:
        caps['bfcl'].setdefault(kind, cap)
    for key in ('max_action_tokens', 'max_context_tokens'):
        if key in config and (type(config[key]) is not int or config[key] < 1):
            raise ValueError('positive integer token limits required')
    for key, value in [('max_state_batch_size', 2), ('memory_peak_budget_gb', 38),
                       ('memory_reserve_gb', 2), ('memory_state_estimate_gb', 16)]:
        config.setdefault(key, value)
    MemoryPolicy.from_config(config)
    estimator, _, _ = validate_source_config(config)
    # Explicit legacy defaults serialize exactly as an omitted v1.1 option.
    if estimator == 'hard2':
        config.pop('source_estimator', None)
        config.pop('cv_cs_mode', None)
    return config


def harness_hash(root, config):
    return digest(evaluation_harness_identity(root, config))


def data_identity(root, config, bank):
    root, bank = Path(root), Path(bank)
    data = root / 'envs/bfcl/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data'
    identity = dict(public=file_hash(bank/'public/requests.json'), integrity=file_hash(bank/'sealed/integrity.json'),
                    support=file_hash(root/config['support_manifest']), official=tree_hash(data))
    if config.get('protocol_version') == '1.1.0':
        from .bank_v11 import CERTIFICATE
        identity['cap_certificate'] = file_hash(bank/CERTIFICATE)
    return digest(identity)


def bank_audit(config, *, build=False):
    bank = ROOT / config['replay_bank_path']
    if config.get('protocol_version') == '1.1.0':
        from .bank_v11 import build_v11_bank, validate_v11_certificate, CERTIFICATE
        if build:
            build_v11_bank(ROOT/'data/rtd/v1_bfcl_c25', bank)
        certificate = validate_v11_certificate(bank)
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
            budget_ceilings=recorded_budget_ceilings(core['budget_denominator'], rounds=config['rounds']),
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


def make_manifest(config, arm, audit, *, smoke=False):
    validate_arm(config, arm)
    from tools.bfcl_hub_merge_export import _snapshot_for_model
    model = Path(config['student'])
    if not model.is_dir():
        # Same local refs/main resolution as cc_three_arms' export. A usable
        # weight snapshot need not contain unrelated Hub README/license files.
        model = _snapshot_for_model(config['student'])
    hardware = hardware_identity()
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
        manifest.update(version='rtd-v1.1.0-run', trajectory_schema_version=2, ledger_schema_version=2,
            acquisition_protocol='batch_common_reference_v1', budget_basis=V11_BUDGET_BASIS,
            budget_denominator=audit['budget_denominator'], budget_rounding='positive_integer_half_up',
            cost_scope=audit['cost_scope'], cap_certificate_sha256=audit['cap_certificate_sha256'],
            public_cost_assumption=audit['public_cost_assumption'],
            exposure_slots_per_window=config['exposure_slots_per_window'],
            max_new_packages_per_window=config['max_new_packages_per_window'],
            replay_semantics='unfilled slots use old data; no fixed empty prior',
            checkpoint_schedule=('cumulative 10/25 percent after rounds 1/2; four windows per round'
                                 if config['rounds'] == 2 else manifest['checkpoint_schedule']))
    if alpha_d_enabled(config):
        manifest.update(trajectory_schema_version=3, distillation_protocol='alpha_d_rev3_1',
            return_objective='temperature_1_stochastic_policy_expected_return',
            exposure_unit='one_state_teacher_record_plus_two_source_actions',
            exposure_mode='full exposure' if config['slots_per_step'] == 40 else 'random exposure',
            planned_exposure_units=config['slots_per_step'], source_action_capacity_per_commit=2*config['slots_per_step'],
            microbatch_states=4, microbatches_per_commit=(config['slots_per_step']+3)//4,
            all_purchased_packages_trained_each_step=False,
            cold_start_missing_teacher='full exposure requires paid inner old pool; random exposure uses available teacher records',
            supervision_record_mapping='adapter.supervision_records(package); default ordered package.behaviors',
            alpha_features='initial_model_state_hidden_projection_and_intercept; before_source_sampling',
            alpha_update='blocked_at_theta_S_d; d_fixed; no_differentiation_through_solver',
            acquisition_reference='separate historical additive insertion surrogate; D9 full-update proxy pending',
            v1_exposure_replay_implemented=False,
            uncertainty_scope='reprojected trajectory contributions; excludes feedback staleness bias')
    for key in ('data_hash', 'base_checkpoint_hash', 'hardware_hash'):
        if config.get('fixed_source_' + key, manifest[key]) != manifest[key]:
            raise ValueError('fixed ledger source differs: ' + key)
    return manifest


def run_command(args):
    from .experiment import BFCLSupport, RTDExperiment
    from .runtime import load_backend
    from .functional_step import lora_parameters
    from ..behavior.deltas import tensor_state_hash
    from tools.behavior_atom.checker_bridge import CheckerBridge
    started = time.monotonic()
    resume = args.command == 'resume'
    smoke = args.command == 'smoke'
    saved = None
    if resume:
        if args.run_dir is None:
            raise ValueError('resume requires --run-dir')
        saved = json.loads((Path(args.run_dir)/'manifest.json').read_text())
        smoke = saved['smoke']
        if args.arm != saved['arm']:
            raise ValueError('resume arm changed')
    config = resume_config(args.config, saved) if resume else load_config(args.config)
    if config['evaluate_after_round'] and not smoke and not args.training_worker:
        return run_campaign(args, config)
    if smoke:
        config = dict(config, smoke_override=dict(parents_per_fold=2, slots=2, rollouts=1, windows=1,
                     baseline='action-independent zero', max_seconds=900))
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
        deadline = started + 900 if smoke else None
        journal = ComputeJournal(directory/'compute.jsonl', cuda=True, deadline=deadline)
        journal.append('device_binding', cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            cuda_device_order=os.environ.get('CUDA_DEVICE_ORDER'), logical_device='cuda:0',
            gpu_uuid=instance(current_hardware)['uuid'], gpu_name=device_class(current_hardware)['gpu'],
            total_memory_bytes=device_class(current_hardware)['memory'], coordinator_gpu_uuid=expected_gpu)
        def timeout(signum, frame):
            raise TimeoutError('smoke exceeded 15 minutes; resume state retained')
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
            support = BFCLSupport(ROOT, config)
            atomic_json(directory/'support_access.json', dict(parents=support.parents,
                runnable_source_feedback_parents=sorted(support.states), unavailable=support.unavailable,
                labels='official truth accessed only by feedback scorer', calibration_training_access=False))
            with _checker_context() as checker:
                experiment = RTDExperiment(config, manifest, directory, backend, support, resume=resume,
                    smoke=smoke, checker=checker, journal=journal)
                result = experiment.run(stop_after_round=args.through_round if args.training_worker else False)
            if smoke:
                elapsed = time.monotonic()-started
                if elapsed >= 900:
                    raise AssertionError('smoke runtime exceeded 15 minutes')
                result['smoke_seconds'] = elapsed
                atomic_json(directory/'audit.json', result)
            print(json.dumps(result, indent=2))
        finally:
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
        if supplied != config and load_config(path) != config:
            raise ValueError('resume config changed; omit --config to use the saved manifest')
    return config


def run_campaign(args, config):
    """Release the training process/GPU before each official endpoint campaign.

    Resume admits each completed round/checkpoint/evaluation once. Evaluation
    metrics never feed a training configuration, schedule or selector.
    """
    from .evaluation import evaluate, report
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
                if command == 'resume' and getattr(args, 'acknowledge_code_drift', False):
                    worker += ['--acknowledge-code-drift']
                subprocess.run(worker, check=True, cwd=ROOT, env=worker_env)
            with exclusive_run(directory):
                evaluate(ROOT, directory, round_number, port=args.port,
                         lock_timeout=getattr(args, 'evaluation_lock_timeout', None),
                         lock_log_interval=getattr(args, 'evaluation_lock_log_interval', None))
        report([directory], directory/'report')
    return 0


def _checker_context():
    from contextlib import contextmanager
    from tools.behavior_atom.checker_bridge import CheckerBridge
    @contextmanager
    def context():
        checker = CheckerBridge()
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
            p.add_argument('--arm', choices=['R0', 'R1', 'V0', 'V1', 'V2'], default=os.environ.get('RTD_ARM', 'R1'))
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
