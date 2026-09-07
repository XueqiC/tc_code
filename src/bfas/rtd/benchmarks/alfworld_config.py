"""C26-E frozen configuration and CPU-only manifest sections, never a trainer."""
from copy import deepcopy
import json
import math
from pathlib import Path

import yaml

from ..memory import MemoryPolicy
from ..persistence import digest, file_hash
from ..scoring import ScoreTolerance
from . import alfworld_identity as identity
from .alfworld_support import validate_support, audit_verified_bank

# Executable copy of readiness §7.2; BFCL files/manifests are never rewritten.
DEFAULTS = {'method': 'rtd_v1',
 'mode': 'sealed_replay',
 'training_seed': 0,
 'same_hardware_required': True,
 'harness_frozen': True,
 'teacher_access': 'text_only',
 'support_split': 'parent_hash_two_fold',
 'certificate_isolation': 'strict',
 'rounds': 3,
 'decision_steps_per_round': [1, 4, 7, 10],
 'max_new_packages_per_decision': 1,
 'replay_is_explicit_choice': True,
 'replay_prior_mass': 0.5,
 'budget_checkpoints_bank_fraction': [0.1, 0.25, 0.5],
 'online_max_teacher_turns': 8,
 'replay_request_lengths': 'recorded_only',
 'source_refresh': 'round',
 'source_samples_per_state': 2,
 'source_temperature': 1.0,
 'source_top_p': 1.0,
 'loss': 'positive_source_teacher_mixture',
 'behavior_probability': 'complete_action_with_termination',
 'exposure_unit': 'complete_source_action_slot',
 'slots_per_step': 8,
 'committed_steps_per_round': 12,
 'gate': 'linear_sigmoid',
 'gate_hidden_projection_dim': 32,
 'gate_initial_logit': 0.0,
 'gate_ridge': 1.0,
 'gate_learning_rate': 0.1,
 'gate_gradient_scale': 'running_rms_from_past_feedback_only',
 'optimizer': 'fixed_preconditioned_single_step',
 'preconditioner': 'train_only_rms_diagonal',
 'preconditioner_refresh': 'round',
 'preconditioner_damping_relative': 0.01,
 'preconditioner_mean_diagonal': 1.0,
 'step_size_selection': 'frozen_training_only_kl_pilot',
 'pilot_kl_target_per_step': 0.005,
 'meta_feedback_steps': [1, 4, 7, 10],
 'insertion_reference': 'virtual_same_start',
 'extra_reference_compute_accounting': 'separate',
 'meta_tasks_per_feedback': 4,
 'rollouts_per_meta_task': 2,
 'reward': 'official_terminal_metric',
 'baseline': 'leave_one_out_same_task',
 'acquisition': 'bayesian_linear_posterior_sampling',
 'acquisition_prior_precision': 1.0,
 'acquisition_prior_noise_variance': 1.0,
 'acquisition_entropy_temperature': 1.0,
 'acquisition_value': 'exposure_conserving_insertion',
 'insertion_fraction': 0.25,
 'acquisition_posterior_refresh': 'round',
 'hard_cost_reservation': True,
 'benchmark': 'alfworld',
 'gate_projection_seed': 0,
 'parameter_space': 'lora_trainables',
 'new_teacher_calls': False,
 'new_teacher_tokens': 0,
 'replay_bank_path': 'data/rtd/v1_alfworld_c26',
 'replay_public_cap_output_tokens_by_class': {'alf_demo_episode': 1310720},
 'replay_cap_scope': 'class_uniform_public_fallback',
 'support_manifest': 'configs/rtd/v1_alfworld_support_c26.json',
 'source_policy_generation': 'same_backend_torch_categorical',
 'feedback_scope': 'rotating_meta_training',
 'experiment_scope': 'exploratory',
 'support_parent_tasks_m': 135,
 'protocol_version': '1.0.1',
 'budget_basis': 'usable_public_cap_sum',
 'student': 'Qwen/Qwen3.5-4B',
 'model_local_files_only': True,
 'output_root': 'results/rtd_v1',
 'max_action_tokens_by_benchmark': {'alfworld': {'agent_action': 256}},
 'max_context_tokens': 32768,
 'max_state_batch_size': 1,
 'memory_peak_budget_gb': 38,
 'memory_reserve_gb': 2,
 'memory_state_estimate_gb': 16,
 'pilot_eta_candidates': [1e-06, 3e-06, 1e-05, 3e-05, 0.0001, 0.0003, 0.001],
 'initial_eta': 1e-05,
 'source_backend': 'hf_generate',
 'score_consistency_tolerance': {'mean_abs': 0.05,
                                 'max_abs': 1.0,
                                 'max_abs_outlier_tokens': 2,
                                 'max_abs_hard': 8.0},
 'lora_rank': 16,
 'lora_alpha': 32,
 'lora_target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
 'evaluate_after_round': True,
 'evaluation_temperature': 0.0,
 'benchmark_schema_version': 'rtd-alfworld-c26-e-v1',
 'alfworld_train_split': 'train',
 'alfworld_evaluation_split': 'valid_seen',
 'alfworld_expected_eval_tasks': 140,
 'alfworld_max_episode_steps': 40,
 'alfworld_student_react': True,
 'alfworld_data_root': 'envs/alfworld/data/json_2.1.1',
 'alfworld_environment_root': 'envs/alfworld',
 'historical_output_tokens_estimated': 189541,
 'historical_success_output_tokens_estimated': 36294,
 'historical_failed_output_tokens_estimated': 153247,
 'historical_missing_response_attempts': 112,
 'historical_missing_attempts': 1}

# These resource locations/limits may change, with a new config identity.
MUTABLE = {"student", "output_root", "replay_bank_path", "support_manifest",
           "max_context_tokens",
           "max_state_batch_size", "memory_peak_budget_gb", "memory_reserve_gb",
           "memory_state_estimate_gb", "gate"}


def default_config():
    return deepcopy(DEFAULTS)


def validate_config(value):
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError("configuration must be a string-keyed mapping")
    if value.get("benchmark") != "alfworld":
        raise ValueError("explicit benchmark: alfworld required")
    unknown = set(value) - set(DEFAULTS) - {'smoke_override'}
    if unknown:
        raise ValueError("unknown ALFWorld configuration keys: " + ", ".join(sorted(unknown)))
    config = default_config()
    config.update(deepcopy(value))
    if 'smoke_override' in value:
        smoke = value['smoke_override']
        expected = dict(parents_per_fold=4, slots=8, rollouts=2, windows=1,
                        baseline='leave_one_out_same_task', max_seconds=900)
        if not isinstance(smoke, dict) or set(smoke) != set(expected):
            raise ValueError('invalid ALFWorld partial smoke override')
        if type(smoke['parents_per_fold']) is not int or smoke['parents_per_fold'] not in (2, 4):
            raise ValueError('ALFWorld partial smoke requires two or four parents per fold')
        expected['parents_per_fold'] = smoke['parents_per_fold']
        if any(type(smoke[k]) is not type(v) or smoke[k] != v for k, v in expected.items()):
            raise ValueError('ALFWorld partial smoke must retain eight slots and K=2 LOO')
    def same(actual, expected):
        if type(actual) is not type(expected):
            return False
        if isinstance(expected, dict):
            return actual.keys() == expected.keys() and all(same(actual[k], v) for k, v in expected.items())
        if isinstance(expected, list):
            return len(actual) == len(expected) and all(same(a, b) for a, b in zip(actual, expected))
        return actual == expected
    for key, expected in DEFAULTS.items():
        if key not in MUTABLE and not same(config[key], expected):
            raise ValueError("frozen protocol value changed: " + key)
    if config["gate"] not in {"linear_sigmoid", "scalar_sigmoid"}:
        raise ValueError("supported gate components: linear_sigmoid/scalar_sigmoid")
    for key in ("student", "output_root", "replay_bank_path", "support_manifest",
                "alfworld_data_root", "alfworld_environment_root"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError("nonempty path/model string required: " + key)
    for key in ("max_state_batch_size", "max_context_tokens"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError("positive integer limit required: " + key)
    for key in ("memory_peak_budget_gb", "memory_reserve_gb", "memory_state_estimate_gb"):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]):
            raise ValueError("finite memory value required: " + key)
    limits = config["max_action_tokens_by_benchmark"]
    if type(limits["alfworld"]["agent_action"]) is not int:
        raise ValueError("positive integer action cap required")
    identity.checked_config(config)
    config["score_consistency_tolerance"] = vars(ScoreTolerance.from_config(config))
    MemoryPolicy.from_config(config)
    if config["memory_reserve_gb"] >= config["memory_peak_budget_gb"]:
        raise ValueError("memory reserve must be smaller than peak budget")
    # Guarantee manifest/round-trip safety before touching any bank.
    json.dumps(config, allow_nan=False)
    return config


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    pairs = loader.construct_pairs(node, deep=True)
    result = {}
    for key, value in pairs:
        if not isinstance(key, str) or key in result:
            raise ValueError("duplicate or non-string YAML key")
        result[key] = value
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def load_config(path):
    return validate_config(yaml.load(Path(path).read_text(), Loader=_UniqueLoader))


def model_directory(root, config):
    model = Path(root) / config["student"]
    if not model.is_dir():
        # Existing local refs/main resolver; no Hub request and no model load.
        from tools.bfcl_hub_merge_export import _snapshot_for_model
        model = Path(_snapshot_for_model(config["student"]))
    return model.resolve()


def bank_audit(root, config):
    config = validate_config(config)
    bank = Path(root) / config["replay_bank_path"]
    verified = audit_verified_bank(bank)
    support = validate_support(json.loads((Path(root) / config["support_manifest"]).read_text()))
    if support != json.loads((bank / "public/support.json").read_text()):
        raise ValueError("external support manifest differs from sealed bank")
    if support["m"] != config["support_parent_tasks_m"]:
        raise ValueError("frozen support parent count differs")
    summary = json.loads((bank / "sealed/audit.json").read_text())
    caps = summary["cap_audit"]
    if any(caps["public_caps"].get(k) != v for k, v in
           config["replay_public_cap_output_tokens_by_class"].items()):
        raise ValueError("sealed bank public cap policy differs")
    historical = summary["historical_inventory"]
    gaps = historical["historical_attempt_gaps"]
    declarations = dict(historical_output_tokens_estimated=summary["legacy_token_estimate"],
        historical_success_output_tokens_estimated=historical["successful_legacy_token_estimate"],
        historical_failed_output_tokens_estimated=historical["failed_legacy_token_estimate"],
        historical_missing_response_attempts=historical["failed_attempts"],
        historical_missing_attempts=sum(len(g["missing_attempt_indices"]) for g in gaps))
    if any(config[k] != v for k, v in declarations.items()):
        raise ValueError("historical estimate/missing-attempt declarations differ from sealed audit")
    return dict(bank_path=str(bank.resolve()), bank_public_cap_sum=caps["bank_public_cap_sum"],
        budget_ceilings=[p["budget"] for p in caps["budget_affordability"]],
        affordability=caps["budget_affordability"], recorded_bank_usage=dict(
            output_tokens_estimated=summary["legacy_token_estimate"],
            usable_output_tokens_estimated=summary["usable_legacy_token_estimate"],
            cost_confidence="estimated", missing_responses=historical["failed_attempts"],
            historical_attempt_gaps=gaps, missing_cost_query_ids=caps["missing_cost_query_ids"],
            provider_usage_exact=False, input_reasoning_discarded_retry_costs="unknown"),
        available_cost_by_confidence={"estimated": summary["usable_legacy_token_estimate"]},
        available_packages=summary["usable_packages"], m=summary["m"], verification=verified,
        support_manifest_hash=support["manifest_hash"],
        unavailable_packages=summary["unavailable_packages"], limitations=summary["limitations"])


def data_identity(root, config, bank):
    bank = Path(bank)
    expected = identity.official_expectations(Path(root) / config["alfworld_data_root"])
    return digest(dict(public=file_hash(bank / "public/requests.json"),
        integrity=file_hash(bank / "sealed/integrity.json"),
        sealed_manifest=file_hash(bank / "sealed/manifest.json"),
        support=file_hash(Path(root) / config["support_manifest"]),
        official_task_list=expected["task_ids_hash"], official=expected["data_manifest_hash"]))


def manifest_section(root, config):
    """Deterministic make_manifest-compatible fields; deliberately no GPU/run identity.

    Revalidate sealed bytes on every call. No clock, hardware query, checkpoint
    creation, renderer/model construction or environment process is involved.
    C26-F must combine these fields with the existing run/source/hardware guards.
    """
    config = validate_config(config)
    audit = bank_audit(root, config)
    model = model_directory(root, config)
    harness = identity.evaluation_harness_identity(root, config,
        data_root=Path(root) / config["alfworld_data_root"], model_path=model,
        tokenizer_path=model, environment_root=Path(root) / config["alfworld_environment_root"])
    support = validate_support(json.loads((Path(root) / config["support_manifest"]).read_text()))
    # Bind current train worlds and the C26-B environment/tokenizer as well as eval.
    for tid in support["training_task_ids"]:
        request = support["tasks"][tid]["request"]
        for name, sha in request["world_files"].items():
            if file_hash(Path(root) / config["alfworld_data_root"] / "train" / tid / name) != sha:
                raise ValueError("training world differs from frozen support")
    from .alfworld_support import environment_identity
    if environment_identity(root, model) != support["environment"]:
        raise ValueError("current environment/tokenizer differs from frozen support")
    return dict(config=config, config_hash=digest(config), model_path=str(model),
        base_checkpoint_hash=harness["model"]["base_checkpoint_hash"],
        tokenizer_hash=harness["tokenizer"]["hash"], evaluation_harness=harness,
        harness_hash=digest(harness), data_hash=data_identity(root, config, audit["bank_path"]),
        **{k: audit[k] for k in ("bank_path", "bank_public_cap_sum", "budget_ceilings",
            "recorded_bank_usage", "available_packages", "m")}, bank_audit=audit,
        benchmark_schema_version=config["benchmark_schema_version"],
        score_consistency=dict(tolerance=dict(config["score_consistency_tolerance"]),
            units="nats/token including EOS",
            records="compute.jsonl: score_consistency, per state/action, including failed checks",
            generation="hf-generate-kv-categorical-v1",
            scoring="torch-functional-teacher-forced-native-ce-v1",
            reinforce_likelihood="same score_tokens native CE tensor as teacher forcing"),
        resources=dict(new_teacher_calls=0, new_teacher_tokens=0,
            support_feedback="rotating meta-training", controller_pretrained=False,
            certification_access="evaluate only; valid_seen exploratory, no independent certificate",
            historical_usage=audit["recorded_bank_usage"]),
        checkpoint_schedule="cumulative 10/25/50 percent after rounds 1/2/3; four windows per round")
