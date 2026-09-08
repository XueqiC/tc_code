"""Strict, standalone configuration; the live RTD validator is never changed."""
from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path

import yaml


RECIPES = ("b1", "b2", "b3_sft", "b3_mix", "b4")
GRIDS = {
    "b1": {"lr": [3e-6, 1e-5, 3e-5], "commits": [36, 72]},
    "b2": {"lr": [1e-6, 5e-6, 1e-5], "commits": [36, 72]},
    "b3_sft": {"lr": [1e-6, 3e-6, 1e-5], "commits": [36]},
    "b3_mix": {"lr": [1e-6, 3e-6, 1e-5], "commits": [36]},
    "b4": {"eta_multiplier": [.3, 1., 3.], "meta_lr": [.03, .1], "commits": [36]},
}
RUNTIME_KEYS = {
    "student", "model_local_files_only", "training_seed", "benchmark", "support_manifest",
    "replay_bank_path", "source_backend", "max_action_tokens", "max_context_tokens",
    "max_action_tokens_by_benchmark", "score_consistency_tolerance", "lora_rank", "lora_alpha",
    "lora_target_modules", "max_state_batch_size", "memory_peak_budget_gb", "memory_reserve_gb",
    "memory_state_estimate_gb", "evaluation_temperature", "pilot_eta_candidates",
    "source_temperature", "source_top_p", "source_samples_per_state",
}


def strict_mapping(value, allowed, *, required=(), label="config"):
    if not isinstance(value, dict) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError(f"{label}: unknown/missing keys; allowed={sorted(allowed)} required={sorted(required)}")
    return value


@dataclass(frozen=True)
class BaselineConfig:
    recipe: str
    view: str = "slots"
    version: str = "c27-baselines-v1"
    seed: int = 0
    commits: int = 36
    slots: int = 288
    lr: float = 1e-5
    optimizer: str = "adamw"
    weight_decay: float = .01
    beta: float = .1
    lambda_pg: float = 1.
    mixture_alpha: float = .5
    features_off: bool = True
    weight_parameterization: str = "per_query_softplus_mean_one"
    meta_lr: float = .1
    ridge: float = 1.
    eta_multiplier: float = 1.
    distribution: str = "strong"
    state_protocol: str = "replay_L1"
    token_cost: str = "prompt_plus_action"
    token_relative_tolerance: float = .01
    # A source-derived value for each round, never inferred from a partial run.
    update_tokens: list = field(default_factory=list)
    pg_reserved_tokens: list = field(default_factory=list)
    evidence_path: str | None = None
    evidence_hash: str | None = None
    pool: str = "A1"
    runtime: dict = field(default_factory=dict)
    new_teacher_calls: bool = False
    evaluate_after_round: bool = False

    def __post_init__(self):
        if self.recipe not in RECIPES or self.version != "c27-baselines-v1" or self.view not in {"slots", "tokens"}:
            raise ValueError("unknown baseline recipe/version/view (D0/D1 are a later integration)")
        if type(self.seed) is not int or self.seed != 0 or self.commits not in (36, 72) or self.slots != 288:
            raise ValueError("seed=0, commits=36/72 and 288 complete slots required")
        if self.recipe in {"b3_sft", "b3_mix", "b4"} and self.commits != 36:
            raise ValueError("B3/B4 require 36 commits")
        if self.optimizer not in {"adamw", "fixed"} or (self.recipe == "b4" and self.optimizer != "fixed"):
            raise ValueError("B4 requires the actual fixed P step, not an AdamW VJP")
        if self.optimizer == "fixed" and self.commits != 36:
            raise ValueError("fixed-step attribution uses 36 commits")
        for name in ("lr", "beta", "meta_lr", "eta_multiplier"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"positive finite {name} required")
        if (self.weight_decay != .01 or self.beta != .1 or self.lambda_pg != 1 or self.mixture_alpha != .5
                or self.ridge != 1 or self.features_off is not True
                or self.weight_parameterization != "per_query_softplus_mean_one"):
            raise ValueError("frozen loss settings; B4 features must be off and weights per query")
        if self.distribution not in {"strong", "p0_attribution"}:
            raise ValueError("unknown slot distribution")
        if self.state_protocol != "replay_L1":
            raise ValueError("only honest replay-L1 states are available; online occupancy needs a new teacher/state path")
        if self.view == "tokens" and self.distribution == "p0_attribution" and self.recipe != "b3_mix":
            raise ValueError("teacher-only empty p0 slots have zero cost: V-T requires the strong positive-cost pT view")
        if self.token_cost != "prompt_plus_action" or self.token_relative_tolerance != .01:
            raise ValueError("full input token cost and final <=1% tolerance required")
        if self.new_teacher_calls is not False or self.evaluate_after_round is not False:
            raise ValueError("no new teacher calls; official evaluation is a separate entry")
        for name in ("update_tokens", "pg_reserved_tokens"):
            values = getattr(self, name)
            if not isinstance(values, list) or (values and (len(values) != 3 or any(type(v) is not int or v < 0 for v in values))):
                raise ValueError(f"{name}: three nonnegative integer round targets required")
        strict_mapping(self.runtime, RUNTIME_KEYS, label="runtime")
        if self.runtime:
            frozen = dict(training_seed=0, model_local_files_only=True, benchmark="bfcl", source_backend="hf_generate",
                          lora_rank=16, lora_alpha=32, evaluation_temperature=0.,
                          source_temperature=1., source_top_p=1., source_samples_per_state=2)
            for key, expected in frozen.items():
                if self.runtime.get(key) != expected:
                    raise ValueError(f"runtime {key} must be {expected}")
            if self.runtime.get("max_context_tokens") != 32768:
                raise ValueError("complete BFCL context cap must be 32768")
            if self.runtime.get("max_action_tokens_by_benchmark") != {"bfcl": {"single_turn": 512, "multi_turn": 1024}}:
                raise ValueError("BFCL action caps must be 512/1024")
            if self.runtime.get("lora_target_modules") != ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
                raise ValueError("frozen RTD LoRA target modules required")
        if (self.evidence_path is None) != (self.evidence_hash is None):
            raise ValueError("evidence path and content hash must be bound together")

    def validate_ready(self):
        if not self.evidence_path or not self.runtime:
            raise ValueError("recipe template: run prepare on a completed source first")
        if self.view == "tokens" and (len(self.update_tokens) != 3 or min(self.update_tokens) <= 0):
            raise ValueError("V-T needs a positive final-source token target for each round")
        if self.recipe.startswith("b3") and self.view == "tokens" and len(self.pg_reserved_tokens) != 3:
            raise ValueError("V-T B3 must reserve the matched feedback score/backprop budget")
        return self

    def to_dict(self):
        return asdict(self)


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    strict_mapping(raw, {f.name for f in fields(BaselineConfig)}, required={"recipe"})
    return BaselineConfig(**raw)
