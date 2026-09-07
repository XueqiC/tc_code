"""Public request-configuration caps; this module never inspects usage/text.

Unknown historical limits use the project lead's v1.0.1 public class caps.
They are NOT provider guarantees or permission to make new API requests.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


PUBLIC_CLASS_CAPS = {"demo_attempt": 65536, "generator_item": 8192}
PROTOCOL_VERSION = "1.0.1"


@dataclass(frozen=True)
class RequestLimits:
    max_output_tokens: int
    max_actions: int = 1
    max_attempts: int = 1  # total attempts, including the first

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("finite positive configured request limits required")

    @property
    def cap(self):
        return self.max_output_tokens * self.max_actions * self.max_attempts


def public_cap(request_class, *, limits=None, evidence=()):
    """Only independently archived CONFIGURATION may supply ``limits``.

    Generator limits describe the historical item approximation. Never divide
    a batch cap by the number of accepted outputs or infer limits from usage.
    Demo a1/a2/a3 are separately charged packages, not three retries of each.
    """
    if request_class not in {"demo_attempt", "generator_item"}:
        raise ValueError("unknown request class")
    provenance = dict(request_class=request_class,
        basis="archived_configuration" if limits else "class_uniform_public_fallback",
        limits=asdict(limits) if limits else None,
        output_token_cap=limits.cap if limits else PUBLIC_CLASS_CAPS[request_class],
        protocol_version=PROTOCOL_VERSION,
        decision="project-lead public replay convention; independent of hidden usage",
        provider_documentation=("https://api-docs.deepseek.com/guides/reasoning_model" if request_class == "demo_attempt"
                                else "https://api-docs.deepseek.com/news/news0725/"),
        evidence=list(evidence), online_authorized=False)
    return provenance["output_token_cap"], json.dumps(provenance, sort_keys=True)


def limits_from_metadata(record, request_class):
    """Inspect only named request CONFIGURATION envelopes, never result/schema/usage.

    max_retries counts additional attempts; max_attempts counts total attempts.
    Demo wrappers need an explicitly finite retry limit. For historical generator
    items the declared per-item approximation uses one configured output allowance.
    """
    envelopes = [record]
    if isinstance(record.get("metadata"), dict):
        envelopes.append(record["metadata"])
    found = []
    for envelope in envelopes:
        for name in ("request_parameters", "request_config"):
            config = envelope.get(name)
            if not isinstance(config, dict):
                continue
            outputs = [config[k] for k in ("max_tokens", "max_completion_tokens") if k in config]
            if not outputs:
                continue
            if len(set(outputs)) != 1:
                raise ValueError("conflicting archived max-output settings")
            attempts = config.get("max_attempts")
            if "max_retries" in config:
                retries = config["max_retries"]
                if type(retries) is not int or retries < 0:
                    return None
                if attempts is not None and attempts != retries + 1:
                    raise ValueError("conflicting archived retry limits")
                attempts = retries + 1
            if request_class == "generator_item":
                attempts = 1
            if attempts is not None:
                found.append(RequestLimits(outputs[0], config.get("max_actions", 1), attempts))
    if len(set(found)) > 1:
        raise ValueError("conflicting archived request configuration envelopes")
    return found[0] if found else None


def archived_cap_evidence(root):
    """C24 audit of the actual archived code paths (content hashes included).

    No automatic keyword search over teacher-authored schemas: those contain
    unrelated tool parameters named max_tokens. Provider defaults are unknown.
    Future recovered request configs can call public_cap with RequestLimits.
    """
    root = Path(root)
    base = "envs/bfcl/gorilla/berkeley-function-call-leaderboard/bfcl_eval/"
    paths = {
        "demo_attempt": ["tools/bfcl_teacher_demos.sh", base + "constants/model_config.py",
            base + "model_handler/api_inference/openai_completion.py", base + "model_handler/utils.py"],
        "generator_item": ["tools/bfcl_generate.py"],
    }
    findings = {
        "demo_attempt": "C24 audit: shell has three separate attempt directories; its max_tokens=5 is only a credential probe. DeepSeek registry uses OpenAICompletionsHandler._query_FC, which omits max_tokens/max_completion_tokens. retry_with_backoff has no finite stop; record metadata has no request limits. No finite provider cap recoverable.",
        "generator_item": "C24 audit: bfcl_generate.ask omits max_tokens/max_completion_tokens (also in HEAD). TEACHER_ATTEMPTS=3 limits item repair, not provider output. Authored rows contain no request settings; provider batch/repair boundaries missing. No configured per-item token cap recoverable.",
    }
    return {kind: [dict(path=p, sha256=hashlib.sha256((root / p).read_bytes()).hexdigest())
                   for p in ps if (root / p).is_file()] + [dict(finding=findings[kind])]
            for kind, ps in paths.items()}


def affordability(records, payloads=None):
    """Public-cap denominator; b is an expected-cost constraint, not a cap filter.

    Window rows are zero-spend illustrations. The live ledger recomputes b from
    actual remaining authorization at every decision. Capacity assumes packages
    of just one class, and is bounded by available inventory.
    """
    usable = [r for r in records if r.unavailable_reason is None]
    total = sum(r.spec.cost_upper_bound for r in usable)
    def kind(r):
        return json.loads(r.spec.cap_provenance)["request_class"]
    def capacity(amount):
        return {k: min(sum(kind(r) == k and not r.dependencies for r in usable), int(amount // cap))
                for k, cap in PUBLIC_CLASS_CAPS.items()}
    points = []
    for percent in (10, 25, 50):
        budget = total * percent // 100
        affordable = [r for r in usable if not r.dependencies and r.spec.cost_upper_bound <= budget]
        points.append(dict(percent=percent, budget=budget, affordable_packages=len(affordable),
            bank_public_cap_sum=total, capacity_by_class=capacity(budget),
            by_class={k: sum(kind(r) == k for r in affordable) for k in PUBLIC_CLASS_CAPS},
            by_inner_fold={str(fold): sum(int(r.parent_hash, 16) % 2 == fold for r in affordable)
                           for fold in (0, 1)},
            windows=[dict(step=step, remaining_windows=4-i, remaining_budget=budget,
                          b=budget/(4-i), capacity_by_class=capacity(budget/(4-i)))
                     for i, step in enumerate((1, 4, 7, 10))],
            basis="sum of usable public caps; individual affordability at empty ownership; window capacities assume no prior spend"))
    return points


def affordability_for(config, records, payloads=None):
    from .benchmarks.registry import get_benchmark
    return get_benchmark(config).affordability(records, payloads)


def cap_policy_for(config):
    from .benchmarks.registry import get_benchmark
    return get_benchmark(config).cap_policy
