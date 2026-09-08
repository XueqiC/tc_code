"""C26 retrospective public configuration, independent of teacher payloads.

These are replay conventions from readiness §2.5, not recovered historical
provider guarantees and not authorization for online calls.
"""
from dataclasses import asdict, dataclass
import json
from types import MappingProxyType


PUBLIC_CLASS_CAPS = MappingProxyType({"alf_demo_episode": 1_310_720, "alf_teacher_turn": 32_768})


@dataclass(frozen=True)
class CapConfiguration:
    max_completion_tokens: int = 2048
    max_episode_steps: int = 40
    empty_response_calls: int = 2
    client_attempts: int = 8

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("positive integer public configuration required")


def public_cap(request_class, *, configuration=CapConfiguration(), evidence=()):
    if request_class not in PUBLIC_CLASS_CAPS:
        raise ValueError("unknown ALFWorld request class")
    actions = configuration.max_episode_steps if request_class == "alf_demo_episode" else 1
    cap = (configuration.max_completion_tokens * actions * configuration.empty_response_calls
           * configuration.client_attempts)
    provenance = dict(
        request_class=request_class, basis="class_uniform_public_fallback",
        configuration=asdict(configuration), output_token_cap=cap,
        version="c26-a", evidence=list(evidence), online_authorized=False,
        derivation=f"{configuration.max_completion_tokens} × {actions} × "
                   f"{configuration.empty_response_calls} × {configuration.client_attempts} = {cap}",
        decision="readiness §2.5 implementation convention; historical request envelope missing",
        source_reason="appworld_teacher.MAX_COMPLETION_TOKENS=2048; episode default 40 steps; "
                      "one extra empty-content call; max(CHAT_COMPLETION_RETRIES+1, "
                      "RATE_LIMIT_RETRIES+1)=8",
        limitations="current-code reconstruction; quota recovery probes and lost responses unknown; "
                    "not a hard upper bound on the complete historical provider bill",
    )
    return cap, json.dumps(provenance, sort_keys=True, ensure_ascii=False)


def affordability(records, payloads=None):
    """Only usable packages enter the denominator. Payloads never set caps."""
    usable = [r for r in records if r.unavailable_reason is None]
    total = sum(r.spec.cost_upper_bound for r in usable)
    return [dict(percent=p, budget=total * p // 100, bank_public_cap_sum=total,
                 by_inner_fold={str(f): sum(int(r.parent_hash, 16) % 2 == f and not r.dependencies
                                           and r.spec.cost_upper_bound <= total * p // 100 for r in usable)
                                for f in (0, 1)},
                 basis="usable_public_cap_sum") for p in (10, 25, 50)]


def cap_audit(records, payloads, *, configuration=CapConfiguration()):
    """Fail closed on cap drift or cost overflow, including unavailable attempts.

    Missing costs remain unknown; they cannot become usable evidence.
    """
    records = tuple(records)
    if len({r.spec.query_id for r in records}) != len(records):
        raise ValueError("duplicate request identity")
    if {r.spec.query_id for r in records} != set(payloads):
        raise ValueError("record/payload inventory mismatch")
    counts = {k: 0 for k in PUBLIC_CLASS_CAPS}
    missing = []
    for record in records:
        provenance = json.loads(record.spec.cap_provenance)
        kind = provenance["request_class"]
        cap, expected = public_cap(kind, configuration=configuration,
                                   evidence=provenance.get("evidence", ()))
        if provenance != json.loads(expected) or cap != record.spec.cost_upper_bound:
            raise ValueError("cap differs from public configuration")
        expected_L = configuration.max_episode_steps if kind == "alf_demo_episode" else 1
        if record.spec.L != expected_L or record.spec.cost_confidence != "estimated":
            raise ValueError("invalid public L/cost convention")
        counts[kind] += 1
        payload = payloads[record.spec.query_id]
        cost = payload["cost"]
        if cost is None:
            missing.append(record.spec.query_id)
            if record.unavailable_reason is None:
                raise ValueError("missing cost cannot be usable")
        elif type(cost) is not int or cost < 0 or cost > cap:
            raise ValueError("recorded cost invalid or exceeds public cap")
        if record.unavailable_reason is None:
            if payload.get("status") != "usable":
                raise ValueError("candidate/unavailable package cannot enter usable denominator")
            if payload.get("exclusion_reasons"):
                raise ValueError("protected package cannot enter usable denominator")
    total = sum(r.spec.cost_upper_bound for r in records if r.unavailable_reason is None)
    example = 107 * public_cap("alf_demo_episode", configuration=configuration)[0]
    return dict(budget_basis="usable_public_cap_sum", bank_public_cap_sum=total,
                B_bank=total, inventory_by_class=counts, missing_cost_query_ids=missing,
                public_caps={k: public_cap(k, configuration=configuration)[0] for k in PUBLIC_CLASS_CAPS},
                cap_provenance_by_class={k: json.loads(public_cap(k, configuration=configuration)[1])
                                         for k in PUBLIC_CLASS_CAPS},
                budget_affordability=affordability(records),
                conditional_107_candidate_example=dict(
                    condition="all 107 pass C26-B state/support audit with no exclusions",
                    B_bank=example, ceilings={str(p): example * p // 100 for p in (10, 25, 50)}))
