"""ReAct's 100-token completion cap, independent of paid episode cost caps."""
from contextlib import contextmanager
import json
from ..bank_build import privileged
from ..caps import recorded_budget_ceilings


@contextmanager
def action_limit(backend, category):
    # Greedy and feedback proxies read the underlying backend's active cap.
    while hasattr(backend, 'backend'):
        backend = backend.backend
    if category != 'agent_action' or backend.action_caps.get(category) != 100:
        raise ValueError('HotpotQA agent_action=100 required')
    previous = backend.max_action_tokens
    backend.max_action_tokens = 100
    try:
        yield
    finally:
        backend.max_action_tokens = previous


def public_cap(request_class, *, limits=None, evidence=()):
    privileged()
    if request_class != 'hotpotqa_demo_episode' or not isinstance(limits, dict) or request_class not in limits:
        raise ValueError('HotpotQA needs certified sealed-bank class caps')
    return limits[request_class], json.dumps(dict(request_class=request_class,
        basis='sealed_bank_class_content_envelope', evidence=list(evidence)))


def affordability(records, *, certificate):
    core = certificate['core']
    usable = [r for r in records if r.unavailable_reason is None]
    if (core['benchmark'] != 'hotpotqa' or len(usable) != core['available_packages']
            or any(r.spec.cost_upper_bound != core['class_caps'][
                json.loads(r.spec.cap_provenance)['request_class']] for r in usable)):
        raise ValueError('affordability requires the matching HotpotQA certificate')
    return recorded_budget_ceilings(core['budget_denominator'], rounds=2)
