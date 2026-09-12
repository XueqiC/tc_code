"""WebShop action horizon and sealed class-cap access."""
from contextlib import contextmanager
import json
from ..bank_build import privileged
from ..caps import recorded_budget_ceilings


@contextmanager
def action_limit(backend, category):
    if category != 'agent_action' or backend.action_caps.get(category) != 128:
        raise ValueError('WebShop agent_action=128 required')
    previous = backend.max_action_tokens
    backend.max_action_tokens = 128
    try:
        yield
    finally:
        backend.max_action_tokens = previous


def public_cap(request_class, *, limits=None, evidence=()):
    privileged()
    if request_class != 'webshop_demo_state' or not isinstance(limits, dict) or request_class not in limits:
        raise ValueError('WebShop needs certified sealed-bank class caps')
    return limits[request_class], json.dumps(dict(request_class=request_class,
        basis='sealed_bank_class_content_envelope', evidence=list(evidence)))


def affordability(records, *, certificate):
    # Public cap sums are reservation envelopes, never the v1.1 denominator.
    core = certificate['core']
    usable = [r for r in records if r.unavailable_reason is None]
    if (core['benchmark'] != 'webshop' or len(usable) != core['available_packages']
            or any(r.spec.cost_upper_bound != core['class_caps'][
                json.loads(r.spec.cap_provenance)['request_class']] for r in usable)):
        raise ValueError('affordability requires the matching bank certificate')
    return recorded_budget_ceilings(core['budget_denominator'], rounds=2)
