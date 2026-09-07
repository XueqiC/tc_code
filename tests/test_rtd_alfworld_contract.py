"""C26-E dispatch contracts, plus explicitly TEST-ONLY C26-F import-deny policy.

The test guard below is not installed by production code. selector._check_import
still needs the §7.3 item 10 edit; passing this suite never certifies that edit.
"""
import ast
import builtins
from copy import deepcopy
from dataclasses import fields
import importlib
import inspect
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from bfas.rtd.benchmarks import registry
from bfas.rtd.benchmarks.alfworld_rollout import IncompleteFeedbackError
from bfas.rtd.benchmarks.alfworld_support import ALFWorldSupport
from bfas.rtd import selector
from bfas.rtd.return_gradient import TaskRollout, reinforce_gradient
from bfas.rtd.persistence import digest
from rtd_alfworld_evaluation_fixtures import campaign, ROOT, put
from test_rtd_alfworld_config import sealed_campaign
from test_rtd_alfworld_state import FakeStepper
from test_rtd_alfworld_rollout import TinyBackend, EnumerableEnv, MemoryJournal, render, parameters


@pytest.mark.parametrize('role', [f.name for f in fields(registry.Providers)])
def test_every_provider_matches_introspected_bfcl_signature(role):
    bfcl = registry.get_benchmark({'benchmark': 'bfcl'})
    alf = registry.get_benchmark({'benchmark': 'alfworld'})
    expected, actual = getattr(bfcl, role), getattr(alf, role)
    assert inspect.signature(actual) == inspect.signature(expected), role
    # No forged __signature__ can hide an incompatible adapter implementation.
    assert not hasattr(actual, '__signature__')
    if role == 'support_protocol':
        assert inspect.signature(actual.feedback) == inspect.signature(expected.feedback)


def test_bfcl_providers_are_original_imported_objects():
    from bfas.rtd import bank, broker, caps, evaluation, identity, scoring_scope, experiment, return_gradient
    p = registry.get_benchmark({})
    assert p == registry.get_benchmark({'benchmark': 'bfcl'})
    originals = dict(bank_builder=bank.build_bfcl_bank, broker_builder=broker.SealedReplayBroker,
        support_protocol=experiment.BFCLSupport, action_limit=return_gradient.TorchPolicyBackend.action_limit,
        feedback_rollout=return_gradient.bfcl_task_rollout, official_evaluation=evaluation.evaluate,
        harness_identity=identity.evaluation_harness_identity, cap_policy=caps.public_cap,
        affordability=caps.affordability, scoring_projection=scoring_scope.scoring_projection)
    assert all(getattr(p, k) is v for k, v in originals.items())
    for value in ('unknown', '', None, []):
        with pytest.raises(ValueError, match='unknown'):
            registry.get_benchmark({'benchmark': value})
    with pytest.raises(TypeError):
        registry.REGISTRY['alfworld']['cap_policy'] = ('bad', 'bad')


def test_support_exact_current_experiment_surface_and_all_parents(sealed_campaign):
    p = registry.get_benchmark({'benchmark': 'alfworld'})
    c = sealed_campaign
    support = p.support_protocol(c.root, c.config)
    from bfas.rtd.experiment import RTDExperiment
    tree = ast.parse(inspect.getsource(RTDExperiment))
    # Derive the consumed surface from the real experiment, not a copied Protocol.
    consumed = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Attribute) and n.value.attr == 'support'
        and isinstance(n.value.value, ast.Name) and n.value.value.id == 'self'}
    assert consumed <= set(dir(support))
    assert len(support.parents) == len(support.states) == 135
    assert len(support.parents) > 107  # failed/no-payload parents remain feedback-eligible
    assert set(support.categories.values()) == {'agent_action'}
    assert support.unavailable == {}
    assert all(len(json.loads(s.history_json)) == 1 for s in support.states.values())


def test_action_policy_dispatch_and_exception_restoration():
    policy = registry.get_benchmark({'benchmark': 'alfworld'}).action_limit
    backend = SimpleNamespace(max_action_tokens=512, action_caps={'agent_action': 256})
    with pytest.raises(RuntimeError):
        with policy(backend, 'agent_action'):
            assert backend.max_action_tokens == 256
            raise RuntimeError('fixture')
    assert backend.max_action_tokens == 512
    with pytest.raises(ValueError, match='unregistered'):
        with policy(backend, 'single_turn'):
            pass
    backend.action_caps['agent_action'] = 512
    with pytest.raises(ValueError, match='256'):
        with policy(backend, 'agent_action'):
            pass


def test_rollout_adapter_loophole_free_rng_resume_and_generic_loo(sealed_campaign, parameters):
    c = sealed_campaign
    p = registry.get_benchmark(c.config)
    support = p.support_protocol(c.root, c.config)
    protocol = support.protocol
    parent = sorted(protocol.parent_hashes(1, use='feedback'))[0]
    journal, backend = MemoryJournal(), TinyBackend()
    class MatchedEnv(EnumerableEnv):
        def observation(self):
            if not self.commands:
                return FakeStepper(self.request).observation(0)
            return super().observation()
    context = registry.ALFWorldFeedbackContext(protocol, 1, render, MatchedEnv, journal)
    generator = torch.Generator().manual_seed(19)
    before = generator.get_state().clone()
    rows = [support.feedback(parent, backend, parameters, generator, context) for _ in range(2)]
    after = generator.get_state().clone()
    assert all(isinstance(r, TaskRollout) and r.from_task_start for r in rows)
    assert not torch.equal(before, after)
    seeds = [e['seed'] for e in journal.events if e['kind'] == 'alfworld_episode']
    assert len(set(seeds)) == 2
    generator.set_state(before)
    again = [support.feedback(parent, backend, parameters, generator, context) for _ in range(2)]
    assert again == rows and torch.equal(generator.get_state(), after)
    g = reinforce_gradient(rows, backend, parameters)
    assert g.parameter_hash and len(g.metadata['rewards']) == 2
    inner = sorted(protocol.parent_hashes(1))[0]
    with pytest.raises(ValueError, match='rejected'):
        support.feedback(inner, backend, parameters, generator, context)
    failed = registry.ALFWorldFeedbackContext(protocol, 1, render,
        lambda: MatchedEnv(fault='step'), journal)
    with pytest.raises(IncompleteFeedbackError):
        support.feedback(parent, backend, parameters, generator, failed)
    assert journal.events[-1]['excluded'] and journal.events[-1]['reward'] is None


def test_native_cap_scoring_and_builder_delegation(monkeypatch):
    from bfas.rtd.benchmarks import alfworld_bank, alfworld_caps, alfworld_identity
    p = registry.get_benchmark({'benchmark': 'alfworld'})
    assert p.cap_policy('alf_demo_episode') == alfworld_caps.public_cap('alf_demo_episode')
    assert p.affordability is alfworld_caps.affordability
    assert p.scoring_projection is alfworld_identity.scoring_projection
    calls = []
    monkeypatch.setattr(alfworld_bank, 'build_alfworld_bank', lambda root, directory: calls.append((root, directory)))
    p.bank_builder('root', 'new')
    assert calls == [('root', 'new')]
    with pytest.raises(ValueError, match='BFCL'):
        p.bank_builder('root', 'new', entries={})


def test_harness_adapter_matches_c26d(sealed_campaign):
    from bfas.rtd.benchmarks import alfworld_identity
    c = sealed_campaign
    p = registry.get_benchmark(c.config)
    assert p.harness_identity(c.root, c.config) == alfworld_identity.evaluation_harness_identity(
        c.root, c.config, data_root=c.data, model_path=c.model, tokenizer_path=c.model)


def test_evaluation_adapter_requires_common_guards_and_delegates(monkeypatch, sealed_campaign):
    from bfas.rtd import identity, hardware
    from bfas.rtd.benchmarks import alfworld_config, alfworld_identity, alfworld_evaluation
    c = sealed_campaign
    directory = c.root / 'new-training'
    saved = dict(config=c.config, config_hash=digest(c.config), bank_path=str(c.bank),
        data_hash=alfworld_config.data_identity(c.root, c.config, c.bank), model_path=str(c.model))
    put(directory / 'manifest.json', saved)
    calls = []
    monkeypatch.setattr(hardware, 'hardware_identity', lambda: c.hardware)
    def guard(*args, **kwargs):
        calls.append('hardware'); return c.hardware
    monkeypatch.setattr(hardware, 'guard_hardware', guard)
    monkeypatch.setattr(identity, 'guard_harness', lambda *a, **k: calls.append('harness') or {})
    monkeypatch.setattr(identity, 'record_code_drift', lambda *a, **k: calls.append('source'))
    binding = {'fixture': 'bound by C26-D'}
    def make(root, cfg, **kwargs):
        assert cfg == c.config and kwargs['run_directory'] == directory and kwargs['round_number'] == 1
        calls.append('checkpoint_binding'); return binding
    monkeypatch.setattr(alfworld_identity, 'make_manifest', make)
    result = {'success_rate': .5, 'success_percent': 50., 'fixture': True}
    def evaluate(root, manifest, **kwargs):
        assert manifest is binding and not kwargs['output_root'].is_relative_to(directory)
        assert kwargs['tag'] == 'round-1' and kwargs['lock_timeout'] == 0
        calls.append('campaign'); return result
    monkeypatch.setattr(alfworld_evaluation, 'evaluate', evaluate)
    p = registry.get_benchmark(c.config)
    assert p.official_evaluation(c.root, directory, 1, lock_timeout=0) == result
    assert calls == ['hardware', 'harness', 'source', 'checkpoint_binding', 'campaign']
    assert json.loads((directory / 'evaluation-1.json').read_text()) == result
    saved['data_hash'] = 'changed'; put(directory / 'manifest.json', saved)
    with pytest.raises(ValueError, match='data differs'):
        p.official_evaluation(c.root, directory, 1)


PRIVILEGED = {'registry', 'alfworld_config', 'alfworld_bank', 'alfworld_state',
              'alfworld_support', 'alfworld_rollout', 'alfworld_evaluation', 'alfworld_identity'}


@pytest.fixture
def proposed_import_guard(monkeypatch):
    """Test-only executable specification for readiness §7.3 item 10.

    Native cached-module imports currently bypass the three-name denylist.
    Patch only this test invocation; no runtime monkeypatch in the CLI/registry.
    """
    original = selector._check_import
    def check(name, fromlist=()):
        original(name, fromlist)
        if selector._active.get() is not None and (set(str(name).split('.')) | set(fromlist or ())) & PRIVILEGED:
            selector._active.get().append({'kind': 'denied_import', 'resource': str(name)})
            raise PermissionError('selector cannot import privileged ALFWorld modules (test-only C26-F policy)')
    monkeypatch.setattr(selector, '_check_import', check)


@pytest.mark.parametrize('module', sorted(PRIVILEGED))
@pytest.mark.parametrize('route', ['cached', 'uncached', 'fromlist', 'relative'])
def test_selector_cannot_import_privileged_alfworld_with_proposed_policy(module, route, proposed_import_guard):
    name = 'bfas.rtd.benchmarks.' + module
    importlib.import_module(name)  # Exercise cached imports, which audit hooks alone miss.
    saved = sys.modules.pop(name) if route == 'uncached' else None
    trace = []
    try:
        with pytest.raises(PermissionError), selector.public_only(trace):
            if route == 'fromlist':
                builtins.__import__('bfas.rtd.benchmarks', fromlist=(module,))
            elif route == 'relative':
                importlib.import_module('.' + module, 'bfas.rtd.benchmarks')
            else:
                importlib.import_module(name)
    finally:
        if saved is not None:
            sys.modules[name] = saved
    assert trace[-1]['kind'] == 'denied_import'


def test_registry_rejects_selector_resolution_without_test_guard():
    with pytest.raises(PermissionError, match='privileged'):
        selector.select_public([], lambda _: registry.get_benchmark({'benchmark': 'alfworld'}))
    tree = ast.parse((ROOT / 'src/bfas/rtd/selector.py').read_text())
    imports = {part for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
        for value in ([n.module or ''] if isinstance(n, ast.ImportFrom) else []) + [a.name for a in n.names]
        for part in value.split('.')}
    assert not imports & PRIVILEGED
