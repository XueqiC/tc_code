"""Every durable C26-F decision phase resumes the same complete ALFWorld window."""
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import json

import numpy as np
import pytest
import torch

from bfas.behavior.deltas import tensor_state_hash
from bfas.rtd import cli
from bfas.rtd.benchmarks.registry import ALFWorldExperimentSupport
from bfas.rtd.experiment import RTDExperiment
from bfas.rtd.functional_step import lora_parameters
from bfas.rtd.persistence import atomic_json
from rtd_alfworld_evaluation_fixtures import campaign
from test_rtd_alfworld_config import sealed_campaign
from rtd_alfworld_runner_fixtures import RunnerBackend, feedback_context


PHASES = ('reference', 'selected', 'revealed', 'actual', 'feedback', 'committed')


def comparable(value):
    if isinstance(value, torch.Tensor):
        return dict(dtype=str(value.dtype), shape=list(value.shape), values=value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return comparable(asdict(value))
    if isinstance(value, dict):
        return {str(k): comparable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [comparable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(comparable(v) for v in value)
    if hasattr(value, '__dict__'):
        return comparable(vars(value))
    return value


def scientific_state(engine):
    keys = ('phase', 'round', 'step', 'parameters', 'source', 'source_id', 'initial',
        'numpy_rng', 'sampling_rng', 'slot_rng', 'inner', 'feedback', 'feedback_tasks',
        'source_cache', 'projection_cache', 'standardizer', 'diagonal', 'eta', 'calibrated',
        'owned', 'owned_before', 'selected', 'label', 'phi', 'controller', 'posterior',
        'cost_model', 'support_return', 'steps', 'old_targets', 'new_targets', 'old_chi',
        'new_chi', 'reference_feedback', 'actual_feedback', 'actual', 'next_phi', 'selection')
    return comparable({k: engine.state[k] for k in keys if k in engine.state})


@pytest.fixture
def integrated(sealed_campaign, monkeypatch):
    c = sealed_campaign
    monkeypatch.setattr(cli, 'ROOT', c.root)
    monkeypatch.setattr(cli, 'hardware_identity', lambda: c.hardware)
    monkeypatch.setattr(ALFWorldExperimentSupport, 'feedback_context', feedback_context)
    c.config = dict(c.config, smoke_override=dict(parents_per_fold=4, slots=8, rollouts=2,
        windows=1, baseline='leave_one_out_same_task', max_seconds=900))
    c.manifest = cli.make_manifest(c.config, 'R1', cli.bank_audit(c.config), smoke=True)
    c.manifest['initial_parameter_hash'] = tensor_state_hash(lora_parameters(RunnerBackend().model))
    def engine(name, *, resume=False, after_save=None):
        directory = c.root / 'results' / name
        if not resume:
            atomic_json(directory / 'manifest.json', c.manifest)
        return RTDExperiment(c.config, c.manifest, directory, RunnerBackend(),
            ALFWorldExperimentSupport(c.root, c.config), smoke=True, resume=resume, after_save=after_save)
    c.engine = engine
    return c


@pytest.mark.parametrize('phase', PHASES)
def test_resume_after_every_durable_phase(integrated, phase):
    c = integrated
    clean = c.engine('clean')
    snapshots = {}
    clean.after_save = lambda current: snapshots.update({current: scientific_state(clean)})
    clean.run()
    class Interrupted(Exception):
        pass
    broken = c.engine('interrupted')
    def stop(current):
        if current == phase:
            assert scientific_state(broken) == snapshots[current]
            raise Interrupted(current)
    broken.after_save = stop
    with pytest.raises(Interrupted):
        broken.run()
    resumed = c.engine('interrupted', resume=True)
    assert scientific_state(resumed) == snapshots[phase]
    resumed.run()
    assert scientific_state(resumed) == scientific_state(clean)
    assert resumed.ledger.charges == clean.ledger.charges
    assert resumed.ledger.owned_ids == clean.ledger.owned_ids
    assert resumed.ledger.spent == clean.ledger.spent
    assert not resumed.ledger.reservations
    assert len(resumed.state['steps']) == 1 and resumed.slots == 8
    assert len(resumed.state['inner']) == len(resumed.state['feedback']) == 4
    for kind in ('source_sample', 'feedback_rollout', 'alfworld_episode', 'return_gradient'):
        def events(e):
            return comparable([{k: v for k, v in row.items() if k not in
                               ('sequence', 'timestamp', 'previous_hash', 'event_hash', 'hash')}
                               for row in e.journal.events if row['kind'] == kind])
        assert events(resumed) == events(clean), kind
    results = [r['metadata'] for r in resumed.journal.events if r['kind'] == 'return_gradient']
    assert results and all(r['tasks'] == 4 and r['rollouts'] == 8 and
                          r['baseline'] == 'leave_one_out_same_task' for r in results)
    # Completed resume consumes no RNG, repeats no episodes, and commits nothing.
    again = c.engine('interrupted', resume=True)
    before = scientific_state(again)
    again.run()
    assert scientific_state(again) == before
    assert again.ledger.charges == clean.ledger.charges


def test_crash_between_reveal_and_phase_save_keeps_paid_prefix_and_rng(integrated, monkeypatch):
    from bfas.rtd.ledger import Ledger
    c = integrated
    clean = c.engine('clean'); clean.run()
    assert clean.ledger.owned_ids, 'fixture must exercise a purchase, not replay only'
    original = Ledger.settle
    class Interrupted(Exception):
        pass
    def settle(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if result:
            raise Interrupted('durable charge before phase save')
        return result
    with monkeypatch.context() as patch:
        patch.setattr(Ledger, 'settle', settle)
        with pytest.raises(Interrupted):
            c.engine('crash').run()
    resumed = c.engine('crash', resume=True); resumed.run()
    assert scientific_state(resumed) == scientific_state(clean)
    assert resumed.ledger.charges == clean.ledger.charges
    assert sum(e['kind'] == 'reveal' for e in resumed.ledger.events) == len(clean.ledger.owned_ids)


def test_saved_alfworld_config_is_exact_and_smoke_cannot_change(integrated, tmp_path):
    import yaml
    c = integrated
    saved = deepcopy(c.manifest)
    path = tmp_path / 'saved.yaml'
    path.write_text(yaml.safe_dump(saved['config']))
    assert cli.resume_config(None, saved) == saved['config']
    assert cli.resume_config(path, saved) == saved['config']
    for name, value in [('gate', 'scalar_sigmoid'), ('max_context_tokens', 30000),
                        ('memory_peak_budget_gb', 37), ('student', 'another-model')]:
        changed = dict(saved['config'], **{name: value})
        path.write_text(yaml.safe_dump(changed))
        with pytest.raises(ValueError, match='resume config changed'):
            cli.resume_config(path, saved)
    saved['config']['smoke_override']['rollouts'] = 1
    with pytest.raises(ValueError, match='saved config hash'):
        cli.resume_config(None, saved)
