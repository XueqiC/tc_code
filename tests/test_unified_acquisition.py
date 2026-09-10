from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from bfas.rtd.ledger import BudgetError
from bfas.rtd.selector import PublicFeatures, PublicQuerySpec, public_only
from bfas.rtd.unified import (Array, Evidence, TeachingObjective, PrequeryContext,
    RevealedEvidence, build_prequery_features, build_acquisition_labels, purchase_or_reveal,
    score_query_batch, build_teaching_problem, solve_exact, empirical_target, ControlObservables, predict_control)
from test_rtd_broker import bank, candidates
from unified_fixtures import from_directions, with_oracle_feedback


def feature(problem, query_id):
    query = PublicQuerySpec(query_id, 'state0', PublicFeatures(), 1, 20, 'exact', 'public cap')
    return build_prequery_features(query, PrequeryContext(problem.theta_hash,
        problem.context.optimizer_version, 0., 30, (.1,), (.2,)))


def test_purchase_boundary_hidden_field_invisibility_charge_once_and_budget(bank, monkeypatch):
    broker, ledger, snapshot, (q, dependent, other), directory = bank
    specs = {s.query_id: s for s in candidates(broker, ledger, snapshot)}
    context = PrequeryContext('theta', 'P', .1, ledger.remaining)
    reads = []
    original = Path.read_text
    def spy(path, *args, **kwargs):
        reads.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', spy)
    with public_only([]):
        features = build_prequery_features(specs[q], context)
        batch = score_query_batch([features], context)
    assert reads == [] and batch.scores == () and 'PLANNED' in batch.status
    assert not {'text', 'teacher_gradient', 'validation', 'actual_cost', 'usage'} & asdict(features).keys()
    # Changing an arbitrary hidden field cannot enter the strict public type.
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError('hidden object read')
    with pytest.raises(TypeError):
        build_prequery_features(Poison(), context)
    assert build_prequery_features(replace(specs[q], cap_provenance='hidden-like ignored string'), context) == features
    package = purchase_or_reveal(specs[q], ledger, broker=broker)
    assert ledger.spent == 7 and ledger.charges[q] == package.cost
    assert len([r for r in reads if r.name == q+'.json']) == 1
    assert purchase_or_reveal(specs[q], ledger, broker=broker) is package and ledger.spent == 7
    assert dependent in {s.query_id for s in candidates(broker, ledger, snapshot)}
    ledger.reserve('unrelated-reservation', 45)
    reads.clear()
    with pytest.raises(BudgetError):
        purchase_or_reveal(specs[other], ledger, broker=broker)
    assert reads == [] and ledger.spent == 7 and ledger.remaining == 8


def test_two_teacher_versions_nested_zero_use_target_increment_and_objective():
    problem = from_directions([[.3, -.1], [.1, .3]], h=[1., .3])
    extra = replace(problem.evidence[0], query_id='q_new', version='teacher-2',
        text='a different teacher continuation', behavior_hash='T2', gradient=Array.of([-2., 0.]))
    saved = feature(problem, extra.query_id)
    revealed = RevealedEvidence((extra,), frozenset({'q0', extra.query_id}), (saved,), 'charged-ledger-hash')
    label = build_acquisition_labels(problem, revealed)
    expanded = label.expanded_problem
    assert expanded.evidence[1].text_hash != expanded.evidence[0].text_hash
    assert label.value >= -1e-10 and label.zero_use_increment_error == 0
    np.testing.assert_array_equal(expanded.a_ref.numpy(), [.5, .5, 0., 0.])
    assert expanded.reference_parameters.hash == problem.reference_parameters.hash
    assert expanded.feedback_hash == problem.feedback_hash
    solution = solve_exact(expanded)
    a = solution.coefficients.numpy()
    assert all(a[list(g)].sum() <= 1+1e-12 for g in expanded.groups)
    assert max(solution.stationarity, solution.complementarity, solution.feasibility) <= 1e-8
    for old_a in ([0, 0], [.2, .8], [1, 1]):
        padded = [*old_a, 0., 0.]
        np.testing.assert_array_equal(TeachingObjective(problem).increment(old_a),
                                      TeachingObjective(expanded).increment(padded))
        assert TeachingObjective(problem).parameters(old_a).hash == TeachingObjective(expanded).parameters(padded).hash
        assert TeachingObjective(problem).value(old_a) == pytest.approx(TeachingObjective(expanded).value(padded), abs=1e-13)
        assert empirical_target(('Y1', 'Y2'), ('T1',), old_a) == empirical_target(
            ('Y1', 'Y2'), ('T1', 'T2'), np.asarray(padded).reshape(2, 2).T)
    q = empirical_target(('Y1', 'Y2'), ('T1', 'T2'), a.reshape(2, 2).T)
    assert min(q.values()) >= 0 and sum(q.values()) == pytest.approx(1.)
    with pytest.raises(ValueError, match='infeasible'):
        TeachingObjective(expanded).increment([.8, .8, .8, .8])


def test_first_evidence_contains_teacher_gain_even_with_identical_student_sources():
    full = from_directions([[1., 1.]], h=[1.])
    old = build_teaching_problem(replace(full.context, owned_query_ids=()), (), full.exposure)
    old = with_oracle_feedback(old, [1.])
    label = build_acquisition_labels(old, RevealedEvidence(full.evidence, frozenset({'q0'}),
        (feature(old, 'q0'),), 'ledger'))
    assert label.value > .4  # teacher gain cannot disappear by re-centering
    assert label.zero_use_increment_error == 0 and label.zero_use_value_error == 0
    assert solve_exact(old).value == 0


def test_no_new_retention_or_reaveraging_and_prequery_version_guards():
    old = from_directions([[.1, .2]], h=[1.])
    # New exposure state is illegal: candidate retention must already be present.
    new = replace(old.evidence[0], query_id='new', state_hash='unseen')
    reveal = RevealedEvidence((new,), frozenset({'q0', 'new'}), (feature(old, 'new'),), 'ledger')
    with pytest.raises(ValueError, match='pre-existing'):
        build_acquisition_labels(old, reveal)
    stale = replace(feature(old, 'new'), context=PrequeryContext('old-theta', 'P', 0., 30))
    with pytest.raises(ValueError, match='versions'):
        build_acquisition_labels(old, replace(reveal, prequery_features=(stale,)))
    with pytest.raises(ValueError, match='purchased'):
        build_teaching_problem(replace(old.context, owned_query_ids=()), old.evidence, old.exposure)
    obs = ControlObservables(old.identity, old.a_ref)
    assert predict_control(obs) is old.a_ref
