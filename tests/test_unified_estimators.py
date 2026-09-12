from itertools import product

import numpy as np
import pytest
import torch

from bfas.rtd.unified import empirical_target, estimate, alpha_d


def test_empirical_target_mass_duplicate_merging_and_teacher_alias():
    for a in product([0., .2, .5, 1.], repeat=2):
        for y in [('A', 'B'), ('A', 'A'), ('T', 'B'), ('T', 'T')]:
            q = empirical_target(y, ('T',), a)
            assert all(v >= 0 for v in q.values())
            assert sum(q.values()) == pytest.approx(1.)
    assert empirical_target(('A', 'A'), ('T',), [.2, .8]) == {'A': .5, 'T': .5}
    # Alpha=.5 fixes transferred source mass, NOT total teacher-behaviour mass.
    assert empirical_target(('A', 'T'), ('T',), [.5, .5]) == {'A': .25, 'T': .75}
    assert empirical_target(('T', 'T'), ('T',), [0., 0.]) == {'T': 1.}
    for bad in ([-.1, .5], [1.01, 0], [np.nan, 0]):
        with pytest.raises(ValueError):
            empirical_target(('A', 'B'), ('T',), bad)


def test_two_versions_share_source_mass_and_zero_added_version_recovers_target():
    a = np.array([[.2, .3], [.7, .1]])
    q = empirical_target(('A', 'T2'), ('T1', 'T2'), a)
    assert sum(q.values()) == pytest.approx(1.) and min(q.values()) >= 0
    assert q == pytest.approx({'A': .25, 'T1': .45, 'T2': .30})
    np.testing.assert_equal(empirical_target(('A', 'B'), ('T1',), a[:, 0]),
        empirical_target(('A', 'B'), ('T1', 'T2'), np.c_[a[:, 0], [0, 0]]))
    with pytest.raises(ValueError, match='sum <= 1'):
        empirical_target(('A', 'B'), ('T1', 'T2'), [[.8, .8], [0., 0.]])


def test_coefficients_stop_gradient_and_reparameterisation():
    a = torch.tensor([.8, .2], requires_grad=True)
    alpha, d = alpha_d(a)
    assert alpha == pytest.approx(.5) and d == pytest.approx(.3)
    got = estimate([[1., 2.], [-2., 3.]], [4., 5.], [.1, .2], a)
    assert np.isfinite(got).all() and a.grad is None


def test_enumerated_arbitrary_adaptive_selector_frozen_target_gradient():
    p = np.array([.2, .3, .5])
    theta = torch.tensor([.2, -.7, .8], dtype=torch.float64, requires_grad=True)
    pi = theta.softmax(0).detach().numpy()
    gs = pi[None, :]-np.eye(3)
    soft, teacher = pi-p, gs[2]
    raw_mean, cv_mean, target_mean = [np.zeros(3) for _ in range(3)]
    # Completely arbitrary sample AND feedback dependent frozen selector.
    selector = np.random.default_rng(12).uniform(size=(3, 3, 2, 2))
    for y1, y2, feedback in product(range(3), range(3), range(2)):
        prob = p[y1]*p[y2]*(.4 if feedback == 0 else .6)
        a = selector[y1, y2, feedback]
        hard = gs[[y1, y2]]
        q = empirical_target((y1, y2), (2,), a)
        frozen_loss = -sum(float(mass)*theta.log_softmax(0)[y] for y, mass in q.items())
        target_grad = torch.autograd.grad(frozen_loss, theta)[0].numpy()
        raw = estimate(hard, teacher, soft, a, estimator='raw')
        cv = estimate(hard, teacher, soft, a)
        np.testing.assert_allclose(raw, target_grad, atol=1e-15)
        np.testing.assert_allclose(cv-raw, soft-hard.mean(0), atol=1e-15)
        raw_mean += prob*raw
        cv_mean += prob*cv
        target_mean += prob*target_grad
    np.testing.assert_allclose(raw_mean, target_mean, atol=1e-15)
    np.testing.assert_allclose(cv_mean, target_mean, atol=1e-15)


def test_old_adaptive_weighted_soft_counterexample_and_signed_batch():
    means = dict(raw=0., cv=0., old=0.)
    for first, second in product(range(2), repeat=2):
        hard = np.array([[-.5 if first == 0 else .5], [-.5 if second == 0 else .5]])
        a = np.repeat(float(first == 1), 2)
        alpha, d = alpha_d(a)
        raw = estimate(hard, [-.5], [0.], a, estimator='raw')[0]
        cv = estimate(hard, [-.5], [0.], a)[0]
        old = (1-alpha)*0 + alpha*(-.5) - d/2*(hard[0, 0]-hard[1, 0])
        for key, value in dict(raw=raw, cv=cv, old=old).items():
            means[key] += .25*value
        assert cv-old == pytest.approx(alpha*(0-hard.mean()))
    assert means == pytest.approx(dict(raw=-.375, cv=-.375, old=-.25))
    # At pair (B,B), CV=-1 lies outside convex hull of ANY hard NLL gradients.
    assert estimate([[.5], [.5]], [-.5], [0.], [1., 1.])[0] == -1.


@pytest.mark.parametrize('endpoint', [0., 1.])
def test_endpoints_values_expectations_and_variances(endpoint):
    raw, cv = [], []
    for hard in product([-.5, .5], repeat=2):
        raw.append(estimate(np.array(hard)[:, None], [-.5], [0.], [endpoint]*2, estimator='raw')[0])
        cv.append(estimate(np.array(hard)[:, None], [-.5], [0.], [endpoint]*2)[0])
        assert cv[-1] == pytest.approx(0. if endpoint == 0 else -.5-np.mean(hard))
        assert raw[-1] == pytest.approx(np.mean(hard) if endpoint == 0 else -.5)
    assert np.mean(raw) == np.mean(cv) == -.5*endpoint
    assert np.var(raw) == pytest.approx(.125 if endpoint == 0 else 0.)
    assert np.var(cv) == pytest.approx(0. if endpoint == 0 else .125)


def test_equal_teacher_and_zero_student_gradient_degeneracies():
    soft = [0., 0.]
    teacher = np.array([1., 2.])
    for mode in ('raw', 'cv'):
        g0 = estimate([[3., 4.], [3., 4.]], teacher, soft, [0, 0], estimator=mode)
        g1 = estimate([[3., 4.], [3., 4.]], teacher, soft, [1, 1], estimator=mode)
        assert not np.array_equal(g0, g1)  # equal students can still learn teacher
        np.testing.assert_allclose(estimate([teacher, teacher], teacher, soft, [0, .9], estimator=mode),
            estimate([teacher, teacher], teacher, soft, [1, 0], estimator=mode))
    np.testing.assert_allclose(estimate([[0., 0.], [0., 0.]], teacher, soft, [1, 1]), teacher)
