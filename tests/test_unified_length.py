from dataclasses import replace

import numpy as np
import pytest
import torch

from bfas.rtd.unified import LossDefinition, hard_loss, soft_retention_loss


def logits(theta, tokens):
    # EOS=0, continue=1; cap=2; independent logits for the two reached prefixes.
    return torch.stack([torch.stack([theta[t]*0, theta[t]]) for t in range(len(tokens))])


def grad(theta, tokens, loss, *, soft=False, snapshot=None, rho=None):
    labels = torch.tensor(tokens)
    new = logits(theta, tokens)
    value = (soft_retention_loss(new, logits(snapshot, tokens), labels, loss, inverse_lengths=rho)
             if soft else hard_loss(new, labels, loss))
    return torch.autograd.grad(value, theta)[0].numpy()


@pytest.mark.parametrize('normalization', ['total_token_nll', 'per_sequence_mean'])
@pytest.mark.parametrize('theta_values', [[0., 0.], [.4, -.3]])
def test_random_length_corrected_soft_matches_source_measure(normalization, theta_values):
    theta = torch.tensor(theta_values, dtype=torch.float64, requires_grad=True)
    source = torch.zeros(2, dtype=torch.float64)
    definition = LossDefinition(normalization, retention_scale=.5)
    expected_hard, expected_soft = np.zeros(2), np.zeros(2)
    for tokens, prob in [((0,), .5), ((1, 0), .25), ((1, 1), .25)]:
        hard = grad(theta, tokens, definition)
        soft = grad(theta, tokens, definition, soft=True, snapshot=source)
        expected_hard += prob*hard
        expected_soft += prob*soft
        total_def = replace(definition, normalization='total_token_nll')
        total_hard = grad(theta, tokens, total_def)
        if normalization == 'per_sequence_mean':
            np.testing.assert_allclose(hard, total_hard/len(tokens), atol=1e-15)
    np.testing.assert_allclose(expected_hard, expected_soft, atol=1e-15)


def test_conditional_length_rao_blackwell_and_snapshot_nonstationarity_counterexample():
    theta = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    source = theta.detach().clone()
    definition = LossDefinition('per_sequence_mean', soft_mode='conditional_length')
    hard_mean, soft_mean, naive_mean = np.zeros(2), np.zeros(2), np.zeros(2)
    for tokens, prob in [((0,), .5), ((1, 0), .25), ((1, 1), .25)]:
        # At root, EOS => L=1, continuation => L=2; next position has L=2.
        rho = torch.tensor([[1., .5], [.5, .5]])[:len(tokens)]
        hard_mean += prob*grad(theta, tokens, definition)
        soft = grad(theta, tokens, definition, soft=True, snapshot=source, rho=rho)
        soft_mean += prob*soft
        np.testing.assert_allclose(soft, [.125, 0.], atol=1e-15)
        naive_mean += prob*grad(theta, tokens, LossDefinition('total_token_nll'), soft=True, snapshot=source)/len(tokens)
    np.testing.assert_allclose(hard_mean, [.125, 0.], atol=1e-15)
    np.testing.assert_allclose(soft_mean, hard_mean, atol=1e-15)
    np.testing.assert_array_equal(naive_mean, [0., 0.])
    with pytest.raises(ValueError, match='conditional E'):
        grad(theta, (0,), definition, soft=True, snapshot=source)


@pytest.mark.parametrize('tokens', [(0,), (1, 0), (1, 1)])
def test_total_nll_soft_zero_at_snapshot_on_every_realised_prefix(tokens):
    theta = torch.tensor([.4, -.7], dtype=torch.float64, requires_grad=True)
    np.testing.assert_array_equal(grad(theta, tokens, LossDefinition('total_token_nll'),
                                     soft=True, snapshot=theta.detach()), [0., 0.])


def test_fixed_length_mean_soft_zero_and_wrong_teacher_prefix_counterexample():
    theta = torch.tensor([.4, -.7], dtype=torch.float64, requires_grad=True)
    definition = LossDefinition('per_sequence_mean', retention_scale=.5)
    np.testing.assert_array_equal(grad(theta, (1, 0), definition, soft=True, snapshot=theta.detach()), [0., 0.])
    definition = LossDefinition('total_token_nll')
    source = torch.zeros(2, dtype=torch.float64)
    expected = sum(prob*grad(theta, tok, definition) for tok, prob in [((0,), .5), ((1, 0), .25), ((1, 1), .25)])
    # Always using the longer teacher prefix doubles the second-prefix occupancy.
    wrong = grad(theta, (1, 0), definition, soft=True, snapshot=source)
    assert wrong[1] == pytest.approx(2*expected[1]) and abs(wrong[1]-expected[1]) > .01
