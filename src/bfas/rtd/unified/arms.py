"""P1 interventions on ONE immutable teaching problem and its full objective."""
from dataclasses import dataclass

import numpy as np

from .solver import solve_exact


@dataclass(frozen=True)
class Arm:
    estimator: str = 'cv'
    restriction: str = 'source'
    geometry: str = 'full'
    pairing: str = 'correct'
    trainer: str = 'unified'


ARMS = {
    'D0': Arm(trainer='sft_kl'),
    'D1': Arm(trainer='v1_1'),
    'D2': Arm(restriction='state_scalar'),
    'D3': Arm(),
    'D3-nocross': Arm(geometry='diagonal'),
    'D3-raw': Arm(estimator='raw'),
    'D3-shuffle': Arm(pairing='shuffle'),
    'D3-fixedmean': Arm(restriction='fixed_mean'),
}


def equality_rows(problem, restriction):
    n = len(problem.coordinates)
    rows = []
    states = {e.slot_id: e.state_hash for e in problem.exposure}
    groups = {}
    for j, c in enumerate(problem.coordinates):
        # P1 has one purchased teacher version per slot. Repeated occurrences
        # of a state share the D2 scalar as well.
        key = states[c.slot_id] if restriction == 'state_scalar' else c.slot_id
        groups.setdefault(key, []).append(j)
    if restriction == 'state_scalar':
        for ids in groups.values():
            if not np.allclose(problem.a_ref.numpy()[ids], problem.a_ref.numpy()[ids[0]]):
                raise ValueError('state-scalar reference must already be tied')
            for j in ids[1:]:
                row = np.zeros(n); row[j], row[ids[0]] = 1., -1.
                rows.append(row)
    elif restriction == 'fixed_mean':
        for ids in groups.values():
            row = np.zeros(n); row[ids] = 1.
            rows.append(row)
    elif restriction != 'source':
        raise ValueError('unknown coefficient restriction')
    return np.asarray(rows).reshape(-1, n) if n else np.zeros((0, 0))


def source_permutation(problem, seed):
    """Permute each state's source pairing, preserving teacher evidence/dose.

    A predeclared random within-pair swap is the two-source permutation null.
    No rejection to force a different action or a nonidentity permutation.
    """
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(problem.coordinates))
    groups = {}
    for j, c in enumerate(problem.coordinates):
        groups.setdefault((c.slot_id, c.evidence_key), []).append(j)
    for ids in groups.values():
        permutation[ids] = rng.permutation(ids)
    return permutation


def solve_arm(problem, arm, *, seed=0):
    preset = ARMS[arm]
    if preset.trainer != 'unified' or preset.estimator != problem.context.estimator:
        raise ValueError('arm does not match the teaching estimator/trainer')
    K = problem.K.numpy()
    control_gram = np.diag(np.diag(K)) if preset.geometry == 'diagonal' else None
    solution = solve_exact(problem, equality=equality_rows(problem, preset.restriction),
                           control_gram=control_gram)
    permutation = (source_permutation(problem, seed) if preset.pairing == 'shuffle'
                   else np.arange(len(problem.coordinates)))
    coefficients = solution.coefficients.numpy()[permutation]
    return solution, coefficients, permutation
