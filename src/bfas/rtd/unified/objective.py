"""ONE definition of update and F for solve, trial, commit and counterfactuals."""
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .immutable import Parameters, Array


@dataclass(frozen=True)
class TeachingObjective:
    problem: object
    control_gram: object = None

    def increment(self, coefficients):
        p = self.problem
        a = p.check_coefficients(coefficients)
        # Stable old-column accumulation makes an appended zero-use teacher
        # recover the exact same floating-point increment, not just its limit.
        result = p.base_increment.numpy().copy()
        for j, value in enumerate(a):
            if value != 0:
                result += value*p.U.numpy()[:, j]
        return result

    def parameters(self, coefficients):
        layout = self.problem.context.theta
        values = layout.values.numpy() + self.increment(coefficients)
        # Match the actual target parameter dtype, including its rounding.
        return Parameters.of(layout.tensors(values=values, requires_grad=False))

    def displacement(self, coefficients):
        p = self.problem
        a = p.check_coefficients(coefficients)
        return p.U.numpy() @ (a-p.a_ref.numpy())

    @cached_property
    def _quadratic(self):
        p = self.problem
        K = p.K.numpy() if self.control_gram is None else self.control_gram
        # U may have millions of rows. Project h once per immutable objective,
        # not on every auxiliary-QP callback. Cached arrays are still read-only.
        Q, linear = p.context.regularization*K, -(p.U.numpy().T @ p.h.numpy())
        if not np.isfinite(Q).all() or not np.isfinite(linear).all():
            raise ValueError('QP coefficients overflowed; rescale the frozen problem explicitly')
        return Array.of(Q).numpy(), Array.of(linear).numpy()

    def quadratic(self):
        return self._quadratic

    def value(self, coefficients):
        p = self.problem
        x = p.check_coefficients(coefficients)-p.a_ref.numpy()
        Q, linear = self.quadratic()
        return float(-linear @ x - .5*x @ Q @ x - p.epsilon.numpy() @ np.abs(x))

    def auxiliary_value_gradient(self, y):
        """Convex minimization form, t >= |x|. Same Q/linear, no copied F."""
        p = self.problem
        n = len(p.coordinates)
        x, t = y[:n], y[n:]
        Q, linear = self.quadratic()
        value = .5*x @ Q @ x + linear @ x + p.epsilon.numpy() @ t
        return float(value), np.r_[Q @ x+linear, p.epsilon.numpy()]

    def regret(self, proposed, exact):
        return self.value(exact)-self.value(proposed)
