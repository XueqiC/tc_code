"""ONE definition of update and F for solve, trial, commit and counterfactuals."""
from dataclasses import dataclass

import numpy as np

from .immutable import Parameters, Array


@dataclass(frozen=True)
class TeachingObjective:
    problem: object

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

    def quadratic(self):
        p = self.problem
        return p.context.regularization*p.K.numpy(), -(p.U.numpy().T @ p.h.numpy())

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
