from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Oracle(Protocol):
    r"""Answers query or weighted contrast about :math:`Q` on a menu."""

    def query(self, X, w, noise, rng) -> float:
        r"""Return :math:`w^\top Q^*(X) + \eta`, :math:`\eta \sim N(0, \text{noise})`.

        Args:
            X (np.ndarray): Menu features, shape ``(N, d)``.
            w (np.ndarray): Query weights or contrast weights, shape ``(N,)``.
            noise (float): Variance :math:`\sigma^2` the plan priced uncertainty from rational inattention.
            rng (np.random.Generator): Source of the answer noise.

        Returns:
            float: The realised answer.
        """
        ...


class VFIOracle:
    def __init__(self, vfi_agent, env, *, gamma=None):
        self.vfi_agent = vfi_agent
        self.env = env
        self.gamma = env.p.disc if gamma is None else float(gamma)

    def q(self, X) -> np.ndarray:
        r"""True action values at the menu features ``X``, shape ``(N,)``."""
        return np.ravel(self.vfi_agent.action_value(X, gamma=self.gamma))

    def query(self, X, w, noise, rng) -> float:
        r"""Project the true :math:`Q^*` on ``w`` and add the priced answer noise.

        Args:
            X (np.ndarray): Menu features, shape ``(N, d)``.
            w (np.ndarray): Contrast weights, shape ``(N,)``.
            noise (float): Answer variance :math:`\sigma^2`; ``0`` answers exactly.
            rng (np.random.Generator): Source of the answer noise.

        Returns:
            float: :math:`w^\top Q^*(X) + \eta`.
        """
        return float(np.ravel(w) @ self.q(X) + rng.normal(0.0, np.sqrt(noise)))
