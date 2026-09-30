"""Module for computing prior means to initialize the GP."""

from abc import ABC, abstractmethod

import numpy as np
from scipy.interpolate import interp1d

try:  # warnings.deprecated is 3.13+; typing_extensions backports it for 3.11/3.12
    from warnings import deprecated
except ImportError:
    from typing_extensions import deprecated


class PriorMean(ABC):
    r"""Abstract base class for prior mean functions."""

    def __init__(self, env):
        self.p = env.p
        self.env = env

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        z, k, i = X[:, 0], X[:, 1], X[:, 2]
        k_next = (1.0 - self.p.DELTA) * k + i
        b_next = np.array([self.env.optimal_b_next(kn) for kn in k_next])

        flow = np.array(
            [self.env.gp_observation(zz, kk, ii, bn) for zz, kk, ii, bn in zip(z, k, i, b_next)]
        )
        return flow + self.p.BETA * self._continuation(k_next)

    @abstractmethod
    def _continuation(self, k_next: np.ndarray) -> np.ndarray:
        pass


class ZeroPrior(PriorMean):
    """Trivial prior mean, zero continuation."""

    def _continuation(self, k_next):
        return np.zeros_like(k_next)


@deprecated(
    "PerpetuityPrior targets the investment environment (z, k, i); use entrepreneur priors instead."
)
class PerpetuityPrior(PriorMean):
    """Perpetuity prior mean, continuation value under heuristic holding fixed capital forever."""

    def _continuation(self, k_next):
        k_safe = np.maximum(k_next, 1e-8)
        i_maint = self.p.DELTA * k_safe
        adj = (self.p.KAPPA / 2.0) * (i_maint**2 / k_safe) if self.p.KAPPA > 0 else 0.0
        d_ss = k_safe**self.p.ALPHA - i_maint - adj  # z=1 at steady state
        return d_ss / (1.0 - self.p.BETA)


class VFIPrior(PriorMean):
    r"""Prior mean of :math:`Q(z, \omega, k', b')` from a solved VFI agent.

    .. math::
        Q(z, \omega, k', b') = u(c) + \gamma \sum_{j} P_{i(z),\, j}\, V_j(\omega'),
        \qquad c = \omega + b' - k', \quad \omega' = z f(k') + (1-\delta)k' - (1+R)b'.

    Wraps ``vfi.action_value`` as a ``prior_mean_fn`` for the GP.

    Args:
        vfi (VFIEntrepreneurAgent): Fitted agent supplying ``V`` and ``env.P``.
        gamma (float, optional): Discount :math:`\gamma`; defaults to ``env.p.disc``.
        prior_z_index (int, optional): Condition the continuation on ``z_grid[prior_z_index]``
            rather than the query's ``z`` (a mis-calibrated type prior). ``None`` is correct.
    """

    def __init__(self, rational_agent):
        super().__init__(rational_agent.env)
        iz_idx = int(np.argmin(np.abs(rational_agent.env.z_grid - 1.0)))
        self._v_interp = interp1d(
            rational_agent.env.k_grid,
            rational_agent.v[iz_idx, :],
            kind="cubic",
            fill_value="extrapolate",
        )

    def _continuation(self, k_next):
        return self._v_interp(np.maximum(k_next, 1e-8))

    def __call__(self, X):
        X = np.atleast_2d(X)
        z, k, i = X[:, 0], X[:, 1], X[:, 2]
        k_next = (1.0 - self.p.DELTA) * k + i
        b_next = np.array([self.env.optimal_b_next(kn) for kn in k_next])

        flow = np.array(
            [self.env.gp_observation(zz, kk, ii, bn) for zz, kk, ii, bn in zip(z, k, i, b_next)]
        )
        return flow + self.p.BETA * self._continuation(k_next)


def make_vfi_prior(vfi, *, gamma=None, prior_z_index=None):
    def prior(X):
        return vfi.action_value(X, gamma=gamma, prior_z_index=prior_z_index)

    return prior
