from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import quantecon as qe

Number = Union[float, np.ndarray]


class Utility(ABC):
    """Per-period utility u(c) with marginal utility and its inverse (for EGM)."""

    @abstractmethod
    def __call__(self, c: Number) -> Number:
        """u(c)."""
        ...

    @abstractmethod
    def prime(self, c: Number) -> Number:
        """u'(c)."""
        ...

    @abstractmethod
    def prime_inv(self, x: Number) -> Number:
        """(u')^{-1}(x)."""
        ...


class CRRA(Utility):
    """u(c) = c^(1-gamma)/(1-gamma); log if gamma == 1. u'(c) = c^(-gamma)."""

    def __init__(self, gamma: float):
        self.gamma = gamma

    def __call__(self, c: Number) -> Number:
        c = np.maximum(c, 1e-12)
        if abs(self.gamma - 1.0) < 1e-9:
            return np.log(c)
        return c ** (1.0 - self.gamma) / (1.0 - self.gamma)

    def prime(self, c: Number) -> Number:
        return np.maximum(c, 1e-12) ** (-self.gamma)

    def prime_inv(self, x: Number) -> Number:
        return np.maximum(x, 1e-12) ** (-1.0 / self.gamma)


class Production(ABC):
    r"""Pure technology :math:`z f(k)`."""

    @abstractmethod
    def output(self, *, z: Number, k: Number) -> Number:
        r"""Output :math:`z f(k)`."""
        ...

    @abstractmethod
    def marginal_product(self, *, z: Number, k: Number) -> Number:
        r"""Marginal product :math:`z f'(k)`."""
        ...

    def __call__(self, *, z: Number, k: Number) -> Number:
        return self.output(z=z, k=k)


class CapitalCobbDouglas(Production):
    r""":math:`z f(k) = z k^{\alpha}`, :math:`\alpha \in (0,1)`."""

    def __init__(self, alpha: float):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.alpha = alpha

    def output(self, *, z: Number, k: Number) -> Number:
        k = np.maximum(k, 1e-12)
        return z * k**self.alpha

    def marginal_product(self, *, z: Number, k: Number) -> Number:
        k = np.maximum(k, 1e-12)
        return self.alpha * z * k ** (self.alpha - 1.0)


class AdjustmentCosts(ABC):
    @abstractmethod
    def __call__(self, *, i: Number, k: Number) -> Number:
        r"""Cost :math:`\psi(i, k)` of investing ``i`` at capital ``k``."""
        ...


class NoAdjustmentCosts(AdjustmentCosts):
    def __call__(self, *, i: Number, k: Number) -> Number:
        return np.zeros_like(np.asarray(i, float))


class QuadraticAdjustmentCosts(AdjustmentCosts):
    r""":math:`\psi(i,k) = \tfrac{\kappa}{2}\, i^2 / k`."""

    def __init__(self, kappa: float):
        self.kappa = kappa

    def __call__(self, *, i: Number, k: Number) -> Number:
        k_safe = np.maximum(k, 1e-8)
        return (self.kappa / 2.0) * (np.asarray(i, float) ** 2 / k_safe)


class Productivity(ABC):
    r"""How ``z`` is initialised and evolved. Vectorized over agents; rng is explicit."""

    @abstractmethod
    def initial(self, *, n: int, rng: np.random.Generator) -> np.ndarray:
        r"""Draw ``n`` initial ``z`` values (one per agent)."""
        ...

    @abstractmethod
    def step(self, *, z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        r"""Evolve a vector of ``z`` one period (identity for permanent types)."""
        ...

    @abstractmethod
    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Discretised support and transition matrix ``(z_grid, P)`` for VFI."""
        ...


class PermanentProductivity(Productivity):
    r"""Firm draws ``z`` once at birth from a fixed set and keeps it forever.

    ``grid()`` returns ``P = I`` (each type absorbing), so VFI solves each type's
    fixed-``z`` problem independently.
    """

    def __init__(self, z_values, probs=None):
        self.z_values = np.asarray(z_values, float)
        k = len(self.z_values)
        self.probs = np.full(k, 1.0 / k) if probs is None else np.asarray(probs, float)

    def initial(self, *, n, rng):
        return rng.choice(self.z_values, size=n, p=self.probs)

    def step(self, *, z, rng):
        return z

    def grid(self):
        return self.z_values, np.eye(len(self.z_values))


class MarkovAR1(Productivity):
    r"""Persistent log-AR(1): ``log z' = rho log z + eps``, Tauchen-discretised."""

    def __init__(self, *, rho, sigma_eps, n_z, n_std=3):
        self.rho, self.sigma_eps, self.n_z = rho, sigma_eps, n_z
        mc = qe.markov.approximation.tauchen(n_z, rho, sigma_eps, mu=0, n_std=n_std)
        self._z_grid = np.exp(mc.state_values)
        self._P = mc.P
        self._stat = self._stationary(self._P)

    @staticmethod
    def _stationary(P):
        vals, vecs = np.linalg.eig(P.T)
        v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
        return v / v.sum()

    def initial(self, *, n, rng):
        return rng.choice(self._z_grid, size=n, p=self._stat)

    def step(self, *, z, rng):
        eps = rng.normal(0.0, self.sigma_eps, size=np.shape(z))
        return np.exp(self.rho * np.log(np.maximum(z, 1e-12)) + eps)

    def grid(self):
        return self._z_grid, self._P


class IIDDraw(Productivity):
    r"""``z`` redrawn i.i.d. each period from a fixed distribution (no persistence)."""

    def __init__(self, z_values, probs=None):
        self.z_values = np.asarray(z_values, float)
        k = len(self.z_values)
        self.probs = np.full(k, 1.0 / k) if probs is None else np.asarray(probs, float)

    def initial(self, *, n, rng):
        return rng.choice(self.z_values, size=n, p=self.probs)

    def step(self, *, z, rng):
        return rng.choice(self.z_values, size=np.shape(z), p=self.probs)

    def grid(self):
        return self.z_values, np.tile(self.probs, (len(self.z_values), 1))


class DeathProcess(ABC):
    @abstractmethod
    def draw(self, *, n: int, rng: np.random.Generator) -> np.ndarray: ...

    @property
    @abstractmethod
    def survival_rate(self) -> float:
        r"""Per-period survival probability :math:`1-\rho`, used in the discount factor."""
        ...


class NoDeath(DeathProcess):
    def draw(self, *, n, rng):
        return np.zeros(n, dtype=bool)

    @property
    def survival_rate(self) -> float:
        return 1.0


class ConstantHazard(DeathProcess):
    def __init__(self, rho_death: float):
        if not 0.0 <= rho_death < 1.0:
            raise ValueError(f"rho_death must be in [0,1), got {rho_death}")
        self.rho_death = rho_death

    def draw(self, *, n, rng):
        return rng.uniform(0.0, 1.0, size=n) < self.rho_death

    @property
    def survival_rate(self) -> float:
        return 1.0 - self.rho_death
