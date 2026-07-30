import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .environment.standard_investment import InvestmentParameters, InvestmentEnvironment
    from .firm import RationalInvestmentAgent


class GPPrior(ABC):
    def __init__(self, env: 'InvestmentEnvironment'):
        self.p = env.p
        self.env = env

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        z, k, i = X[:, 0], X[:, 1], X[:, 2]
        k_next = (1.0 - self.p.DELTA) * k + i
        b_next = np.array([self.env.optimal_b_next(kn) for kn in k_next])

        flow = np.array([
            self.env.gp_observation(zz, kk, ii, bn)
            for zz, kk, ii, bn in zip(z, k, i, b_next)
        ])
        return flow + self.p.BETA * self._continuation(k_next)

    @abstractmethod
    def _continuation(self, k_next: np.ndarray) -> np.ndarray:
        pass


class ZeroPrior(GPPrior):
    def _continuation(self, k_next):
        return np.zeros_like(k_next)


class PerpetuityPrior(GPPrior):
    def _continuation(self, k_next):
        k_safe = np.maximum(k_next, 1e-8)
        i_maint = self.p.DELTA * k_safe
        adj = (self.p.KAPPA / 2.0) * (i_maint**2 / k_safe) if self.p.KAPPA > 0 else 0.0
        d_ss = k_safe**self.p.ALPHA - i_maint - adj  # z=1 at steady state
        return d_ss / (1.0 - self.p.BETA)

    
class TrueValueFunctionPrior(GPPrior):
    """
    A prior that decomposes Q into an exact flow payoff from the chosen
    investment action and a discounted continuation value at the resulting
    capital stock k' = (1 - delta)*k + i. The continuation value is
    approximated by a power-law fitted to the rational agent's value
    function V*(k) evaluated at z=1.
    """
    def __init__(self, rational_agent):
        super().__init__(rational_agent.env)
        iz_idx = int(np.argmin(np.abs(rational_agent.env.z_grid - 1.0)))
        self._v_interp = interp1d(
            rational_agent.env.k_grid,
            rational_agent.v[iz_idx, :],
            kind='cubic',
            fill_value='extrapolate'
        )

    def _continuation(self, k_next):
        return self._v_interp(np.maximum(k_next, 1e-8))

    def __call__(self, X):
        X = np.atleast_2d(X)
        z, k, i = X[:, 0], X[:, 1], X[:, 2]
        k_next = (1.0 - self.p.DELTA) * k + i
        b_next = np.array([self.env.optimal_b_next(kn) for kn in k_next])

        flow = np.array([
            self.env.gp_observation(zz, kk, ii, bn)
            for zz, kk, ii, bn in zip(z, k, i, b_next)
        ])
        return flow + self.p.BETA * self._continuation(k_next)
class Kernel(ABC):
    @abstractmethod
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def diag(self, X: np.ndarray) -> np.ndarray: ...


class RBFKernel(Kernel):
    r"""Squared-exponential kernel. `length_scales` length = input dimension."""
    def __init__(self, sigma0: float, length_scales: list[float]):
        self.sigma0_sq = sigma0 ** 2
        self.length_scales = np.asarray(length_scales, float)

    def __call__(self, X1, X2):
        X1s = np.atleast_2d(X1) / self.length_scales
        X2s = np.atleast_2d(X2) / self.length_scales
        sq = (np.sum(X1s**2, 1)[:, None] + np.sum(X2s**2, 1) - 2 * X1s @ X2s.T)
        return self.sigma0_sq * np.exp(-0.5 * np.maximum(sq, 0.0))

    def diag(self, X):
        return np.full(np.atleast_2d(X).shape[0], self.sigma0_sq)


class LaplacianKernel(Kernel):
    r"""Matern-1/2 kernel (Ilut-Vachev original). `length_scales` length = input dim."""
    def __init__(self, sigma0: float, length_scales: list[float]):
        self.sigma0_sq = sigma0 ** 2
        self.length_scales = np.asarray(length_scales, float)

    def __call__(self, X1, X2):
        X1s = np.atleast_2d(X1) / self.length_scales
        X2s = np.atleast_2d(X2) / self.length_scales
        sq = (np.sum(X1s**2, 1)[:, None] + np.sum(X2s**2, 1) - 2 * X1s @ X2s.T)
        return self.sigma0_sq * np.exp(-np.sqrt(np.maximum(sq, 0.0)))

    def diag(self, X):
        return np.full(np.atleast_2d(X).shape[0], self.sigma0_sq)


@dataclass
class GPBeliefParameters:
    kernel: Kernel
    sigma_n: float = 0.01


class GPBelief:
    r"""Gaussian-Process Temporal-Difference belief over an action-value function Q.

    The agent never observes Q; it observes the flow payoff u(c) each step, which is the
    TD residual u = Q(decision) - gamma * Q(outcome). This class places a GP prior on Q
    and forms the induced covariance of the *observable* residuals (the 4-term functional
    covariance). Works for any feature dimension and any discount `gamma`.

    Parameters
    ----------
    gamma : float
        Per-period discount used in the TD residual. Investment: BETA. Entrepreneur:
        disc = BETA*(1-rho_death).
    gp_params : GPBeliefParameters
        Kernel + observation noise.
    input_dim : int
        Feature-vector width (3 for (z,k,i); 4 for (z,omega,k',b')).
    prior_mean_fn : callable, optional
        X -> prior mean of Q at those features. Default: zero.
    """

    def __init__(self, gamma: float, gp_params: GPBeliefParameters,
                 input_dim: int, prior_mean_fn: Callable = None):
        self.gamma = float(gamma)
        self.gp_params = gp_params
        self.kernel = gp_params.kernel
        self.sigma_n_sq = gp_params.sigma_n ** 2
        self.input_dim = int(input_dim)
        self.prior_mean_fn = prior_mean_fn if prior_mean_fn else (lambda X: np.zeros(np.atleast_2d(X).shape[0]))

        self.X_dec = np.empty((0, self.input_dim))     # decision points (s_t, a_t)
        self.X_out = np.empty((0, self.input_dim))     # outcome points  (s_{t+1}, a_{t+1})
        self.Y = np.empty(0)                           # observed flow payoffs u(c_t)
        self.C_inv = np.empty((0, 0))
        self.alpha = np.empty(0)

    def _functional_cov(self, Xd1, Xo1, Xd2, Xo2):
        g = self.gamma
        k_dd = self.kernel(Xd1, Xd2)
        k_do = self.kernel(Xd1, Xo2)
        k_od = self.kernel(Xo1, Xd2)
        k_oo = self.kernel(Xo1, Xo2)
        return k_dd - g * k_do - g * k_od + (g ** 2) * k_oo

    def _prior_residual(self, Xd, Xo):
        r"""Prior mean of the TD residual: m(d) - γ m(o)."""
        return self.prior_mean_fn(Xd) - self.gamma * self.prior_mean_fn(Xo)

    def _first_observation(self, x_dec, x_out, flow):
        v_self = self._functional_cov(x_dec, x_out, x_dec, x_out)[0, 0]
        self.C_inv = np.array([[1.0 / (v_self + self.sigma_n_sq)]])
        self.X_dec, self.X_out, self.Y = x_dec, x_out, np.array([flow])
        m0 = float(np.ravel(self._prior_residual(x_dec, x_out))[0])
        self.alpha = self.C_inv @ np.array([flow - m0])

    def add_observation(self, x_dec, x_out, flow, return_gain=False):
        r"""Incorporate one TD tuple: decision features, outcome features, observed flow u(c).
        Recursively updates the inverse Gram matrix and prediction weights."""
        x_dec = np.atleast_2d(x_dec); x_out = np.atleast_2d(x_out)
        if len(self.Y) == 0:
            self._first_observation(x_dec, x_out, flow)
            return None

        v = self._functional_cov(x_dec, x_out, self.X_dec, self.X_out).T
        w = self._functional_cov(x_dec, x_out, x_dec, x_out)[0, 0] + self.sigma_n_sq
        q = self.C_inv @ v
        S = max(w - float((v.T @ q)[0, 0]), 1e-10)

        self.C_inv = np.block([[self.C_inv + (q @ q.T) / S, -q / S],
                               [-q.T / S,                    np.array([[1.0 / S]])]])
        self.X_dec = np.vstack([self.X_dec, x_dec])
        self.X_out = np.vstack([self.X_out, x_out])
        self.Y = np.append(self.Y, flow)

        M = self._prior_residual(self.X_dec, self.X_out)
        self.alpha = self.C_inv @ (self.Y - M)
        return (S / w) if return_gain else None

    def _k_star(self, X_q):
        r"""Cross-covariance between Q(X_q) and the observed residuals: k(d,q) - γ k(o,q)."""
        return self.kernel(self.X_dec, X_q) - self.gamma * self.kernel(self.X_out, X_q)

    def _predict_no_observations(self, X_q, return_std):
        prior_q = self.prior_mean_fn(X_q)
        if return_std:
            return prior_q, np.sqrt(self.kernel.diag(X_q))
        return prior_q

    def predict(self, X_query, return_std=False):
        X_q = np.atleast_2d(X_query)
        if len(self.Y) == 0:
            return self._predict_no_observations(X_q, return_std)
        ks = self._k_star(X_q)
        post_mean = self.prior_mean_fn(X_q) + ks.T @ self.alpha
        if not return_std:
            return post_mean
        var = self.kernel.diag(X_q) - np.sum(ks * (self.C_inv @ ks), axis=0)
        return post_mean, np.sqrt(np.maximum(var, 0))

    def predict_full(self, X_query):
        X_q = np.atleast_2d(X_query)
        if len(self.Y) == 0:
            return self.prior_mean_fn(X_q), self.kernel(X_q, X_q)
        ks = self._k_star(X_q)
        post_mean = self.prior_mean_fn(X_q) + ks.T @ self.alpha
        post_cov = self.kernel(X_q, X_q) - ks.T @ self.C_inv @ ks
        return post_mean, post_cov

    def reset(self):
        self.X_dec = np.empty((0, self.input_dim))
        self.X_out = np.empty((0, self.input_dim))
        self.Y = np.empty(0)
        self.C_inv = np.empty((0, 0))
        self.alpha = np.empty(0)