r"""Gaussian-Process over an action-value function Q.

This module implements a Gaussian-Process belief over an action-value function :math:`Q`.
It provides all necessary functionality to maintain and update this belief as new observations
are made.

We assume :math:`Q(\mathbf{X})` follows a Gaussian process with mean function :math:`m(\mathbf{X})` and
covariance function :math:`K(\mathbf{X}, \mathbf{X}')`. Equivalently, the decision maker has a prior
belief over :math:`Q` characterized by these functions. :math:`Q(\mathbf{X}) \sim \mathcal{N}(m(\mathbf{X}), K(\mathbf{X}, \mathbf{X}))`

By assumption, for any finite collection of points :math:`\mathbf{x}_1, \dots, \mathbf{x}_n`, the corresponding
function values :math:`Q(\mathbf{x}_1), \dots, Q(\mathbf{x}_n)` are jointly Gaussian.
The GP posterior given a set of observations :math:`\mathbf{r}` can be computed:

.. math::
    Q(\mathbf{X}_*) \mid \mathbf{r} \sim \mathcal{N}(\bar{m}(\mathbf{X}_*), \bar{K}(\mathbf{X}_*, \mathbf{X}_*))

where

.. math::
    \bar{m}(\mathbf{X}_*) &= m(\mathbf{X}_*) + K(\mathbf{X}_*, \mathbf{X}) [K(\mathbf{X}, \mathbf{X}) + \sigma_n^2 I]^{-1} (\mathbf{r} - m(\mathbf{X})) \\
    \bar{K}(\mathbf{X}_*, \mathbf{X}_*) &= K(\mathbf{X}_*, \mathbf{X}_*) - K(\mathbf{X}_*, \mathbf{X}) [K(\mathbf{X}, \mathbf{X}) + \sigma_n^2 I]^{-1} K(\mathbf{X}, \mathbf{X}_*)

:math:`\mathbf{X}_*` denotes the query points at which we want to evaluate the posterior.

To generalize to any input signal, we consider linear functionals of the form
:math:`r_i = \mathbf{w}_i^\top Q(\mathbf{X}_i) + \varepsilon_i`, where :math:`\varepsilon_i \sim \mathcal{N}(0, \sigma_i^2)`
is considered to be some independent Gaussian noise with variance :math:`\sigma_i^2` and could
plausibly be :math:`0`.

Note that a temporal-difference observation is a special case of this linear functional, with
:math:`r_i = Q(\mathbf{x}_{\text{decision}}) - \gamma\, Q(\mathbf{x}_{\text{outcome}})` and :math:`\varepsilon_i = 0`,
:math:`\mathbf{x}_{\text{decision}}` and :math:`\mathbf{x}_{\text{outcome}}` are the decision and outcome vectors, :math:`\mathbf{w}_i = (1, -\gamma)`.

The vector :math:`\mathbf{r}` reports the realized observations of the linear functionals.
Collecting all known weight vectors and input points, let :math:`\mathbf{X}_A` stack every input point. The linear system is then:

.. math::
    \mathbf{r} = W Q(\mathbf{X}_A) + \boldsymbol{\varepsilon}

Finally, :math:`\mathbf{r}` is a Gaussian random vector and jointly Gaussian with the function
values :math:`Q(\mathbf{X}_*)` at any set of query points :math:`\mathbf{X}_*`:

.. math::
    \begin{pmatrix}
    Q(\mathbf{X}_*) \\
    \mathbf{r}
    \end{pmatrix}
    \sim \mathcal{N}\left(
    \begin{pmatrix}
    m(\mathbf{X}_*) \\
    W m(\mathbf{X}_A)
    \end{pmatrix},
    \begin{pmatrix}
    K(\mathbf{X}_*, \mathbf{X}_*) & K(\mathbf{X}_*, \mathbf{X}_A) W^\top \\
    W K(\mathbf{X}_A, \mathbf{X}_*) & W K(\mathbf{X}_A, \mathbf{X}_A) W^\top + \operatorname{diag}(\sigma_1^2, \dots, \sigma_n^2)
    \end{pmatrix}
    \right)

Using the known formula for conditioning a jointly Gaussian vector on one of its blocks, the
posterior at the query points is

.. math::
    Q(\mathbf{X}_*) \mid \mathbf{r} \sim \mathcal{N}\left(\bar{m}(\mathbf{X}_*), \bar{K}(\mathbf{X}_*, \mathbf{X}_*)\right)

with

.. math::
    \bar{m}(\mathbf{X}_*) &= m(\mathbf{X}_*) + K(\mathbf{X}_*, \mathbf{X}_A) W^\top
        \left[W K(\mathbf{X}_A, \mathbf{X}_A) W^\top + \operatorname{diag}(\sigma_1^2, \dots, \sigma_n^2)\right]^{-1}
        \left(\mathbf{r} - W m(\mathbf{X}_A)\right) \\
    \bar{K}(\mathbf{X}_*, \mathbf{X}_*) &= K(\mathbf{X}_*, \mathbf{X}_*) - K(\mathbf{X}_*, \mathbf{X}_A) W^\top
        \left[W K(\mathbf{X}_A, \mathbf{X}_A) W^\top + \operatorname{diag}(\sigma_1^2, \dots, \sigma_n^2)\right]^{-1}
        W K(\mathbf{X}_A, \mathbf{X}_*)

We then can compute the posterior of :math:`Q(\mathbf{X}_*)` at any set of query points :math:`\mathbf{X}_*`.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .kernels import Kernel


@dataclass
class GPBeliefParameters:
    kernel: Kernel
    sigma_n: float = 0.01


class GPBelief:
    r"""Gaussian-Process Temporal-Difference belief over an action-value function Q.

    Args:
        gamma (float): Per-period discount used in the TD residual. Investment: ``BETA``.
            Entrepreneur: ``disc = BETA * (1 - rho_death)``.
        gp_params (GPBeliefParameters): Kernel function and jitter (observation noise).
        input_dim (int): Feature-vector width (e.g., 3 for ``(z, k, i)``; 4 for ``(z, omega, k', b')``).
        prior_mean_fn (callable, optional): ``X -> prior mean of Q`` at those features.
            Default: zero.
    """

    def __init__(
        self,
        gamma: float,
        gp_params: GPBeliefParameters,
        input_dim: int,
        prior_mean_fn: Callable | None = None,
    ):
        self.gamma = float(gamma)
        self.gp_params = gp_params
        self.kernel = gp_params.kernel
        self.sigma_n_sq = gp_params.sigma_n**2
        self.input_dim = int(input_dim)
        self.prior_mean_fn = (
            prior_mean_fn if prior_mean_fn else (lambda X: np.zeros(np.atleast_2d(X).shape[0]))
        )
        self.reset()

    def _functional_cov(self, X1, w1, X2, w2):
        r"""Calculate covariance between two linear functionals of Q.

        .. math::
            \operatorname{Cov}(w_1^\top Q(X_1), w_2^\top Q(X_2)) = w_1^\top K(X_1, X_2) w_2

        Args:
            X1 (np.ndarray): First set of input features.
            w1 (np.ndarray): Weight vector for the first functional.
            X2 (np.ndarray): Second set of input features.
            w2 (np.ndarray): Weight vector for the second functional.

        Returns:
            float: Covariance between the two linear functionals of Q.
        """
        return float(w1 @ self.kernel(X1, X2) @ w2)

    def _prior_obs(self):
        r"""Calculate prior mean of every observation.

        .. math::
            \mathbf{W} m(\mathbf{X}_{A})

        Returns:
            np.ndarray: Prior mean of every observation.
        """
        return self.W @ self.prior_mean_fn(self.X_atoms)

    def _k_star(self, X_q):
        r"""Calculate covariance between the observations and the query points Q(X_q).

        .. math::
            \operatorname{Cov}(\mathbf{r}, Q(\mathbf{X}_*)) = W K(\mathbf{X}_A, \mathbf{X}_*)

        Returns:
            np.ndarray: Covariance between the observations and the query points. Shape ``(n_obs, n_q)``.
        """
        return self.W @ self.kernel(self.X_atoms, X_q)

    def _first_functional(self, X, w, s):
        r"""Initialize the first functional observation.

        Args:
            X (np.ndarray): Input features for the functional observation.
            w (np.ndarray): Weight vector for the functional observation.
            s (float): Covariance of the functional observation.

        Returns:
            float: Covariance of the first functional observation.
        """
        self.C_inv = np.array([[1.0 / s]])
        self.X_atoms, self.W = X, w[None, :]
        return s

    def add_functional(self, X, w, r, noise, return_gain=False):
        r"""Create a new functional observation.

        .. math::
            r_i = \mathbf{w}_i^\top Q(\mathbf{X}_i) + \varepsilon_i, \qquad
            \varepsilon_i \sim \mathcal{N}(0, \sigma_i^2)

        Args:
            X (np.ndarray): Input features for the functional observation. :math:`\mathbf{X}_i`.
            w (np.ndarray): Weight vector for the functional observation. :math:`\mathbf{w}_i`.
            r (float): Observed value of the functional. :math:`r_i`.
            noise (float): Observation noise variance. :math:`\sigma_i^2`.
            return_gain (bool, optional): Whether to return the gain factor. Defaults to False.

        Returns:
            float or None: Gain factor if `return_gain` is True, otherwise None.
        """
        X = np.atleast_2d(X)
        w = np.ravel(w)
        s = self._functional_cov(X, w, X, w) + noise

        if len(self.R) == 0:
            S = self._first_functional(X, w, s)
        else:
            v = self.W @ self.kernel(self.X_atoms, X) @ w
            q = self.C_inv @ v
            S = max(s - float(v @ q), 1e-10)  # clip above zero
            self.C_inv = np.block(
                [
                    [self.C_inv + np.outer(q, q) / S, -q[:, None] / S],
                    [-q[None, :] / S, np.array([[1.0 / S]])],
                ]
            )
            M_old = self.X_atoms.shape[0]
            W = np.zeros((len(self.R) + 1, M_old + X.shape[0]))
            W[:-1, :M_old] = self.W
            W[-1, M_old:] = w
            self.W = W
            self.X_atoms = np.vstack([self.X_atoms, X])

        self.R = np.append(self.R, r)
        self.noise = np.append(self.noise, noise)
        self.alpha = self.C_inv @ (self.R - self._prior_obs())
        return (S / s) if return_gain else None

    def add_observation(self, x_dec, x_out, reward, return_gain=False):
        r"""Add an experience (GPTD) observation.

        .. math::
            r = Q(\mathbf{x}_\text{decision}) - \gamma\, Q(\mathbf{x}_\text{outcome}) + \varepsilon,
            \quad \varepsilon \sim \mathcal{N}(0, \sigma_n^2)

        Args:
            x_dec (np.ndarray): Decision features :math:`\mathbf{x}_\text{decision} = (\mathbf{s}_t, \mathbf{a}_t)`.
            x_out (np.ndarray): Outcome features :math:`\mathbf{x}_\text{outcome} = (\mathbf{s}_{t+1}, \mathbf{a}_{t+1})`.
            reward (float): Observed one-period reward.
            return_gain (bool, optional): Whether to return the gain factor. Defaults to False.

        Returns:
            float or None: Gain factor if ``return_gain`` is True, otherwise None.
        """
        X = np.vstack([np.atleast_2d(x_dec), np.atleast_2d(x_out)])
        w = np.array([1.0, -self.gamma])
        return self.add_functional(X, w, reward, self.sigma_n_sq, return_gain)

    def _predict_no_observations(self, X_q, return_std):
        r"""Prior mean and (optionally) prior std at ``\mathbf{X}_*``: :math:`m(\mathbf{X}_*),\ \sqrt{k(\mathbf{x}_*, \mathbf{x}_*)}`.

        This is used to provide the prior mean and variance for query points when no observations have been added yet.

        Args:
            X_q (np.ndarray): Query features, shape ``(n_q, d)``.
            return_std (bool): Whether to also return the prior std.

        Returns:
            np.ndarray or tuple: Prior mean, or ``(mean, std)`` if ``return_std``.
        """
        prior_q = self.prior_mean_fn(X_q)
        if return_std:
            return prior_q, np.sqrt(self.kernel.diag(X_q))
        return prior_q

    def predict(self, X_q, return_std=False):
        r"""Posterior mean and (optionally) marginal std of Q at query points.

        .. math::
            \bar{m}(\mathbf{X}_*) = m(\mathbf{X}_*) + k_*^\top \boldsymbol{\alpha}, \qquad
            \bar{\sigma}^2(\mathbf{x}_*) = k(\mathbf{x}_*, \mathbf{x}_*) - k_*^\top C^{-1} k_*

        Args:
            X_q (np.ndarray): Query points :math:`\mathbf{X}_*`, shape ``(n_q, d)``.
            return_std (bool, optional): Whether to also return the std. Defaults to False.

        Returns:
            np.ndarray or tuple: Posterior mean, or ``(mean, std)`` if ``return_std``.
        """
        X_q = np.atleast_2d(X_q)
        if len(self.R) == 0:
            return self._predict_no_observations(X_q, return_std)
        ks = self._k_star(X_q)
        post_mean = self.prior_mean_fn(X_q) + ks.T @ self.alpha
        if not return_std:
            return post_mean
        var = self.kernel.diag(X_q) - np.sum(ks * (self.C_inv @ ks), axis=0)
        return post_mean, np.sqrt(np.maximum(var, 0))

    def predict_full(self, X_query):
        r"""Posterior mean and full covariance of Q at query points.

        .. math::
            \bar{m}(\mathbf{X}_*) = m(\mathbf{X}_*) + k_*^\top \boldsymbol{\alpha}, \qquad
            \bar{K}(\mathbf{X}_*, \mathbf{X}_*) = K(\mathbf{X}_*, \mathbf{X}_*) - k_*^\top C^{-1} k_*

        Args:
            X_query (np.ndarray): Query points :math:`\mathbf{X}_*`, shape ``(n_q, d)``.

        Returns:
            tuple: ``(mean, cov)`` with shapes ``(n_q,)`` and ``(n_q, n_q)``.
        """
        X_q = np.atleast_2d(X_query)
        if len(self.R) == 0:
            return self.prior_mean_fn(X_q), self.kernel(X_q, X_q)
        ks = self._k_star(X_q)
        post_mean = self.prior_mean_fn(X_q) + ks.T @ self.alpha
        post_cov = self.kernel(X_q, X_q) - ks.T @ self.C_inv @ ks
        return post_mean, post_cov

    def reset(self):
        r"""Discard all observations and return to the prior."""
        self.X_atoms = np.empty((0, self.input_dim))
        self.W = np.empty((0, 0))
        self.noise = np.empty(0)
        self.R = np.empty(0)
        self.C_inv = np.empty((0, 0))
        self.alpha = np.empty(0)
