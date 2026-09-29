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

    The agent observes the flow payoff u(c) each step, which is the TD residual
    :math:`u = Q(\text{decision}) - \gamma\, Q(\text{outcome})`. This class places
    a GP prior on Q and forms the induced covariance of the observable residuals.

    Every observation is a linear functional of Q,

    .. math::
        y_i = w_i^\top Q(X_i) + \varepsilon_i, \qquad \varepsilon_i \sim N(0, \sigma_i^2),

    so :math:`\operatorname{Cov}(y_i, y_j) = w_i^\top K(X_i, X_j)\, w_j`.

    * Experience: :math:`X_i = [x_{\text{dec}}; x_{\text{out}}]`, :math:`w_i = (1, -\gamma)`,
      :math:`\sigma_i^2 = \sigma_n^2`.
    * Reasoning: :math:`X_i = X_q` (menu), :math:`w_i = v_i`,
      :math:`\sigma_i^2 = \lambda_i \theta / (\lambda_i - \theta)`.

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
            W m(X_{\text{atoms}})

        Args:
            W (np.ndarray): Weight matrix for the observations.
            X_atoms (np.ndarray): Input features for the atomic observations.

        Returns:
            np.ndarray: Prior mean of every observation.
        """
        return self.W @ self.prior_mean_fn(self.X_atoms)

    def _k_star(self, X_q):
        r"""Calculate covariance between the observations and the query points Q(X_q).

        .. math::
            \operatorname{Cov}(\text{observations}, Q(X_q)) = W K(X_{\text{atoms}}, X_q)

        Returns:
            np.ndarray: Covariance between the observations and the query points.
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

    def add_functional(self, X, w, y, noise, return_gain=False):
        r"""Create a new functional observation.

        .. math::
            y = w @ Q(X) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \text{noise})

        Args:
            X (np.ndarray): Input features for the functional observation.
            w (np.ndarray): Weight vector for the functional observation.
            y (float): Observed value of the functional.
            noise (float): Observation noise variance.
            return_gain (bool, optional): Whether to return the gain factor. Defaults to False.

        Returns:
            float or None: Gain factor if `return_gain` is True, otherwise None.
        """
        X = np.atleast_2d(X)
        w = np.ravel(w)
        s = self._functional_cov(X, w, X, w) + noise

        if len(self.Y) == 0:
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
            W = np.zeros((len(self.Y) + 1, M_old + X.shape[0]))
            W[:-1, :M_old] = self.W
            W[-1, M_old:] = w
            self.W = W
            self.X_atoms = np.vstack([self.X_atoms, X])

        self.Y = np.append(self.Y, y)
        self.noise = np.append(self.noise, noise)
        self.alpha = self.C_inv @ (self.Y - self._prior_obs())
        return (S / s) if return_gain else None

    def add_observation(self, x_dec, x_out, reward, return_gain=False):
        r"""Add an experience (GPTD) observation.

        .. math::
            r = Q(x_\text{dec}) - \gamma\, Q(x_\text{out}) + \epsilon,
            \quad \epsilon \sim \mathcal{N}(0, \sigma_n^2)

        Args:
            x_dec (np.ndarray): Decision features :math:`(s_t, a_t)`.
            x_out (np.ndarray): Outcome features :math:`(s_{t+1}, a_{t+1})`.
            reward (float): Observed one-period reward.
            return_gain (bool, optional): Whether to return the gain factor. Defaults to False.

        Returns:
            float or None: Gain factor if ``return_gain`` is True, otherwise None.
        """
        X = np.vstack([np.atleast_2d(x_dec), np.atleast_2d(x_out)])
        w = np.array([1.0, -self.gamma])
        return self.add_functional(X, w, reward, self.sigma_n_sq, return_gain)

    def _predict_no_observations(self, X_q, return_std):
        r"""Prior mean and (optionally) prior std at ``X_q``: :math:`m(X_q),\ \sqrt{k(x, x)}`.

        This is used to provide the prior mean and variance for query points when no observations have been added yet.
        """
        prior_q = self.prior_mean_fn(X_q)
        if return_std:
            return prior_q, np.sqrt(self.kernel.diag(X_q))
        return prior_q

    def predict(self, X_query, return_std=False):
        r"""Posterior mean and (optionally) marginal std of Q at query points.

        .. math::
            \hat{Q}(X_q) = m(X_q) + k_*^\top \alpha, \qquad
            \sigma^2(x_q) = k(x_q, x_q) - k_*^\top C^{-1} k_*

        Args:
            X_query (np.ndarray): Query features, shape ``(n_q, d)``.
            return_std (bool, optional): Whether to also return the std. Defaults to False.

        Returns:
            np.ndarray or tuple: Posterior mean, or ``(mean, std)`` if ``return_std``.
        """
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
        r"""Posterior mean and full covariance of Q at query points.

        .. math::
            \hat{Q}(X_q) = m(X_q) + k_*^\top \alpha, \qquad
            \Sigma_q = K(X_q, X_q) - k_*^\top C^{-1} k_*

        Args:
            X_query (np.ndarray): Query features, shape ``(n_q, d)``.

        Returns:
            tuple: ``(mean, cov)`` with shapes ``(n_q,)`` and ``(n_q, n_q)``.
        """
        X_q = np.atleast_2d(X_query)
        if len(self.Y) == 0:
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
        self.Y = np.empty(0)
        self.C_inv = np.empty((0, 0))
        self.alpha = np.empty(0)
