import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class Signal:
    r"""Immutable object representing a single contrast question about :math:`Q` on a menu, and the precision it is acquired at.

    The oracle returns :math:`y = w^\top Q^*(X) + \eta`, :math:`\eta \sim N(0, \sigma^2)`, with
    :math:`\sigma^2 = \lambda\theta/(\lambda-\theta)` set so that the posterior eigenvalue falls
    to the water level :math:`\theta`.

    The weights are the eigenvectors :math:`\psi` of the contrast matrix
    :math:`\tilde K = W^{1/2} P_W \Sigma P_W^\top W^{1/2}`, rescaled by cell measure,
    :math:`w = W^{1/2}\psi`; on a uniform mesh :math:`w \propto \psi`. Then
    :math:`\sum_j w_j Q_j \approx \int \phi\, Q\, \mathrm{d}c` is mesh-invariant and
    :math:`\sum_j w_j = 0`.

    Args:
        w (np.ndarray): Node weights, shape ``(N,)``; sum to zero.
        noise (float): Signal noise variance :math:`\sigma^2`.
    """

    w: np.ndarray
    noise: float


@dataclass(frozen=True)
class Plan:
    r"""Immutable object representing the agent's one period reasoning plan.

    Args:
        delta (float): Temperature :math:`\delta_t`.
        theta (float): Water level :math:`\theta_t`.
        signals (tuple): The reasoning :class:`Signal` s to acquire this period.
    """

    delta: float
    theta: float
    signals: tuple[Signal, ...] = field(default=())


@dataclass(frozen=True)
class Decision:
    r"""Immutable object representing the agent's one-period decision.

    Args:
        probs (np.ndarray): Choice probabilities :math:`p_t \propto e^{\hat Q_t/\delta_t}\Delta`,
            shape ``(N,)``, from the post-reasoning mean.
        plan (Plan): The agent's reasoning plan that produced the choice probabilities.
    """

    probs: np.ndarray
    plan: Plan


class Planner:
    r"""Builds and solves the experience-reasoning plan for the agent.

    Jointly solves for :math:`\delta_t` and :math:`\theta_t` given the menu and the reasoning cost.

    Args:
        kappa (float): Shadow price of one unit of reasoning. Alternatively, regularization parameter for reasoning.
        h (float): Entropy-floor scaling parameter.
    """

    def __init__(self, kappa: float, h: float):
        self.kappa = float(kappa)
        self.h = float(h)

    @staticmethod
    def softmax(q, delta, areas):
        r"""Choice probabilities at temperature ``delta`` against the mesh measure.

        .. math::
            \pi_j = \frac{\exp(q_j/\delta)\, \Delta_j}{\sum_k \exp(q_k/\delta)\, \Delta_k}

        Args:
            q (np.ndarray): Action values of all actions on the menu, shape ``(N,)``.
            delta (float): Temperature.
            areas (np.ndarray): Cell areas :math:`\Delta_j`, shape ``(N,)``.

        Returns:
            np.ndarray: Probabilities, shape ``(N,)``.
        """
        z = np.ravel(q) / np.maximum(delta, 1e-300)
        p = np.exp(z - z.max()) * np.ravel(areas)
        return p / p.sum()

    @staticmethod
    def entropy(probs, areas):
        r"""Differential entropy of ``probs`` on the mesh, in nats.

        .. math::
            H = -\sum_j \pi_j \ln \pi_j + \sum_j \pi_j \ln \Delta_j

        Args:
            probs (np.ndarray): Probability vector, shape ``(N,)``.
            areas (np.ndarray): Cell areas, shape ``(N,)``.

        Returns:
            float: Entropy in nats.
        """
        p, a = np.ravel(probs), np.ravel(areas)
        return float(
            -np.sum(p * np.log(np.maximum(p, 1e-300))) + np.sum(p * np.log(np.maximum(a, 1e-300)))
        )

    @staticmethod
    def greedy(q, areas):
        r"""Find best believed action :math:`\delta \to 0` limit of :meth:`softmax`.

        Tie-breaking is done proportionally to the areas of the tied actions.

        Args:
            q (np.ndarray): Action values, shape ``(N,)``.
            areas (np.ndarray): Cell areas, shape ``(N,)``.

        Returns:
            np.ndarray: Probabilities concentrated on the maximisers, shape ``(N,)``.
        """
        q, a = np.ravel(q), np.ravel(areas)
        tied = (q >= q.max() - 1e-12) * a
        return tied / tied.sum()

    @staticmethod
    def contrast_trace(Sigma, areas):
        r"""Total advantage uncertainty :math:`V` on the menu.

        .. math::
            V = \sum_j \Delta_j \big[P_W \Sigma P_W^\top\big]_{jj}
              = \Delta^\top \operatorname{diag}\Sigma - \frac{\Delta^\top \Sigma \Delta}{L}

        Args:
            Sigma (np.ndarray): Prior covariance on the menu, shape ``(N, N)``.
            areas (np.ndarray): Cell areas, shape ``(N,)``.

        Returns:
            float: :math:`V`, equal to the sum of the :meth:`spectrum`.
        """
        a = np.ravel(areas)
        return float(a @ np.diag(Sigma) - (a @ Sigma @ a) / a.sum())

    @staticmethod
    def spectrum(Sigma, areas, *, eigen_tolerance=1e-12, return_vectors=False):
        r"""Eigendecompose the weighted contrast covariance, clip negatives, and sort eigenvalues in descending order.

        .. math::
            \tilde K = W^{1/2} P_W \Sigma P_W^\top W^{1/2}, \qquad
            P_W = I - \frac{\mathbf{1}\Delta^\top}{L}

        Args:
            Sigma (np.ndarray): Prior covariance on the menu, shape ``(N, N)``.
            areas (np.ndarray): Cell areas, shape ``(N,)``.
            eigen_tolerance (float): Tolerance for a "numerically nonzero" eigenvalue.
            return_vectors (bool): Return eigenvectors. Optional. (default: ``False``)

        Returns:
            np.ndarray or tuple: ``lam`` ``(m,)``, or ``(lam, psi)`` with ``psi`` ``(N, m)``.
        """
        Sigma = np.atleast_2d(Sigma)
        a = np.ravel(areas)
        N = Sigma.shape[0]
        # trivial case of single action
        if N < 2:
            return (np.zeros(0), np.zeros((N, 0))) if return_vectors else np.zeros(0)
        L = a.sum()
        sa = Sigma @ a
        ones = np.ones(N)
        Sc = Sigma - np.outer(ones, sa) / L - np.outer(sa, ones) / L + (a @ sa) / L**2
        rw = np.sqrt(a)
        w, V = np.linalg.eigh(rw[:, None] * Sc * rw)  # W^{1/2} P_W Sigma P_W' W^{1/2}
        order = np.argsort(w)[::-1]  # sort highest to lowest
        # clip negatives
        w, V = (
            np.maximum(w[order], 0.0)[: N - 1],
            V[:, order][:, : N - 1],
        )
        # return early if no positive eigenvalues remain
        if w.size == 0 or w[0] <= 0.0:
            return (np.zeros(0), np.zeros((N, 0))) if return_vectors else np.zeros(0)
        keep = w > eigen_tolerance * w[0]
        return (w[keep], V[:, keep]) if return_vectors else w[keep]

    def floor(self, eigenvalues):
        r"""Entropy floor :math:`\ln(hV)`, with :math:`V` the total advantage uncertainty.

        .. math::
            \mathcal{H}(\pi) \ge \ln(hV), \qquad V = \sum_i \tilde\lambda_i

        Args:
            eigenvalues (np.ndarray or float): Contrast spectrum, or :math:`V` directly.

        Returns:
            float: The floor in nats; :math:`-\infty` when it cannot bind.
        """
        V = float(np.sum(eigenvalues))
        return np.log(self.h * V) if (V > 0 and self.h > 0) else -np.inf

    def theta(self, delta, lam):
        r"""Water level implied by a temperature.

        .. math::
            \frac{\delta}{\kappa} = \sum_i \min\!\left(1, \frac{\tilde\lambda_i}{\theta}\right)

        [Need to link this to formula]

        Args:
            delta (float): Temperature.
            lam (np.ndarray): Contrast eigenvalues, descending.
        """
        lam = np.ravel(lam)
        n = lam.size
        if float(delta) <= 0.0:
            return np.inf
        if self.kappa <= 0.0:  # no reasoning technology: delta/kappa = inf
            return None
        r = float(delta) / self.kappa
        if n == 0 or r >= n:
            return None
        k = np.arange(n)
        tail = lam.sum() - np.concatenate([[0.0], np.cumsum(lam)])[:n]  # S_k
        room = r - k
        th = tail / np.where(room > 0.0, room, 1.0)  # theta = S_k / (delta/kappa - k)
        upper = np.concatenate([[np.inf], lam[:-1]])
        ok = (room > 0.0) & (lam <= th) & (th < upper)
        j = int(np.argmax(ok))
        return float(th[j]) if ok[j] else None

    def entropy_policy(self, q, floor, areas):
        r"""Temperature that puts the choice entropy exactly on a floor.

        .. math::
            H\big(\text{softmax}(q/\delta)\big) = \text{floor}

        Args:
            q (np.ndarray): Action values on the menu, shape ``(N,)``.
            floor (float): Entropy floor in nats.
            areas (np.ndarray): Cell areas, shape ``(N,)``.

        Returns:
            Tuple[np.ndarray, float]: Probabilities and ``delta`` (``0.0`` greedy,
            ``np.inf`` uniform fallback).
        """
        floor = np.asarray(floor, dtype=float)
        if floor.ndim != 0:
            raise TypeError(
                "entropy_policy takes a scalar entropy floor in nats, not a per-action std"
                "vector. Use: planner.entropy_policy(mu, planner.floor(lam), areas)."
            )
        floor = float(floor)
        a = np.ravel(areas)
        L = float(a.sum())
        if not np.isfinite(floor) or floor <= self.entropy(self.greedy(q, a), a):
            return self.greedy(q, a), 0.0
        if floor >= np.log(L) - 1e-12:
            warnings.warn(
                f"[entropy] floor {floor:.4g} exceeds ln L = {np.log(L):.4g}: the "
                f"constraint cannot be met on this menu. Returning uniform."
            )
            return a / L, np.inf

        def f(d):
            return self.entropy(self.softmax(q, d, a), a) - floor

        lo, hi = 1e-12, 1e6
        k = 0
        while f(hi) <= 0.0 and k < 40:
            hi *= 10.0
            k += 1
        delta = float(brentq(f, lo, hi, maxiter=300))
        return self.softmax(q, delta, a), delta

    def solve(self, q, Sigma, areas):
        r"""Joint solve for the temperature :math:`\delta` and the water level :math:`\theta`.

        .. math::
            f(\delta) = H\big(\text{softmax}(q/\delta)\big)
                        - \ln\!\big(h\, V^{\text{post}}(\delta)\big),
            \qquad V^{\text{post}}(\delta) = \sum_i \min\{\theta(\delta), \tilde\lambda_i\},

        where :math:`\theta(\delta)` solves :math:`\delta/\kappa = \sum_i \min(1,
        \tilde\lambda_i/\theta)`.

        Early exits: greedy (:math:`\delta = 0`) when the floor
        cannot bind (``h <= 0``, ``V <= 0``, or a floor already below the greedy entropy).

        Args:
            q (np.ndarray): Prior (pre-reasoning) action values on the menu, shape ``(N,)``.
            Sigma (np.ndarray): Prior covariance on the menu, shape ``(N, N)``.
            areas (np.ndarray): Cell areas, shape ``(N,)``.

        Returns:
            Decision: Resulting decision, the chosen action probabilities.
        """
        q = np.ravel(q)
        a = np.ravel(areas)
        N = q.size
        tied = self.greedy(q, a)
        V_E = self.contrast_trace(Sigma, a) if N > 1 else 0.0
        if self.floor(V_E) <= self.entropy(tied, a):
            return Decision(probs=tied, plan=Plan(delta=0.0, theta=np.inf))

        if self.kappa <= 0.0:  # experience only: no reasoning to price
            probs, delta = self.entropy_policy(q, self.floor(V_E), a)
            return Decision(probs=probs, plan=Plan(delta=delta, theta=np.inf))

        lam, psi = self.spectrum(Sigma, a, return_vectors=True)
        if lam.size == 0:
            probs, delta = self.entropy_policy(q, self.floor(V_E), a)
            return Decision(probs=probs, plan=Plan(delta=delta, theta=np.inf))

        def f(d):
            th = self.theta(d, lam)
            if th is None:
                return np.inf
            V_post = float(np.minimum(lam, th).sum())
            if V_post <= 0.0:
                return np.inf
            return self.entropy(self.softmax(q, d, a), a) - np.log(self.h * V_post)

        lo = 1e-12
        hi = self.kappa * lam.size * (1.0 - 1e-9)
        if f(lo) >= 0.0:
            return Decision(probs=tied, plan=Plan(delta=0.0, theta=np.inf))
        if not np.isfinite(f(hi)) or f(hi) <= 0.0:
            hi = np.nextafter(self.kappa * lam.size, 0.0)
            if f(hi) <= 0.0:
                warnings.warn(
                    "[entropy] no interior root below the water-filling corner; returning uniform."
                )
                return Decision(probs=a / a.sum(), plan=Plan(delta=np.inf, theta=np.inf))
        delta = float(brentq(f, lo, hi, maxiter=300))
        th = self.theta(delta, lam)
        return Decision(
            probs=self.softmax(q, delta, a),
            plan=Plan(
                delta=delta,
                theta=np.inf if th is None else float(th),
                signals=self._signals(lam, psi, th, a),
            ),
        )

    @staticmethod
    def _signals(lam, psi, theta, areas):
        r"""One signal per active direction, priced to land its eigenvalue on :math:`\theta`.

        .. math::
            \sigma_i^2 = \frac{\tilde\lambda_i \theta}{\tilde\lambda_i - \theta},
            \qquad w_i = W^{1/2}\psi_i

        Args:
            lam (np.ndarray): Contrast eigenvalues, descending.
            psi (np.ndarray): Matching orthonormal eigenvectors, columns.
            theta (float or None): Water level; ``inf`` or ``None`` buys nothing.
            areas (np.ndarray): Cell areas, shape ``(N,)``.

        Returns:
            tuple: :class:`Signal` s for the directions with
            :math:`\tilde\lambda_i > \theta`.
        """
        if theta is None or not np.isfinite(theta):
            return ()
        rw = np.sqrt(np.ravel(areas))
        active = np.flatnonzero(lam > theta)
        # sigma^2 = lam theta / (lam - theta) cuts the posterior eigenvalue down to theta
        return tuple(
            Signal(w=rw * psi[:, i], noise=float(lam[i] * theta / (lam[i] - theta))) for i in active
        )
