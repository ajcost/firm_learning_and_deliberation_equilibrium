from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar

from .environment.standard_investment import (
    InvestmentParameters,
    SimResult,
    InvestmentEnvironment,
)
from .gaussian_process import GPBelief # type: ignore[import]

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .gaussian_process import GPBeliefParameters


class InvestmentAgent(ABC):
    """Abstract base class for an investment agent."""

    def __init__(self, env: InvestmentEnvironment, name: str):
        self.env = env
        self.p = env.p   # convenience shorthand
        self.name = name

    def fit(self):
        """Optional training/solving step. Default: do nothing."""
        return self

    @abstractmethod
    def policy(self, z: float, k: float, b: float = 0.0, t: int | None = None) -> tuple[float, float]:
        """Given state (z, k, b), return (k_next, b_next)."""
        raise NotImplementedError

    @abstractmethod
    def fixed_point(self) -> float:
        """Return the steady-state capital k* (with z=1). b* = THETA * k*."""
        raise NotImplementedError

    def simulate(self, env: InvestmentEnvironment | None = None, T=60, z0=1.0, k0=10.0, b0=0.0, seed=0) -> SimResult:
        """Simulate the agent. Uses the agent's own environment if env is not provided."""
        env = env or self.env
        rng = np.random.default_rng(seed)

        t = np.arange(T, dtype=int)
        z = np.empty(T, dtype=float)
        k = np.empty(T, dtype=float)
        k_next = np.empty(T, dtype=float)
        i = np.empty(T, dtype=float)
        d = np.empty(T, dtype=float)
        b = np.empty(T, dtype=float)
        b_next = np.empty(T, dtype=float)

        z[0], k[0], b[0] = float(z0), float(k0), float(b0)

        for tt in range(T):
            kp, bp = self.policy(z[tt], k[tt], b[tt], t=tt)
            k_next[tt], b_next[tt] = kp, bp
            i[tt] = kp - (1.0 - self.p.DELTA) * k[tt]
            d[tt] = env.dividend(z[tt], k[tt], i[tt], b[tt], bp)

            if tt < T - 1:
                z[tt+1], k[tt+1], b[tt+1] = env.transition(z[tt], kp, bp, custom_rng=rng)

        return SimResult(t=t, z=z, k=k, k_next=k_next, i=i, d=d, b=b, b_next=b_next)

    def simulate_irf(
        self,
        env: InvestmentEnvironment | None = None,
        T: int = 40,
        shock_size_log: float = 0.05,
        shock_time: int = 0,
        z0: float = 1.0,
        k0: float | None = None,
        b0: float | None = None,
        deterministic_after: bool = True,
        seed: int = 0,
    ) -> dict:
        """Impulse response function. Uses the agent's own environment if env is not provided."""
        env = env or self.env
        p = self.p

        if k0 is None:
            k0 = float(self.fixed_point())
        if b0 is None:
            b0 = p.THETA * k0

        tgrid = np.arange(T, dtype=int)
        rng_base = np.random.default_rng(seed)
        rng_shk  = np.random.default_rng(seed)

        def step_z(z_now: float, rng) -> float:
            if p.SIGMA_EPS <= 0:
                return z_now
            if deterministic_after:
                return float(np.exp(p.RHO * np.log(max(z_now, 1e-12))))
            return float(np.exp(p.RHO * np.log(max(z_now, 1e-12)) + rng.normal(0.0, p.SIGMA_EPS)))

        def run_path(apply_shock: bool, rng) -> SimResult:
            z = np.empty(T, dtype=float)
            k = np.empty(T, dtype=float)
            k_next = np.empty(T, dtype=float)
            i_arr = np.empty(T, dtype=float)
            d = np.empty(T, dtype=float)
            b = np.empty(T, dtype=float)
            b_next = np.empty(T, dtype=float)

            z[0], k[0], b[0] = float(z0), float(k0), float(b0)

            if apply_shock and shock_time == 0:
                z[0] = float(np.exp(np.log(max(z[0], 1e-12)) + shock_size_log))

            for tt in range(T):
                kp, bp = self.policy(z[tt], k[tt], b[tt], t=tt)
                k_next[tt], b_next[tt] = kp, bp
                i_arr[tt] = kp - (1.0 - p.DELTA) * k[tt]
                d[tt] = float(env.dividend(z[tt], k[tt], i_arr[tt], b[tt], bp))

                if tt < T - 1:
                    k[tt+1] = max(kp, 1e-8)
                    b[tt+1] = float(bp)
                    z[tt+1] = step_z(z[tt], rng)
                    if apply_shock and (tt + 1) == shock_time:
                        z[tt+1] = float(np.exp(np.log(max(z[tt+1], 1e-12)) + shock_size_log))

            return SimResult(t=tgrid, z=z, k=k, k_next=k_next, i=i_arr, d=d, b=b, b_next=b_next)

        base = run_path(False, rng_base)
        shk  = run_path(True,  rng_shk)

        diff = SimResult(
            t=tgrid,
            z=shk.z - base.z, k=shk.k - base.k,
            k_next=shk.k_next - base.k_next, i=shk.i - base.i,
            d=shk.d - base.d, b=shk.b - base.b, b_next=shk.b_next - base.b_next,
        )
        pct = {
            "z": 100.0 * (shk.z / base.z - 1.0),
            "k": 100.0 * (shk.k / base.k - 1.0),
            "i": 100.0 * (shk.i / base.i - 1.0),
            "d": 100.0 * (shk.d / base.d - 1.0),
            "b": 100.0 * (shk.b / np.maximum(base.b, 1e-12) - 1.0),
        }
        return {"t": tgrid, "base": base, "shk": shk, "diff": diff, "pct": pct}

    def policy_curve(self, z_grid: np.ndarray, k_grid: np.ndarray, z_idx: int = 0, b: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """Return (k_grid, kprime_grid) for plotting at a given z index and debt b."""
        z_val = float(z_grid[z_idx])
        kprime_vals = np.array([self.policy(z_val, float(kv), b)[0] for kv in k_grid], dtype=float)
        return k_grid, kprime_vals


class RationalInvestmentAgent(InvestmentAgent):

    def __init__(self, env: InvestmentEnvironment):
        super().__init__(env, name="Rational (VFI)")
        self.initialize_value_function()

    def initialize_value_function(self):
        self.v = np.zeros((self.env.actual_nz, self.env.p.N_k))
        self.policy_k = np.zeros((self.env.actual_nz, self.env.p.N_k))
        for i_z in range(self.env.actual_nz):
            self.v[i_z, :] = self.env.production(self.env.z_grid[i_z], self.env.k_grid) / self.env.p.R

    def _vfi_dividend(self, k_next, k_current, z):
        """VFI objective dividend for W(z,k) = V(z,k,0).

        Since V(z,k,b) = W(z,k) - (1+R)*b, the VFI objective at b=0 is
        exactly the GP observation target: f(z,k) - i - psi(i,k) + b_next*(1 - beta*(1+R)).
        """
        i = k_next - (1 - self.env.p.DELTA) * k_current
        b_next = self.env.optimal_b_next(k_next)
        return self.env.gp_observation(z, k_current, i, b_next)

    def fit(self, tol=1e-5, max_iter=500):
        for _ in range(max_iter):
            v_new = np.zeros_like(self.v)
            policy_new = np.zeros_like(self.policy_k)

            v_expected = self.env.P @ self.v
            ev_funcs = [
                interp1d(self.env.k_grid, v_expected[i, :], kind="linear", fill_value="extrapolate") # type: ignore[reportUnknownMemberType]
                for i in range(self.env.actual_nz)
            ]

            for i_z in range(self.env.actual_nz):
                z = self.env.z_grid[i_z]
                cont_func = ev_funcs[i_z]

                for i_k, k in enumerate(self.env.k_grid):
                    def objective(k_prime, _k=k, _z=z):
                        return -(self._vfi_dividend(k_prime, _k, _z) + self.env.p.BETA * cont_func(k_prime))

                    res = minimize_scalar(objective, bounds=(self.env.p.K_min, self.env.p.K_max), method="bounded")
                    v_new[i_z, i_k] = -res.fun
                    policy_new[i_z, i_k] = res.x

            if np.max(np.abs(v_new - self.v)) < tol:
                break

            self.v = v_new
            self.policy_k = policy_new

        self._pol_funcs = [
            interp1d(self.env.k_grid, self.policy_k[i, :], kind="linear", fill_value="extrapolate") # type: ignore[reportUnknownMemberType]
            for i in range(self.env.actual_nz)
        ]
        return self

    def _z_to_idx(self, z):
        return int(np.argmin(np.abs(self.env.z_grid - z)))

    def policy(self, z: float, k: float, b: float = 0.0, t=None) -> tuple[float, float]:
        z_idx = self._z_to_idx(z)
        k_next = float(self._pol_funcs[z_idx](k))
        return k_next, self.env.optimal_b_next(k_next)

    def fixed_point(self) -> float:
        def objective(k):
            return self.policy(1.0, k)[0] - k
        return brentq(objective, self.env.k_grid[0], self.env.k_grid[-1]) # pyright: ignore[reportReturnType]


@dataclass
class InvestmentAgentParameters:
    H: float = 0.05  # exploration regularization parameter
    KAPPA_R: float = 0.01


def calibrated_entropy_regularization_parameter(env: InvestmentEnvironment, gp_params: GPBeliefParameters) -> float:
    """Calibrates h from environment primitives so the initial entropy target
    h * tr(Sigma_0) = 0.5 * ln(N_k), where tr(Sigma_0) = N_k * sigma0^2.

        h = 0.5 * ln(N_k) / (N_k * sigma0^2)
    """
    N = env.p.N_k
    return 0.5 * np.log(N) / (N * gp_params.kernel.sigma0_sq)


class ExperienceReasoningAgent(EntrepreneurAgent):
    r"""GP-belief agent with uncertainty-driven exploration and reasoning.
    Environment-agnostic: state/action are opaque; all domain access goes through the
    env's generic interface (candidate_actions, featurize, gp_target). The GP learns
    Q(features); the entropy layer sets exploration temperature from posterior uncertainty;
    reasoning does eigenvalue water-filling on the belief covariance."""

    def __init__(self, env, gp, agent_params, experience_only=False, seed=42):
        super().__init__(env, name="Experience-Reasoning Agent")
        self.gp = gp
        self.agent_params = agent_params
        self.experience_only = experience_only
        self.rng = np.random.default_rng(seed)

    # ---- beliefs: query GP over the feasible candidate actions -----------------------
    def get_beliefs(self, state):
        actions = self.env.candidate_actions(state=state)          # (n, action_dim), opaque
        X_q = self.env.featurize(state=state, actions=actions)     # (n, feature_dim)
        mean, std = self.gp.predict(X_q, return_std=True)
        return actions, X_q, mean, std

    # ---- entropy policy: temperature delta s.t. H(pi) = H * sum(std^2) ----------------
    def _entropy_policy(self, q, std):
        N = len(q); H_max = np.log(N)
        lb = self.agent_params.H * float(np.sum(std ** 2))         # uncertainty-driven entropy floor
        if lb <= 1e-8:                                             # no uncertainty -> greedy
            p = np.zeros(N); p[int(np.argmax(q))] = 1.0
            return p, 0.0
        if lb >= H_max - 1e-8:                                     # saturated -> uniform
            return np.ones(N) / N, np.inf

        def f(delta):
            logits = q / max(delta, 1e-12); logits -= logits.max()
            pp = np.exp(logits); pp /= pp.sum()
            return -np.sum(pp * np.log(pp + 1e-12)) - lb          # entropy(delta) - floor

        try:
            delta_star = brentq(f, 1e-6, 1e6, maxiter=200)
        except ValueError:
            delta_star = 1e-6 if abs(f(1e-6)) < abs(f(1e6)) else 1e6
        logits = q / max(delta_star, 1e-12); logits -= logits.max()
        p = np.exp(logits); p /= p.sum()
        return p, float(delta_star)

    def reason(self, X_q, delta_E, std_E):
        kappa = self.agent_params.KAPPA_R
        if self.experience_only or kappa <= 0 or delta_E <= 0 or self.agent_params.H <= 0:
            return std_E
        _, Sigma_E = self.gp.predict_full(X_q)                    # full belief covariance
        vals_E, V = np.linalg.eigh(Sigma_E)
        water_level = kappa / (self.agent_params.H * delta_E)     # reasoning budget
        vals_R = np.minimum(vals_E, water_level)                  # cap high-variance directions
        Sigma_R = V @ np.diag(vals_R) @ V.T
        return np.sqrt(np.maximum(np.diag(Sigma_R), 0))

    def eigenvalues(self, X_q):
        _, Sigma_E = self.gp.predict_full(X_q)
        return np.maximum(np.linalg.eigh(Sigma_E)[0], 0)

    # ---- policy: draft (experience) -> reason -> final -> sample ---------------------
    def policy(self, *, z=1.0, omega=None, t=None):
        omega = self.p.OMEGA_ZERO if omega is None else omega
        state = (z, omega)
        actions, X_q, mean, std_E = self.get_beliefs(state)
        _, delta_E = self._entropy_policy(mean, std_E)            # draft temperature
        std_R = self.reason(X_q, delta_E, std_E)                 # reasoning shrinks uncertainty
        probs_R, _ = self._entropy_policy(mean, std_R)           # final policy
        j = self.rng.choice(len(actions), p=probs_R)
        kp, bp = actions[j]
        return float(kp), float(bp)

    # ---- diagnostics -----------------------------------------------------------------
    def get_greedy_action(self, *, z=1.0, omega=None):
        omega = self.p.OMEGA_ZERO if omega is None else omega
        actions = self.env.candidate_actions(state=(z, omega))
        mean = self.gp.predict(self.env.featurize(state=(z, omega), actions=actions), return_std=False)
        kp, bp = actions[int(np.argmax(mean))]
        return float(kp), float(bp)

    def get_expected_action(self, *, z=1.0, omega=None):
        omega = self.p.OMEGA_ZERO if omega is None else omega
        actions, X_q, mean, std_E = self.get_beliefs((z, omega))
        _, delta_E = self._entropy_policy(mean, std_E)
        std_R = self.reason(X_q, delta_E, std_E)
        probs_R, _ = self._entropy_policy(mean, std_R)
        kp, bp = probs_R @ actions                               # probability-weighted (kp, bp)
        return float(kp), float(bp)

    def fixed_point(self, *, z=1.0):
        og = self.env.omega_grid
        diffs = [abs(self.omega_next(z=z, omega=float(w)) - float(w)) for w in og]
        return float(og[int(np.argmin(diffs))])

    # ---- experience update: observe one transition, teach the GP ---------------------
    def observe(self, *, state, action, reward, rng=None):
        r"""Feed one realized (state, action, reward) into the GP. Target from env.gp_target
        (u(c) for the entrepreneur — no debt correction since b' is explicit)."""
        X = self.env.featurize(state=state, actions=np.atleast_2d(action))
        y = self.env.gp_target(state=state, action=action, reward=reward)
        self.gp.add_observation(X, np.atleast_1d(y))
        return self.env.transition(state=state, action=action, rng=rng)