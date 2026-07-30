from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import quantecon as qe

from tqdm import tqdm

Number = float | np.ndarray

@dataclass
class InvestmentParameters:
    ALPHA: float = 0.33 # capital elasticity in production
    DELTA: float = 0.04 # depreciation rate
    R: float = 0.03     # interest rate
    BETA: float = 0.96  # discount factor (set independently of R)
 
    # Frictions
    KAPPA: float = 0.1    # quadratic adjustment cost parameter
    THETA_K: float = 0.95 # resale price of capital (partial irreversibility)
 
    # Financing
    ZETA: float = 1.0 # credit state: fraction of collateral pledgeable (b' <= ZETA * THETA_K * k)
    # 
 
    # Idiosyncratic shock parameters
    RHO: float = 0.9         # AR(1) persistence of log z
    SIGMA_EPS: float = 0.0   # innovation std (0 => deterministic)
 
    # Grids
    N_k: int = 100
    N_z: int = 7
    K_min: float = 0.01
    K_max: float = 20.0
    B_min: float = 0.0
    B_max: float = 10.0
    N_b: int = 1 # 1 => debt state inert (no-debt problems)
 

@dataclass
class SimResult:
    t: np.ndarray
    z: np.ndarray
    k: np.ndarray
    k_next: np.ndarray
    i: np.ndarray
    d: np.ndarray
    b: np.ndarray
    b_next: np.ndarray


class AdjustmentCosts(ABC):
    @abstractmethod
    def __call__(self, i: float, k: float) -> float:
        """Returns psi(i, k)."""
        raise NotImplementedError

class NoAdjustmentCosts(AdjustmentCosts):
    def __call__(self, i: float, k: float) -> float:
        return 0.0

class QuadraticAdjustmentCosts(AdjustmentCosts):
    def __init__(self, kappa: float):
        self.kappa = kappa

    def __call__(self, i: float, k: float) -> float:
        k_safe = max(k, 1e-8)
        return (self.kappa / 2.0) * (i**2 / k_safe)
    
class Utility(ABC):
    """Per-period utility u(c) with marginal utility and its inverse (for EGM)."""

    @abstractmethod
    def __call__(self, c: float | np.ndarray) -> float | np.ndarray:
        """Returns u(c)."""
        raise NotImplementedError

    @abstractmethod
    def prime(self, c: float | np.ndarray) -> float | np.ndarray:
        """Returns u'(c)."""
        raise NotImplementedError

    @abstractmethod
    def prime_inv(self, x: float | np.ndarray) -> float | np.ndarray:
        """Returns (u')^{-1}(x)."""
        raise NotImplementedError

class CRRA(Utility):
    """u(c) = c^(1-gamma)/(1-gamma); log if gamma == 1. u'(c) = c^(-gamma)."""

    def __init__(self, gamma: float):
        self.gamma = gamma

    def __call__(self, c):
        c = np.maximum(c, 1e-12)
        if abs(self.gamma - 1.0) < 1e-9:
            return np.log(c)
        return c ** (1.0 - self.gamma) / (1.0 - self.gamma)

    def prime(self, c):
        return np.maximum(c, 1e-12) ** (-self.gamma)

    def prime_inv(self, x):
        return np.maximum(x, 1e-12) ** (-1.0 / self.gamma)


class ProductivityProcess(ABC):
    r"""How ``z`` is initialised and evolved. Vectorized over agents."""

    @abstractmethod
    def initial(self, *, n: int, rng: np.random.Generator) -> np.ndarray:
        r"""Draw ``n`` initial ``z`` values (one per agent)."""
        ...

    @abstractmethod
    def step(self, *, z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        r"""Evolve a vector of ``z`` one period. (Identity for permanent types.)"""
        ...

    @abstractmethod
    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Discretised support and transition matrix ``(z_grid, P)`` for VFI.

        Permanent/i.i.d. processes return ``P`` = rows of the stationary dist (no persistence).
        """
        ...


class PermanentType(ProductivityProcess):
    r"""Firm draws its ``z`` once at birth from a fixed set and keeps it forever.

    (Your 'high / mid / low, drawn at the beginning and that's it' case.)
    """

    def __init__(self, z_values, probs=None):
        self.z_values = np.asarray(z_values, float)
        k = len(self.z_values)
        self.probs = np.full(k, 1.0 / k) if probs is None else np.asarray(probs, float)

    def initial(self, *, n, rng):
        return rng.choice(self.z_values, size=n, p=self.probs)

    def step(self, *, z, rng):
        return z                                    # permanent: never changes

    def grid(self):
        # each type is absorbing: P = identity (a firm stays its type)
        return self.z_values, np.eye(len(self.z_values))


class MarkovAR1(ProductivityProcess):
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
        # continuous AR(1) update (keeps agents off-grid; env can snap for policy lookup)
        eps = rng.normal(0.0, self.sigma_eps, size=np.shape(z))
        return np.exp(self.rho * np.log(np.maximum(z, 1e-12)) + eps)

    def grid(self):
        return self._z_grid, self._P


class IIDDraw(ProductivityProcess):
    r"""``z`` redrawn i.i.d. each period from a fixed distribution (no persistence)."""

    def __init__(self, z_values, probs=None):
        self.z_values = np.asarray(z_values, float)
        k = len(self.z_values)
        self.probs = np.full(k, 1.0 / k) if probs is None else np.asarray(probs, float)

    def initial(self, *, n, rng):
        return rng.choice(self.z_values, size=n, p=self.probs)

    def step(self, *, z, rng):
        return rng.choice(self.z_values, size=np.shape(z), p=self.probs)   # ignores current z

    def grid(self):
        # i.i.d.: every row of P is the same draw distribution
        return self.z_values, np.tile(self.probs, (len(self.z_values), 1))

class Production(ABC):
    r"""Pure technology :math:`z f(k)`. ``z`` is always an argument, never stored."""

    @abstractmethod
    def output(self, *, z: Number, k: Number) -> Number: ...
    @abstractmethod
    def marginal_product(self, *, z: Number, k: Number) -> Number: ...

    def __call__(self, *, z: Number, k: Number) -> Number:
        return self.output(z=z, k=k)


class CapitalCobbDouglas(Production):
    def __init__(self, alpha: float):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.alpha = alpha

    def output(self, *, z, k):
        k = np.maximum(k, 1e-12); return z * k ** self.alpha

    def marginal_product(self, *, z, k):
        k = np.maximum(k, 1e-12); return self.alpha * z * k ** (self.alpha - 1.0)


class InvestmentEnvironment:
    def __init__(self, params: InvestmentParameters, adjustment_costs: AdjustmentCosts, seed: int = 42):
        self.p = params
        self.adjustment_costs = adjustment_costs
        self.rng = np.random.default_rng(seed)
        self.setup_grids()

    def setup_grids(self):
        if self.p.SIGMA_EPS > 0 and self.p.N_z > 1:
            mc = qe.markov.approximation.tauchen(
                self.p.N_z, self.p.RHO, self.p.SIGMA_EPS, mu=0, n_std=3
            )
            self.z_grid = np.exp(mc.state_values)
            self.P = mc.P
            self.actual_nz = self.p.N_z
        else:
            warnings.warn("Zero volatility or zero z states: using degenerate z grid with a single point at 1.0.")
            self.z_grid = np.array([1.0])
            self.P = np.array([[1.0]])
            self.actual_nz = 1

        self.k_grid = np.linspace(self.p.K_MIN, self.p.K_MAX, self.p.N_k)

    def action_query_grid(self, z_t, k_t):
        """Returns a grid of (z, k, i) points for querying the GP given current state (z_t, k_t).
        Generally used to discretize the action space for the ExperienceReasoningAgent's policy.
        """
        i_grid = self.k_grid - (1 - self.p.DELTA) * k_t
        return np.column_stack([
            np.full_like(i_grid, z_t),
            np.full_like(i_grid, k_t),
            i_grid
        ])

    def production(self, z, k):
        return z * (k ** self.p.ALPHA)

        def optimal_b_next(self, k_next: float) -> float:
            """Optimal next-period debt given k_next.

            With linear utility, the net PV of a unit of debt is (1 - beta*(1+R)):
            - beta*(1+R) <= 1: borrow at collateral constraint
            - beta*(1+R) >  1: don't borrow
            """
            if self.p.BETA * (1.0 + self.p.R) <= 1.0:
                return self.p.THETA * k_next
            return 0.0

    def dividend(self, z, k, i, b=0.0, b_next=0.0):
        """d = f(z,k) - i - psi(i,k) + b_{t+1} - (1+R)*b_t"""
        k_safe = max(k, 1e-8)
        adj_cost = self.adjustment_costs(i, k_safe)
        return self.production(z, k_safe) - i - adj_cost + b_next - (1 + self.p.R) * b

    # TODO: Double check this
    def gp_observation(self, z, k, i, b_next):
        """Dividend corrected for debt terms for GP updates.

        The GP learns Q(z,k,b,i) = GP(z,k,i) - (1+R)*b.
        The correct GP observation target is:
            d + (1+R)*b - beta*(1+R)*b_next = f(z,k) - i - psi(i,k) + b_next*(1 - beta*(1+R))
        """
        k_safe = max(k, 1e-8)
        adj_cost = self.adjustment_costs(i, k_safe)
        correction = b_next * (1.0 - self.p.BETA * (1.0 + self.p.R))
        return self.production(z, k_safe) - i - adj_cost + correction

    def transition(self, z, k_prime, b_next=0.0, custom_rng=None):
        k_next = max(k_prime, 1e-8)
        if self.p.SIGMA_EPS > 0:
            rng = custom_rng if custom_rng is not None else self.rng
            eps = rng.normal(0.0, self.p.SIGMA_EPS)
            z_next = np.exp(self.p.RHO * np.log(max(z, 1e-12)) + eps)
        else:
            z_next = z
        return z_next, k_next, float(b_next)

def _eig_summary(eigs):
    return {
        "eig_max": eigs[-1],
        "eig_min": eigs[0],
        "eig_mean": eigs.mean(),
        "eig_std": eigs.std(),
        "eig_tr": eigs.sum(),
        "eig_n_active": int((eigs > 1e-10).sum()),
    }

def run_simulation(env, agents: list, T: int = 100, z0: float = 1.0, firm_exit_rate: float = 0.0, seed: int = 0) -> pd.DataFrame:
    from .firm import RationalInvestmentAgent

    p = env.p
    rational_agent = RationalInvestmentAgent(env).fit()
    k_ss = rational_agent.fixed_point()
    b_ss = env.optimal_b_next(k_ss)

    records = []
    firm_exit_on = firm_exit_rate is not None and firm_exit_rate > 0.0

    for j, agent_j in enumerate(tqdm(agents)):
        if firm_exit_on:
            exit_rng = np.random.default_rng(42 + j)
            exit_ = exit_rng.uniform(0, 1, size=T) < firm_exit_rate

        shock_rng = np.random.default_rng(seed + j)

        z_t, k_t, b_t = float(z0), k_ss, b_ss
        z_rat, k_rat, b_rat = float(z0), k_ss, b_ss
        did_reset = False

        for t in range(T):

            kp_t, b_next = agent_j.policy(z_t, k_t, b_t)
            i_t = kp_t - (1.0 - p.DELTA) * k_t
            d_t = env.dividend(z_t, k_t, i_t, b_t, b_next)
            gp_obs_t = env.gp_observation(z_t, k_t, i_t, b_next)

            kp_rat, b_next_rat = rational_agent.policy(z_rat, k_rat, b_rat)
            i_rat = kp_rat - (1.0 - p.DELTA) * k_rat
            d_rat = env.dividend(z_rat, k_rat, i_rat, b_rat, b_next_rat)

            q_chosen = agent_j.gp.predict(np.array([[z_t, k_t, i_t]]), return_std=False)[0]

            # Pre-update beliefs (what firm knew when deciding)
            _, X_q, mean, std_E = agent_j.get_beliefs(z_t, k_t, b_t)
            _, delta_E = agent_j._entropy_policy(mean, std_E)
            eigs = agent_j.eigenvalues(X_q)

            # Transition and TD update
            z_next, _, _ = env.transition(z_t, kp_t, b_next, custom_rng=shock_rng)
            kp_greedy, _ = agent_j.get_greedy_action(z_next, kp_t, b_next)
            i_greedy = kp_greedy - (1.0 - p.DELTA) * kp_t
            x_dec = np.array([z_t, k_t, i_t])
            x_out = np.array([z_next, kp_t, i_greedy])
            alpha_gain = agent_j.gp.add_observation(x_dec, x_out, gp_obs_t, return_gain=True)

            # Record everything on same period
            records.append({
                "agent_id": j,
                "t": t,
                "z": z_t,
                "k": k_t, "i": i_t, "d": d_t, "q_chosen": q_chosen,
                "k_rat": k_rat, "i_rat": i_rat, "d_rat": d_rat,
                "reset": int(did_reset),
                "delta_E": delta_E,
                "alpha_gain": alpha_gain,
                **_eig_summary(eigs),
            })

            # Firm exit
            did_reset = firm_exit_on and exit_[t] # type: ignore
            if did_reset:
                agent_j.gp.reset()

            z_t, k_t, b_t = z_next, kp_t, b_next
            z_rat, k_rat, b_rat = z_next, kp_rat, b_next_rat

    return pd.DataFrame(records)