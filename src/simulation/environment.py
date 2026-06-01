from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import quantecon as qe

from tqdm import tqdm



@dataclass
class InvestmentParameters:
    ALPHA: float = 0.33 # Capital production elasticity
    DELTA: float = 0.04 # Depreciation rate
    R: float = 0.03     # Interest rate
    THETA: float = 0.0  # Collateral constraint fraction (b_{t+1} <= theta * k_{t+1})
    BETA: float = 0.96  # Discount factor (set independently of R)

    # These will be set based on the scenario
    KAPPA: float = 0.1   # Adjustment cost parameter
    RHO: float = 0.9     # Persistence
    SIGMA_EPS: float = 0.0 # Volatility (0 = Deterministic)

    # Grid Parameters
    N_k: int = 100
    N_z: int = 7
    K_min: float = 0.01
    K_max: float = 20.0


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

        self.k_grid = np.linspace(self.p.K_min, self.p.K_max, self.p.N_k)

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