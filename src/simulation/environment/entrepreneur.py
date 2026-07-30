"""Entrepreneur's problem: net worth omega, productivity z, action (k', b')."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .investment_elements import (
    Utility, CRRA, Production, CapitalCobbDouglas,
    Productivity, PermanentProductivity, MarkovAR1, IIDDraw,
    DeathProcess, NoDeath, ConstantHazard,
)

@dataclass
class EntrepreneurParameters:
    DELTA: float = 0.08
    R: float = 0.04
    BETA: float = 0.96
    THETA_K: float = 0.95
    OMEGA_ZERO: float = 1.0

    N_omega: int = 100
    OMEGA_MIN: float = 1e-3
    OMEGA_MAX: float = 20.0

    N_k: int = 150
    K_MIN: float = 1e-3
    K_MAX: float | None = None
    N_b: int = 150
    B_MIN: float | None = None
    B_MAX: float | None = None
    C_MIN: float = 1e-4

    # composed primitives
    utility: Utility = field(default_factory=lambda: CRRA(1.0))
    production: Production = field(default_factory=lambda: CapitalCobbDouglas(0.33))
    productivity: Productivity = field(default_factory=lambda: PermanentProductivity([1.0]))
    death: DeathProcess = field(default_factory=lambda: ConstantHazard(0.01))

    @property
    def survival(self) -> float:
        return self.death.survival_rate

    @property
    def disc(self) -> float:
        r""":math:`\beta(1-\rho)`."""
        return self.BETA * self.survival

    @property
    def knife_edge(self) -> bool:
        return abs(self.disc * (1.0 + self.R) - 1.0) < 1e-9

    @property
    def impatient(self) -> bool:
        return self.disc * (1.0 + self.R) < 1.0 - 1e-9

    def set_beta_knife_edge(self):
        self.BETA = 1.0 / (self.survival * (1.0 + self.R))

    def set_r_knife_edge(self):
        self.R = 1.0 / self.disc - 1.0


class EntrepreneurEnvironment:
    r"""Environment for Entrepreneur's problem."""
    def __init__(self, params, *, seed: int = 42):
        self.p = params
        self.utility = params.utility
        self.production = params.production
        self.productivity = params.productivity
        self.death = params.death
        self.omega0 = params.OMEGA_ZERO
        self.rng = np.random.default_rng(seed)
        self._adaptive_menu: bool = False
        self.setup_grids()

    def _arr(self, x, default) -> np.ndarray:
        return np.asarray(default if x is None else x, dtype=float)

    def setup_grids(self, *, calibrate: bool = False, k_headroom: float = 1.3,
                    b_headroom: float = 3.0, omega_headroom: float = 1.5):
        self.z_grid, self.P = self.productivity.grid()
        self.actual_nz = len(self.z_grid)
        if calibrate:
            self.calibrate_grids(k_headroom=k_headroom, b_headroom=b_headroom,
                                 omega_headroom=omega_headroom)
            return
        p = self.p
        self.omega_grid = np.linspace(p.OMEGA_MIN, p.OMEGA_MAX, p.N_omega)
        k_max = p.K_MAX if p.K_MAX is not None else 1.5 * self.k_star()
        b_max = p.B_MAX if p.B_MAX is not None else float(np.ravel(self.collateral_cap(kp=k_max))[0])
        b_min = p.B_MIN if p.B_MIN is not None else -k_max
        self.k_grid = np.linspace(p.K_MIN, k_max, p.N_k)
        self.b_grid = np.linspace(b_min, b_max, p.N_b)

    def calibrate_grids(self, *, k_headroom=1.3, b_headroom=3.0, omega_headroom=1.5, verbose=True):
        p = self.p
        z_nodes = np.atleast_1d(self.z_grid)
        ks = np.array([self.k_star(z=float(zz)) for zz in z_nodes])
        wb = np.array([self.omega_bar(z=float(zz)) for zz in z_nodes])
        k_top, wbar_top = float(ks.max()), float(wb.max())
        p.K_MIN = min(p.K_MIN, 1e-3); p.K_MAX = k_headroom * k_top
        p.B_MAX = float(np.ravel(self.collateral_cap(kp=p.K_MAX))[0])
        p.B_MIN = -b_headroom * k_top
        p.OMEGA_MIN = min(p.OMEGA_MIN, 1e-3); p.OMEGA_MAX = omega_headroom * wbar_top
        self.omega_grid = np.linspace(p.OMEGA_MIN, p.OMEGA_MAX, p.N_omega)
        self.k_grid = np.linspace(p.K_MIN, p.K_MAX, p.N_k)
        self.b_grid = np.linspace(p.B_MIN, p.B_MAX, p.N_b)
        caps = np.array([float(np.ravel(self.collateral_cap(kp=k))[0]) for k in ks])
        self.k_grid = np.unique(np.r_[self.k_grid, ks])
        self.b_grid = np.unique(np.r_[self.b_grid, caps, 0.0])
        self.omega_grid = np.unique(np.r_[self.omega_grid, wb])
        if verbose:
            print(f"[calibrate] z∈[{z_nodes.min():.3f},{z_nodes.max():.3f}] "
                  f"k*∈[{ks.min():.3f},{ks.max():.3f}] ω̄∈[{wb.min():.3f},{wbar_top:.3f}]  "
                  f"K n={len(self.k_grid)} B n={len(self.b_grid)} Ω n={len(self.omega_grid)}")
            
    def reset_grids(self, step_size=0.1, ignore_b_grid=False, ignore_k_grid=False, ignore_omega_grid=False, verbose=False):
        p = self.p
        if not ignore_k_grid:
            self.k_grid = np.arange(p.K_MIN, p.K_MAX + step_size, step_size)
        if not ignore_b_grid:
            self.b_grid = np.arange(p.B_MIN, p.B_MAX + step_size, step_size)
        if not ignore_omega_grid:
            self.omega_grid = np.arange(p.OMEGA_MIN, p.OMEGA_MAX + step_size, step_size)
        if verbose:
            print(f"[reset_grids] K n={len(self.k_grid)} B n={len(self.b_grid)} Ω n={len(self.omega_grid)}")

    def assert_grids_admit_optimum(self, tol=1e-8):
        for zz in np.atleast_1d(self.z_grid):
            ks = self.k_star(z=float(zz))
            assert np.min(np.abs(self.k_grid - ks)) < tol, f"k*({zz:.3f}) not on k_grid"
            wb = self.omega_bar(z=float(zz))
            assert np.min(np.abs(self.omega_grid - wb)) < tol, f"omega_bar({zz:.3f}) not on grid"
            assert self.b_grid[0] <= ks - self.omega_grid[-1], f"B_MIN too tight at z={zz:.3f}"
        return True
    
    def set_adaptive_menu(self, value: bool):
        self._adaptive_menu = value

    @property
    def downpayment(self) -> float:
        p = self.p
        return 1.0 - p.THETA_K * (1.0 - p.DELTA) / (1.0 + p.R)
    
    @property
    def adaptive_menu(self) -> bool:
        return self._adaptive_menu

    def k_star(self, *, z: float = 1.0):
        p = self.p
        z = np.asarray(z, float)
        alpha = getattr(self.production, "alpha", None)
        if alpha is not None:
            return (z * alpha / (p.R + p.DELTA)) ** (1.0 / (1.0 - alpha))
        flat = np.atleast_1d(z)
        out = np.array([brentq(lambda k: self.production.marginal_product(z=zi, k=k) - (p.R + p.DELTA),
                               1e-8, 1e6) for zi in flat])
        return out.reshape(z.shape) if z.ndim else float(out[0])

    def pi_star(self, *, z: float = 1.0):
        k = self.k_star(z=z)
        return self.production(z=z, k=k) - (self.p.R + self.p.DELTA) * k

    def omega_bar(self, *, z: float = 1.0):
        return (1.0 + self.p.R) * self.downpayment * self.k_star(z=z) + self.pi_star(z=z)

    def consumption(self, *, omega=None, kp=None, bp=None):
        return self._arr(omega, self.omega0) + self._arr(bp, 0.0) - self._arr(kp, self.k_star())

    def equity(self, *, kp=None, bp=None):
        return self._arr(kp, self.k_star()) - self._arr(bp, 0.0)

    def collateral_cap(self, *, kp=None):
        p = self.p
        return p.THETA_K * (1.0 - p.DELTA) * self._arr(kp, self.k_star()) / (1.0 + p.R)

    def feasible(self, *, omega=None, kp=None, bp=None):
        bp = self._arr(bp, 0.0)
        c_ok = self.consumption(omega=omega, kp=kp, bp=bp) > self.p.C_MIN
        b_ok = bp <= self.collateral_cap(kp=kp) + 1e-10
        return c_ok & b_ok

    def next_worth(self, *, kp=None, bp=None, z: float = 1.0):
        p = self.p
        z = np.asarray(z, float)
        kp = self._arr(kp, self.k_star(z=z)); bp = self._arr(bp, 0.0)
        return self.production(z=z, k=kp) + kp * (1.0 - p.DELTA) - bp * (1.0 + p.R)

    def flow_utility(self, *, omega=None, kp=None, bp=None):
        return self.utility(self.consumption(omega=omega, kp=kp, bp=bp))

    def action_query_grid(self, *, omega=None, z: float = 1.0) -> np.ndarray:
        r"""Feasible (kp, bp) mesh at (z, omega). Cols (z, omega, kp, bp)."""
        omega = self.omega0 if omega is None else omega
        K, B = np.meshgrid(self.k_grid, self.b_grid)
        kp, bp = K.ravel(), B.ravel()
        m = self.feasible(omega=omega, kp=kp, bp=bp)
        kp, bp = kp[m], bp[m]
        if kp.size == 0:
            kp, bp = np.array([self.k_grid[0]]), np.array([0.0])
        return np.column_stack([np.full_like(kp, z), np.full_like(kp, omega), kp, bp])

    def _z_step(self, z, rng):
        return float(np.ravel(self.productivity.step(z=np.array([z]), rng=rng))[0])

    def draw_z_paths(self, *, n_agents, T, seed=None):
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        Z = np.empty((T, n_agents)); Z[0] = self.productivity.initial(n=n_agents, rng=rng)
        for t in range(T - 1):
            Z[t + 1] = self.productivity.step(z=Z[t], rng=rng)
        return Z

    def draw_death_paths(self, *, n_agents, T, seed=None):
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        return np.array([self.death.draw(n=n_agents, rng=rng) for _ in range(T)])

    def initial_state(self, *, z0: float = 1.0):
        return (float(z0), float(self.omega0))

    def candidate_actions(self, *, state) -> np.ndarray:
        if self.adaptive_menu:
            z, omega = state
            return self.candidate_actions_adaptive(self, z=z, omega=omega)
        z, omega = state
        return self.action_query_grid(omega=omega, z=z)[:, 2:4]     # (kp, bp) pairs
    
    def candidate_actions_adaptive(self, env, z, omega, nk=25, nb=25):
        r"""Build the (k',b') menu INSIDE the feasible box at (z, omega):
        k' in [tiny, max affordable at full leverage], b' in [lend, collateral cap]."""
        p = env.p
        wp = self.downpayment
        k_max = min((omega - p.C_MIN) / wp * 0.999, env.k_grid[-1])
        k_max = max(k_max, 0.02)
        kg = np.linspace(0.01, k_max, nk)
        rows = []
        for k_ in kg:
            b_hi = float(np.ravel(env.collateral_cap(kp=k_))[0])  # borrow to cap
            b_lo = max(k_ - omega + p.C_MIN + 1e-9, -3.0 * k_max) # c floor / lending floor
            if b_hi <= b_lo:
                continue
            for b_ in np.linspace(b_lo, b_hi, nb):
                rows.append((k_, b_))
        if not rows:
            rows = [(0.01, 0.0)]
        return np.asarray(rows)

    def featurize(self, *, state, actions) -> np.ndarray:
        z, omega = state
        actions = np.atleast_2d(actions)
        n = len(actions)
        return np.column_stack([np.full(n, z), np.full(n, omega), actions[:, 0], actions[:, 1]])

    def reward(self, *, state, action) -> float:
        z, omega = state; kp, bp = action
        return float(np.ravel(self.flow_utility(omega=omega, kp=kp, bp=bp))[0])

    def transition(self, *, state, action, rng=None):
        z, omega = state; kp, bp = action
        rng = self.rng if rng is None else rng
        omega_next = float(np.ravel(self.next_worth(kp=kp, bp=bp, z=z))[0])
        return (self._z_step(z, rng), omega_next)

    def gp_target(self, *, state, action, reward) -> float:
        r"""What the GP learns for (state, action): just u(c). b' is an explicit action,
        so no debt correction. Override per-env if needed."""
        return reward


class EntrepreneurAgent(ABC):
    r"""Base for entrepreneur agents. State (z, omega), action (kp, bp)."""

    def __init__(self, env, name: str):
        self.env = env
        self.p = env.p
        self.name = name

    def fit(self, **kwargs):
        return self

    @abstractmethod
    def policy(self, *, z=1.0, omega=None, t=None) -> tuple[float, float]: ...

    @abstractmethod
    def fixed_point(self, *, z=1.0) -> float: ...

    def policy_vec(self, *, z, omega):
        z = np.atleast_1d(np.asarray(z, float)); omega = np.atleast_1d(np.asarray(omega, float))
        out = [self.policy(z=float(zi), omega=float(wi)) for zi, wi in zip(z, omega)]
        kp, bp = (np.array(v) for v in zip(*out))
        return kp, bp

    def omega_next(self, *, z=1.0, omega=None):
        kp, bp = self.policy(z=z, omega=omega)
        return float(np.ravel(self.env.next_worth(kp=kp, bp=bp, z=z))[0])

    def _z_index(self, z):
        return int(np.argmin(np.abs(self.env.z_grid - z)))

    def simulate(self, *, T=80, z0=None, omega0=None, seed=0):
        env = self.env
        omega0 = env.omega0 if omega0 is None else omega0
        Z = env.draw_z_paths(n_agents=1, T=T, seed=seed)[:, 0]
        if z0 is not None:
            Z[0] = z0
        t = np.arange(T); omega = np.empty(T); c = np.empty(T); kp = np.empty(T); bp = np.empty(T)
        omega[0] = float(omega0)
        for tt in range(T):
            kp[tt], bp[tt] = self.policy(z=float(Z[tt]), omega=float(omega[tt]), t=tt)
            c[tt] = float(np.ravel(env.consumption(omega=omega[tt], kp=kp[tt], bp=bp[tt]))[0])
            if tt < T - 1:
                omega[tt + 1] = float(np.ravel(env.next_worth(kp=kp[tt], bp=bp[tt], z=Z[tt]))[0])
        return {"t": t, "z": Z, "omega": omega, "c": c, "kp": kp, "bp": bp}

    def simulate_panel(self, *, n_agents, T=80, omega0=None, z_seed=0, death_seed=1):
        env = self.env
        omega0 = env.omega0 if omega0 is None else omega0
        rng_z = np.random.default_rng(z_seed)
        Z = np.empty((T, n_agents)); Z[0] = env.productivity.initial(n=n_agents, rng=rng_z)
        death = env.draw_death_paths(n_agents=n_agents, T=T, seed=death_seed)
        OM = np.empty((T, n_agents)); C = np.empty((T, n_agents))
        KP = np.empty((T, n_agents)); BP = np.empty((T, n_agents))
        OM[0] = omega0
        for tt in range(T):
            KP[tt], BP[tt] = self.policy_vec(z=Z[tt], omega=OM[tt])
            C[tt] = env.consumption(omega=OM[tt], kp=KP[tt], bp=BP[tt])
            if tt < T - 1:
                OM[tt + 1] = np.ravel(env.next_worth(kp=KP[tt], bp=BP[tt], z=Z[tt])) # inlined
                Z[tt + 1] = env.productivity.step(z=Z[tt], rng=rng_z)
                nb = death[tt + 1]
                OM[tt + 1] = np.where(nb, omega0, OM[tt + 1])
                if nb.any(): # newborns redraw their type
                    Z[tt + 1][nb] = env.productivity.initial(n=int(nb.sum()), rng=rng_z)
        return {"t": np.arange(T), "z": Z, "omega": OM, "c": C, "kp": KP, "bp": BP, "death": death}

class VFIEntrepreneurAgent(EntrepreneurAgent):
    def __init__(self, env):
        super().__init__(env, name="VFI (kp, bp)")
        nz, no = env.actual_nz, len(env.omega_grid)
        self.V = np.tile(env.utility(np.maximum(env.omega_grid, 1e-8)) / (1.0 - env.p.disc), (nz, 1))
        self.kp_pol = np.zeros((nz, no)); self.bp_pol = np.zeros((nz, no))
        self._cache = None

    def _build_cache(self):
        env = self.env; cache = {}
        for iz, z in enumerate(env.z_grid):
            for io, omega in enumerate(env.omega_grid):
                A = env.action_query_grid(omega=omega, z=z)
                kp, bp = A[:, 2], A[:, 3]
                flow = env.flow_utility(omega=omega, kp=kp, bp=bp)      # <- flow_utility, not reward
                wn = env.next_worth(kp=kp, bp=bp, z=z)
                cache[(iz, io)] = (kp, bp, flow, wn)
        self._cache = cache

    def _refine_policy_continuous_b(self, *, n_gold=60, tie=1e-8):
        r"""Re-extract the policy with b' solved CONTINUOUSLY given each k' (concave 1-D
        inner problem, vectorized golden-section). Kills the lattice-consumption-tuning
        overshoot: on the discrete (k,b) mesh the firm over-holds capital to fine-tune c,
        because the cost of k'>k* is second-order while hitting the right c is first-order."""
        env = self.env; p = env.p
        og, kg = env.omega_grid, env.k_grid
        EV_all = env.P @ self.V
        invphi = (np.sqrt(5) - 1) / 2
        for iz, zz in enumerate(env.z_grid):
            z = float(zz); EV = EV_all[iz]
            cap = env.collateral_cap(kp=kg)                       # b' upper bound per k'
            prod = env.production(z=z, k=kg) + kg * (1 - p.DELTA)
            for io, omega in enumerate(og):
                b_hi = cap.copy()
                b_lo = np.maximum(p.C_MIN + kg - omega + 1e-12, env.b_grid[0])  # c > C_MIN
                ok = b_hi > b_lo
                a, b = b_lo.copy(), b_hi.copy()
                def val(bv):
                    c = omega + bv - kg
                    wn = prod - bv * (1 + p.R)
                    return np.where(ok, env.utility(np.maximum(c, 1e-12))
                                    + p.disc * np.interp(wn, og, EV), -1e18)
                c1 = b - invphi * (b - a); c2 = a + invphi * (b - a)
                f1, f2 = val(c1), val(c2)
                for _ in range(n_gold):
                    m = f1 > f2
                    b = np.where(m, c2, b); a = np.where(m, a, c1)
                    c1 = b - invphi * (b - a); c2 = a + invphi * (b - a)
                    f1, f2 = val(c1), val(c2)
                bstar = 0.5 * (a + b); vals = val(bstar)
                j = int(np.argmax(vals - tie * kg))
                self.kp_pol[iz, io], self.bp_pol[iz, io] = kg[j], bstar[j]

    def fit(self, *, tol=1e-6, max_iter=1000):
        env = self.env; disc = env.p.disc; og = env.omega_grid; no = len(og)
        if self._cache is None: self._build_cache()
        for _ in range(max_iter):
            EV = env.P @ self.V
            Vn = np.empty_like(self.V)
            for iz in range(env.actual_nz):
                evz = EV[iz]
                for io in range(no):
                    kp, bp, flow, wn = self._cache[(iz, io)]
                    vals = flow + disc * np.interp(wn, og, evz)
                    j = int(np.argmax(vals - 1e-8 * kp))              # tie-break: never over-invest
                    Vn[iz, io] = vals[j]
                    self.kp_pol[iz, io], self.bp_pol[iz, io] = kp[j], bp[j]
            if np.max(np.abs(Vn - self.V)) < tol:
                self.V = Vn; break
            self.V = Vn

        self._refine_policy_continuous_b()
        return self

    def policy(self, *, z=1.0, omega=None, t=None):
        omega = self.p.OMEGA_ZERO if omega is None else omega
        iz = self._z_index(z); og = self.env.omega_grid
        return float(np.interp(omega, og, self.kp_pol[iz])), float(np.interp(omega, og, self.bp_pol[iz]))

    def policy_vec(self, *, z, omega):
        z = np.atleast_1d(np.asarray(z, float)); omega = np.atleast_1d(np.asarray(omega, float))
        og = self.env.omega_grid
        iz = np.abs(self.env.z_grid[:, None] - z[None, :]).argmin(axis=0)
        kp = np.empty_like(omega); bp = np.empty_like(omega)
        for j in np.unique(iz):
            m = iz == j
            kp[m] = np.interp(omega[m], og, self.kp_pol[j])
            bp[m] = np.interp(omega[m], og, self.bp_pol[j])
        return kp, bp

    def value(self, *, z=1.0, omega=None):
        omega = self.p.OMEGA_ZERO if omega is None else omega
        iz = self._z_index(z)
        return float(np.interp(omega, self.env.omega_grid, self.V[iz]))

    def fixed_point(self, *, z=1.0):
        return float(np.ravel(self.env.omega_bar(z=z))[0])
    

class ExperienceReasoningAgent(EntrepreneurAgent):

    def __init__(self, env, gp, agent_params, experience_only=False, seed=42):
        super().__init__(env, name="Experience-Reasoning Agent")
        self.gp = gp
        self.agent_params = agent_params
        self.experience_only = experience_only
        self.rng = np.random.default_rng(seed)

    def get_beliefs(self, state):
        actions = self.env.candidate_actions(state=state)
        X_q = self.env.featurize(state=state, actions=actions)
        mean, std = self.gp.predict(X_q, return_std=True)
        return actions, X_q, mean, std

    def _entropy_policy(self, q, std):
        N = len(q); H_max = np.log(N)
        lb = self.agent_params.H * float(np.sum(std ** 2))
        if lb <= 1e-8:
            p = np.zeros(N); p[int(np.argmax(q))] = 1.0
            return p, 0.0
        if lb >= H_max - 1e-8:
            return np.ones(N) / N, np.inf

        def f(delta):
            logits = q / max(delta, 1e-12); logits -= logits.max()
            pp = np.exp(logits); pp /= pp.sum()
            return -np.sum(pp * np.log(pp + 1e-12)) - lb

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
        _, Sigma_E = self.gp.predict_full(X_q)
        vals_E, V = np.linalg.eigh(Sigma_E)
        water_level = kappa / (self.agent_params.H * delta_E)
        vals_R = np.minimum(vals_E, water_level)
        Sigma_R = V @ np.diag(vals_R) @ V.T
        return np.sqrt(np.maximum(np.diag(Sigma_R), 0))

    def eigenvalues(self, X_q):
        _, Sigma_E = self.gp.predict_full(X_q)
        return np.maximum(np.linalg.eigh(Sigma_E)[0], 0)

    def policy(self, *, z=1.0, omega=None, t=None):
        omega = self.p.OMEGA_ZERO if omega is None else omega
        state = (z, omega)
        actions, X_q, mean, std_E = self.get_beliefs(state)
        _, delta_E = self._entropy_policy(mean, std_E)
        std_R = self.reason(X_q, delta_E, std_E)
        probs_R, _ = self._entropy_policy(mean, std_R)
        j = self.rng.choice(len(actions), p=probs_R)
        kp, bp = actions[j]
        return float(kp), float(bp)

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
        kp, bp = probs_R @ actions
        return float(kp), float(bp)

    def fixed_point(self, *, z=1.0):
        og = self.env.omega_grid
        diffs = [abs(self.omega_next(z=z, omega=float(w)) - float(w)) for w in og]
        return float(og[int(np.argmin(diffs))])

    def observe(self, *, state, action, reward, rng=None):
        X = self.env.featurize(state=state, actions=np.atleast_2d(action))
        y = self.env.gp_target(state=state, action=action, reward=reward)
        self.gp.add_observation(X, np.atleast_1d(y))
        return self.env.transition(state=state, action=action, rng=rng)