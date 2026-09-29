"""Entrepreneur's problem: net worth omega, productivity z, action (k', b')."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq

from .investment_elements import (
    CRRA,
    CapitalCobbDouglas,
    ConstantHazard,
    DeathProcess,
    PermanentProductivity,
    Production,
    Productivity,
    Utility,
)
from .menu import Menu


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
        self._mesh_sig = None  # invalidates the cached (k,b) action mesh
        self.setup_grids()

    def _arr(self, x, default) -> np.ndarray:
        return np.asarray(default if x is None else x, dtype=float)

    def setup_grids(
        self,
        *,
        calibrate: bool = False,
        k_headroom: float = 1.3,
        b_headroom: float = 3.0,
        omega_headroom: float = 1.5,
    ):
        self.z_grid, self.P = self.productivity.grid()
        self.actual_nz = len(self.z_grid)
        if calibrate:
            self.calibrate_grids(
                k_headroom=k_headroom, b_headroom=b_headroom, omega_headroom=omega_headroom
            )
            return
        p = self.p
        self.omega_grid = np.linspace(p.OMEGA_MIN, p.OMEGA_MAX, p.N_omega)
        k_max = p.K_MAX if p.K_MAX is not None else 1.5 * self.k_star()
        b_max = (
            p.B_MAX if p.B_MAX is not None else float(np.ravel(self.collateral_cap(kp=k_max))[0])
        )
        b_min = p.B_MIN if p.B_MIN is not None else -k_max
        self.k_grid = np.linspace(p.K_MIN, k_max, p.N_k)
        self.b_grid = np.linspace(b_min, b_max, p.N_b)

    def calibrate_grids(self, *, k_headroom=1.3, b_headroom=3.0, omega_headroom=1.5, verbose=True):
        p = self.p
        z_nodes = np.atleast_1d(self.z_grid)
        ks = np.array([self.k_star(z=float(zz)) for zz in z_nodes])
        wb = np.array([self.omega_bar(z=float(zz)) for zz in z_nodes])
        k_top, wbar_top = float(ks.max()), float(wb.max())
        p.K_MIN = min(p.K_MIN, 1e-3)
        p.K_MAX = k_headroom * k_top
        p.B_MAX = float(np.ravel(self.collateral_cap(kp=p.K_MAX))[0])
        p.B_MIN = -b_headroom * k_top
        p.OMEGA_MIN = min(p.OMEGA_MIN, 1e-3)
        p.OMEGA_MAX = omega_headroom * wbar_top
        self.omega_grid = np.linspace(p.OMEGA_MIN, p.OMEGA_MAX, p.N_omega)
        self.k_grid = np.linspace(p.K_MIN, p.K_MAX, p.N_k)
        self.b_grid = np.linspace(p.B_MIN, p.B_MAX, p.N_b)
        caps = np.array([float(np.ravel(self.collateral_cap(kp=k))[0]) for k in ks])
        self.k_grid = np.unique(np.r_[self.k_grid, ks])
        self.b_grid = np.unique(np.r_[self.b_grid, caps, 0.0])
        self.omega_grid = np.unique(np.r_[self.omega_grid, wb])
        if verbose:
            print(
                f"[calibrate] z∈[{z_nodes.min():.3f},{z_nodes.max():.3f}] "
                f"k*∈[{ks.min():.3f},{ks.max():.3f}] ω̄∈[{wb.min():.3f},{wbar_top:.3f}]  "
                f"K n={len(self.k_grid)} B n={len(self.b_grid)} Ω n={len(self.omega_grid)}"
            )

    def reset_grids(
        self,
        step_size=0.1,
        ignore_b_grid=False,
        ignore_k_grid=False,
        ignore_omega_grid=False,
        verbose=False,
    ):
        p = self.p
        if not ignore_k_grid:
            self.k_grid = np.arange(p.K_MIN, p.K_MAX + step_size, step_size)
        if not ignore_b_grid:
            self.b_grid = np.arange(p.B_MIN, p.B_MAX + step_size, step_size)
        if not ignore_omega_grid:
            self.omega_grid = np.arange(p.OMEGA_MIN, p.OMEGA_MAX + step_size, step_size)
        if verbose:
            print(
                f"[reset_grids] K n={len(self.k_grid)} B n={len(self.b_grid)} Ω n={len(self.omega_grid)}"
            )

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
        out = np.array(
            [
                brentq(
                    lambda k: self.production.marginal_product(z=zi, k=k) - (p.R + p.DELTA),
                    1e-8,
                    1e6,
                )
                for zi in flat
            ]
        )
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
        kp = self._arr(kp, self.k_star(z=z))
        bp = self._arr(bp, 0.0)
        return self.production(z=z, k=kp) + kp * (1.0 - p.DELTA) - bp * (1.0 + p.R)

    def flow_utility(self, *, omega=None, kp=None, bp=None):
        return self.utility(self.consumption(omega=omega, kp=kp, bp=bp))

    def _action_mesh(self):
        r"""Raveled (kp, bp) over meshgrid(k_grid, b_grid) plus the omega-INDEPENDENT
        collateral mask (b' <= cap(k')). Built once and reused until a grid is replaced;
        only the consumption feasibility mask depends on omega (recomputed per query)."""
        sig = (id(self.k_grid), id(self.b_grid), self.k_grid.shape[0], self.b_grid.shape[0])
        if self._mesh_sig != sig:
            K, B = np.meshgrid(self.k_grid, self.b_grid)
            kp_full, bp_full = K.ravel(), B.ravel()
            b_ok = bp_full <= np.ravel(self.collateral_cap(kp=kp_full)) + 1e-10
            self._mesh_cache = (kp_full, bp_full, b_ok)
            self._mesh_sig = sig
        return self._mesh_cache

    def _grid_cell_measure(self) -> float:
        r"""Cell measure :math:`\mathrm{d}k\,\mathrm{d}b` of the fixed (k', b') product mesh."""
        dk = (self.k_grid[-1] - self.k_grid[0]) / max(len(self.k_grid) - 1, 1)
        db = (self.b_grid[-1] - self.b_grid[0]) / max(len(self.b_grid) - 1, 1)
        return float(dk * db)

    def action_query_grid(self, *, omega=None, z: float = 1.0):
        r"""Feasible (kp, bp) mesh at (z, omega), with the measure each node stands for.

        The mesh is a product grid, so every cell is the same rectangle
        :math:`\mathrm{d}k\,\mathrm{d}b` and the measures are constant across rows.

        Args:
            omega (float, optional): Net worth; defaults to the environment's.
            z (float): Productivity.

        Returns:
            tuple: ``(X, actions, measures)`` with ``X`` the features, cols
            ``(z, omega, kp, bp)``; ``actions`` ``(N, 2)``; ``measures`` ``(N,)``.
        """
        omega = self.omega0 if omega is None else omega
        kp_full, bp_full, b_ok = self._action_mesh()
        m = ((omega + bp_full - kp_full) > self.p.C_MIN) & b_ok  # c_ok & b_ok
        kp, bp = kp_full[m], bp_full[m]
        if kp.size == 0:
            kp, bp = np.array([self.k_grid[0]]), np.array([0.0])
        X = np.column_stack([np.full_like(kp, z), np.full_like(kp, omega), kp, bp])
        return X, np.column_stack([kp, bp]), np.full(len(kp), self._grid_cell_measure())

    def _z_step(self, z, rng):
        return float(np.ravel(self.productivity.step(z=np.array([z]), rng=rng))[0])

    def draw_z_paths(self, *, n_agents, T, seed=None):
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        Z = np.empty((T, n_agents))
        Z[0] = self.productivity.initial(n=n_agents, rng=rng)
        for t in range(T - 1):
            Z[t + 1] = self.productivity.step(z=Z[t], rng=rng)
        return Z

    def draw_death_paths(self, *, n_agents, T, seed=None):
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        return np.array([self.death.draw(n=n_agents, rng=rng) for _ in range(T)])

    def initial_state(self, *, z0: float = 1.0):
        return (float(z0), float(self.omega0))

    def candidate_actions(self, *, state) -> np.ndarray:
        r"""The menu's actions at ``state``, shape ``(N, 2)``; a thin wrapper over
        :meth:`menu` for callers that do not need the cells."""
        return self.menu(state).actions

    def candidate_actions_adaptive(self, *, z, omega, nk=35, nb=30):
        r"""Midpoint-rule menu on the feasible :math:`(k', b')` set at ``(z, omega)``.

        Nodes are cell centers, so the mesh covers the set exactly:

        .. math::
            k_i = \underline k + \mathrm{d}k\,(i + \tfrac12), \qquad
            b_{ij} = \underline b(k_i) + \mathrm{d}b(k_i)\,(j + \tfrac12), \qquad
            \Delta_{ij} = \mathrm{d}k\, \mathrm{d}b(k_i),

        with :math:`\mathrm{d}k = (\bar k - \underline k)/n_k` and
        :math:`\mathrm{d}b(k) = (\bar b(k) - \underline b(k))/n_b`. The :math:`b'` span depends
        on :math:`k'`, so :math:`\Delta_{ij}` varies across slices and
        :math:`\sum_{ij}\Delta_{ij} = \sum_i \mathrm{d}k\,(\bar b(k_i) - \underline b(k_i))`.

        Args:
            z (float): Productivity; unused (feasibility depends on ``omega`` only).
            omega (float): Net worth.
            nk (int): Number of :math:`k'` cells.
            nb (int): Number of :math:`b'` cells per :math:`k'` slice.

        Returns:
            tuple: ``(actions, measures)`` with shapes ``(N, 2)`` and ``(N,)``.
        """
        p = self.p
        wp = self.downpayment
        k_lo = 0.01
        k_hi = min((omega - p.C_MIN) / wp * 0.999, self.k_grid[-1])
        k_hi = max(k_hi, 0.02)
        dk = (k_hi - k_lo) / nk
        kg = k_lo + dk * (np.arange(nk) + 0.5)  # midpoints, not edges
        rows = []
        measures = []
        for k_ in kg:
            b_hi = float(np.ravel(self.collateral_cap(kp=k_))[0])  # borrow to cap
            b_lo = max(k_ - omega + p.C_MIN + 1e-9, -3.0 * k_hi)  # c floor / lending floor
            if b_hi <= b_lo:
                continue
            db = (b_hi - b_lo) / nb
            for b_ in b_lo + db * (np.arange(nb) + 0.5):
                rows.append((k_, b_))
                measures.append(dk * db)  # Delta = dk db(k)
        if not rows:
            rows, measures = [(k_lo, 0.0)], [1.0]  # degenerate: one cell, nominal measure
        return np.asarray(rows), np.asarray(measures)

    def menu(self, state) -> Menu:
        r"""The feasible menu at ``state``, as a quadrature rule over the action space.

        Args:
            state (tuple): ``(z, omega)``.

        Returns:
            Menu: Actions ``(N, 2)``, measures ``(N,)``, and features ``(N, 4)``.
        """
        z, omega = state
        if self.adaptive_menu:
            actions, measures = self.candidate_actions_adaptive(z=z, omega=omega)
        else:
            _, actions, measures = self.action_query_grid(omega=omega, z=z)
        return Menu(
            state=(float(z), float(omega)),
            actions=actions,
            measures=measures,
            points=self.featurize(state=state, actions=actions),
        )

    def featurize(self, *, state, actions) -> np.ndarray:
        z, omega = state
        actions = np.atleast_2d(actions)
        n = len(actions)
        return np.column_stack([np.full(n, z), np.full(n, omega), actions[:, 0], actions[:, 1]])

    def reward(self, *, state, action) -> float:
        z, omega = state
        kp, bp = action
        return float(np.ravel(self.flow_utility(omega=omega, kp=kp, bp=bp))[0])

    def step(self, *, state, action, rng=None):
        r"""One period: draw :math:`z'` and roll net worth forward under ``action``.

        .. math::
            \omega' = z f(k') + (1-\delta) k' - (1+R) b'

        Args:
            state (tuple): ``(z, omega)``.
            action (array_like): ``(k', b')``.
            rng (np.random.Generator, optional): For the productivity draw; defaults to the
                environment's own generator.

        Returns:
            tuple: ``state_next = (z', omega')``.
        """
        z, omega = state
        kp, bp = action
        rng = self.rng if rng is None else rng
        omega_next = float(np.ravel(self.next_worth(kp=kp, bp=bp, z=z))[0])
        return (self._z_step(z, rng), omega_next)

    def gp_target(self, *, state, action, reward) -> float:
        r"""What the GP learns for (state, action): just u(c). b' is an explicit action,
        so no debt correction. Override per-env if needed."""
        return reward
