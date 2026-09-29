from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import interp1d

from .base import Agent


class VFIEntrepreneurAgent(Agent):
    def __init__(self, env):
        super().__init__(env, name="VFI (kp, bp)")
        nz, no = env.actual_nz, len(env.omega_grid)
        self.V = np.tile(
            env.utility(np.maximum(env.omega_grid, 1e-8)) / (1.0 - env.p.disc), (nz, 1)
        )
        self.kp_pol = np.zeros((nz, no))
        self.bp_pol = np.zeros((nz, no))
        self._cache = None
        self._v_interp = None
        self._v_interp_for = None

    def _build_cache(self, *, n_b_cache: int = 48):
        env = self.env
        p = env.p
        kg = env.k_grid
        b_coarse = np.linspace(float(env.b_grid[0]), float(env.b_grid[-1]), n_b_cache)
        K, B = np.meshgrid(kg, b_coarse)
        kp_full, bp_full = K.ravel(), B.ravel()
        b_ok = bp_full <= np.ravel(env.collateral_cap(kp=kp_full)) + 1e-10  # omega-independent

        self._omega_cache = {}
        self._wn_cache = {}
        for io, omega in enumerate(env.omega_grid):
            m = ((omega + bp_full - kp_full) > p.C_MIN) & b_ok
            kp, bp = kp_full[m], bp_full[m]
            if kp.size == 0:
                kp, bp = kg[:1].copy(), np.zeros(1)
            flow = np.asarray(env.flow_utility(omega=omega, kp=kp, bp=bp), dtype=np.float64)
            self._omega_cache[io] = (kp.astype(np.float32), bp.astype(np.float32), flow)
            for iz, z in enumerate(env.z_grid):
                wn = env.next_worth(kp=kp, bp=bp, z=float(z))
                self._wn_cache[(iz, io)] = np.asarray(wn, dtype=np.float32)
        self._cache = True  # sentinel: cache is built

    def _refine_policy_continuous_b(self, *, n_gold=60, tie=1e-8):
        env = self.env
        p = env.p
        og, kg = env.omega_grid, env.k_grid
        EV_all = env.P @ self.V
        invphi = (np.sqrt(5) - 1) / 2
        for iz, zz in enumerate(env.z_grid):
            z = float(zz)
            EV = EV_all[iz]
            cap = env.collateral_cap(kp=kg)  # b' upper bound per k'
            prod = env.production(z=z, k=kg) + kg * (1 - p.DELTA)
            for io, omega in enumerate(og):
                b_hi = cap.copy()
                b_lo = np.maximum(p.C_MIN + kg - omega + 1e-12, env.b_grid[0])  # c > C_MIN
                ok = b_hi > b_lo
                if not ok.any():  # no feasible (k',b') at this node
                    self.kp_pol[iz, io] = kg[0]  # least-infeasible fallback:
                    self.bp_pol[iz, io] = float(cap[0])  # smallest k', max consumption (b'=cap)
                    continue
                a, b = b_lo.copy(), b_hi.copy()

                def val(bv):
                    c = omega + bv - kg
                    wn = prod - bv * (1 + p.R)
                    return np.where(
                        ok,
                        env.utility(np.maximum(c, 1e-12)) + p.disc * np.interp(wn, og, EV),
                        -1e18,
                    )

                c1 = b - invphi * (b - a)
                c2 = a + invphi * (b - a)
                f1, f2 = val(c1), val(c2)
                for _ in range(n_gold):
                    m = f1 > f2
                    b = np.where(m, c2, b)
                    a = np.where(m, a, c1)
                    c1 = b - invphi * (b - a)
                    c2 = a + invphi * (b - a)
                    f1, f2 = val(c1), val(c2)
                bstar = 0.5 * (a + b)
                vals = val(bstar)
                j = int(np.argmax(vals - tie * kg))
                kc, bc = kg[j], bstar[j]  # discrete-k optimum (fallback)
                if 0 < j < len(kg) - 1:  # parabolic sub-grid refine of k'
                    x0, x1, x2 = kg[j - 1], kg[j], kg[j + 1]
                    y0, y1, y2 = vals[j - 1], vals[j], vals[j + 1]
                    denom = (x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0)
                    if abs(denom) > 1e-300:
                        kv = (
                            x1
                            - 0.5
                            * ((x1 - x0) ** 2 * (y1 - y2) - (x1 - x2) ** 2 * (y1 - y0))
                            / denom
                        )
                        kv = min(max(kv, x0), x2)  # keep inside the bracket
                        bv, vv = self._best_b(z, omega, kv, EV, n_gold=n_gold)
                        if vv >= y1:  # accept only if it improves
                            kc, bc = kv, bv
                self.kp_pol[iz, io], self.bp_pol[iz, io] = kc, bc

    def _best_b(self, z, omega, k, EV, *, n_gold=60):
        env = self.env
        p = env.p
        og = env.omega_grid
        cap = float(np.ravel(env.collateral_cap(kp=k))[0])
        prod = float(np.ravel(env.production(z=z, k=k))[0]) + k * (1 - p.DELTA)
        b_lo = max(p.C_MIN + k - omega + 1e-12, float(env.b_grid[0]))
        b_hi = cap
        if b_hi <= b_lo:
            return b_lo, -1e18
        invphi = (np.sqrt(5) - 1) / 2

        def v(bv):
            c = max(omega + bv - k, 1e-12)
            wn = prod - bv * (1 + p.R)
            return float(env.utility(c) + p.disc * np.interp(wn, og, EV))

        a, b = b_lo, b_hi
        c1 = b - invphi * (b - a)
        c2 = a + invphi * (b - a)
        f1, f2 = v(c1), v(c2)
        for _ in range(n_gold):
            if f1 > f2:
                b = c2
            else:
                a = c1
            c1 = b - invphi * (b - a)
            c2 = a + invphi * (b - a)
            f1, f2 = v(c1), v(c2)
        bstar = 0.5 * (a + b)
        return bstar, v(bstar)

    def fit(self, *, tol=1e-6, max_iter=1000, n_eval=40, n_howard=20, pol_tol=1e-5):
        env = self.env
        disc = env.p.disc
        og = env.omega_grid
        no = len(og)
        nz = env.actual_nz
        if self._cache is None:
            self._build_cache()
        flow_star = np.empty((nz, no))
        wn_star = np.empty((nz, no))
        for _ in range(max_iter):
            # --- greedy maximization sweep: argmax over the cached action set ----------
            EV = env.P @ self.V
            Vn = np.empty_like(self.V)
            for iz in range(nz):
                evz = EV[iz]
                for io in range(no):
                    kp, bp, flow = self._omega_cache[io]
                    wn = self._wn_cache[(iz, io)]
                    vals = flow + disc * np.interp(wn, og, evz)  # flat clamp: contraction-safe
                    j = int(np.argmax(vals - 1e-8 * kp))  # tie-break: never over-invest
                    Vn[iz, io] = vals[j]
                    self.kp_pol[iz, io] = kp[j]
                    self.bp_pol[iz, io] = bp[j]
                    flow_star[iz, io] = flow[j]
                    wn_star[iz, io] = wn[j]
            converged = np.max(np.abs(Vn - self.V)) < tol
            self.V = Vn
            if converged:
                break
            # --- Howard evaluation sweeps: fixed greedy policy, one interp per cell ----
            for _ in range(n_eval):
                EVp = env.P @ self.V
                Ve = np.empty_like(self.V)
                for iz in range(nz):
                    Ve[iz] = flow_star[iz] + disc * np.interp(wn_star[iz], og, EVp[iz])
                gap = np.max(np.abs(Ve - self.V))
                self.V = Ve
                if gap < tol:
                    break

        # continuous policy iteration: refine (k',b') off-grid, re-value, repeat until the
        # policy stops moving. Smooths V' (hence the b'/velocity sawtooth) and leaves V
        # consistent with the policy the agent actually plays.
        prev = None
        for _ in range(n_howard):
            self._refine_policy_continuous_b()
            self._evaluate_policy(tol=tol)
            cur = np.concatenate([self.kp_pol.ravel(), self.bp_pol.ravel()])
            if prev is not None and np.max(np.abs(cur - prev)) < pol_tol:
                break
            prev = cur
        return self

    def _evaluate_policy(self, *, tol=1e-8, max_iter=2000):
        env = self.env
        p = env.p
        disc = p.disc
        og = env.omega_grid
        c = np.maximum(og[None, :] + self.bp_pol - self.kp_pol, 1e-12)  # (nz, no)
        flow = env.utility(c)
        z = np.atleast_1d(env.z_grid).astype(float)[:, None]  # (nz, 1)
        wn = (
            env.production(z=z, k=self.kp_pol)
            + self.kp_pol * (1 - p.DELTA)
            - self.bp_pol * (1 + p.R)
        )
        for _ in range(max_iter):
            EV = env.P @ self.V
            Vn = np.empty_like(self.V)
            for iz in range(env.actual_nz):
                Vn[iz] = flow[iz] + disc * np.interp(wn[iz], og, EV[iz])
            gap = np.max(np.abs(Vn - self.V))
            self.V = Vn
            if gap < tol:
                break
        wbar_max = float(
            max(np.ravel(env.omega_bar(z=float(zz)))[0] for zz in np.atleast_1d(env.z_grid))
        )
        if og[-1] < wbar_max - 1e-9:
            warnings.warn(
                f"[VFI] omega_grid top ({og[-1]:.3g}) is below the highest rest point "
                f"omega_bar={wbar_max:.3g}: the ergodic region is off-grid and V/policy are "
                f"biased there. Widen OMEGA_MAX / omega_headroom (calibrate_grids does this)."
            )

    def policy(self, state):
        r"""Optimal ``(k', b')`` at ``state``, linearly interpolated in :math:`\omega`.

        Args:
            state (tuple): ``(z, omega)``.

        Returns:
            Tuple[float, float]: ``(kp, bp)``.
        """
        z, omega = state
        iz = self._z_index(z)
        og = self.env.omega_grid
        return float(np.interp(omega, og, self.kp_pol[iz])), float(
            np.interp(omega, og, self.bp_pol[iz])
        )

    def policy_vec(self, *, z, omega):
        z = np.atleast_1d(np.asarray(z, float))
        omega = np.atleast_1d(np.asarray(omega, float))
        og = self.env.omega_grid
        iz = np.abs(self.env.z_grid[:, None] - z[None, :]).argmin(axis=0)
        kp = np.empty_like(omega)
        bp = np.empty_like(omega)
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
        r"""Analytic rest point :math:`\bar\omega(z)` of net worth."""
        return float(np.ravel(self.env.omega_bar(z=z))[0])

    def action_value(self, X, *, gamma=None, prior_z_index=None):
        r"""Action value :math:`Q^*(z, \omega, k', b')` implied by the solved ``V``.

        Bellman decomposition with the stochastic continuation:

        .. math::
            Q^*(z, \omega, k', b') = u(c) + \gamma\, \mathbb{E}\big[V(z', \omega') \mid z\big]
                                   = u(c) + \gamma \sum_j P_{iz(z), j}\, V_j(\omega'),

        with :math:`c = \omega + b' - k'` and
        :math:`\omega' = z f(k') + (1-\delta)k' - (1+R)b'`.

        Args:
            X (np.ndarray): Query features, rows ``(z, omega, k', b')``.
            gamma (float, optional): Continuation discount. Defaults to the environment's
                ``disc = BETA * (1 - rho)``.
            prior_z_index (int, optional): If set, the continuation conditions on
                ``z_grid[prior_z_index]`` instead of the query's ``z``: beliefs calibrated to
                that type while living at the real ``z`` (the flow and :math:`\omega'` still
                use the actual query ``z``). ``None`` (default) gives the correct value.

        Returns:
            np.ndarray: :math:`Q^*` at each row of ``X``, shape ``(n,)``.
        """
        env = self.env
        p = env.p
        g = p.disc if gamma is None else gamma
        z_grid, P = env.z_grid, env.P
        X = np.atleast_2d(X)
        z, omega, kp, bp = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        c = omega + bp - kp
        flow = env.utility(np.maximum(c, p.C_MIN))
        omega_next = env.next_worth(z=z, kp=kp, bp=bp)

        if prior_z_index is None:
            iz = np.abs(z_grid[:, None] - z[None, :]).argmin(axis=0)
        else:
            iz = np.full(len(z), prior_z_index, dtype=int)

        # continuation: E[V(z', w') | z_iz] = sum_j P[iz, j] * V_j(w')
        Vmat = np.column_stack([f(omega_next) for f in self._value_interpolators()])
        cont = np.sum(P[iz, :] * Vmat, axis=1)  # row-wise dot
        return flow + g * cont

    def _value_interpolators(self):
        r"""Per-z linear interpolants of ``V`` in :math:`\omega`."""
        if self._v_interp_for is not self.V:
            og = self.env.omega_grid
            self._v_interp = [
                interp1d(og, self.V[iz], kind="linear", fill_value="extrapolate")
                for iz in range(len(self.env.z_grid))
            ]
            self._v_interp_for = self.V
        return self._v_interp
