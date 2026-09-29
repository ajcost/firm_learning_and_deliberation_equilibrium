from abc import ABC, abstractmethod

import numpy as np


class Agent(ABC):
    r"""Base for entrepreneur agents. State ``(z, omega)``, action ``(k', b')``."""

    def __init__(self, env, name: str):
        self.env = env
        self.p = env.p
        self.name = name

    def fit(self, **kwargs):
        return self

    @abstractmethod
    def policy(self, state) -> tuple[float, float]:
        r"""Action ``(k', b')`` at ``state = (z, omega)``."""

    @abstractmethod
    def fixed_point(self, *, z=1.0) -> float:
        r"""Rest point of net worth, :math:`\omega'(\bar\omega) = \bar\omega`, at ``z``."""

    def policy_vec(self, *, z, omega):
        r"""Vectorize :meth:`policy` over matched ``(z, omega)`` arrays.

        Args:
            z (array_like): Productivities, shape ``(n,)``.
            omega (array_like): Net worths, shape ``(n,)``.

        Returns:
            tuple: ``(kp, bp)``, each shape ``(n,)``.
        """
        z = np.atleast_1d(np.asarray(z, float))
        omega = np.atleast_1d(np.asarray(omega, float))
        out = [self.policy((float(zi), float(wi))) for zi, wi in zip(z, omega)]
        kp, bp = (np.array(v) for v in zip(*out))
        return kp, bp

    def omega_next(self, state):
        r"""Net worth one period on from ``state`` under this agent's action.

        Args:
            state (tuple): ``(z, omega)``.

        Returns:
            float: :math:`\omega'`.
        """
        z, _ = state
        kp, bp = self.policy(state)
        return float(np.ravel(self.env.next_worth(kp=kp, bp=bp, z=z))[0])

    def _z_index(self, z):
        return int(np.argmin(np.abs(self.env.z_grid - z)))

    def simulate(self, *, T=80, z0=None, omega0=None, seed=0):
        r"""One life path of ``T`` periods; no death, no learning.

        Args:
            T (int): Number of periods.
            z0 (float, optional): Overrides the first drawn productivity.
            omega0 (float, optional): Initial net worth; defaults to the environment's.
            seed (int): Seed for the productivity path.

        Returns:
            dict: ``t``, ``z``, ``omega``, ``c``, ``kp``, ``bp``, each of length ``T``.
        """
        env = self.env
        omega0 = env.omega0 if omega0 is None else omega0
        Z = env.draw_z_paths(n_agents=1, T=T, seed=seed)[:, 0]
        if z0 is not None:
            Z[0] = z0
        t = np.arange(T)
        omega = np.empty(T)
        c = np.empty(T)
        kp = np.empty(T)
        bp = np.empty(T)
        omega[0] = float(omega0)
        for tt in range(T):
            kp[tt], bp[tt] = self.policy((float(Z[tt]), float(omega[tt])))
            c[tt] = float(np.ravel(env.consumption(omega=omega[tt], kp=kp[tt], bp=bp[tt]))[0])
            if tt < T - 1:
                omega[tt + 1] = float(np.ravel(env.next_worth(kp=kp[tt], bp=bp[tt], z=Z[tt]))[0])
        return {"t": t, "z": Z, "omega": omega, "c": c, "kp": kp, "bp": bp}

    def simulate_panel(self, *, n_agents, T=80, omega0=None, z_seed=0, death_seed=1):
        r"""Cross section of ``n_agents`` paths with death and rebirth at ``omega0``.

        Args:
            n_agents (int): Panel width.
            T (int): Number of periods.
            omega0 (float, optional): Endowment at birth; defaults to the environment's.
            z_seed (int): Seed for the productivity draws.
            death_seed (int): Seed for the death draws.

        Returns:
            dict: ``t`` ``(T,)`` plus ``z``, ``omega``, ``c``, ``kp``, ``bp``, ``death``,
            each ``(T, n_agents)``.
        """
        env = self.env
        omega0 = env.omega0 if omega0 is None else omega0
        rng_z = np.random.default_rng(z_seed)
        Z = np.empty((T, n_agents))
        Z[0] = env.productivity.initial(n=n_agents, rng=rng_z)
        death = env.draw_death_paths(n_agents=n_agents, T=T, seed=death_seed)
        OM = np.empty((T, n_agents))
        C = np.empty((T, n_agents))
        KP = np.empty((T, n_agents))
        BP = np.empty((T, n_agents))
        OM[0] = omega0
        for tt in range(T):
            KP[tt], BP[tt] = self.policy_vec(z=Z[tt], omega=OM[tt])
            C[tt] = env.consumption(omega=OM[tt], kp=KP[tt], bp=BP[tt])
            if tt < T - 1:
                OM[tt + 1] = np.ravel(env.next_worth(kp=KP[tt], bp=BP[tt], z=Z[tt]))  # inlined
                Z[tt + 1] = env.productivity.step(z=Z[tt], rng=rng_z)
                nb = death[tt + 1]
                OM[tt + 1] = np.where(nb, omega0, OM[tt + 1])
                if nb.any():  # newborns redraw their type
                    Z[tt + 1][nb] = env.productivity.initial(n=int(nb.sum()), rng=rng_z)
        return {"t": np.arange(T), "z": Z, "omega": OM, "c": C, "kp": KP, "bp": BP, "death": death}
