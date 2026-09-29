from dataclasses import replace

import numpy as np

from .base import Agent


class ExperienceReasoningAgent(Agent):
    r"""Economic agent that learns from experience and from reasoning."""

    def __init__(self, env, belief, planner, oracle=None, seed: int = 0):
        super().__init__(env, name="Experience-Reasoning Agent")
        self.belief = belief
        self.planner = planner
        self.oracle = oracle
        self.rng = np.random.default_rng(seed)

    @property
    def experience_only(self) -> bool:
        return self.oracle is None

    def _can_reason(self, plan, reason: bool) -> bool:
        r"""Check if plan can be reasoned about; it is not degenerate.

        Checks for degenerate conditions: no oracle, no signals, or degenerate temperature :math:`\delta = 0` greedy, :math:`\delta = \infty` uniform.

        Args:
            plan (Plan): The plan to check.
            reason (bool): Whether reasoning is allowed for this plan.

        Returns:
            bool: ``True`` if the plan is not degenerate and reasoning is possible.
        """
        return (
            reason and not self.experience_only and bool(plan.signals) and 0.0 < plan.delta < np.inf
        )

    def _query_oracle(self, points, signals) -> int:
        r"""Query the oracle for each signal (eigenvector) and file the answer as a functional observation.

        Args:
            points (np.ndarray): Menu features, shape ``(N, d)``.
            signals (tuple): :class:`Signal` s from the plan.

        Returns:
            int: Number of signals bought.
        """
        for s in signals:
            r = self.oracle.query(points, s.w, s.noise, self.rng)
            self.belief.add_functional(X=points, w=s.w, r=r, noise=s.noise)
        return len(signals)

    def act(self, state, *, reason: bool = True):
        r"""Run a decision period at ``state``.

        .. math::
            p_t(a) = \frac{\exp(\hat Q_t^{\text{post}}(a)/\delta_t)\, \Delta_a}
                          {\sum_{a'} \exp(\hat Q_t^{\text{post}}(a')/\delta_t)\, \Delta_{a'}} .

        Args:
            state (tuple): ``(z, omega)``.
            reason (bool): Set ``False`` to price the menu without buying anything.

        Returns:
            Tuple[Menu, Decision]: The menu and the decision actually acted on;
            ``decision.probs`` are post-reasoning choice probabilities and
            ``decision.plan`` is the plan they were bought under.
        """
        menu = self.env.menu(state)
        mean, Sigma = self.belief.predict_full(menu.points)
        decision = self.planner.solve(np.ravel(mean), Sigma, menu.measures)
        if not self._can_reason(decision.plan, reason):
            return menu, decision
        self._query_oracle(menu.points, decision.plan.signals)
        probs = self.planner.softmax(
            self.belief.predict(menu.points), decision.plan.delta, menu.measures
        )
        return menu, replace(decision, probs=probs)

    def observe(self, state, action, reward, state_next) -> None:
        r"""File one GPTD (experience) observation.

        Decision point :math:`x_{\text{dec}} = (s_t, a_t)`; outcome point
        :math:`x_{\text{out}} = (s_{t+1}, \tilde a_{t+1})` with :math:`\tilde a_{t+1}` the
        greedy action under the current belief (Q-learning target). The reward is filed as

        .. math::
            r_t = Q(x_{\text{dec}}) - \gamma\, Q(x_{\text{out}}) + \epsilon,
            \quad \epsilon \sim \mathcal{N}(0, \sigma_n^2).

        Args:
            state (tuple): ``(z, omega)``.
            action (array_like): ``(kp, bp)`` taken.
            reward (float): Realised flow utility :math:`u(c_t)`.
            state_next (tuple): ``(z', omega')`` the environment returned.
        """
        env = self.env
        x_dec = env.featurize(state=state, actions=np.atleast_2d(action))
        y = env.gp_target(state=state, action=action, reward=reward)
        nxt = env.menu(state_next)
        c_tilde = nxt.actions[int(np.argmax(self.belief.predict(nxt.points)))]
        x_out = env.featurize(state=state_next, actions=np.atleast_2d(c_tilde))
        self.belief.add_observation(x_dec, x_out, float(np.ravel(y)[0]))

    def policy(self, state):
        r"""Sample one action from the entropy-constrained policy at ``state``.

        Runs a full decision period (:meth:`act`) and draws :math:`(k', b') \sim p_t`.

        Args:
            state (tuple): ``(z, omega)``.

        Returns:
            Tuple[float, float]: ``(kp, bp)``.
        """
        menu, decision = self.act(state)
        kp, bp = menu.actions[self.rng.choice(len(menu.actions), p=decision.probs)]
        return float(kp), float(bp)

    def greedy(self, state):
        r"""Argmax of the current posterior mean at ``state``.

        The greedy action which maximizes the current posterior mean Q-value. Perhaps
        not the one chosen under :meth:`policy`.

        .. math::
            (k', b') = \arg\max_a \hat Q_t(a)

        Args:
            state (tuple): ``(z, omega)``.

        Returns:
            Tuple[float, float]: ``(kp, bp)``.
        """
        menu = self.env.menu(state)
        kp, bp = menu.actions[int(np.argmax(self.belief.predict(menu.points)))]
        return float(kp), float(bp)

    def expected(self, state):
        r"""Mean action under the policy at ``state``.

        .. math::
            \mathbb{E}_{p_t}[(k', b')] = \sum_a p_t(a)\, a

        Args:
            state (tuple): ``(z, omega)``.

        Returns:
            Tuple[float, float]: ``(kp, bp)``.
        """
        menu, decision = self.act(state, reason=False)
        kp, bp = decision.probs @ menu.actions
        return float(kp), float(bp)

    def fixed_point(self, *, z=1.0):
        r"""Find grid point closest to a rest point of net worth, :math:`\omega'(\omega) = \omega`.

        Args:
            z (float): Productivity.

        Returns:
            float: :math:`\bar\omega` on ``omega_grid``.
        """
        env = self.env
        og = env.omega_grid
        diffs = []
        for w in og:
            kp, bp = self.expected((z, float(w)))
            diffs.append(abs(float(np.ravel(env.next_worth(kp=kp, bp=bp, z=z))[0]) - float(w)))
        return float(og[int(np.argmin(diffs))])
