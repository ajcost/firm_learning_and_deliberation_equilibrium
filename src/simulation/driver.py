r"""The period loop: act, draw, pay, step, file.

This is the only place the four layers meet in sequence. Everything it does is deliberately
visible — the agent is never asked to advance the world, and the environment is never asked
what the agent believes:

.. code-block:: text

    menu, decision = agent.act(state)      # price the menu, buy what the plan asks for
    action     = draw from decision.probs  # the driver owns the choice randomness
    reward     = env.reward(state, action)
    state_next = env.step(state, action)   # the environment owns the transition
    agent.observe(state, action, reward, state_next)
"""

import numpy as np


def run_episode(env, agent, T, rng, *, state=None):
    r"""Run one learning episode of ``T`` periods and return its path.

    Args:
        env (EntrepreneurEnvironment): The world.
        agent (ExperienceReasoningAgent): Learner; its belief is updated in place.
        T (int): Number of periods.
        rng (np.random.Generator): Drives the choice draw and the productivity transition.
        state (tuple, optional): Starting ``(z, omega)``; defaults to ``env.initial_state()``.

    Returns:
        dict: ``t``, ``z``, ``omega``, ``kp``, ``bp``, ``reward``, ``delta``, ``n_signals``,
        each of length ``T``.
    """
    state = env.initial_state() if state is None else state
    z = np.empty(T)
    omega = np.empty(T)
    kp = np.empty(T)
    bp = np.empty(T)
    reward = np.empty(T)
    delta = np.empty(T)
    n_signals = np.empty(T, dtype=int)
    for t in range(T):
        menu, decision = agent.act(state)
        action = menu.actions[rng.choice(len(menu.actions), p=decision.probs)]
        r = env.reward(state=state, action=action)
        state_next = env.step(state=state, action=action, rng=rng)
        agent.observe(state, action, r, state_next)
        z[t], omega[t] = state
        kp[t], bp[t] = action
        reward[t] = r
        delta[t] = decision.plan.delta
        n_signals[t] = len(decision.plan.signals)
        state = state_next
    return {
        "t": np.arange(T),
        "z": z,
        "omega": omega,
        "kp": kp,
        "bp": bp,
        "reward": reward,
        "delta": delta,
        "n_signals": n_signals,
    }


def run_panel(env, agents, T, rng, *, states=None):
    r"""Run one episode per agent and stack the paths into a cross section.

    The agents are independent learners: each carries its own belief, so the panel is a
    population of separate histories rather than one agent seen many times.

    Args:
        env (EntrepreneurEnvironment): The world.
        agents (list): Learners, one episode each.
        T (int): Number of periods.
        rng (np.random.Generator): Drives every agent's choices and transitions.
        states (list, optional): Per-agent starting ``(z, omega)``; defaults to
            ``env.initial_state()`` for all.

    Returns:
        dict: ``t`` ``(T,)`` plus ``z``, ``omega``, ``kp``, ``bp``, ``reward``, ``delta``,
        ``n_signals``, each ``(T, n_agents)``.
    """
    states = [None] * len(agents) if states is None else states
    paths = [run_episode(env, a, T, rng, state=s) for a, s in zip(agents, states)]
    out = {"t": np.arange(T)}
    for key in ("z", "omega", "kp", "bp", "reward", "delta", "n_signals"):
        out[key] = np.column_stack([p[key] for p in paths])
    return out
