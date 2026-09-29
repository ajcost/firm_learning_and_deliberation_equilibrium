"""Does the GP Experience-Reasoning agent LEARN the optimal (VFI) policy?

Three figures:
  (1) learned_vs_rational_z1 — 6-panel phase-diagram-style comparison at z=1: the rational
      VFI policy with the trained agent's learned policy (greedy argmax Q̂ and expected Σπ·a)
      overlaid. If the agent has learned, the two coincide.
  (2) anatomy_matched_z1 — belief V̂(ω) + choice-anatomy snapshots every 5 steps, matched
      beliefs (endowed prior z=1.0, living z=1.0).
  (3) anatomy_biased_true1.26_belief1.0 — same anatomy for the BIASED agent: endowed with a
      z=1.0 "manual" but actually a high (z=1.26) firm; watch beliefs re-learn toward the truth.

Run from anywhere:  python scripts/entrepreneur_env_test_learning.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.agents import ExperienceReasoningAgent, VFIEntrepreneurAgent
from src.simulation.belief import (
    GPBelief, GPBeliefParameters, RBFKernel, make_vfi_prior)
from src.simulation.environment.entrepreneur import (
    EntrepreneurParameters, EntrepreneurEnvironment)
from src.simulation.reasoning import Planner, VFIOracle
from src.simulation.environment.investment_elements import (
    CRRA, CapitalCobbDouglas, PermanentProductivity, ConstantHazard)
from src.visualization.entrepreneur_visualization import apply_style, save_fig, landmarks, ECON

OUT = os.path.dirname(os.path.abspath(__file__))
apply_style(latex=False, base_size=10)
B, O, G, GY, R, GRY = ECON["blue"], ECON["purple"], ECON["green"], ECON["yellow"], ECON["red"], ECON["grey"]


# ============================ model + rational benchmark =============================
def build_env():
    p = EntrepreneurParameters(
        N_k=400, N_b=400, N_omega=200,
        BETA=1.0, THETA_K=0.5, OMEGA_ZERO=5.0,
        utility=CRRA(2.0), production=CapitalCobbDouglas(0.6),
        productivity=PermanentProductivity([0.79, 1.00, 1.26], probs=[1 / 3, 1 / 3, 1 / 3]),
        death=ConstantHazard(0.05),
    )
    p.set_r_knife_edge()
    p.BETA = 0.999                       # impatient: disc*(1+R) < 1
    env = EntrepreneurEnvironment(p)
    env.setup_grids(calibrate=True)
    env.reset_grids(step_size=0.2, ignore_b_grid=True)     # 0.2 keeps the VFI solve quick
    return env, p


@dataclass
class AgentParams:
    H: float = 0.01          # entropy exploration weight
    KAPPA_R: float = 1.0       # reasoning price (unused when experience_only=True)


def make_agent(env, vfi, prior_iz, *, seed=7, experience_only=True):
    r"""GP agent whose prior mean is the VFI Q for productivity node `prior_iz`."""
    kernel = RBFKernel(sigma0=5.0, length_scales=[0.5, 34, 40, 120])
    gp = GPBelief(gamma=env.p.disc,
                  gp_params=GPBeliefParameters(kernel=kernel, sigma_n=0.025),
                  prior_mean_fn=make_vfi_prior(vfi, prior_z_index=prior_iz), input_dim=4)
    return ExperienceReasoningAgent(env, gp, AgentParams(), experience_only=experience_only, seed=seed)


def train_agent(env, agent, z_live, *, n_episodes=30, T=20, omega_span=(2.0, 38.0), seed=0):
    r"""Explore the omega-space: run episodes from random omega0, filing one GPTD tuple per
    step through agent.observe (decision -> greedy outcome -> flow u(c))."""
    rng = np.random.default_rng(seed)
    for _ in range(n_episodes):
        state = (float(z_live), float(rng.uniform(*omega_span)))
        for _ in range(T):
            kp, bp = agent.policy(z=state[0], omega=state[1])         # entropy-sampled action
            u = env.reward(state=state, action=(kp, bp))
            state = agent.observe(state=state, action=(kp, bp), reward=u, rng=rng)
    return agent


# ============================ learned-policy extraction ==============================
def learned_curves(env, agent, z, Wg):
    r"""Greedy (argmax Q̂) and expected (Σπ·a) policy across omega, plus belief std at greedy."""
    kp_g = np.empty_like(Wg); bp_g = np.empty_like(Wg); sd_g = np.empty_like(Wg)
    kp_e = np.full_like(Wg, np.nan); bp_e = np.full_like(Wg, np.nan)
    for i, w in enumerate(Wg):
        A = env.candidate_actions(state=(z, float(w)))
        mu, sd = agent.gp.predict(env.featurize(state=(z, float(w)), actions=A), return_std=True)
        j = int(np.argmax(mu))
        kp_g[i], bp_g[i], sd_g[i] = A[j, 0], A[j, 1], sd[j]
        probs, delta = agent._entropy_policy(mu, sd)
        if np.isfinite(delta):                                       # informed region
            kp_e[i], bp_e[i] = probs @ A[:, 0], probs @ A[:, 1]
    return kp_g, bp_g, sd_g, kp_e, bp_e


def belief_V(env, agent, z, Wg):
    r"""Belief-implied V̂(ω)=max_a Q̂ and the std at that greedy action."""
    Vm, Vs = np.empty(len(Wg)), np.empty(len(Wg))
    for i, w in enumerate(Wg):
        A = env.candidate_actions(state=(z, float(w)))
        mu, sd = agent.gp.predict(env.featurize(state=(z, float(w)), actions=A), return_std=True)
        j = int(np.argmax(mu)); Vm[i], Vs[i] = mu[j], sd[j]
    return Vm, Vs


# ============================ figure 1: learned vs rational ==========================
def plot_learned_vs_rational(env, vfi, agent, z, *, path):
    p = env.p
    iz = int(np.argmin(np.abs(env.z_grid - z)))
    Z, KS, WB = landmarks(env)
    ks, wb = KS[iz], WB[iz]
    Wg = np.linspace(2.0, min(env.omega_grid[-1], 1.35 * wb), 220)
    prod = lambda k: env.production(z=z, k=k) + k * (1 - p.DELTA)
    mp = env.production.marginal_product

    kp_g, bp_g, sd_g, kp_e, bp_e = learned_curves(env, agent, z, Wg)
    kp_v = np.interp(Wg, env.omega_grid, vfi.kp_pol[iz])
    bp_v = np.interp(Wg, env.omega_grid, vfi.bp_pol[iz])
    wn_g = prod(kp_g) - bp_g * (1 + p.R)
    wn_v = prod(kp_v) - bp_v * (1 + p.R)

    fig, ax = plt.subplots(2, 3, figsize=(15.5, 8.6))

    a = ax[0, 0]                                              # transition map
    a.plot(Wg, Wg, ls="--", lw=.8, color=ECON["dark"], label="45°")
    a.plot(Wg, wn_v, color=ECON["dark"], lw=2, label="VFI  ω'")
    a.plot(Wg, wn_g, color=B, lw=2, ls="--", label="learned ω'")
    a.axvline(wb, color=GRY, ls=":", lw=.8)
    a.set_title("Transition map ω'=Φ(ω)"); a.set_xlabel("ω"); a.set_ylabel("ω'"); a.legend()

    a = ax[0, 1]                                              # k' & b' policy
    a.plot(Wg, kp_v, color=ECON["dark"], lw=2, label="VFI k'")
    a.plot(Wg, kp_g, color=B, lw=2, ls="--", label="learned k' (greedy)")
    a.plot(Wg, kp_e, color=B, lw=1.1, ls=":", label="learned k' (expected)")
    a.plot(Wg, bp_v, color=ECON["dark"], lw=1.4, alpha=.6, label="VFI b'")
    a.plot(Wg, bp_g, color=O, lw=2, ls="--", label="learned b'")
    a.axhline(ks, color=GRY, ls=":", lw=.7); a.axvline(wb, color=GRY, ls=":", lw=.7)
    a.set_title("Capital & debt policy"); a.set_xlabel("ω"); a.set_ylabel("k', b'"); a.legend(fontsize=7)

    a = ax[0, 2]                                              # velocity
    a.plot(Wg, wn_v - Wg, color=ECON["dark"], lw=2, label="VFI")
    a.plot(Wg, wn_g - Wg, color=B, lw=2, ls="--", label="learned")
    a.axhline(0, color=ECON["dark"], lw=.6); a.axvline(wb, color=GRY, ls=":", lw=.8)
    a.set_title("Velocity ω'−ω"); a.set_xlabel("ω"); a.set_ylabel("ω'−ω"); a.legend()

    a = ax[1, 0]                                              # MPK wedge
    a.plot(Wg, mp(z=z, k=np.maximum(kp_v, 1e-9)) - (p.R + p.DELTA), color=ECON["dark"], lw=2, label="VFI")
    a.plot(Wg, mp(z=z, k=np.maximum(kp_g, 1e-9)) - (p.R + p.DELTA), color=B, lw=2, ls="--", label="learned")
    a.axhline(0, color=ECON["dark"], lw=.6); a.axvline(wb, color=GRY, ls=":", lw=.8)
    a.set_title("MPK wedge zf'(k')−(r+δ)"); a.set_xlabel("ω"); a.set_ylabel("wedge"); a.legend()

    a = ax[1, 1]                                              # capital-debt locus
    a.plot(bp_v, kp_v, color=ECON["dark"], lw=2, label="VFI")
    a.plot(bp_g, kp_g, color=B, lw=2, ls="--", label="learned")
    a.axhline(ks, color=GRY, ls=":", lw=.7); a.axvline(0, color=ECON["dark"], lw=.6)
    a.set_title("Capital–debt locus"); a.set_xlabel("b'"); a.set_ylabel("k'"); a.legend()

    a = ax[1, 2]                                              # policy error + uncertainty
    a.plot(Wg, kp_g - kp_v, color=B, lw=2, label="greedy − VFI (k')")
    a.axhline(0, color=ECON["dark"], lw=.6); a.axvline(wb, color=GRY, ls=":", lw=.8)
    a2 = a.twinx(); a2.fill_between(Wg, sd_g, color=GRY, alpha=.25)
    a2.set_ylabel("belief std at greedy", color=GRY)
    a.set_title("Policy error vs rational (shade = uncertainty)")
    a.set_xlabel("ω"); a.set_ylabel("Δk'"); a.legend(fontsize=8)

    mae = float(np.mean(np.abs(kp_g - kp_v)))
    fig.suptitle(f"Learned vs rational policy — z={z:.2f}, n={len(agent.gp.Y)} obs, "
                 f"mean|Δk'|={mae:.3f}", y=1.01)
    fig.tight_layout()
    save_fig(fig, path)
    plt.close(fig)
    return mae


# ============================ figures 2/3: belief + choice anatomy ===================
def plot_belief_choice_anatomy(env, vfi, prior_iz, z_live, *, path, n_steps=50, snap_every=5, seed=7):
    p = env.p
    Z, KS, WB = landmarks(env)
    iz_live = int(np.argmin(np.abs(env.z_grid - z_live)))
    agent = make_agent(env, vfi, prior_iz, seed=seed, experience_only=True)

    Wg = np.linspace(2.0, min(env.omega_grid[-1], 1.15 * max(WB[iz_live], WB[prior_iz])), 60)
    V_true = np.interp(Wg, env.omega_grid, vfi.V[iz_live])
    V_prior = np.interp(Wg, env.omega_grid, vfi.V[prior_iz])
    biased = iz_live != prior_iz

    snaps = list(range(0, n_steps, snap_every))
    nrow_b = int(np.ceil(len(snaps) / 2))
    rng = np.random.default_rng(0)
    state = env.initial_state(z0=z_live)

    fig = plt.figure(figsize=(13, 3.1 * nrow_b))
    outer = gridspec.GridSpec(nrow_b, 2, wspace=0.18, hspace=0.55)

    for step in range(n_steps):
        z_t, om_t = state
        A, X_q, mu, sd = agent.get_beliefs(state)
        probs, delta = agent._entropy_policy(mu, sd)
        j = agent.rng.choice(len(A), p=probs)
        action = tuple(A[j])
        xdec_t = env.featurize(state=state, actions=np.atleast_2d(action))
        u_t = env.reward(state=state, action=action)
        a_star = vfi.policy(z=z_t, omega=om_t)

        if step in snaps:
            Vm, Vs = belief_V(env, agent, z_live, Wg)
            blk = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[snaps.index(step)], wspace=0.25)

            a1 = fig.add_subplot(blk[0])                       # belief V(ω)
            a1.plot(Wg, Vm, color=B, lw=1.6)
            a1.fill_between(Wg, Vm - 2 * Vs, Vm + 2 * Vs, color=B, alpha=.2)
            a1.plot(Wg, V_true, "k--", lw=1.0)
            if biased:
                a1.plot(Wg, V_prior, color=GRY, ls=":", lw=1.0)
            a1.plot(om_t, float(np.interp(om_t, Wg, Vm)), "o", color=R, ms=7, zorder=5)
            a1.axvline(WB[iz_live], color=GRY, ls=":", lw=.7)
            a1.set_title(f"t={step}: ω={om_t:.1f} (n={len(agent.gp.Y)})", fontsize=8)
            a1.set_xlabel("ω", fontsize=7); a1.set_ylabel("V", fontsize=7); a1.tick_params(labelsize=6)

            a2 = fig.add_subplot(blk[1])                       # P(k') anatomy
            eff_n = 1.0 / np.sum(probs ** 2)
            if eff_n > 3 and np.unique(A[:, 0]).size > 1:
                kde = gaussian_kde(A[:, 0], weights=probs)
                kr = np.linspace(A[:, 0].min(), A[:, 0].max(), 200)
                a2.plot(kr, kde(kr), color=B, lw=1.6)
                a2.fill_between(kr, kde(kr), color=B, alpha=.2)
            else:
                kk = np.unique(A[:, 0])
                pk = np.array([probs[np.isclose(A[:, 0], k_)].sum() for k_ in kk])
                a2.bar(kk, pk, width=0.9 * np.diff(kk).mean() if len(kk) > 1 else .5, color=B, alpha=.7)
            a2.axvline(action[0], color=R, lw=2)                          # sampled
            a2.axvline(A[int(np.argmax(mu)), 0], color="k", lw=1.2, ls="--")   # greedy
            a2.axvline(a_star[0], color=G, lw=1.5, ls="-.")               # VFI
            a2.set_title(f"a=({action[0]:.1f},{action[1]:.1f})  a*=({a_star[0]:.1f},{a_star[1]:.1f})", fontsize=8)
            a2.set_xlabel("k'", fontsize=7); a2.set_ylabel("P(k')", fontsize=7)
            a_fil = A[:, 0][probs > 1e-3]
            if a_fil.size > 1:
                a2.set_xlim(a_fil.min() - 0.25, a_fil.max() + 0.25)
            a2.tick_params(labelsize=6)

        # ---- learn: GPTD outcome point = greedy next action ---------------------------
        state_next = env.transition(state=state, action=action, rng=rng)
        A_next = env.candidate_actions(state=state_next)
        mu_next = agent.gp.predict(env.featurize(state=state_next, actions=A_next))
        c_tilde = A_next[int(np.argmax(mu_next))]
        agent.gp.add_observation(xdec_t, env.featurize(state=state_next, actions=np.atleast_2d(c_tilde)), u_t)
        state = state_next

    handles = [Line2D([0], [0], color=B, lw=2, label="belief V̂(ω) / P(k')"),
               Line2D([0], [0], color="k", ls="--", lw=1.2, label=f"VFI V (z={z_live:.2f}) / greedy k'"),
               Line2D([0], [0], color=R, lw=2, label="agent ω / sampled k'"),
               Line2D([0], [0], color=G, ls="-.", lw=1.5, label="VFI k'*")]
    if biased:
        handles.insert(2, Line2D([0], [0], color=GRY, ls=":", lw=1.2,
                                 label=f"endowed prior V (z={float(env.z_grid[prior_iz]):.2f})"))
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=8, bbox_to_anchor=(0.5, 1.0))
    tag = f"biased: prior z={float(env.z_grid[prior_iz]):.2f}, living z={z_live:.2f}" if biased \
        else f"matched: z={z_live:.2f}"
    fig.suptitle(f"Belief V̂ and choice anatomy every {snap_every} steps — {tag}", y=1.03)
    save_fig(fig, path)
    plt.close(fig)


# =================================== driver ==========================================
def main():
    print("solving rational VFI benchmark ...")
    env, p = build_env()
    vfi = VFIEntrepreneurAgent(env).fit()
    env.set_adaptive_menu(True)
    print(f"  grids: K={len(env.k_grid)} Ω={len(env.omega_grid)}; z={np.round(env.z_grid, 3)}")

    # (1) learned vs rational at z=1 (matched belief) -------------------------------------
    print("training matched agent (belief z=1.0, living z=1.0) ...")
    agent = make_agent(env, vfi, prior_iz=1, seed=7, experience_only=True)
    train_agent(env, agent, z_live=1.0, n_episodes=30, T=20, seed=0)
    mae = plot_learned_vs_rational(env, vfi, agent, z=1.0,
                                   path=os.path.join(OUT, "learned_vs_rational_z1.pdf"))
    print(f"  learned policy mean|Δk'| vs VFI = {mae:.4f}  (n={len(agent.gp.Y)} obs)")

    # (2) belief choice anatomy — matched z=1 --------------------------------------------
    print("belief/choice anatomy — matched (z=1) ...")
    plot_belief_choice_anatomy(env, vfi, prior_iz=1, z_live=1.0,
                               path=os.path.join(OUT, "anatomy_matched_z1.pdf"))

    # (3) belief choice anatomy — biased (true z=1.26, belief z=1.0) ----------------------
    print("belief/choice anatomy — biased (true z=1.26, belief z=1.0) ...")
    plot_belief_choice_anatomy(env, vfi, prior_iz=1, z_live=1.26,
                               path=os.path.join(OUT, "anatomy_biased_true1.26_belief1.0.pdf"))

    print("done.")


main()


import numpy as np
from typing import List


a = SnapshotArray(length=10)  # specify the length of the array


a.set(0, 3)

a.snap()

a.get(0, 0)  # get the value at index 0 from the first snapshot




t = Bank(balance=[100, 200, 300])  # initialize with some balances


t.transfer(1, 2, 50)  # transfer 50 from account 1 to account 2

t.balance

import defaultdict

class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:



        most = []

        prev = None

        sorted(dictUsers.keys(), key=lambda x:x.lower())

        for k, v in dict(sorted(x.items(), key=lambda item: item[1], reverse=True)):
            i = 0
            cur = val

            if i + 1 <= k or cur == prev:
                most.append(cur)

            prev = cur


        most = sorted(counts.values())[:k]


        final = ["" for]

words = ["i","love","leetcode","i","love","coding"]

counts = dict()

for word in words:
    if word in counts.keys():
        counts[word] = counts[word] + 1
    else:
        counts[word] = 1

items = sorted(counts.items(), reverse=True)   # names Z→A
items.sort(key=lambda kv: kv[1], reverse=True)
