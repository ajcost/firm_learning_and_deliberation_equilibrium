"""Entrepreneur VFI — permanent low/mid/high types: phase diagrams (continuous-b policy)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import medfilt
import sys
sys.path.append("..")
from src.simulation.agents import *
from src.simulation.environment.entrepreneur import *
from src.simulation.environment.investment_elements import (
    CRRA, CapitalCobbDouglas, PermanentProductivity, ConstantHazard)
from src.visualization.entrepreneur_visualization import *
import pandas as pd

G, O, B, P, R = "#269E1D", "#D85A30", "#3B65A5", "#8153A6", "#E23333"
COL, LBL = [R, B, G], ["low", "mid", "high"]

# ===================== solve =========================================================
p = EntrepreneurParameters(
    N_k=600, N_b=600, N_omega=300,
    BETA=1.0, THETA_K=0.5, OMEGA_ZERO=5.0,
    utility=CRRA(2.0),
    production=CapitalCobbDouglas(0.6),
    productivity=PermanentProductivity([0.79, 1.00, 1.26], probs=[1/3, 1/3, 1/3]),
    death=ConstantHazard(0.05),
)
p.set_r_knife_edge()          # R pinned at BETA=1.0
p.BETA = 0.999                # nudge impatient: product = 0.999
assert p.impatient
print("product:", p.disc * (1 + p.R))

env = EntrepreneurEnvironment(p)
env.setup_grids(calibrate=True)

env.reset_grids(step_size=0.1, ignore_b_grid=True, verbose=True)

vfi = VFIEntrepreneurAgent(env).fit()

env.set_adaptive_menu(True)

# import pickle

# vfi = None

# with open("vfi_agent.pkl", "wb") as f:
#     pickle.dump(vfi, f)

# vfi = pickle.load(open("vfi_agent.pkl", "rb"))


%matplotlib inline

fig, WSS = plot_phase_diagrams(vfi, path="./entrepreneur_phase_diagrams.pdf")
pop = vfi.simulate_panel(n_agents=5_000, T=1_000, omega0=p.OMEGA_ZERO, z_seed=0, death_seed=1)
fig, gini = plot_wealth_hist_lorenz(pop, env, burn=300, path="./wealth_hist_lorenz.pdf")
print("rest points:", WSS, " gini:", round(gini, 3))

from dataclasses import dataclass
from src.simulation.belief import *
from dataclasses import dataclass
from matplotlib.lines import Line2D

sc  = lambda x: float(np.ravel(x)[0])
Z   = [float(z) for z in env.z_grid]
KS  = [sc(env.k_star(z=z)) for z in Z]
WB  = [sc(env.omega_bar(z=z)) for z in Z]

@dataclass
class AgentParams:
    H: float = 0.0005
    KAPPA_R: float = 1.0

# --- experiment switch: set z_live different from prior_z_index for the MISMATCH run --
prior_iz = 1                                          # endowed beliefs: mid type
z_live   = 1.0                                        # agent's actual z  (1.26 => mismatch)
iz_live  = int(np.argmin(np.abs(env.z_grid - z_live)))

kernel = RBFKernel(sigma0=5.0, length_scales=[0.5, 34, 40, 120])
gp = GPBelief(gamma=p.disc, gp_params=GPBeliefParameters(kernel=kernel, sigma_n=0.025),
              prior_mean_fn=make_vfi_prior(vfi, prior_z_index=prior_iz), input_dim=4)
agent = ExperienceReasoningAgent(env, gp, AgentParams(), experience_only=True, seed=7)

def belief_V(agent, z, Wgrid):
    r"""Belief-implied V(ω)=max_a Q̂(z,ω,a); std at the greedy action."""
    Vm, Vs = np.empty(len(Wgrid)), np.empty(len(Wgrid))
    for i, w in enumerate(Wgrid):
        A  = agent.env.candidate_actions(state=(z, float(w)))
        mu, sd = agent.gp.predict(agent.env.featurize(state=(z, float(w)), actions=A),
                                  return_std=True)
        j = int(np.argmax(mu))
        Vm[i], Vs[i] = mu[j], sd[j]
    return Vm, Vs

Wg = np.linspace(2.0, 40.0, 60)
V_true = np.interp(Wg, env.omega_grid, vfi.V[iz_live])     # truth at the LIVED z

n_steps, ncol = 50, 5
nrow = n_steps // ncol
rng = np.random.default_rng(0)
state = env.initial_state(z0=z_live)


# ============ paired snapshots: V(ω) belief + P(k') choice anatomy, every 5 steps =====
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

n_steps, snap_every = 50, 5
snaps = list(range(0, n_steps, snap_every))            # 8 snapshots
nrow_b = int(np.ceil(len(snaps) / 2))                  # 2 blocks per row -> 4x2 outer

rng = np.random.default_rng(0)
state = env.initial_state(z0=z_live)

fig = plt.figure(figsize=(13, 3.1 * nrow_b))
outer = gridspec.GridSpec(nrow_b, 2, wspace=0.18, hspace=0.55)
from scipy.stats import gaussian_kde

for step in range(n_steps):
    z_t, om_t = state

    A, X_q, mu, sd = agent.get_beliefs(state)
    probs, delta = agent._entropy_policy(mu, sd)
    j = agent.rng.choice(len(A), p=probs)
    action = tuple(A[j])
    u_t = env.reward(state=state, action=action)
    xdec_t = env.featurize(state=state, actions=np.atleast_2d(action))
    a_star = vfi.policy(z=z_t, omega=om_t)

    if step in snaps:
        Vm, Vs = belief_V(agent, z_live, Wg)
        blk = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[snaps.index(step)], wspace=0.25)

        a1 = fig.add_subplot(blk[0])
        a1.plot(Wg, Vm, color="#3B65A5", lw=1.6)
        a1.fill_between(Wg, Vm - 2 * Vs, Vm + 2 * Vs, color="#3B65A5", alpha=.2)
        a1.plot(Wg, V_true, "k--", lw=1.0)
        a1.plot(om_t, float(np.interp(om_t, Wg, Vm)), "o", color="#E23333", ms=7, zorder=5)
        a1.axvline(WB[iz_live], color="gray", ls=":", lw=.7)
        a1.set_title(f"t={step}: ω={om_t:.1f} (n={len(agent.gp.Y)})", fontsize=8)
        a1.set_xlabel("ω", fontsize=7); a1.set_ylabel("V", fontsize=7)
        a1.tick_params(labelsize=6)

        a2 = fig.add_subplot(blk[1])                   # right: P(k') anatomy

        kde = gaussian_kde(A[:, 0], weights=probs)
        k_range = np.linspace(A[:, 0].min(), A[:, 0].max(), 200)
        a2.plot(k_range, kde(k_range), color="#3B65A5", alpha=0.2, lw=1.6)
        a2.fill_between(k_range, kde(k_range), color="#3B65A5", alpha=.2)
        a2.axvline(action[0], color="#E23333", lw=2) # sampled
        a2.axvline(A[int(np.argmax(mu)), 0], color="k", lw=1.2, ls="--")   # greedy
        a2.axvline(a_star[0], color="#269A8B", lw=1.5, ls="-.")             # VFI
        a2.set_title(f"a=({action[0]:.1f},{action[1]:.1f})  "
                     f"a*=({a_star[0]:.1f},{a_star[1]:.1f})", fontsize=8)
        a2.set_xlabel("k'", fontsize=7); a2.set_ylabel("P(k')", fontsize=7)
        a_fil = A[:,0][probs > 1e-3]
        a2.set_xlim(a_fil.min() - 0.25, a_fil.max() + 0.25)
        a2.tick_params(labelsize=6)

    # ---- learn: Q-learning outcome point, file immediately --------------------------
    state_next = env.transition(state=state, action=action, rng=rng)
    A_next = env.candidate_actions(state=state_next)
    mu_next = agent.gp.predict(env.featurize(state=state_next, actions=A_next))
    c_tilde = A_next[int(np.argmax(mu_next))]
    agent.gp.add_observation(
        xdec_t, env.featurize(state=state_next, actions=np.atleast_2d(c_tilde)), u_t)
    state = state_next

handles = [Line2D([0], [0], color="#3B65A5", lw=2, label="belief V̂(ω) / P(k')"),
           Line2D([0], [0], color="k", ls="--", lw=1.2, label="VFI V / greedy k'"),
           Line2D([0], [0], color="#E23333", lw=2, label="agent ω / sampled k'"),
           Line2D([0], [0], color="#269A8B", ls=":", lw=1.2, label="VFI k'*")]
fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8,
           bbox_to_anchor=(0.5, 1.00))
fig.suptitle(f"Belief V̂ and choice anatomy every {snap_every} steps — "
             f"prior node {prior_iz}, living z={z_live}", y=1.02)
fig.savefig("value_choice_anatomy.png", dpi=120, bbox_inches="tight")

print("saved: value_choice_anatomy.png")

fig.savefig("value_function_evolution_and_choice_anatomy_z0=1.0_static_z.pdf", dpi=120, bbox_inches="tight")
print("saved: value_function_evolution_and_choice_anatomy_z0=1.0_static_z.pdf")

fig.savefig("value_evolution_20_biased_prior_true=hightype_z0=1.26.pdf", dpi=115)
####################################

z_a = 1.0
iz_a = int(np.argmin(np.abs(env.z_grid - z_a)))
Wg   = np.linspace(2.0, 55.0, 200)

kp_g = np.empty_like(Wg); bp_g = np.empty_like(Wg)     # greedy: argmax_a Q̂
kp_e = np.empty_like(Wg); bp_e = np.empty_like(Wg)     # expected: Σ π(a)·a (NaN if saturated)
wn_e = np.empty_like(Wg)                               # exact E[ω'] under the choice dist
sd_g = np.empty_like(Wg)                               # uncertainty at the greedy action

for i, w in enumerate(Wg):
    A  = env.candidate_actions(state=(z_a, float(w)))
    Xq = env.featurize(state=(z_a, float(w)), actions=A)
    mu, sd = agent.gp.predict(Xq, return_std=True)
    j = int(np.argmax(mu))
    kp_g[i], bp_g[i], sd_g[i] = A[j, 0], A[j, 1], sd[j]
    probs, delta = agent._entropy_policy(mu, sd)
    wn_all = env.production(z=z_a, k=A[:, 0]) + A[:, 0] * (1 - p.DELTA) - A[:, 1] * (1 + p.R)
    if np.isfinite(delta):                             # informed region: real expectation
        kp_e[i], bp_e[i] = probs @ A[:, 0], probs @ A[:, 1]
        wn_e[i] = probs @ wn_all                       # E[φ(a)] — exact, no Jensen gap
    else:                                              # saturated: mask (menu-geometry artifact)
        kp_e[i] = bp_e[i] = wn_e[i] = np.nan

kp_v = np.interp(Wg, env.omega_grid, vfi.kp_pol[iz_a])  # VFI benchmark at the SAME z
bp_v = np.interp(Wg, env.omega_grid, vfi.bp_pol[iz_a])
ks_a, wb_a = KS[iz_a], WB[iz_a]

wn_g = env.production(z=z_a, k=kp_g) + kp_g * (1 - p.DELTA) - bp_g * (1 + p.R)
wn_v = env.production(z=z_a, k=kp_v) + kp_v * (1 - p.DELTA) - bp_v * (1 + p.R)

fig, ax = plt.subplots(2, 2, figsize=(17.5, 10))

a = ax[0, 0]
a.plot(Wg, kp_g - kp_v, color=B, lw=2, label="greedy − VFI (k')")
a.axhline(0, color="k", lw=.6); a.axvline(wb_a, color="gray", ls=":", lw=.8)
a2 = a.twinx(); a2.fill_between(Wg, sd_g, color="gray", alpha=.2)
a2.set_ylabel("belief std at greedy", color="gray")
a.set_title("Policy error vs rational (shade = uncertainty)")
a.set_xlabel("ω"); a.set_ylabel("Δk'"); a.legend(fontsize=8)

a = ax[0, 1]
a.plot(Wg, kp_g, color=B, lw=2, label="greedy k' (argmax Q̂)")
a.plot(Wg, bp_g, color=O, lw=2, label="greedy b'")
a.plot(Wg, kp_v, "k--", lw=1.4, label="VFI k'")
a.plot(Wg, bp_v, "k-.", lw=1.4, label="VFI b'")
a.axhline(ks_a, color="gray", ls=":", lw=.8); a.axvline(wb_a, color="gray", ls=":", lw=.8)
a.set_title("Capital & debt policy"); a.set_xlabel("ω"); a.set_ylabel("k', b'"); a.legend(fontsize=8)

a = ax[1, 0]                                           # 5) implied velocity (rest points)
a.plot(Wg, wn_g - Wg, color=B, lw=2, label="greedy")
a.plot(Wg, wn_v - Wg, "k--", lw=1.4, label="VFI")
a.axhline(0, color="k", lw=.6); a.axvline(wb_a, color="gray", ls=":", lw=.8)
# mark the agent's implied rest (greedy velocity zero-crossing)
g_ = wn_g - Wg
s_ = np.where((g_[:-1] > 0) & (g_[1:] <= 0))[0]
if len(s_):
    a.axvline(Wg[s_[0]], color=B, lw=1.2)
    a.text(Wg[s_[0]], a.get_ylim()[1]*.85, f" agent rest≈{Wg[s_[0]]:.0f}", color=B, fontsize=8)
a.set_title("Implied velocity ω'−ω  (zero-crossing = rest)")
a.set_xlabel("ω"); a.set_ylabel("ω'−ω"); a.legend(fontsize=8)

a = ax[1, 1]                                           # 6) MPK wedge (UNCLIPPED, log scale)
mp = env.production.marginal_product
a.semilogy(Wg, np.maximum(mp(z=z_a, k=np.maximum(kp_g, 1e-9)) - (p.R + p.DELTA), 1e-4),
           color=B, lw=2, label="greedy")
a.semilogy(Wg, np.maximum(mp(z=z_a, k=np.maximum(kp_v, 1e-9)) - (p.R + p.DELTA), 1e-4),
           "k--", lw=1.4, label="VFI")
a.axvline(wb_a, color="gray", ls=":", lw=.8)
a.set_title("MPK wedge (log scale)"); a.set_xlabel("ω"); a.set_ylabel("wedge")
a.legend(fontsize=8)

fig.suptitle(f"Learned policy diagnostics — z={z_a:.2f}, n={len(agent.gp.Y)} observations", y=.995)
fig.tight_layout(); fig.savefig("agent_policy.png", dpi=120)

fig.savefig("learned_policy_diagnostics_z0=1.0_static_z.pdf")

# ---- numeric sanity: the three claims the figure makes -------------------------------
print("max visited ω:", float(agent.gp.X_dec[:, 1].max()))
print("agent implied rest:", float(Wg[s_[0]]) if len(s_) else None, " vs rational:", wb_a)



# ============ paired snapshots: V(ω) belief + P(k') choice anatomy, every 5 steps =====

from src.simulation.belief import *


@dataclass
class AgentParams:
    H: float = 0.0005
    KAPPA_R: float = 1.0

# --- experiment switch: set z_live different from prior_z_index for the MISMATCH run --
prior_iz = 2                                          # endowed beliefs: mid type
z_live   = 1.0                                        # agent's actual z  (1.26 => mismatch)
iz_live  = int(np.argmin(np.abs(env.z_grid - z_live)))

kernel = RBFKernel(sigma0=5.0, length_scales=[0.5, 34, 40, 120])
gp = GPBelief(gamma=p.disc, gp_params=GPBeliefParameters(kernel=kernel, sigma_n=0.025),
              prior_mean_fn=make_vfi_prior(vfi, prior_z_index=prior_iz), input_dim=4)
agent = ExperienceReasoningAgent(env, gp, AgentParams(), experience_only=True, seed=7)


Wg = np.linspace(2.0, 40.0, 60)                       # one definition
V_true = np.interp(Wg, env.omega_grid, vfi.V[iz_live])  # ALWAYS rebuilt from Wg

import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

n_steps, snap_every = 50, 5
snaps = list(range(0, n_steps, snap_every))            # 8 snapshots
nrow_b = int(np.ceil(len(snaps) / 2))                  # 2 blocks per row -> 4x2 outer

rng = np.random.default_rng(0)
state = env.initial_state(z0=z_live)

fig = plt.figure(figsize=(13, 3.1 * nrow_b))
outer = gridspec.GridSpec(nrow_b, 2, wspace=0.18, hspace=0.55)
from scipy.stats import gaussian_kde

for step in range(n_steps):
    z_t, om_t = state

    A, X_q, mu, sd = agent.get_beliefs(state)
    probs, delta = agent._entropy_policy(mu, sd)
    j = agent.rng.choice(len(A), p=probs)
    action = tuple(A[j])
    u_t = env.reward(state=state, action=action)
    xdec_t = env.featurize(state=state, actions=np.atleast_2d(action))
    a_star = vfi.policy(z=z_t, omega=om_t)

    if step in snaps:
        Vm, Vs = belief_V(agent, z_live, Wg)
        blk = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[snaps.index(step)], wspace=0.25)

        a1 = fig.add_subplot(blk[0])
        a1.plot(Wg, Vm, color="#3B65A5", lw=1.6)
        a1.fill_between(Wg, Vm - 2 * Vs, Vm + 2 * Vs, color="#3B65A5", alpha=.2)
        a1.plot(Wg, V_true, "k--", lw=1.0)
        a1.plot(om_t, float(np.interp(om_t, Wg, Vm)), "o", color="#E23333", ms=7, zorder=5)
        a1.axvline(WB[iz_live], color="gray", ls=":", lw=.7)
        a1.set_title(f"t={step}: ω={om_t:.1f} (n={len(agent.gp.Y)})", fontsize=8)
        a1.set_xlabel("ω", fontsize=7); a1.set_ylabel("V", fontsize=7)
        a1.tick_params(labelsize=6)

        a2 = fig.add_subplot(blk[1])                   # right: P(k') anatomy

@dataclass
class AgentParams:
    H: float = 0.0001
    KAPPA_R: float = 1.0

# --- experiment switch ---------------------------------------------------------------
prior_iz = 2                                          # endowed beliefs: HIGH type (1.26)
z_live   = 1.0                                        # agent's actual z
iz_live  = int(np.argmin(np.abs(env.z_grid - z_live)))

kernel = RBFKernel(sigma0=5.0, length_scales=[0.5, 34, 40, 120])
gp = GPBelief(gamma=p.disc, gp_params=GPBeliefParameters(kernel=kernel, sigma_n=0.025),
              prior_mean_fn=make_vfi_prior(vfi, prior_z_index=prior_iz), input_dim=4)
agent = ExperienceReasoningAgent(env, gp, AgentParams(), experience_only=True, seed=7)

Wg      = np.linspace(2.0, 40.0, 60)
V_true  = np.interp(Wg, env.omega_grid, vfi.V[iz_live])   # truth at the LIVED z
V_prior = np.interp(Wg, env.omega_grid, vfi.V[prior_iz])  # the endowed (wrong) V

import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

n_steps, snap_every = 50, 5
snaps = list(range(0, n_steps, snap_every))               # 10 snapshots -> 5x2 outer
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
    u_t = env.reward(state=state, action=action)
    xdec_t = env.featurize(state=state, actions=np.atleast_2d(action))
    a_star = vfi.policy(z=z_t, omega=om_t)

    if step in snaps:
        Vm, Vs = belief_V(agent, z_live, Wg)
        blk = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[snaps.index(step)],
                                               wspace=0.25)

        a1 = fig.add_subplot(blk[0])                      # left: belief V(ω)
        a1.plot(Wg, Vm, color="#3B65A5", lw=1.6)
        a1.fill_between(Wg, Vm - 2 * Vs, Vm + 2 * Vs, color="#3B65A5", alpha=.2)
        a1.plot(Wg, V_true, "k--", lw=1.0)                # truth (lived z)
        a1.plot(Wg, V_prior, color="#999999", ls=":", lw=1.0)   # endowed prior V
        a1.plot(om_t, float(np.interp(om_t, Wg, Vm)), "o", color="#E23333", ms=7, zorder=5)
        a1.axvline(WB[iz_live], color="gray", ls=":", lw=.7)
        a1.set_title(f"t={step}: ω={om_t:.1f} (n={len(agent.gp.Y)})", fontsize=8)
        a1.set_xlabel("ω", fontsize=7); a1.set_ylabel("V", fontsize=7)
        a1.tick_params(labelsize=6)

        a2 = fig.add_subplot(blk[1])                      # right: P(k') anatomy
        eff_n = 1.0 / np.sum(probs**2)                    # KDE degenerate when near-greedy
        if eff_n > 3:
            kde = gaussian_kde(A[:, 0], weights=probs)
            k_range = np.linspace(A[:, 0].min(), A[:, 0].max(), 200)
            a2.plot(k_range, kde(k_range), color="#3B65A5", lw=1.6)
            a2.fill_between(k_range, kde(k_range), color="#3B65A5", alpha=.2)
        else:
            kk = np.unique(A[:, 0])
            pk = np.array([probs[np.isclose(A[:, 0], k_)].sum() for k_ in kk])
            a2.bar(kk, pk, width=0.9 * np.diff(kk).mean() if len(kk) > 1 else .5,
                   color="#3B65A5", alpha=.7)
        a2.axvline(action[0], color="#E23333", lw=2)                      # sampled
        a2.axvline(A[int(np.argmax(mu)), 0], color="k", lw=1.2, ls="--")  # greedy
        a2.axvline(a_star[0], color="#269A8B", lw=1.5, ls="-.")           # VFI
        a2.set_title(f"a=({action[0]:.1f},{action[1]:.1f})  "
                     f"a*=({a_star[0]:.1f},{a_star[1]:.1f})", fontsize=8)
        a2.set_xlabel("k'", fontsize=7); a2.set_ylabel("P(k')", fontsize=7)
        a_fil = A[:, 0][probs > 1e-3]
        if a_fil.size > 1:
            a2.set_xlim(a_fil.min() - 0.25, a_fil.max() + 0.25)
        a2.tick_params(labelsize=6)

    # ---- learn: Q-learning outcome point, file immediately --------------------------
    state_next = env.transition(state=state, action=action, rng=rng)
    A_next = env.candidate_actions(state=state_next)
    mu_next = agent.gp.predict(env.featurize(state=state_next, actions=A_next))
    c_tilde = A_next[int(np.argmax(mu_next))]
    agent.gp.add_observation(
        xdec_t, env.featurize(state=state_next, actions=np.atleast_2d(c_tilde)), u_t)
    state = state_next

handles = [Line2D([0], [0], color="#3B65A5", lw=2, label="belief V̂(ω) / P(k')"),
           Line2D([0], [0], color="k", ls="--", lw=1.2, label=f"VFI V (z={z_live})"),
           Line2D([0], [0], color="#999999", ls=":", lw=1.2,
                  label=f"endowed prior V (z={float(env.z_grid[prior_iz]):.2f})"),
           Line2D([0], [0], color="#E23333", lw=2, label="agent ω / sampled k'"),
           Line2D([0], [0], color="#269A8B", ls="-.", lw=1.5, label="VFI k'*")]
fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=8,
           bbox_to_anchor=(0.5, 1.00))
fig.suptitle(f"Belief V̂ and choice anatomy every {snap_every} steps — "
             f"prior node {prior_iz} (z={float(env.z_grid[prior_iz]):.2f}), living z={z_live}",
             y=1.02)

fname = f"value_choice_anatomy_zlive={z_live}_prioriz={prior_iz}_static_z.pdf"
fig.savefig(fname, dpi=120, bbox_inches="tight")
print("saved:", fname)
####################################




@dataclass
class AgentParams:
    H: float = 0.0005
    KAPPA_R: float = 1.0

# --- experiment switch: set z_live different from prior_z_index for the MISMATCH run --
prior_iz = 1                                          # endowed beliefs: mid type
z_live   = 1.0                                        # agent's actual z  (1.26 => mismatch)
iz_live  = int(np.argmin(np.abs(env.z_grid - z_live)))

kernel = RBFKernel(sigma0=5.0, length_scales=[0.5, 34, 40, 120])
gp = GPBelief(gamma=p.disc, gp_params=GPBeliefParameters(kernel=kernel, sigma_n=0.025),
              prior_mean_fn=make_vfi_prior(vfi, prior_z_index=prior_iz), input_dim=4)
agent = ExperienceReasoningAgent(env, gp, AgentParams(), experience_only=True, seed=7)


Wg = np.linspace(2.0, 40.0, 60)                       # one definition
V_true = np.interp(Wg, env.omega_grid, vfi.V[iz_live])  # ALWAYS rebuilt from Wg

# ============ paired snapshots: V(ω) + P(k') anatomy w/ counterfactual reasoning ======
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

n_steps, snap_every = 50, 5
snaps = list(range(0, n_steps, snap_every))
nrow_b = int(np.ceil(len(snaps) / 2))

rng = np.random.default_rng(0)
state = env.initial_state(z0=z_live)

fig = plt.figure(figsize=(13, 3.1 * nrow_b))
outer = gridspec.GridSpec(nrow_b, 2, wspace=0.18, hspace=0.55)

def plot_choice_dist(a2, A, probs, color, z=2):
    r"""KDE of the k'-marginal if non-degenerate, else bars. Returns nothing."""
    eff_n = 1.0 / np.sum(probs**2)
    if eff_n > 3:
        kde = gaussian_kde(A[:, 0], weights=probs)
        kr = np.linspace(A[:, 0].min(), A[:, 0].max(), 200)
        a2.plot(kr, kde(kr), color=color, lw=1.6, zorder=z)
        a2.fill_between(kr, kde(kr), color=color, alpha=.2, zorder=z)
    else:
        kk = np.unique(A[:, 0])
        pk = np.array([probs[np.isclose(A[:, 0], k_)].sum() for k_ in kk])
        a2.bar(kk, pk, width=0.9 * np.diff(kk).mean() if len(kk) > 1 else .5,
               color=color, alpha=.45, zorder=z)

for step in range(n_steps):
    z_t, om_t = state

    A, X_q, mu, sd = agent.get_beliefs(state)
    probs, delta = agent._entropy_policy(mu, sd)          # EXPERIENCE-ONLY: acts on this
    j = agent.rng.choice(len(A), p=probs)
    action = tuple(A[j])
    u_t = env.reward(state=state, action=action)
    xdec_t = env.featurize(state=state, actions=np.atleast_2d(action))
    a_star = vfi.policy(z=z_t, omega=om_t)

    if step in snaps:
        # ---- counterfactual reasoning: what WOULD the choice dist be? (display only) ----
        vals_E, w = agent._spectrum(X_q)                  # eigenvalues + decision-relevance weights
        N_ = len(vals_E)
        tau = agent.agent_params.KAPPA_R * N_ / (agent.agent_params.H * delta * np.maximum(w, 1e-12))
        vals_R = np.minimum(vals_E, tau)                  # water-filled (forced here, for display)

        eigens = np.sort(vals_E)[::-1]
        print(f"Eigenvalues - Max: {eigens.max():.4f}, Mean: {eigens.mean():.4f}, Min: {eigens.min():.4f}")
        print(f"Top 5 eigenvalues: {eigens[:5]}")
        print(f"Water level (median tau): {np.median(tau):.4f}")

        probs_R, delta_R = agent._entropy_policy(mu, agent._floor(vals_R, w, centered=True))

        Vm, Vs = belief_V(agent, z_live, Wg)
        blk = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[snaps.index(step)],
                                               wspace=0.25)

        a1 = fig.add_subplot(blk[0])
        a1.plot(Wg, Vm, color="#3B65A5", lw=1.6)
        a1.fill_between(Wg, Vm - 2 * Vs, Vm + 2 * Vs, color="#3B65A5", alpha=.2)
        a1.plot(Wg, V_true, "k--", lw=1.0)
        a1.plot(om_t, float(np.interp(om_t, Wg, Vm)), "o", color="#E23333", ms=7, zorder=5)
        a1.axvline(WB[iz_live], color="gray", ls=":", lw=.7)
        a1.set_title(f"t={step}: ω={om_t:.1f} (n={len(agent.gp.Y)})", fontsize=8)
        a1.set_xlabel("ω", fontsize=7); a1.set_ylabel("V", fontsize=7)
        a1.tick_params(labelsize=6)

        a2 = fig.add_subplot(blk[1])
        plot_choice_dist(a2, A, probs,   "#3B65A5", z=2)   # experience policy (blue)
        plot_choice_dist(a2, A, probs_R, "#EBB434", z=3)   # after reasoning (yellow)
        a2.axvline(action[0], color="#E23333", lw=2)                      # sampled
        a2.axvline(A[int(np.argmax(mu)), 0], color="k", lw=1.2, ls="--")  # greedy
        a2.axvline(a_star[0], color="#269A8B", lw=1.5, ls="-.")           # VFI
        a2.set_title(f"a=({action[0]:.1f},{action[1]:.1f})  "
                     f"a*=({a_star[0]:.1f},{a_star[1]:.1f})", fontsize=8)
        a2.set_xlabel("k'", fontsize=7); a2.set_ylabel("P(k')", fontsize=7)
        a_fil = A[:, 0][np.maximum(probs, probs_R) > 1e-3]
        lo = min(a_fil.min(), a_star[0]) - 0.25 if a_fil.size else a_star[0] - 1
        hi = max(a_fil.max(), a_star[0]) + 0.25 if a_fil.size else a_star[0] + 1
        a2.set_xlim(lo, hi)
        a2.tick_params(labelsize=6)

    # ---- learn: experience-only Q-learning (reasoning NOT applied to behavior) ------
    state_next = env.transition(state=state, action=action, rng=rng)
    A_next = env.candidate_actions(state=state_next)
    mu_next = agent.gp.predict(env.featurize(state=state_next, actions=A_next))
    c_tilde = A_next[int(np.argmax(mu_next))]
    agent.gp.add_observation(
        xdec_t, env.featurize(state=state_next, actions=np.atleast_2d(c_tilde)), u_t)
    state = state_next

handles = [Line2D([0], [0], color="#3B65A5", lw=2, label="P(k') experience"),
           Line2D([0], [0], color="#EBB434", lw=2, label="P(k') after reasoning"),
           Line2D([0], [0], color="k", ls="--", lw=1.2, label="VFI V / greedy k'"),
           Line2D([0], [0], color="#E23333", lw=2, label="agent ω / sampled k'"),
           Line2D([0], [0], color="#269A8B", ls="-.", lw=1.5, label="VFI k'*")]
fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=8,
           bbox_to_anchor=(0.5, 1.00))
fig.suptitle(f"Belief V̂ and choice anatomy every {snap_every} steps — "
             f"prior node {prior_iz}, living z={z_live} (yellow: counterfactual reasoning)",
             y=1.02)
# fig.savefig("value_choice_anatomy_reasoning_overlay.png", dpi=120, bbox_inches="tight")
# print("saved: value_choice_anatomy_reasoning_overlay.png")

plt.show()




@dataclass
class AgentParams:
    H: float = 0.1
    KAPPA_R: float = 1.0

# --- experiment switch: set z_live different from prior_z_index for the MISMATCH run --
prior_iz = 1                                          # endowed beliefs: mid type
z_live   = 1.0                                        # agent's actual z  (1.26 => mismatch)
iz_live  = int(np.argmin(np.abs(env.z_grid - z_live)))

kernel = RBFKernel(sigma0=5.0, length_scales=[0.25, 17, 30, 64])
gp = GPBelief(gamma=p.disc, gp_params=GPBeliefParameters(kernel=kernel, sigma_n=0.025),
              prior_mean_fn=make_vfi_prior(vfi, prior_z_index=prior_iz), input_dim=4)
agent = ExperienceReasoningAgent(env, gp, AgentParams(), experience_only=True, seed=7)
# ===== mismatch experiment: prior z=1.0 beliefs, TRUE z=1.26 — V(ω) + P(k') anatomy ===

Wg = np.linspace(2.0, 40.0, 60)                       # one definition
V_true = np.interp(Wg, env.omega_grid, vfi.V[iz_live])  # ALWAYS rebuilt from Wg

# ============ paired snapshots: V(ω) + P(k') anatomy w/ counterfactual reasoning ======
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

n_steps, snap_every = 50, 5
snaps = list(range(0, n_steps, snap_every))
nrow_b = int(np.ceil(len(snaps) / 2))

rng = np.random.default_rng(0)
state = env.initial_state(z0=z_live)

fig = plt.figure(figsize=(13, 3.1 * nrow_b))
outer = gridspec.GridSpec(nrow_b, 2, wspace=0.18, hspace=0.55)
# --- experiment: LOW-manual-on-HIGH-firm ----------------------------------------------
prior_iz = 1                                          # endowed beliefs: MID type (z=1.0)
z_live   = 1.26                                       # agent's actual z: HIGH type
iz_live  = int(np.argmin(np.abs(env.z_grid - z_live)))

kernel = RBFKernel(sigma0=5.0, length_scales=[1.0, 34, 40, 120])
gp = GPBelief(gamma=p.disc, gp_params=GPBeliefParameters(kernel=kernel, sigma_n=0.025),
              prior_mean_fn=make_vfi_prior(vfi, prior_z_index=prior_iz), input_dim=4)
agent = ExperienceReasoningAgent(env, gp, AgentParams(), experience_only=True, seed=7)

Wg      = np.linspace(2.0, 55.0, 60)                  # extend past wbar(1.26)=52.8
V_true  = np.interp(Wg, env.omega_grid, vfi.V[iz_live])   # truth for the 1.26 firm
V_prior = np.interp(Wg, env.omega_grid, vfi.V[prior_iz])  # the endowed 1.0 manual

n_steps, snap_every = 50, 5
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
    u_t = env.reward(state=state, action=action)
    xdec_t = env.featurize(state=state, actions=np.atleast_2d(action))
    a_star = vfi.policy(z=z_t, omega=om_t)

    if step in snaps:
        Vm, Vs = belief_V(agent, z_live, Wg)
        blk = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[snaps.index(step)],
                                               wspace=0.25)

        a1 = fig.add_subplot(blk[0])                  # left: belief V(ω)
        a1.plot(Wg, Vm, color="#3B65A5", lw=1.6)
        a1.fill_between(Wg, Vm - 2 * Vs, Vm + 2 * Vs, color="#3B65A5", alpha=.2)
        a1.plot(Wg, V_true, "k--", lw=1.0)            # what a 1.26 firm is really worth
        a1.plot(Wg, V_prior, color="#999999", ls=":", lw=1.0)   # the 1.0 manual
        a1.plot(om_t, float(np.interp(om_t, Wg, Vm)), "o", color="#E23333", ms=7, zorder=5)
        a1.axvline(WB[iz_live], color="gray", ls=":", lw=.7)    # true rest 52.8
        a1.axvline(WB[prior_iz], color="#999999", ls=":", lw=.7)  # manual's rest 29.6
        a1.set_title(f"t={step}: ω={om_t:.1f} (n={len(agent.gp.Y)})", fontsize=8)
        a1.set_xlabel("ω", fontsize=7); a1.set_ylabel("V", fontsize=7)
        a1.tick_params(labelsize=6)

        a2 = fig.add_subplot(blk[1])                  # right: P(k') anatomy
        eff_n = 1.0 / np.sum(probs**2)
        if eff_n > 3:
            kde = gaussian_kde(A[:, 0], weights=probs)
            kr = np.linspace(A[:, 0].min(), A[:, 0].max(), 200)
            a2.plot(kr, kde(kr), color="#3B65A5", lw=1.6)
            a2.fill_between(kr, kde(kr), color="#3B65A5", alpha=.2)
        else:
            kk = np.unique(A[:, 0])
            pk = np.array([probs[np.isclose(A[:, 0], k_)].sum() for k_ in kk])
            a2.bar(kk, pk, width=0.9 * np.diff(kk).mean() if len(kk) > 1 else .5,
                   color="#3B65A5", alpha=.7)
        a2.axvline(action[0], color="#E23333", lw=2)                      # sampled
        a2.axvline(A[int(np.argmax(mu)), 0], color="k", lw=1.2, ls="--")  # belief-greedy
        a2.axvline(a_star[0], color="#269A8B", lw=1.5, ls="-.")           # VFI (1.26)
        a2.set_title(f"a=({action[0]:.1f},{action[1]:.1f})  "
                     f"a*=({a_star[0]:.1f},{a_star[1]:.1f})", fontsize=8)
        a2.set_xlabel("k'", fontsize=7); a2.set_ylabel("P(k')", fontsize=7)
        a_fil = A[:, 0][probs > 1e-3]
        lo = min(a_fil.min(), a_star[0]) - 0.25 if a_fil.size else a_star[0] - 1
        hi = max(a_fil.max(), a_star[0]) + 0.25 if a_fil.size else a_star[0] + 1
        a2.set_xlim(lo, hi)
        a2.tick_params(labelsize=6)

    # ---- learn: Q-learning outcome point, file immediately --------------------------
    state_next = env.transition(state=state, action=action, rng=rng)
    A_next = env.candidate_actions(state=state_next)
    mu_next = agent.gp.predict(env.featurize(state=state_next, actions=A_next))
    c_tilde = A_next[int(np.argmax(mu_next))]
    agent.gp.add_observation(
        xdec_t, env.featurize(state=state_next, actions=np.atleast_2d(c_tilde)), u_t)
    state = state_next

handles = [Line2D([0], [0], color="#3B65A5", lw=2, label="belief V̂(ω) / P(k')"),
           Line2D([0], [0], color="k", ls="--", lw=1.2, label=f"VFI V (z={z_live})"),
           Line2D([0], [0], color="#999999", ls=":", lw=1.2, label="endowed prior V (z=1.00)"),
           Line2D([0], [0], color="#E23333", lw=2, label="agent ω / sampled k'"),
           Line2D([0], [0], color="#269A8B", ls="-.", lw=1.5, label="VFI k'*")]
fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=8,
           bbox_to_anchor=(0.5, 1.00))
fig.suptitle(f"Belief V̂ and choice anatomy every {snap_every} steps — "
             f"endowed z=1.00 beliefs, living z={z_live}", y=1.02)
plt.show()

plt.savefig("./value_function_evolution_and_choice_anatomy_z0=1.26_prior_iz=1.0_static_z.pdf", dpi=120)



%matplotlib inline
