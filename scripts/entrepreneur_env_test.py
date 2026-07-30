"""Entrepreneur VFI — permanent low/mid/high types: phase diagrams (continuous-b policy)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.signal import medfilt
import sys
sys.path.append("..")
from src.simulation.environment.entrepreneur import *
from src.simulation.environment.investment_elements import (
    CRRA, CapitalCobbDouglas, PermanentProductivity, ConstantHazard)
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

# vfi = VFIEntrepreneurAgent(env).fit()

env.set_adaptive_menu(True)

import pickle

with open("vfi_agent.pkl", "wb") as f:
    pickle.dump(vfi, f)

vfi = pickle.load(open("vfi_agent.pkl", "rb"))

sc = lambda x: float(np.ravel(x)[0])
slope = (1.0 + p.R) / (p.THETA_K * (1.0 - p.DELTA))
Z  = [float(z) for z in env.z_grid]
KS = [sc(env.k_star(z=z)) for z in Z]
WB = [sc(env.omega_bar(z=z)) for z in Z]
z_lo, z_mid, z_hi = Z
W  = env.omega_grid
Wd = np.linspace(W[0], W[-1], 2000)

def smooth_below(y, wbar_z, med=5, s_frac=0.02, k=3):
    r"""Light spline on the constrained branch (W < wbar), where the discrete k-grid
    leaves one-spacing stairs. The region above wbar (k'=k*, continuous b') is exact —
    leave it raw."""
    y = medfilt(np.asarray(y, float), med)
    out = y.copy()
    m = W < wbar_z
    n = int(m.sum())
    if n > k:
        sp = UnivariateSpline(W[m], y[m], k=k, s=n * np.var(y[m]) * s_frac)
        out[m] = sp(W[m])
    return out

def curves(iz):
    z, ks, wb = Z[iz], KS[iz], WB[iz]
    kp = np.minimum(smooth_below(vfi.kp_pol[iz], wb), ks)
    bp = smooth_below(vfi.bp_pol[iz], wb)
    wn = env.production(z=z, k=kp) + kp * (1 - p.DELTA) - bp * (1 + p.R)
    kp_d = np.interp(Wd, W, kp); bp_d = np.interp(Wd, W, bp)
    wn_d = env.production(z=z, k=kp_d) + kp_d * (1 - p.DELTA) - bp_d * (1 + p.R)
    return dict(kp=kp, bp=bp, wn=wn, kp_d=kp_d, bp_d=bp_d, wn_d=wn_d)

C = [curves(0), curves(1), curves(2)]

def rest_point(wn, wbar_z):
    g = wn - W
    s = np.where((g[:-1] > 0) & (g[1:] <= 0))[0]
    return W[s[np.argmin(np.abs(W[s] - wbar_z))]] if len(s) else np.nan

WSS = [rest_point(c["wn"], wb) for c, wb in zip(C, WB)]
print("rest vs wbar:")
for l, ws, wb in zip(LBL, WSS, WB):
    print(f"  {l}: wss={ws:.3f}  wbar={wb:.3f}  gap={ws-wb:+.4f}")

w0, T = 5.0, 80
om_p = np.empty(T); kp_p = np.empty(T); bp_p = np.empty(T); om_p[0] = w0
for t in range(T):
    kp_p[t] = np.interp(om_p[t], W, C[1]["kp"]); bp_p[t] = np.interp(om_p[t], W, C[1]["bp"])
    if t < T - 1:
        om_p[t + 1] = sc(env.next_worth(kp=kp_p[t], bp=bp_p[t], z=z_mid))
a_p = kp_p - bp_p

fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))

a = ax[0, 0]
a.plot(Wd, Wd, "k--", lw=.8, label="45°")
for c, z, c_ in zip(C, Z, COL):
    a.plot(Wd, c["wn_d"], color=c_, lw=2, label=f"z={z:.2f}")
x = w0; xs, ys = [x], [x]
for _ in range(80):
    y = float(np.interp(x, W, C[1]["wn"])); xs += [x, y]; ys += [y, y]
    if abs(y - x) < 1e-3: break
    x = y
a.plot(xs, ys, color=P, lw=.6, label="cobweb")
for wb, c_ in zip(WB, COL): a.axvline(wb, color=c_, ls=":", lw=.8)
a.set_title("Transition map ω'=Φ(ω;z)"); a.set_xlabel("ω"); a.set_ylabel("ω'")
a.legend(fontsize=8); a.set_xlim(0, 35); a.set_ylim(0, 35)

a = ax[0, 1]
a.plot(Wd, C[1]["kp_d"], color=B, lw=2, label="k' mid")
a.plot(Wd, C[2]["kp_d"], color=G, lw=1.4, ls="--", label="k' high")
a.plot(Wd, C[0]["kp_d"], color=R, lw=1.4, ls="--", label="k' low")
a.plot(Wd, C[1]["bp_d"], color=O, lw=2, label="b' mid")
for ks, c_ in zip(KS, COL): a.axhline(ks, color=c_, ls=":", lw=.6)
a.axhline(0, color="k", lw=.6); a.axvline(WB[1], color="gray", ls=":", lw=.8)
a.set_title("Capital & debt policy"); a.set_xlabel("ω"); a.legend(fontsize=8)
a.set_xlim(-5, 60); a.set_ylim(-5, 85)

a = ax[0, 2]
for c, c_, l in zip(C, COL, LBL):
    a.plot(Wd, c["wn_d"] - Wd, color=c_, lw=2, label=l)
a.axhline(0, color="k", lw=.6)
for ws, c_ in zip(WSS, COL):
    if np.isfinite(ws): a.axvline(ws, color=c_, lw=1.1)
for wb, c_ in zip(WB, COL): a.axvline(wb, color=c_, ls=":", lw=.8)
a.set_title("Velocity ω'−ω  (solid=rest, dotted=ω̄)")
a.set_xlabel("ω"); a.set_ylabel("ω'−ω"); a.legend(fontsize=8); a.set_xlim(0, 60)

a = ax[1, 0]
mp = env.production.marginal_product
for c, z_x, c_, l in zip(C, Z, COL, LBL):
    a.plot(Wd, mp(z=z_x, k=c["kp_d"]) - (p.R + p.DELTA), color=c_, lw=2, label=l)
a.axhline(0, color="k", lw=.6)
for wb, c_ in zip(WB, COL): a.axvline(wb, color=c_, ls=":", lw=.8)
a.set_title("MPK wedge zf'(k')−(r+δ)"); a.set_xlabel("ω"); a.set_ylabel("wedge")
a.legend(fontsize=8); a.set_ylim(-.05, .8); a.set_xlim(0, 30)

a = ax[1, 1]
bb = np.linspace(C[1]["bp"].min(), C[1]["bp"].max(), 50)
a.plot(bb, slope * bb, "k--", lw=.8, label="collateral line")
a.plot(C[1]["bp_d"], C[1]["kp_d"], color=B, lw=2, label="locus (mid)")
a.axhline(KS[1], color=B, ls=":", lw=.8); a.axvline(0, color="k", lw=.6)
a.set_title("Capital–debt locus"); a.set_xlabel("b'"); a.set_ylabel("k'")
a.legend(fontsize=8); a.set_xlim(-25, 25); a.set_ylim(0, 60)

a = ax[1, 2]
t = np.arange(T)
a.plot(t, om_p, color=B, lw=2, label="ω")
a.plot(t, kp_p, color=G, lw=1.5, label="k'")
a.plot(t, bp_p, color=O, lw=1.5, label="b'")
a.plot(t, a_p,  color=P, lw=1.5, label="a'=k'−b'")
a.axhline(WB[1], color=B, ls=":", lw=.8); a.axhline(KS[1], color=G, ls=":", lw=.6)
a.set_title(f"Simulated path (mid) from ω₀={w0}"); a.set_xlabel("t"); a.legend(fontsize=8)

fig.suptitle("Entrepreneur — phase diagrams (VFI, permanent types, continuous-b policy)", y=.995)
fig.tight_layout(); fig.savefig("entrepreneur_phase_diagrams.png", dpi=125)
print("saved: entrepreneur_phase_diagrams.png")

#pdf

fig.savefig("entrepreneur_phase_diagrams.pdf", dpi=125)

########################################################################################
from scipy.stats import gaussian_kde

N, Tp, burn = 5_000, 1000, 300
pop = vfi.simulate_panel(n_agents=N, T=Tp, omega0=p.OMEGA_ZERO, z_seed=0, death_seed=1)

om  = pop["omega"][burn:].ravel()                       # pooled stationary sample
zz  = pop["z"][burn:].ravel()
kpp = pop["kp"][burn:].ravel()
bpp = pop["bp"][burn:].ravel()

eq    = kpp - bpp                                       # equity a' = k' - b'
d2e   = bpp / np.where(np.abs(eq) < 1e-8, np.nan, eq)   # debt-to-equity
mp    = env.production.marginal_product
wedge = mp(z=zz, k=kpp) - (p.R + p.DELTA)               # MPK wedge; >1e-4 = constrained
constrained = wedge > 1e-4

print(f"\nmean ω: {om.mean():.3f}   mean k': {kpp.mean():.3f}   "
      f"share constrained: {constrained.mean():.3f}")
for z, ks, wb, l in zip(Z, KS, WB, LBL):
    m = np.isclose(zz, z)
    print(f"  {l} (z={z:.2f}): pop share {m.mean():.3f}  mean ω {om[m].mean():.3f} "
          f"(ω̄={wb:.2f})  constrained {constrained[m].mean():.3f}")

df = pd.DataFrame(
    {
        "omega": om,
        "z": zz,
        "kp": kpp,
        "bp": bpp,
        "equity": eq,
        "debt_to_equity": d2e,
        "wedge": wedge,
        "constrained": constrained,
    }
)


df[df.z == 1.0]["kp"].std()
df[df.z == 1.0]["bp"].std()
df[df.z == 1.0]["omega"].std()


vfi.policy(z=1.0, omega=23.9740834524991)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

a = ax[0]
bins = np.linspace(om.min(), np.percentile(om, 99.5), 35)          # fewer bars
for z, c_, l in zip(Z, COL, LBL):
    s = om[np.isclose(zz, z)]
    a.hist(s, bins=bins, density=True, color=c_, alpha=0.3,
           histtype="stepfilled", label=l)
    a.hist(s, bins=bins, density=True, color=c_, alpha=0.9,
           histtype="step", lw=1.2)                                # crisp outline on top
for wb, c_ in zip(WB, COL):
    a.axvline(wb, color=c_, ls=":", lw=1.0)
a.set_title("Stationary wealth distribution by type (dotted = ω̄)")
a.set_xlabel("ω"); a.set_ylabel("density"); a.legend(fontsize=8)

a = ax[1]
ws = np.sort(om)
cum = np.cumsum(ws) / ws.sum()
pc = np.arange(1, len(ws) + 1) / len(ws)
gini = 1 - 2 * np.trapz(cum, pc)
a.plot([0, 1], [0, 1], "k--", lw=.8, label="perfect equality")
a.plot(pc, cum, color=P, lw=2, label=f"wealth (Gini = {gini:.3f})")
a.fill_between(pc, cum, pc, alpha=.15, color=P)
a.set_title("Lorenz curve — wealth")
a.set_xlabel("cumulative share of firms"); a.set_ylabel("cumulative share of ω")
a.legend(fontsize=8)

fig.suptitle(f"Population (N={N}, post burn-in t>{burn})", y=.99)
fig.tight_layout(); fig.savefig("wealth_hist_lorenz.png", dpi=125)
print(f"saved: wealth_hist_lorenz.png   Gini={gini:.3f}")

fig.savefig("wealth_hist_lorenz.pdf", dpi=125)
# ============================================================

def make_vfi_prior(vfi, *, gamma=None):
    r"""Prior mean of Q(z, omega, k', b') from a solved VFI agent.

    Reconstructs Q via the one-step Bellman decomposition:
        Q(z, omega, k', b') = u(c) + gamma * V(z, omega'),
    with c = omega + b' - k', omega' = z f(k') + k'(1-delta) - b'(1+r).
    Flow u(c) is exact; the continuation V(z, omega') interpolates the VFI value
    over omega for each z-node (nearest z, linear in omega, extrapolated at edges).

    Parameters
    ----------
    vfi : VFIEntrepreneurAgent
        Already .fit(); provides V on (z_grid, omega_grid).
    gamma : float, optional
        Discount for the continuation. Defaults to the env's disc = beta*(1-rho).
    """
    env = vfi.env
    p = env.p
    g = p.disc if gamma is None else gamma
    og = env.omega_grid
    z_grid = env.z_grid

    # one linear interpolator of V(.,omega) per z-node (extrapolate past the grid edges)
    v_interp = [interp1d(og, vfi.V[iz], kind="linear", fill_value="extrapolate")
                for iz in range(len(z_grid))]

    def prior(X):
        X = np.atleast_2d(X)
        z, omega, kp, bp = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        # flow u(c), exact
        c = omega + bp - kp
        flow = env.utility(np.maximum(c, p.C_MIN))
        # next net worth omega' = z f(k') + k'(1-delta) - b'(1+r), exact
        omega_next = env.production(z=z, k=kp) + kp * (1 - p.DELTA) - bp * (1 + p.R)
        # continuation gamma * V(z, omega'): pick nearest z-node, interpolate in omega
        iz = np.abs(z_grid[:, None] - z[None, :]).argmin(axis=0)
        cont = np.empty_like(omega_next)
        for j in np.unique(iz):
            m = iz == j
            cont[m] = v_interp[j](omega_next[m])
        return flow + g * cont

    return prior

def make_vfi_prior_wrong_z(vfi, prior_z_index, gamma=None):
    r"""Prior that ALWAYS uses vfi.V[prior_z_index] for continuation, regardless of the
    query's actual z. Endows an agent with beliefs calibrated to prior_z while it lives
    at its real z. (make_vfi_prior instead auto-selects the nearest z-node — no mismatch.)"""
    env = vfi.env; p = env.p
    g = p.disc if gamma is None else gamma
    og = env.omega_grid
    v_fixed = interp1d(og, vfi.V[prior_z_index], kind="linear", fill_value="extrapolate")

    def prior(X):
        X = np.atleast_2d(X)
        z, omega, kp, bp = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        c = omega + bp - kp
        flow = env.utility(np.maximum(c, p.C_MIN))               # ACTUAL z's output below
        omega_next = env.production(z=z, k=kp) + kp * (1 - p.DELTA) - bp * (1 + p.R)
        return flow + g * v_fixed(omega_next)                    # but ALWAYS prior_z's V
    return prior

from dataclasses import dataclass
from src.simulation.gaussian_process import *

@dataclass
class AgentParams:
    H: float = 0.005        # entropy weight (uncertainty -> exploration)
    KAPPA_R: float = 1.0     # reasoning budget

kernel  = RBFKernel(sigma0=5.0, length_scales=[0.234, 17, 19, 64])

gp = GPBelief(gamma=p.disc, gp_params=GPBeliefParameters(kernel=kernel, sigma_n=0.01), prior_mean_fn=make_vfi_prior_wrong_z(vfi,prior_z_index=1), input_dim=4)
agent = ExperienceReasoningAgent(env, gp, AgentParams(), experience_only=True, seed=7)
print("agent built. GP has", len(gp.Y), "observations (fresh — knows nothing).")
# ===================== 20-step V(ω) belief-evolution grid (4 rows x 5 cols) ===========
def belief_V(agent, z, Wgrid):
    r"""Belief-implied V(ω) = max_a Q̂(z,ω,a), std at the greedy action, per ω."""
    Vm = np.empty(len(Wgrid)); Vs = np.empty(len(Wgrid))
    for i, w in enumerate(Wgrid):
        A = agent.env.candidate_actions(state=(z, float(w)))
        Xq = agent.env.featurize(state=(z, float(w)), actions=A)
        mu, sd = agent.gp.predict(Xq, return_std=True)
        j = int(np.argmax(mu))
        Vm[i] = mu[j]; Vs[i] = sd[j]
    return Vm, Vs

z0 = 1.0
Wg = np.linspace(2.0, 40.0, 60)
V_true = np.interp(Wg, env.omega_grid, vfi.V[1])       # VFI truth (mid type)

n_steps, ncol =35, 5
nrow = n_steps // ncol
rng = np.random.default_rng(0)
state = env.initial_state(z0=z0)
xdec_prev, u_prev = None, None

fig, ax = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow),
                       sharex=True, sharey=True)
ax = ax.ravel()

# for step in range(n_steps):
#     z_t, om_t = state
#     Vm, Vs = belief_V(agent, z0, Wg)

#     a = ax[step]
#     a.plot(Wg, Vm, color="#3B65A5", lw=1.8)
#     a.fill_between(Wg, Vm - 2 * Vs, Vm + 2 * Vs, color="#3B65A5", alpha=.2)
#     a.plot(Wg, V_true, "k--", lw=1.1)
#     v_here = float(np.interp(om_t, Wg, Vm))
#     a.plot(om_t, v_here, "o", color="#E23333", ms=8, zorder=5)
#     a.axvline(WB[1], color="gray", ls=":", lw=.7)
#     a.set_title(f"t={step}:  ω={om_t:.1f}   (n={len(agent.gp.Y)})", fontsize=8)
#     if step % ncol == 0: a.set_ylabel("V")
#     if step >= n_steps - ncol: a.set_xlabel("ω")

#     # act, learn (SARSA lag), move
#     action = agent.policy(z=z_t, omega=om_t, t=step)
#     u_t = env.reward(state=state, action=action)
#     xdec_t = env.featurize(state=state, actions=np.atleast_2d(action))
#     if xdec_prev is not None:
#         agent.gp.add_observation(xdec_prev, xdec_t, u_prev)
#     xdec_prev, u_prev = xdec_t, u_t
#     state = env.transition(state=state, action=action, rng=rng)
for step in range(n_steps):
    z_t, om_t = state
    Vm, Vs = belief_V(agent, z0, Wg)                  # snapshot BEFORE acting

    # act
    action = agent.policy(z=z_t, omega=om_t, t=step)
    u_t = env.reward(state=state, action=action)
    xdec_t = env.featurize(state=state, actions=np.atleast_2d(action))
    optimal_action = vfi.policy(z=z_t, omega=om_t)     # (t kwarg not needed)

    # ---- plot ----
    a = ax[step]
    a.plot(Wg, Vm, color="#3B65A5", lw=1.8)
    a.fill_between(Wg, Vm - 2 * Vs, Vm + 2 * Vs, color="#3B65A5", alpha=.2)
    a.plot(Wg, V_true, "k--", lw=1.1)
    v_here = float(np.interp(om_t, Wg, Vm))
    a.plot(om_t, v_here, "o", color="#E23333", ms=8, zorder=5)
    a.axvline(WB[1], color="gray", ls=":", lw=.7)
    a.set_title(f"t={step}: ω={om_t:.1f} (n={len(agent.gp.Y)})\n"
                f"a=({action[0]:.1f},{action[1]:.1f})  "
                f"a*=({optimal_action[0]:.1f},{optimal_action[1]:.1f})", fontsize=7)
    if step % ncol == 0: a.set_ylabel("V")
    if step >= n_steps - ncol: a.set_xlabel("ω")

    # transition FIRST (need next state for the outcome point)
    state_next = env.transition(state=state, action=action, rng=rng)

    # Q-LEARNING outcome point: greedy action at s_{t+1} under current beliefs
    A_next = env.candidate_actions(state=state_next)
    mu_next = agent.gp.predict(env.featurize(state=state_next, actions=A_next))
    c_tilde = A_next[int(np.argmax(mu_next))]
    xout_t = env.featurize(state=state_next, actions=np.atleast_2d(c_tilde))
    agent.gp.add_observation(xdec_t, xout_t, u_t)

    state = state_next

# shared legend (one, not per-panel)
from matplotlib.lines import Line2D
handles = [Line2D([0],[0], color="#3B65A5", lw=2, label="belief V̂(ω)"),
           Line2D([0],[0], color="#3B65A5", lw=8, alpha=.2, label="±2σ"),
           Line2D([0],[0], color="k", ls="--", lw=1.2, label="VFI V(ω)"),
           Line2D([0],[0], marker="o", color="#E23333", lw=0, ms=8, label="agent's ω")]
fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, 1.005))
fig.suptitle("Belief value function V̂(ω) over 20 steps", y=1.03)
fig.tight_layout(); fig.savefig("value_evolution_20.png", dpi=115)
print("saved: value_evolution_20.png")

fig.savefig("value_evolution_20_biased_prior_true=hightype_z0=1.26.pdf", dpi=115)
####################################

# ===================== 20-step V(ω): wrong-z prior, Q-learning, corrected plot ========
from scipy.interpolate import interp1d

def make_vfi_prior_wrong_z(vfi, prior_z_index, gamma=None):
    r"""Prior ALWAYS uses vfi.V[prior_z_index] for continuation, ignoring the query z.
    Endows beliefs calibrated to prior_z on an agent living at a different z."""
    env = vfi.env; p = env.p
    g = p.disc if gamma is None else gamma
    og = env.omega_grid
    v_fixed = interp1d(og, vfi.V[prior_z_index], kind="linear", fill_value="extrapolate")
    def prior(X):
        X = np.atleast_2d(X)
        z, omega, kp, bp = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        c = omega + bp - kp
        flow = env.utility(np.maximum(c, p.C_MIN))
        omega_next = env.production(z=z, k=kp) + kp * (1 - p.DELTA) - bp * (1 + p.R)
        return flow + g * v_fixed(omega_next)
    return prior

def belief_V(agent, z, Wgrid):
    r"""Belief-implied V(ω) = max_a Q̂(z,ω,a), std at the greedy action, per ω."""
    Vm = np.empty(len(Wgrid)); Vs = np.empty(len(Wgrid))
    for i, w in enumerate(Wgrid):
        A = agent.env.candidate_actions(state=(z, float(w)))
        Xq = agent.env.featurize(state=(z, float(w)), actions=A)
        mu, sd = agent.gp.predict(Xq, return_std=True)
        j = int(np.argmax(mu))
        Vm[i] = mu[j]; Vs[i] = sd[j]
    return Vm, Vs

# ---- experiment setup: 1.0 prior, agent LIVES at z=1.26 -------------------------------
z_true   = 1.26                                       # the agent's actual productivity
z_true_i = int(np.argmin(np.abs(env.z_grid - z_true)))# index of the true-z value function
prior_i  = 1                                          # z=1.0 beliefs (the "wrong manual")

kernel = RBFKernel(sigma0=5.0, length_scales=[0.234, 17, 19, 64])
gp = GPBelief(gamma=p.disc, gp_params=GPBeliefParameters(kernel=kernel, sigma_n=0.01),
              prior_mean_fn=make_vfi_prior_wrong_z(vfi, prior_z_index=prior_i), input_dim=4)
agent = ExperienceReasoningAgent(env, gp, AgentParams(), experience_only=True, seed=7)

Wg = np.linspace(2.0, 40.0, 60)
V_true = np.interp(Wg, env.omega_grid, vfi.V[z_true_i])   # TRUE z=1.26 value (what it should learn)
V_prior_line = np.interp(Wg, env.omega_grid, vfi.V[prior_i])  # the 1.0 line it was endowed with

n_steps, ncol = 20, 5
nrow = n_steps // ncol
rng = np.random.default_rng(0)
state = env.initial_state(z0=z_true)                  # START at z=1.26
xdec_prev, u_prev = None, None

fig, ax = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.0 * nrow),
                       sharex=True, sharey=True)
ax = ax.ravel()

for step in range(n_steps):
    z_t, om_t = state
    Vm, Vs = belief_V(agent, z_true, Wg)              # snapshot BEFORE acting

    action = agent.policy(z=z_t, omega=om_t, t=step)
    u_t = env.reward(state=state, action=action)
    xdec_t = env.featurize(state=state, actions=np.atleast_2d(action))
    a_star = vfi.policy(z=z_t, omega=om_t)

    a = ax[step]
    a.plot(Wg, Vm, color="#3B65A5", lw=1.8)
    a.fill_between(Wg, Vm - 2 * Vs, Vm + 2 * Vs, color="#3B65A5", alpha=.2)
    a.plot(Wg, V_true, "k--", lw=1.2)                                     # true 1.26 target
    a.plot(Wg, V_prior_line, color="#999999", ls=":", lw=1.0)            # endowed 1.0 line
    v_here = float(np.interp(om_t, Wg, Vm))
    a.plot(om_t, v_here, "o", color="#E23333", ms=8, zorder=5)
    a.axvline(WB[z_true_i], color="gray", ls=":", lw=.7)
    a.set_title(f"t={step}: ω={om_t:.1f} (n={len(agent.gp.Y)})\n"
                f"a=({action[0]:.1f},{action[1]:.1f})  a*=({a_star[0]:.1f},{a_star[1]:.1f})",
                fontsize=7)
    if step % ncol == 0: a.set_ylabel("V")
    if step >= n_steps - ncol: a.set_xlabel("ω")

    # transition, then Q-LEARNING outcome point (greedy at next state)
    state_next = env.transition(state=state, action=action, rng=rng)
    A_next = env.candidate_actions(state=state_next)
    mu_next = agent.gp.predict(env.featurize(state=state_next, actions=A_next))
    c_tilde = A_next[int(np.argmax(mu_next))]
    xout_t = env.featurize(state=state_next, actions=np.atleast_2d(c_tilde))
    agent.gp.add_observation(xdec_t, xout_t, u_t)
    state = state_next

ax[0].set_ylim(-15, 5)                                # RIGHT-SIDE-UP (was inverted!)

from matplotlib.lines import Line2D
handles = [Line2D([0],[0], color="#3B65A5", lw=2, label="belief V̂(ω)"),
           Line2D([0],[0], color="#3B65A5", lw=8, alpha=.2, label="±2σ"),
           Line2D([0],[0], color="k", ls="--", lw=1.2, label=f"TRUE V (z={z_true})"),
           Line2D([0],[0], color="#999999", ls=":", lw=1.0, label=f"endowed prior (z={env.z_grid[prior_i]:.2f})"),
           Line2D([0],[0], marker="o", color="#E23333", lw=0, ms=8, label="agent's ω")]
fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, 1.005))
fig.suptitle(f"Belief V̂(ω): endowed z={env.z_grid[prior_i]:.2f} beliefs, living at z={z_true} — Q-learning", y=1.03)
fig.tight_layout(); fig.savefig("value_mismatch_20.png", dpi=115)
print("saved: value_mismatch_20.png")

fig.savefig("value_mismatch_20.pdf", dpi=115)
#####################################

# after the run, inspect the path and the greedy policy
print("ω path:", np.round([s for s in om_history], 2))   # if you logged it; else re-run logging om_t
# is the greedy action holding low capital?
for w in [5, 10, 20, 30]:
    kg, bg = agent.get_greedy_action(z=1.26, omega=float(w))
    kv, bv = vfi.policy(z=1.26, omega=float(w))
    print(f"ω={w}: agent greedy k={kg:.1f} vs VFI k={kv:.1f}")

w=5.0; a_star=vfi.policy(z=1.26, omega=w)
q_at_star = float(make_vfi_prior(vfi)(np.array([[1.26, w, *a_star]])))
v_true    = float(np.interp(w, env.omega_grid, vfi.V[2]))   # z=1.26 is index 2!
print(q_at_star, v_true, q_at_star - v_true)
# ── Cell 1: BELIEFS at one state — what does an ignorant agent believe? ────────
state = (1.0, 10.0)
actions, X_q, mean, std = agent.get_beliefs(state)
print(f"candidate actions at (z=1, ω=10): {len(actions)} feasible (k',b') pairs")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sc0 = ax[0].scatter(actions[:, 0], actions[:, 1], c=mean, s=4, cmap="viridis")
plt.colorbar(sc0, ax=ax[0], label="belief mean Q")
ax[0].set_title("Belief MEAN over actions (all ~0: zero prior)")
ax[0].set_xlabel("k'"); ax[0].set_ylabel("b'")
sc1 = ax[1].scatter(actions[:, 0], actions[:, 1], c=std, s=4, cmap="magma")
plt.colorbar(sc1, ax=ax[1], label="belief std")
ax[1].set_title("Belief UNCERTAINTY (all = prior σ₀: no data yet)")
ax[1].set_xlabel("k'"); ax[1].set_ylabel("b'")
plt.tight_layout(); plt.savefig("tut1_beliefs.png", dpi=110); print("saved tut1")

probs_E, delta_E = agent._entropy_policy(mean, std)
print(f"total uncertainty Σσ² = {np.sum(std**2):.1f}")
print(f"entropy floor H·Σσ²  = {agent.agent_params.H*np.sum(std**2):.2f} (max possible {np.log(len(mean)):.2f})")
print(f"solved temperature δ_E = {delta_E}")
print(f"policy entropy: {-np.sum(probs_E*np.log(probs_E+1e-12)):.3f}")
print(f"-> max prob on any single action: {probs_E.max():.4f}  (uniform would be {1/len(probs_E):.4f})")
# With zero data, uncertainty saturates the floor -> near-uniform exploration.

# ── Cell 3: REASONING — water-filling shrinks uncertainty before acting ────────
std_R = agent.reason(X_q, delta_E, std)
eigs = agent.eigenvalues(X_q)
water = agent.agent_params.KAPPA_R / (agent.agent_params.H * max(delta_E, 1e-12)) if np.isfinite(delta_E) else np.inf

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].semilogy(np.sort(eigs)[::-1], "o-", ms=3, color="#3B65A5", label="belief eigenvalues")
if np.isfinite(water): ax[0].axhline(water, color="#E23333", ls="--", label=f"water level {water:.2g}")
ax[0].set_title("Reasoning = cap the largest eigen-directions")
ax[0].set_xlabel("eigen index"); ax[0].set_ylabel("variance"); ax[0].legend(fontsize=8)
ax[1].scatter(std, std_R, s=4, color="#8153A6")
ax[1].plot([0, std.max()], [0, std.max()], "k--", lw=.7)
ax[1].set_title("Per-action std: before vs after reasoning")
ax[1].set_xlabel("std (experience)"); ax[1].set_ylabel("std (reasoned)")
plt.tight_layout(); plt.savefig("tut2_reasoning.png", dpi=110); print("saved tut2")

# ── Cell 4: one full POLICY call — draft -> reason -> final -> sample ──────────
probs_R, delta_R = agent._entropy_policy(mean, std_R)
kp0, bp0 = agent.policy(z=1.0, omega=10.0)
print(f"reasoned temperature δ: {delta_R} (vs experience δ: {delta_E})")
print(f"sampled action: k'={kp0:.3f}, b'={bp0:.3f}   (VFI would pick "
      f"k'={np.interp(10.0, env.omega_grid, vfi.kp_pol[1]):.3f}, "
      f"b'={np.interp(10.0, env.omega_grid, vfi.bp_pol[1]):.3f})")
# Ignorant agent samples near-randomly; the rational benchmark is shown for contrast.

def train(agent, env, *, T, z0=1.0, seed=0, record_every=1):
    rng = np.random.default_rng(seed)
    state = env.initial_state(z0=z0)
    x_prev, u_prev = None, None
    path = {k: [] for k in ("t", "omega", "kp", "bp", "u", "nobs")}
    for t in range(T):
        action = agent.policy(z=state[0], omega=state[1], t=t)
        u_t = env.reward(state=state, action=action)
        x_t = env.featurize(state=state, actions=np.atleast_2d(action))
        if x_prev is not None:
            agent.gp.add_observation(x_prev, x_t, u_prev)
        for k, v in zip(("t", "omega", "kp", "bp", "u", "nobs"),
                        (t, state[1], action[0], action[1], u_t, len(agent.gp.Y))):
            path[k].append(v)
        x_prev, u_prev = x_t, u_t
        state = env.transition(state=state, action=action, rng=rng)
    return {k: np.array(v) for k, v in path.items()}


path = train(agent, env, T=100, z0=1.0, seed=0)
print(f"trained {len(path['t'])} periods; GP holds {len(agent.gp.Y)} TD observations")

agent.gp.reset()




# ── Cell 6: the agent's LIFE — wealth path + actions vs the rational benchmark ─
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
a = ax[0]
a.plot(path["t"], path["omega"], color=B, lw=1.5, label="learning agent ω")
a.axhline(WB[1], color="gray", ls=":", lw=.8, label="ω̄ (rational rest)")
a.set_title("Net-worth path while learning"); a.set_xlabel("t"); a.legend(fontsize=8)

a = ax[1]
a.plot(path["t"], path["kp"], color=G, lw=1, alpha=.8, label="k' chosen")
kp_star_path = np.interp(path["omega"], env.omega_grid, vfi.kp_pol[1])
a.plot(path["t"], kp_star_path, color="k", lw=1, ls="--", label="VFI k' at same ω")
a.axhline(KS[1], color="gray", ls=":", lw=.8)
a.set_title("Capital choice vs rational"); a.set_xlabel("t"); a.legend(fontsize=8)

a = ax[2]
a.plot(path["t"], path["u"], color=O, lw=1)
a.set_title("Flow utility u(c) (the ONLY thing the GP ever observes)")
a.set_xlabel("t")
plt.tight_layout(); plt.savefig("tut3_life.png", dpi=110); print("saved tut3")

# ── Cell 7: what did it LEARN? beliefs revisited + greedy policy vs VFI ─────────
actions2, X_q2, mean2, std2 = agent.get_beliefs((1.0, 10.0))
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sc0 = ax[0].scatter(actions2[:, 0], actions2[:, 1], c=mean2, s=4, cmap="viridis")
plt.colorbar(sc0, ax=ax[0], label="mean Q"); ax[0].set_title("Belief mean AFTER learning")
ax[0].set_xlabel("k'"); ax[0].set_ylabel("b'")
sc1 = ax[1].scatter(actions2[:, 0], actions2[:, 1], c=std2, s=4, cmap="magma")
plt.colorbar(sc1, ax=ax[1], label="std"); ax[1].set_title("Uncertainty AFTER (low where it visited)")
ax[1].set_xlabel("k'"); ax[1].set_ylabel("b'")

a = ax[2]                                              # greedy policy across ω vs VFI
Wt = np.linspace(1, 40, 40)
kp_g = np.array([agent.get_greedy_action(z=1.0, omega=float(w))[0] for w in Wt])
a.plot(Wt, kp_g, color=B, lw=2, label="agent greedy k'(ω)")
a.plot(Wt, np.interp(Wt, env.omega_grid, vfi.kp_pol[1]), "k--", lw=1.5, label="VFI k'(ω)")
a.axvline(WB[1], color="gray", ls=":", lw=.8); a.axhline(KS[1], color="gray", ls=":", lw=.6)
a.set_title("Learned greedy policy vs rational"); a.set_xlabel("ω"); a.set_ylabel("k'")
a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("tut4_learned.png", dpi=110); print("saved tut4")

agent.gp.reset()



def q_slice(agent, z, omega):
    r"""Belief over Q as a function of k', with b' at its per-k' best (envelope).
    Returns kk (sorted unique k'), mean_env, std_env, and the raw candidate arrays."""
    actions, X_q, mean, std = agent.get_beliefs((z, omega))
    kk = np.unique(actions[:, 0])
    m_env = np.empty_like(kk); s_env = np.empty_like(kk)
    for i, k_ in enumerate(kk):
        rows = np.isclose(actions[:, 0], k_)
        j = np.argmax(mean[rows])                     # best b' for this k'
        m_env[i] = mean[rows][j]; s_env[i] = std[rows][j]
    return kk, m_env, s_env, actions, mean

def greedy_policies(agent, Wt, z):
    kp_g = np.empty_like(Wt); bp_g = np.empty_like(Wt)
    for i, w in enumerate(Wt):
        kp_g[i], bp_g[i] = agent.get_greedy_action(z=z, omega=float(w))
    return kp_g, bp_g

n_steps = 5
z0 = 1.0                                              # mid type for the whole exercise
Wt = np.linspace(1.0, 40.0, 30)                       # ω-axis for policy panels
rng = np.random.default_rng(0)
state = env.initial_state(z0=z0)
xdec_prev, u_prev = None, None

fig, ax = plt.subplots(n_steps, 3, figsize=(15, 3.2 * n_steps))

for step in range(n_steps):
    z_t, om_t = state

    # ---- BEFORE acting: snapshot beliefs and policies -------------------------------
    kk, m_env, s_env, actions, mean_all = q_slice(agent, z_t, om_t)
    kp_g, bp_g = greedy_policies(agent, Wt, z0)

    # ---- act: red dot ---------------------------------------------------------------
    action = agent.policy(z=z_t, omega=om_t, t=step)
    u_t = env.reward(state=state, action=action)
    xdec_t = env.featurize(state=state, actions=np.atleast_2d(action))

    # column 1: belief Q over k' at current state
    a = ax[step, 0]
    a.plot(kk, m_env, color=B, lw=2, label="belief mean")
    a.fill_between(kk, m_env - 2 * s_env, m_env + 2 * s_env, alpha=.2, color=B, label="±2σ")
    if len(agent.gp.Y):                               # observed decision points so far
        seen = agent.gp.X_dec
        a.plot(seen[:, 2], np.zeros(len(seen)), "k|", ms=10, label="observed k'")
    q_at = mean_all[np.argmin(np.abs(actions[:, 0] - action[0])
                              + np.abs(actions[:, 1] - action[1]))]
    a.plot(action[0], q_at, "o", color=R, ms=9, label="chosen")
    a.axvline(float(env.k_star(z=z_t)), color="gray", ls=":", lw=.8)
    a.set_title(f"t={step}: belief Q(k') at (z={z_t:.2f}, ω={om_t:.2f})", fontsize=9)
    a.set_xlabel("k'"); a.set_ylabel("Q")
    if step == 0: a.legend(fontsize=7)

    # column 2: greedy k'(ω) vs VFI
    a = ax[step, 1]
    a.plot(Wt, kp_g, color=B, lw=2, label="agent greedy k'")
    a.plot(Wt, np.interp(Wt, env.omega_grid, vfi.kp_pol[1]), "k--", lw=1.2, label="VFI")
    a.plot(om_t, action[0], "o", color=R, ms=9, label="chosen")
    a.axvline(WB[1], color="gray", ls=":", lw=.8); a.axhline(KS[1], color="gray", ls=":", lw=.6)
    a.set_title(f"t={step}: k'(ω) policy", fontsize=9)
    a.set_xlabel("ω"); a.set_ylabel("k'")
    if step == 0: a.legend(fontsize=7)

    # column 3: greedy b'(ω) vs VFI
    a = ax[step, 2]
    a.plot(Wt, bp_g, color=O, lw=2, label="agent greedy b'")
    a.plot(Wt, np.interp(Wt, env.omega_grid, vfi.bp_pol[1]), "k--", lw=1.2, label="VFI")
    a.plot(om_t, action[1], "o", color=R, ms=9, label="chosen")
    a.axvline(WB[1], color="gray", ls=":", lw=.8)
    a.set_title(f"t={step}: b'(ω) policy", fontsize=9)
    a.set_xlabel("ω"); a.set_ylabel("b'")
    if step == 0: a.legend(fontsize=7)

    # ---- learn (SARSA lag) and move -------------------------------------------------
    if xdec_prev is not None:
        agent.gp.add_observation(xdec_prev, xdec_t, u_prev)
    xdec_prev, u_prev = xdec_t, u_t
    state = env.transition(state=state, action=action, rng=rng)

fig.suptitle("Learning agent: beliefs and policies over 5 steps (red = chosen action)", y=.999)
fig.tight_layout()
fig.savefig("learning_steps.png", dpi=120)
print("saved: learning_steps.png")


actions, X_q, mean, std_E = agent.get_beliefs((1.0, 6.0))
probs_E, delta_E = agent._entropy_policy(mean, std_E)
std_R = agent.reason(X_q, delta_E, std_E)
probs_R, delta_R = agent._entropy_policy(mean, std_R)
print(f"Σσ²={np.sum(std_E**2):.1f}  H·Σσ²={agent.agent_params.H*np.sum(std_E**2):.2f}  "
      f"H_max={np.log(len(mean)):.2f}")
print(f"δ_E={delta_E:.3g}  δ_R={delta_R:.3g}  max prob={probs_R.max():.3f}")

water_level = kappa / (self.agent_params.H * delta_E)

env.featurize(state=(1.0, 6.0), actions=np.atleast_2d(actions))



# ===================== DEBUG: experience-only agent, VFI prior, per-step printout =====
# run after your solve script (needs env, p, vfi, and your GPBelief/make_vfi_prior)

# ---- FIX: state-adaptive candidate menu (replaces the global-grid subsample) ---------
def candidate_actions_adaptive(env, z, omega, nk=25, nb=25):
    r"""Build the (k',b') menu INSIDE the feasible box at (z, omega):
    k' in [tiny, max affordable at full leverage], b' in [lend, collateral cap].
    A poor firm gets a fine local menu instead of 1 surviving global node."""
    p = env.p
    wp = 1.0 - p.THETA_K * (1.0 - p.DELTA) / (1.0 + p.R)        # downpayment
    k_max = min((omega - p.C_MIN) / wp * 0.999, env.k_grid[-1]) # affordability ceiling
    k_max = max(k_max, 0.02)
    kg = np.linspace(0.01, k_max, nk)
    rows = []
    for k_ in kg:
        b_hi = float(np.ravel(env.collateral_cap(kp=k_))[0])     # borrow to cap
        b_lo = max(k_ - omega + p.C_MIN + 1e-9, -3.0 * k_max)    # c floor / lending floor
        if b_hi <= b_lo:
            continue
        for b_ in np.linspace(b_lo, b_hi, nb):
            rows.append((k_, b_))
    if not rows:
        rows = [(0.01, 0.0)]
    return np.asarray(rows)

# monkey-patch it in (or edit your env class)
import types
env.candidate_actions = types.MethodType(
    lambda self, *, state: candidate_actions_adaptive(self, state[0], state[1]), env)

env.set_adaptive_menu(True)
# ---- per-step debug loop -------------------------------------------------------------
prior = make_vfi_prior(vfi)
kern  = RBFKernel(sigma0=5.0, length_scales=[0.234, 17, 19, 64])
gp    = GPBelief(gamma=p.disc, gp_params=GPBeliefParameters(kernel=kern, sigma_n=0.5),
                 input_dim=4, prior_mean_fn=prior)
agent = ExperienceReasoningAgent(env, gp, AgentParams(H=0.0001, KAPPA_R=1.0),
                                 experience_only=True, seed=7)

rng = np.random.default_rng(0)
z0, om = 1.0, 5.0
xdec_prev, u_prev = None, None

print("=== per-step debug: experience-only, VFI prior ===")
for t in range(15):
    A, X_q, mu, sd = agent.get_beliefs((z0, om))
    probs, delta = agent._entropy_policy(mu, sd)
    j = agent.rng.choice(len(A), p=probs)
    kp, bp = A[j]
    c = om + bp - kp
    u_t = float(np.ravel(env.utility(np.maximum(c, 1e-12)))[0])

    kv = float(np.interp(om, env.omega_grid, vfi.kp_pol[1]))
    bv = float(np.interp(om, env.omega_grid, vfi.bp_pol[1]))
    q_vfi = float(gp.predict(env.featurize(state=(z0, om), actions=np.array([[kv, bv]])))[0])
    jg = int(np.argmax(mu))

    print(f"t={t:2d} ω={om:7.3f} nA={len(A):4d} | chose(k={kp:6.2f},b={bp:6.2f}) c={c:7.3f} "
          f"u={u_t:9.3f} | δ={delta:9.3g} maxP={probs.max():.3f} | "
          f"Q(chosen)={mu[j]:8.2f} Q(greedy)={mu[jg]:8.2f} Q(vfi)={q_vfi:8.2f} | "
          f"greedy_k={A[jg,0]:6.2f} vfi_k={kv:6.2f}")
    if len(A) < 20:
        print(f"   *** WARNING: menu has only {len(A)} actions — exploration crippled ***")
    if abs(u_t) > 1e4:
        print(f"   *** CATASTROPHIC u(c) — one obs like this poisons the GP ***")

    xdec_t = env.featurize(state=(z0, om), actions=np.array([[kp, bp]]))
    if xdec_prev is not None:
        gp.add_observation(xdec_prev, xdec_t, u_prev)
        resid = float((gp.Y - (gp.prior_mean_fn(gp.X_dec)
                       - gp.gamma * gp.prior_mean_fn(gp.X_out)))[-1])
        print(f"      filed TD: n={len(gp.Y)}  |alpha|max={np.abs(gp.alpha).max():.3g}  "
              f"prior-residual={resid:+.3f}  (≈0 means prior is Bellman-consistent)")
    xdec_prev, u_prev = xdec_t, u_t
    om = float(np.ravel(env.next_worth(kp=kp, bp=bp, z=z0))[0])


# ===================== Viz 1: belief evolution at a fixed reference state =============
from matplotlib.cm import viridis
from matplotlib.colors import Normalize

ref_state = (1.0, 10.0)                                # FIXED probe state
snap_every, T = 10, 100
snaps, act_log = [], []

rng = np.random.default_rng(0)
state = env.initial_state(z0=1.0)
xdec_prev, u_prev = None, None
for t in range(T):
    if t % snap_every == 0:                            # snapshot beliefs at ref_state
        A, X_q, mu, sd = agent.get_beliefs(ref_state)
        kk = np.unique(A[:, 0])
        m_env = np.array([mu[np.isclose(A[:, 0], k_)].max() for k_ in kk])
        s_env = np.array([sd[np.isclose(A[:, 0], k_)][np.argmax(mu[np.isclose(A[:, 0], k_)])]
                          for k_ in kk])
        snaps.append((t, kk, m_env, s_env))
    action = agent.policy(z=state[0], omega=state[1], t=t)
    u_t = env.reward(state=state, action=action)
    xdec_t = env.featurize(state=state, actions=np.atleast_2d(action))
    if xdec_prev is not None:
        agent.gp.add_observation(xdec_prev, xdec_t, u_prev)
    act_log.append((t, state[1], action[0]))
    xdec_prev, u_prev = xdec_t, u_t
    state = env.transition(state=state, action=action, rng=rng)

fig, a = plt.subplots(figsize=(9, 5.5))
norm = Normalize(0, T)
for (t, kk, m_env, s_env) in snaps:
    col = viridis(norm(t))
    a.plot(kk, m_env, color=col, lw=2, label=f"t={t}")
    a.fill_between(kk, m_env - 2 * s_env, m_env + 2 * s_env, color=col, alpha=.10)
kv = float(np.interp(ref_state[1], env.omega_grid, vfi.kp_pol[1]))
a.axvline(kv, color="k", ls="--", lw=1, label="VFI k' at ref state")
a.set_title(f"Belief Q(k') at fixed state (z=1, ω={ref_state[1]}) — color = time")
a.set_xlabel("k'"); a.set_ylabel("Q envelope"); a.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig("belief_evolution_ribbon.png", dpi=120)

# ===================== Viz 2: phase-space trail over uncertainty field ================
Wg = np.linspace(1, 35, 45); Kg = np.linspace(0.5, 30, 45)
SD = np.full((len(Kg), len(Wg)), np.nan)
for i_w, w in enumerate(Wg):                           # std field at CURRENT beliefs
    A = env.candidate_actions(state=(1.0, float(w)))
    Xq = env.featurize(state=(1.0, float(w)), actions=A)
    _, sd = agent.gp.predict(Xq, return_std=True)
    for i_k, k_ in enumerate(Kg):                      # nearest-candidate std at (w, k)
        j = np.argmin(np.abs(A[:, 0] - k_))
        if abs(A[j, 0] - k_) < 1.0:
            SD[i_k, i_w] = sd[j]

fig, a = plt.subplots(figsize=(9, 5.5))
pc = a.pcolormesh(Wg, Kg, SD, cmap="Greys", shading="auto")
plt.colorbar(pc, ax=a, label="belief std (posterior uncertainty)")
tt_, ww_, kk_ = zip(*act_log)
scp = a.scatter(ww_, kk_, c=tt_, cmap="viridis", s=18, zorder=3)
plt.colorbar(scp, ax=a, label="t")
a.plot(env.omega_grid, vfi.kp_pol[1], "r--", lw=1.5, label="VFI k'(ω)", zorder=2)
a.axvline(WB[1], color="gray", ls=":", lw=.8)
a.set_title("Agent's life in policy space — trail over uncertainty field")
a.set_xlabel("ω"); a.set_ylabel("k' chosen"); a.legend(fontsize=8)
a.set_xlim(1, 35); a.set_ylim(0, 30)
fig.tight_layout(); fig.savefig("phase_trail.png", dpi=120)


# ===================== debug run with recording + diagnostic plots ====================
def debug_run(agent, gp, env, vfi, *, T=40, z0=1.0, om0=5.0, seed=0,
              outcome="sarsa"):                      # "sarsa" | "qlearn"
    rng = np.random.default_rng(seed)
    om = om0
    xdec_prev, u_prev = None, None
    log = {k: [] for k in ("t", "omega", "kp", "bp", "c", "u", "delta", "maxP",
                            "q_vfi", "q_greedy", "greedy_k", "vfi_k", "alpha_max", "resid")}
    for t in range(T):
        A, X_q, mu, sd = agent.get_beliefs((z0, om))
        probs, delta = agent._entropy_policy(mu, sd)
        j = agent.rng.choice(len(A), p=probs)
        kp, bp = A[j]
        c = om + bp - kp
        u_t = float(np.ravel(env.utility(np.maximum(c, 1e-12)))[0])

        kv = float(np.interp(om, env.omega_grid, vfi.kp_pol[1]))
        bv = float(np.interp(om, env.omega_grid, vfi.bp_pol[1]))
        q_vfi = float(gp.predict(env.featurize(state=(z0, om), actions=np.array([[kv, bv]])))[0])
        jg = int(np.argmax(mu))

        xdec_t = env.featurize(state=(z0, om), actions=np.array([[kp, bp]]))
        resid = np.nan
        if xdec_prev is not None:
            if outcome == "qlearn":                  # outcome point = greedy at current state
                kg_, bg_ = A[jg]
                x_out = env.featurize(state=(z0, om), actions=np.array([[kg_, bg_]]))
            else:                                    # sarsa: realized decision
                x_out = xdec_t
            gp.add_observation(xdec_prev, x_out, u_prev)
            M = gp.prior_mean_fn(gp.X_dec) - gp.gamma * gp.prior_mean_fn(gp.X_out)
            resid = float((gp.Y - M)[-1])

        for k_, v_ in zip(log.keys(),
                          (t, om, kp, bp, c, u_t, delta, probs.max(), q_vfi, mu[jg],
                           A[jg, 0], kv, np.abs(gp.alpha).max() if len(gp.Y) else 0.0, resid)):
            log[k_].append(v_)
        xdec_prev, u_prev = xdec_t, u_t
        om = float(np.ravel(env.next_worth(kp=kp, bp=bp, z=z0))[0])
    return {k: np.array(v) for k, v in log.items()}

log = debug_run(agent, gp, env, vfi, T=40, outcome="sarsa")   # or outcome="qlearn"

# ===================== diagnostic figure (2x3) =======================================
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
tt = log["t"]

a = ax[0, 0]                                          # 1) wealth path + consumption
a.plot(tt, log["omega"], color=B, lw=2, label="ω")
a.plot(tt, log["c"], color=O, lw=1.2, label="c")
a.axhline(WB[1], color="gray", ls=":", lw=.8, label="ω̄")
a.set_title("Wealth & consumption path"); a.set_xlabel("t"); a.legend(fontsize=8)

a = ax[0, 1]                                          # 2) chosen vs greedy vs VFI capital
a.plot(tt, log["kp"], "o", color=R, ms=4, alpha=.6, label="chosen k' (sampled)")
a.plot(tt, log["greedy_k"], color=B, lw=2, label="greedy k' (beliefs)")
a.plot(tt, log["vfi_k"], "k--", lw=1.2, label="VFI k' at same ω")
a.set_title("Capital: sampled / greedy / rational"); a.set_xlabel("t"); a.legend(fontsize=8)

a = ax[0, 2]                                          # 3) belief health: Q at VFI action
a.plot(tt, log["q_vfi"], color=P, lw=2, label="Q̂(VFI action)")
a.plot(tt, log["q_greedy"], color=G, lw=1.2, label="Q̂(greedy)")
a.set_title("Belief at the rational action\n(downtrend = prior corruption)")
a.set_xlabel("t"); a.legend(fontsize=8)

a = ax[1, 0]                                          # 4) TD residuals (should hover ~0)
a.plot(tt, log["resid"], "o-", color=O, ms=3, lw=.8)
a.axhline(0, color="k", lw=.6)
a.set_title("Prior residual per obs\n(systematic <0 = SARSA vs Q* mismatch)")
a.set_xlabel("t"); a.set_ylabel("u − [m(d)−γm(o)]")

a = ax[1, 1]                                          # 5) alpha growth (GP stability)
a.semilogy(tt, np.maximum(log["alpha_max"], 1e-3), color=R, lw=2)
a.set_title("|α|max (log)\n(growth = Gram system amplifying)")
a.set_xlabel("t")

a = ax[1, 2]                                          # 6) exploration state
a2 = a.twinx()
a.plot(tt, log["maxP"], color=B, lw=2, label="max prob")
a.axhline(1 / 625, color="gray", ls=":", lw=.8, label="uniform")
a2.semilogy(tt, np.minimum(np.nan_to_num(log["delta"], posinf=1e6), 1e6),
            color=G, lw=1.2, alpha=.7)
a.set_title("Exploration: maxP (blue, left) & δ (green, log right)")
a.set_xlabel("t"); a.legend(fontsize=8, loc="upper left")

fig.suptitle(f"Learning-agent diagnostics — outcome={'SARSA' if True else ''}", y=.995)
fig.tight_layout(); fig.savefig("agent_diagnostics.png", dpi=120)

%matplotlib inline


# ===================== rational length-scales (Prop 3) for the entrepreneur ==========
def rational_length_scales(vfi, env, p, *, sigma0, displacements=None, x_star=None):
    r"""Prop-3 RBF calibration: measure true |ΔQ*| along each dimension at displacement
    d_j from the anchor x*, then l_j = d_j / sqrt(-2 ln(1 - |ΔQ*|²/(2σ0²))).
    Dimensions: (z, ω, k', b'). Q* evaluated via the VFI Bellman decomposition."""
    Q = make_vfi_prior(vfi)                             # Q*(z, ω, k', b')
    if x_star is None:                                  # anchor: mid type at its rest
        z_s = 1.0
        w_s = float(np.ravel(env.omega_bar(z=z_s))[0])
        k_s = float(np.ravel(env.k_star(z=z_s))[0])
        b_s = k_s + 2.0 - w_s                           # b' financing k* with c≈2 at rest
        x_star = np.array([z_s, w_s, k_s, b_s])
    if displacements is None:                           # ~1 typical std per dimension
        displacements = np.array([0.2, 5.0, 5.0, 3.0])  # (dz, dω, dk, db)

    q0 = float(Q(x_star[None, :])[0])
    ells = np.empty(4)
    for j, d in enumerate(displacements):
        xp = x_star.copy(); xp[j] += d
        dQ = abs(float(Q(xp[None, :])[0]) - q0)
        arg = 1.0 - dQ**2 / (2 * sigma0**2)
        if arg <= 0:                                    # variation exceeds prior scale
            raise ValueError(f"dim {j}: |ΔQ*|={dQ:.2f} too large for σ0={sigma0} "
                             f"(raise σ0 or shrink d_j)")
        ells[j] = d / np.sqrt(-2.0 * np.log(arg))
        print(f"dim {j} ({'zωkb'[j]}): d={d:5.2f}  |ΔQ*|={dQ:7.3f}  ->  ℓ={ells[j]:.3f}")
    return ells


Q = make_vfi_prior(vfi)
z_s = 1.0; w_s = float(np.ravel(env.omega_bar(z=z_s))[0])
k_s = float(np.ravel(env.k_star(z=z_s))[0])
b_s = k_s + 2.0 - w_s
x_star = np.array([z_s, w_s, k_s, b_s])

q_0 = float(Q(x_star[None, :])[0])

q_0

def abs_q(x):
    return abs(float(Q(x[None, :])[0]) - q_0)


from scipy.interpolate import interp1d
def rational_length_scales(vfi, env, p, *, sigma0, x_ref=None, displacements=None,
                           two_sided=True, verbose=True):
    r"""Prop-3 RBF calibration at an ARBITRARY reference point in (z, omega, kp, bp).

    Measures the true variation of Q* along each dimension at displacement d_j from
    x_ref, then inverts the RBF believed-variation formula:

        l_j = d_j / sqrt( -2 ln( 1 - |dQ*_j|^2 / (2 sigma0^2) ) )

    Q* is evaluated via the Bellman decomposition (make_vfi_prior), so any reference
    point works: a rest point (flat Q, long l), a constrained climber (steep Q, short l),
    or anywhere else.

    Parameters
    ----------
    sigma0 : float
        Prior std of the GP kernel.
    x_ref : array-like (4,), optional
        Reference point (z, omega, kp, bp). Default: the mid type's constrained state
        at birth wealth, with the VFI policy's action there (where learning matters).
    displacements : array-like (4,), optional
        d_j per dimension. Default (0.2, 5, 5, 3). For the ergodic-std discipline,
        pass stds from your panel simulation.
    two_sided : bool
        Evaluate |dQ*| in both directions and use the LARGER (calibrates to the steep
        side, e.g. toward the consumption cliff in bp). One-sided uses +d_j only.

    Returns
    -------
    ells : np.ndarray (4,)  — rational length scales for (z, omega, kp, bp).
    """
    Q = make_vfi_prior(vfi)

    if x_ref is None:                                  # default: constrained climber
        z_s = 1.0
        w_s = float(p.OMEGA_ZERO)
        k_s = float(np.interp(w_s, env.omega_grid, vfi.kp_pol[1]))
        b_s = float(np.interp(w_s, env.omega_grid, vfi.bp_pol[1]))
        x_ref = np.array([z_s, w_s, k_s, b_s])
    x_ref = np.asarray(x_ref, float)

    if displacements is None:
        displacements = np.array([0.2, 5.0, 5.0, 3.0])
    displacements = np.asarray(displacements, float)

    q0 = float(Q(x_ref[None, :])[0])
    ells = np.empty(4)
    for j, d in enumerate(displacements):
        xp = x_ref.copy(); xp[j] += d
        dQ_plus = abs(float(Q(xp[None, :])[0]) - q0)
        if two_sided:
            xm = x_ref.copy(); xm[j] -= d
            dQ_minus = abs(float(Q(xm[None, :])[0]) - q0)
            dQ = max(dQ_plus, dQ_minus)
        else:
            dQ = dQ_plus
        arg = 1.0 - dQ**2 / (2.0 * sigma0**2)
        if arg <= 0:
            raise ValueError(
                f"dim {j} ('zwkb'[j]): |dQ*|={dQ:.3f} >= sqrt(2)*sigma0={np.sqrt(2)*sigma0:.3f}; "
                f"raise sigma0 or shrink d_j — the prior scale can't accommodate this variation.")
        ells[j] = d / np.sqrt(-2.0 * np.log(arg))
        if verbose:
            print(f"  dim {j} ({'zωkb'[j]}): d={d:6.3f}  |ΔQ*|={dQ:8.3f}  ->  ℓ={ells[j]:.3f}")
    return ells


x_ref = (1.0, 16.914, *vfi.policy(z=1.0, omega=16.914))
c0 = x_ref[1] + x_ref[3] - x_ref[2]                      # consumption at the anchor
ells = rational_length_scales(vfi, env, p, sigma0=2.269668387668483, x_ref=x_ref,
    displacements=[0.3, df[(df.z == 1.0) & (abs(df.omega - 16.914) < 2)]["omega"].std(),
                        df[(df.z == 1.0) & (abs(df.omega - 16.914) < 2)]["kp"].std(),
                        df[(df.z == 1.0) & (abs(df.omega - 16.914) < 2)]["bp"].std()])


# σ0 = std of Q* over the ergodic (state, action) sample (local window or full type)
Q = make_vfi_prior(vfi)
m = (df.z == 1.0)
Xerg = np.column_stack([df[m]["z"], df[m]["omega"], df[m]["kp"], df[m]["bp"]])
q_erg = Q(Xerg)
sigma0 = float(np.std(q_erg))
print(f"σ0 from ergodic Q* variation: {sigma0:.3f}")

ells = rational_length_scales(vfi, env, p, sigma0=sigma0, x_ref=x_ref,
    displacements=[0.3, df[mloc]["omega"].std(), df[mloc]["kp"].std(), df[mloc]["bp"].std()])

m = (df.z == 1.0) & (abs(df.omega - 16.914) < 2)         # define once, reuse
print(f"window rows: {m.sum()}   stds: ω={df[m]['omega'].std():.3f}  "
      f"kp={df[m]['kp'].std():.3f}  bp={df[m]['bp'].std():.3f}   c0={c0:.3f}")
ells = rational_length_scales(vfi, env, p, sigma0=5.0, x_ref=x_ref,
    displacements=[0.3, df[m]["omega"].std(), df[m]["kp"].std(), df[m]["bp"].std()])


# σ0 = std of Q* over the ergodic (state, action) sample (local window or full type)
Q = make_vfi_prior(vfi)
m = (df.z == 1.0)
Xerg = np.column_stack([df[m]["z"], df[m]["omega"], df[m]["kp"], df[m]["bp"]])
q_erg = Q(Xerg)
sigma0 = float(np.std(q_erg))
print(f"σ0 from ergodic Q* variation: {sigma0:.3f}")

ells = rational_length_scales(vfi, env, p, sigma0=sigma0, x_ref=x_ref,
    displacements=[0.3, df[m]["omega"].std(), df[m]["kp"].std(), df[m]["bp"].std()])

Xerg = np.column_stack([df["z"], df["omega"], df["kp"], df["bp"]])   # no z filter
sigma0 = float(np.std(make_vfi_prior(vfi)(Xerg)))                     # ~2-3 likely

sigma0



# ===================== Viz 3: choice-distribution anatomy for 5 selected steps ========
show_at = [0, 5, 10, 20, 40]
fig, ax = plt.subplots(len(show_at), 1, figsize=(9, 2.0 * len(show_at)), sharex=True)
rng = np.random.default_rng(0)
state = env.initial_state(z0=1.0); xdec_prev, u_prev = None, None; row = 0
# (rebuild agent/gp fresh before running this cell)
for t in range(max(show_at) + 1):
    A, X_q, mu, sd = agent.get_beliefs(state)
    probs, delta = agent._entropy_policy(mu, sd)
    j = agent.rng.choice(len(A), p=probs)
    if t in show_at:
        a = ax[row]
        # aggregate prob mass by k' (marginal over b')
        kk = np.unique(A[:, 0])
        pk = np.array([probs[np.isclose(A[:, 0], k_)].sum() for k_ in kk])
        a.bar(kk, pk, width=0.9 * np.diff(kk).mean(), color="#3B65A5", alpha=.7)
        a.axvline(A[j, 0], color="#E23333", lw=2, label="sampled")
        a.axvline(A[int(np.argmax(mu)), 0], color="k", lw=1.2, ls="--", label="greedy")
        a.axvline(float(np.interp(state[1], env.omega_grid, vfi.kp_pol[1])),
                  color="#269E1D", lw=1.2, ls=":", label="VFI")
        a.set_ylabel(f"t={t}\nP(k')", fontsize=8)
        if row == 0: a.legend(fontsize=7)
        row += 1
    u_t = env.reward(state=state, action=A[j])
    xdec_t = env.featurize(state=state, actions=np.atleast_2d(A[j]))
    if xdec_prev is not None:
        agent.gp.add_observation(xdec_prev, xdec_t, u_prev)
    xdec_prev, u_prev = xdec_t, u_t
    state = env.transition(state=state, action=A[j], rng=rng)
ax[-1].set_xlabel("k'")
fig.suptitle("Anatomy of choice: P(k') marginal, sampled vs greedy vs VFI", y=.995)
fig.tight_layout(); fig.savefig("choice_anatomy.png", dpi=120)


agent

# ===================== agent policy diagnostics (2x3) ================================
z_a = z_true                                          # the agent's productivity (1.26)
iz_a = int(np.argmin(np.abs(env.z_grid - z_a)))
Wg = np.linspace(2.0, 40.0, 45)                       # ω-axis (45 pts: GP calls are the cost)

kp_g = np.empty_like(Wg); bp_g = np.empty_like(Wg)     # greedy: argmax_a Q̂
kp_e = np.empty_like(Wg); bp_e = np.empty_like(Wg)     # expected: Σ π(a) a
sd_g = np.empty_like(Wg)                               # uncertainty at the greedy action
for i, w in enumerate(Wg):
    A = env.candidate_actions(state=(z_a, float(w)))
    Xq = env.featurize(state=(z_a, float(w)), actions=A)
    mu, sd = agent.gp.predict(Xq, return_std=True)
    j = int(np.argmax(mu))
    kp_g[i], bp_g[i], sd_g[i] = A[j, 0], A[j, 1], sd[j]
    probs, _ = agent._entropy_policy(mu, sd)           # the agent's actual choice dist
    kp_e[i], bp_e[i] = probs @ A[:, 0], probs @ A[:, 1]

kp_v = np.interp(Wg, env.omega_grid, vfi.kp_pol[iz_a])  # VFI benchmark at the SAME z
bp_v = np.interp(Wg, env.omega_grid, vfi.bp_pol[iz_a])
ks_a, wb_a = KS[iz_a], WB[iz_a]

fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))

a = ax[0, 0]                                           # 1) capital policy
a.plot(Wg, kp_g, color=B, lw=2, label="greedy k' (argmax Q̂)")
a.plot(Wg, kp_e, color=O, lw=1.8, ls="-.", label="expected k' (E[a])")
a.plot(Wg, kp_v, "k--", lw=1.4, label="VFI k'")
a.axhline(ks_a, color="gray", ls=":", lw=.8); a.axvline(wb_a, color="gray", ls=":", lw=.8)
a.set_title("Capital policy k'(ω)"); a.set_xlabel("ω"); a.set_ylabel("k'"); a.legend(fontsize=8)

a = ax[0, 1]                                           # 2) debt policy
a.plot(Wg, bp_g, color=B, lw=2, label="greedy b'")
a.plot(Wg, bp_e, color=O, lw=1.8, ls="-.", label="expected b'")
a.plot(Wg, bp_v, "k--", lw=1.4, label="VFI b'")
a.axhline(0, color="k", lw=.6); a.axvline(wb_a, color="gray", ls=":", lw=.8)
a.set_title("Debt policy b'(ω)"); a.set_xlabel("ω"); a.set_ylabel("b'"); a.legend(fontsize=8)

a = ax[0, 2]                                           # 3) policy gap + uncertainty
a.plot(Wg, kp_g - kp_v, color=B, lw=2, label="greedy − VFI (k')")
a.plot(Wg, kp_e - kp_v, color=O, lw=1.8, ls="-.", label="expected − VFI (k')")
a.axhline(0, color="k", lw=.6); a.axvline(wb_a, color="gray", ls=":", lw=.8)
a2 = a.twinx(); a2.fill_between(Wg, sd_g, color="gray", alpha=.2)
a2.set_ylabel("belief std at greedy", color="gray")
a.set_title("Policy error vs rational (bars = uncertainty)")
a.set_xlabel("ω"); a.set_ylabel("Δk'"); a.legend(fontsize=8)

a = ax[1, 0]                                           # 4) implied transition map
wn_g = env.production(z=z_a, k=kp_g) + kp_g * (1 - p.DELTA) - bp_g * (1 + p.R)
wn_e = env.production(z=z_a, k=kp_e) + kp_e * (1 - p.DELTA) - bp_e * (1 + p.R)
wn_v = env.production(z=z_a, k=kp_v) + kp_v * (1 - p.DELTA) - bp_v * (1 + p.R)
a.plot(Wg, Wg, "k:", lw=.8, label="45°")
a.plot(Wg, wn_g, color=B, lw=2, label="greedy")
a.plot(Wg, wn_e, color=O, lw=1.8, ls="-.", label="expected")
a.plot(Wg, wn_v, "k--", lw=1.4, label="VFI")
a.axvline(wb_a, color="gray", ls=":", lw=.8)
a.set_title("Implied transition ω'=Φ(ω)"); a.set_xlabel("ω"); a.set_ylabel("ω'"); a.legend(fontsize=8)

a = ax[1, 1]                                           # 5) implied velocity (rest points)
a.plot(Wg, wn_g - Wg, color=B, lw=2, label="greedy")
a.plot(Wg, wn_e - Wg, color=O, lw=1.8, ls="-.", label="expected")
a.plot(Wg, wn_v - Wg, "k--", lw=1.4, label="VFI")
a.axhline(0, color="k", lw=.6); a.axvline(wb_a, color="gray", ls=":", lw=.8)
a.set_title("Implied velocity ω'−ω  (zero-crossing = rest)")
a.set_xlabel("ω"); a.set_ylabel("ω'−ω"); a.legend(fontsize=8)

a = ax[1, 2]                                           # 6) MPK wedge under each policy
mp = env.production.marginal_product
a.plot(Wg, mp(z=z_a, k=np.maximum(kp_g, 1e-9)) - (p.R + p.DELTA), color=B, lw=2, label="greedy")
a.plot(Wg, mp(z=z_a, k=np.maximum(kp_e, 1e-9)) - (p.R + p.DELTA), color=O, lw=1.8, ls="-.", label="expected")
a.plot(Wg, mp(z=z_a, k=np.maximum(kp_v, 1e-9)) - (p.R + p.DELTA), "k--", lw=1.4, label="VFI")
a.axhline(0, color="k", lw=.6); a.axvline(wb_a, color="gray", ls=":", lw=.8)
a.set_title("MPK wedge under each policy"); a.set_xlabel("ω"); a.set_ylabel("wedge")
a.set_ylim(-.05, 1.0); a.legend(fontsize=8)

fig.suptitle(f"Learned policy diagnostics — z={z_a:.2f}, n={len(agent.gp.Y)} observations", y=.995)
fig.tight_layout(); fig.savefig("agent_policy_2x3.png", dpi=120)
print("saved: agent_policy_2x3.png")


# ===================== agent policy diagnostics (2x3) — full corrected version ========
z_a = 1.0                                   # the agent's ACTUAL z for this run
iz_a = int(np.argmin(np.abs(env.z_grid - z_a)))    # -> 1, not 2
Wg   = np.linspace(2.0, 55.0, 55)                      # extended past ω̄(1.26)=52.8

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

fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))

a = ax[0, 0]                                           # 1) capital policy
a.plot(Wg, kp_g, color=B, lw=2, label="greedy k' (argmax Q̂)")
a.plot(Wg, kp_e, color=O, lw=1.8, ls="-.", label="expected k' (E[a])")
a.plot(Wg, kp_v, "k--", lw=1.4, label="VFI k'")
a.axhline(ks_a, color="gray", ls=":", lw=.8); a.axvline(wb_a, color="gray", ls=":", lw=.8)
a.set_title("Capital policy k'(ω)"); a.set_xlabel("ω"); a.set_ylabel("k'"); a.legend(fontsize=8)

a = ax[0, 1]                                           # 2) debt policy
a.plot(Wg, bp_g, color=B, lw=2, label="greedy b'")
a.plot(Wg, bp_e, color=O, lw=1.8, ls="-.", label="expected b'")
a.plot(Wg, bp_v, "k--", lw=1.4, label="VFI b'")
a.axhline(0, color="k", lw=.6); a.axvline(wb_a, color="gray", ls=":", lw=.8)
a.set_title("Debt policy b'(ω)"); a.set_xlabel("ω"); a.set_ylabel("b'"); a.legend(fontsize=8)

a = ax[0, 2]                                           # 3) policy gap + uncertainty
a.plot(Wg, kp_g - kp_v, color=B, lw=2, label="greedy − VFI (k')")
a.plot(Wg, kp_e - kp_v, color=O, lw=1.8, ls="-.", label="expected − VFI (k')")
a.axhline(0, color="k", lw=.6); a.axvline(wb_a, color="gray", ls=":", lw=.8)
a2 = a.twinx(); a2.fill_between(Wg, sd_g, color="gray", alpha=.2)
a2.set_ylabel("belief std at greedy", color="gray")
a.set_title("Policy error vs rational (shade = uncertainty)")
a.set_xlabel("ω"); a.set_ylabel("Δk'"); a.legend(fontsize=8)

a = ax[1, 0]                                           # 4) implied transition map
a.plot(Wg, Wg, "k:", lw=.8, label="45°")
a.plot(Wg, wn_g, color=B, lw=2, label="greedy")
a.plot(Wg, wn_e, color=O, lw=1.8, ls="-.", label="expected")
a.plot(Wg, wn_v, "k--", lw=1.4, label="VFI")
a.axvline(wb_a, color="gray", ls=":", lw=.8)
a.set_title("Implied transition ω'=Φ(ω)"); a.set_xlabel("ω"); a.set_ylabel("ω'"); a.legend(fontsize=8)

a = ax[1, 1]                                           # 5) implied velocity (rest points)
a.plot(Wg, wn_g - Wg, color=B, lw=2, label="greedy")
a.plot(Wg, wn_e - Wg, color=O, lw=1.8, ls="-.", label="expected")
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

a = ax[1, 2]                                           # 6) MPK wedge (UNCLIPPED, log scale)
mp = env.production.marginal_product
a.semilogy(Wg, np.maximum(mp(z=z_a, k=np.maximum(kp_g, 1e-9)) - (p.R + p.DELTA), 1e-4),
           color=B, lw=2, label="greedy")
a.semilogy(Wg, np.maximum(mp(z=z_a, k=np.maximum(kp_e, 1e-9)) - (p.R + p.DELTA), 1e-4),
           color=O, lw=1.8, ls="-.", label="expected")
a.semilogy(Wg, np.maximum(mp(z=z_a, k=np.maximum(kp_v, 1e-9)) - (p.R + p.DELTA), 1e-4),
           "k--", lw=1.4, label="VFI")
a.axvline(wb_a, color="gray", ls=":", lw=.8)
a.set_title("MPK wedge (log scale)"); a.set_xlabel("ω"); a.set_ylabel("wedge")
a.legend(fontsize=8)

fig.suptitle(f"Learned policy diagnostics — z={z_a:.2f}, n={len(agent.gp.Y)} observations", y=.995)
fig.tight_layout(); fig.savefig("agent_policy_2x3.png", dpi=120)
print("saved: agent_policy_2x3.png")

# ---- numeric sanity: the three claims the figure makes -------------------------------
print("max visited ω:", float(agent.gp.X_dec[:, 1].max()))
print("agent implied rest:", float(Wg[s_[0]]) if len(s_) else None, " vs rational:", wb_a)