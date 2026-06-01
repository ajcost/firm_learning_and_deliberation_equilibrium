#!/usr/bin/env python
"""
Cross-sectional ergodic analysis of firm investment dynamics.

Simulates N firms facing common macro shocks under GP-based learning,
benchmarked against a rational (VFI) agent.  Produces an 8-panel
diagnostic dashboard.
"""

import sys
sys.path.append("..")  # for relative imports from src/

import numpy as np
import matplotlib.pyplot as plt

from src.simulation.environment import (
    InvestmentParameters,
    InvestmentEnvironment,
    QuadraticAdjustmentCosts,
    run_simulation,
)
from src.simulation.firm import (
    RationalInvestmentAgent,
    ExperienceReasoningAgent,
    InvestmentAgentParameters,
)
from src.simulation.gaussian_process import (
    GPBelief,
    GPBeliefParameters,
    LaplacianKernel,
    TrueValueFunctionPrior,
    RBFKernel,
)
from src.simulation.postprocessing import cluster_firm_policies


p = InvestmentParameters(
    KAPPA=2.0, SIGMA_EPS=0.05, RHO=0.9, DELTA=0.04,
    R=0.01, N_z=5, BETA=0.97, THETA=0.3,
    K_min=0.0, K_max=30
)

env = InvestmentEnvironment(p, QuadraticAdjustmentCosts(p.KAPPA), seed=42)
rational_agent = RationalInvestmentAgent(env).fit()
k_ss = rational_agent.fixed_point()
b_ss = env.optimal_b_next(k_ss)

kernel = RBFKernel(sigma0=5.0, length_scales=[2.29, 4.50, 5.88])
gp_params = GPBeliefParameters(kernel=kernel, sigma_n=0.3)
true_value_prior = TrueValueFunctionPrior(rational_agent)
H_cal = 0.001

N_FIRMS = 500
T = 300

print(f"Rational steady state: k*={k_ss:.4f}, b*={b_ss:.4f}")

print(f"Creating {N_FIRMS} firms...")
firm_agents = []
for j in range(N_FIRMS):
    gp_j = GPBelief(env_params=p, gp_params=gp_params, prior_mean_fn=true_value_prior)
    agent_j = ExperienceReasoningAgent(
        env=env, gp=gp_j,
        agent_params=InvestmentAgentParameters(H=H_cal, KAPPA_R=0.001),
        experience_only=True,
        seed=j,
    )
    agent_j.name = f"Firm {j}"
    firm_agents.append(agent_j)

print(f"Simulating {N_FIRMS} firms x {T} periods...")
print(f"  k* = {k_ss:.4f},  b* = {b_ss:.4f},  H = {H_cal}")
print(f"  GP: RBF, sigma0^2={gp_params.kernel.sigma0_sq}, "
      f"sigma_n={gp_params.sigma_n}, "
      f"l_z={kernel.length_scales[0]}, l_k={kernel.length_scales[1]}, "
      f"l_i={kernel.length_scales[2]}")

df = run_simulation(env, firm_agents, T=T, z0=1.0, firm_exit_rate=0.02, seed=42)

df

import numpy as np
from scipy.optimize import brentq

z_ss = 1.0
i_ss = rational_agent.policy(z_ss, k_ss)[0] - (1 - p.DELTA) * k_ss
q0 = true_value_prior(np.array([[z_ss, k_ss, i_ss]]))[0]
sigma0 = np.sqrt(kernel.sigma0_sq) * 0.5

def q_at(z, k, i):
    return true_value_prior(np.array([[z, k, i]]))[0]


# ── Ergodic standard deviations from rational simulation ─────
std_z = df['z'].std()
std_k = df['k_rat'].std()
std_i = df['i_rat'].std()


# ── True Q* variation at 1 ergodic std ───────────────────────
dq_z = abs(q_at(z_ss + std_z, k_ss, i_ss) - q0)
dq_k = abs(q_at(z_ss, k_ss + std_k, i_ss) - q0)
dq_i = abs(q_at(z_ss, k_ss, i_ss + std_i) - q0)

# ── Solve for belief-rational ell_j ──────────────────────────
# From proposition: ell_j = d_j / sqrt(-2 ln(1 - |dQ|^2 / (2 sigma0^2)))
# Here d_j = std_j (ergodic std is our chosen displacement)

def belief_rational_ell(d_j, dq_j, sigma0):
    ratio = dq_j**2 / (2 * sigma0**2)
    if ratio >= 1:
        print(f"  Warning: |dQ*| = {dq_j:.4f} >= sqrt(2)*sigma_0 = {np.sqrt(2)*sigma0:.4f}")
        print(f"  No RBF length scale can achieve belief-rationality at this displacement.")
        return np.nan
    return d_j / np.sqrt(-2 * np.log(1 - ratio))

ell_z = belief_rational_ell(std_z, dq_z, sigma0)
ell_k = belief_rational_ell(std_k, dq_k, sigma0)
ell_i = belief_rational_ell(std_i, dq_i, sigma0)

# ── Print ────────────────────────────────────────────────────
print(f"Steady state: z={z_ss}, k={k_ss:.2f}, i={i_ss:.2f}, Q*={q0:.4f}")
print(f"sigma_0 = {sigma0}")
print(f"")
print(f"{'Dim':<6} {'std_j':>8} {'|dQ*|':>8} {'ell_j':>8} {'current':>8}")
print(f"{'-'*40}")
print(f"{'z':<6} {std_z:>8.4f} {dq_z:>8.4f} {ell_z:>8.4f} {kernel.length_scales[0]:>8.4f}")
print(f"{'k':<6} {std_k:>8.4f} {dq_k:>8.4f} {ell_k:>8.4f} {kernel.length_scales[1]:>8.4f}")
print(f"{'i':<6} {std_i:>8.4f} {dq_i:>8.4f} {ell_i:>8.4f} {kernel.length_scales[2]:>8.4f}")

# ── Verify ───────────────────────────────────────────────────
def sigma_hat(dx, ell, sigma0):
    return sigma0 * np.sqrt(2 * (1 - np.exp(-dx**2 / (2 * ell**2))))

print(f"\nVerification (believed should equal true):")
print(f"  z: believed={sigma_hat(std_z, ell_z, sigma0):.4f}  true={dq_z:.4f}")
print(f"  k: believed={sigma_hat(std_k, ell_k, sigma0):.4f}  true={dq_k:.4f}")
print(f"  i: believed={sigma_hat(std_i, ell_i, sigma0):.4f}  true={dq_i:.4f}")

# Isolate the behavioral firm data (long format for k, i, d)
# Ensure data is sorted by agent then time for proper reshaping
firm_df = df.sort_values(["agent_id", "t"])

# The rational baseline is identical across all agents in this simulation structure.
# Extract the rational baseline from the first agent's paired columns.
rat_df = firm_df[firm_df["agent_id"] == 0]
Z     = rat_df["z"].values
K_RAT = rat_df["k_rat"].values
I_RAT = rat_df["i_rat"].values
D_RAT = rat_df["d_rat"].values

# Extract and reshape behavioral firms to (N_FIRMS, T) matrices
K = firm_df.pivot(index="agent_id", columns="t", values="k").values
I = firm_df.pivot(index="agent_id", columns="t", values="i").values
D = firm_df.pivot(index="agent_id", columns="t", values="d").values


print("\n--- Simulation complete ---")
print(f"Final-period cross-section:")
print(f"  Experience K:  mean={K[:, -1].mean():.2f}  "
      f"median={np.median(K[:, -1]):.2f}  std={K[:, -1].std():.2f}")
print(f"  Rational   K:  mean={K_RAT[-1]:.2f}")



# z and k: ergodic std (these are good)
print(f"ell_z = {ell_z:.4f}  (from ergodic std)")
print(f"ell_k = {ell_k:.4f}  (from ergodic std)")

# i: use curvature instead
# Q* drops by sigma_0 at displacement sqrt(2 * sigma_0 * k / phi)
curv = p.KAPPA / k_ss  # |d2Q/di2| = phi / k
d_i_curv = np.sqrt(2 * sigma0 / curv)
ell_i_curv = d_i_curv / np.sqrt(2 * np.log(2))

# Or: solve exactly from Q*
try:
    d_i_exact = brentq(lambda d: q0 - q_at(z_ss, k_ss, i_ss + d) - sigma0, 0.01, 30)
except:
    d_i_exact = brentq(lambda d: q0 - q_at(z_ss, k_ss, i_ss - d) - sigma0, 0.01, 30)
ell_i_exact = d_i_exact / np.sqrt(2 * np.log(2))

print(f"\nell_i options:")
print(f"  From ergodic std:   {ell_i:.4f}  (meaningless — FOC)")
print(f"  From curvature:     {ell_i_curv:.4f}")
print(f"  From exact Q* drop: {ell_i_exact:.4f}")

print(f"\nRecommended belief-rational calibration:")
print(f"  ell_z = {ell_z:.4f}")
print(f"  ell_k = {ell_k:.4f}")
print(f"  ell_i = {ell_i_exact:.4f}")

ell_z_local = sigma0 / abs(dq_dz)  # 7.5 / 2.18
ell_k_local = sigma0 / abs(dq_dk)  # 7.5 / 1.10
# ell_i: dQ/di = 0, so ell_i = infinity

print(f"Local belief-rational (Delta x -> 0):")
print(f"  ell_z = {ell_z_local:.4f}  (sigma_0 / |dQ/dz| = {sigma0}/{abs(dq_dz):.2f})")
print(f"  ell_k = {ell_k_local:.4f}  (sigma_0 / |dQ/dk| = {sigma0}/{abs(dq_dk):.2f})")
print(f"  ell_i = inf        (dQ/di = 0 at optimum)")

# ── Steady state ─────────────────────────────────────────────
z_ss = 1.0
i_ss = rational_agent.policy(z_ss, k_ss)[0] - (1 - p.DELTA) * k_ss
q0 = true_value_prior(np.array([[z_ss, k_ss, i_ss]]))[0]
sigma0 = np.sqrt(kernel.sigma0_sq)

def q_at(z, k, i):
    return true_value_prior(np.array([[z, k, i]]))[0]

# ── Numerical derivatives at steady state ────────────────────
eps = 1e-4
dq_dz = (q_at(z_ss + eps, k_ss, i_ss) - q0) / eps
dq_dk = (q_at(z_ss, k_ss + eps, i_ss) - q0) / eps
dq_di = (q_at(z_ss, k_ss, i_ss + eps) - q0) / eps

print(f"Steady state: z={z_ss}, k={k_ss:.2f}, i={i_ss:.2f}")
print(f"Q*(x*) = {q0:.4f}")
print(f"sigma_0 = {sigma0}")
print(f"")
print(f"Gradients at x*:")
print(f"  dQ/dz = {dq_dz:.4f}")
print(f"  dQ/dk = {dq_dk:.4f}")
print(f"  dQ/di = {dq_di:.4f}")

# ── Local belief-rational: ell_j = sigma_0 / |dQ/dx_j| ──────
ell_z_local = sigma0 / abs(dq_dz) if abs(dq_dz) > 1e-6 else np.nan
ell_k_local = sigma0 / abs(dq_dk) if abs(dq_dk) > 1e-6 else np.nan
ell_i_local = sigma0 / abs(dq_di) if abs(dq_di) > 1e-6 else np.nan

print(f"\nLocal belief-rational (Delta x -> 0):")
print(f"  ell_z = {ell_z_local:.4f}   (current: {kernel.length_scales[0]})")
print(f"  ell_k = {ell_k_local:.4f}   (current: {kernel.length_scales[1]})")
print(f"  ell_i = {ell_i_local:.4f}   (current: {kernel.length_scales[2]})")
print(f"")
if abs(dq_di) < 1e-4:
    print(f"  Note: dQ/di ~ 0 (FOC), so ell_i is meaningless here.")
    print(f"  Use curvature or exact method for i instead.")
from sklearn.cluster import KMeans

# ---------------------------------------------------------
# SMART AVERAGING: Cluster Firms by Learned Policy
# ---------------------------------------------------------
print("\nExtracting and clustering learned policies...")
eval_k_grid = np.linspace(p.K_min, 25.0, 40)
firm_policies = np.zeros((N_FIRMS, len(eval_k_grid)))

# 1. Evaluate every firm's final expected policy at z=1.0
for j, agent in enumerate(firm_agents):
    for i_k, k_val in enumerate(eval_k_grid):
        kp, _ = agent.get_expected_action(z=1.0, k=k_val, b=0.0)
        firm_policies[j, i_k] = kp

# 2. Cluster the policies into 3 behavioral groups
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(firm_policies)

# 3. Calculate the True Rational Policy for comparison
rational_policy = np.array([rational_agent.policy(1.0, k, 0.0)[0] for k in eval_k_grid])
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import UnivariateSpline

def smooth(x, y, s_factor=0.005):
    spl = UnivariateSpline(x, y, s=len(x) * s_factor)
    return spl(x)

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 0.6,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.4,
})

colors = ["#B33030", "#2C5F9E", "#1A7A56"]
cluster_names = ["Expected Firm Policy Group A", "Expected Firm Policy Group B", "Expected Firm Policy Group C"]

fig, ax = plt.subplots(figsize=(6.5, 5.2))

# 45-degree line
ax.plot(eval_k_grid, eval_k_grid, color="0.45", ls="--", lw=0.8, label="45° line")

# Collateral constraint
from scipy.optimize import brentq, minimize_scalar

feas_kp = []
for k in eval_k_grid:
    b_now = env.optimal_b_next(k)
    def d_of_kp(kp):
        i = kp - (1 - p.DELTA) * k
        return env.dividend(1.0, k, i, b_now, env.optimal_b_next(kp))

    # Find k' that maximizes dividend (near zero-investment point)
    res = minimize_scalar(lambda kp: -d_of_kp(kp), bounds=(p.K_min, p.K_max), method="bounded")
    kp_peak = res.x
    d_peak = d_of_kp(kp_peak)

    if d_peak < 0:
        feas_kp.append(np.nan)
    elif d_of_kp(p.K_max) >= 0:
        feas_kp.append(p.K_max)
    else:
        feas_kp.append(brentq(d_of_kp, kp_peak, p.K_max))

feas_kp = np.array(feas_kp)
ax.plot(eval_k_grid, feas_kp, color="0.55", ls="-.", lw=0.8, label="Feasibility ceiling ($d=0$)")

# Rational policy
rat_smooth = smooth(eval_k_grid, rational_policy, s_factor=0.005)
ax.plot(eval_k_grid, rat_smooth, color="k", lw=1.6, ls="--", label="Rational $E[k_{t+1}|k_t]$")
ax.scatter([k_ss], [k_ss], color="k", s=35, zorder=6)

# Collect all equilibrium crossings for zoom
all_crossings = [k_ss]

# Cluster policies
for c in range(n_clusters):
    mask = (cluster_labels == c)
    n_c = mask.sum()
    avg = firm_policies[mask].mean(axis=0)
    std = firm_policies[mask].std(axis=0)
    avg_s = smooth(eval_k_grid, avg, s_factor=0.005)
    std_s = smooth(eval_k_grid, std, s_factor=0.01)

    ax.plot(eval_k_grid, avg_s, color=colors[c], lw=1.6,
            label=f"{cluster_names[c]} (N={n_c})")
    ax.fill_between(eval_k_grid, avg_s - std_s, avg_s + std_s,
                    color=colors[c], alpha=0.08)

    # Equilibrium dots: crossings with 45-degree line
    diff = avg_s - eval_k_grid
    crossings = np.where(np.diff(np.sign(diff)))[0]
    for cx in crossings:
        k_cross = eval_k_grid[cx] + (-diff[cx]) / (diff[cx + 1] - diff[cx]) * (eval_k_grid[cx + 1] - eval_k_grid[cx])
        ax.scatter([k_cross], [k_cross], color=colors[c], s=35, zorder=6,
                   edgecolors="k", linewidths=0.4)
        all_crossings.append(k_cross)

# Zoom: min/max of all equilibrium points ± 3
k_lo = max(0, min(all_crossings) - 3)
k_hi = max(all_crossings) + 3
ax.set_xlim(k_lo, k_hi)
ax.set_ylim(k_lo, k_hi)

ax.set_xlabel("$k_{t}$")
ax.set_ylabel("$k_{t+1}$")
ax.set_title("Clustered Firm Policies (Experience-Only) in Expectation ($E[k_{t+1}|k_t]$)", fontweight="medium")
ax.legend(frameon=False, loc="upper left")
ax.grid(True)

fig.tight_layout()
fig.savefig("../figures/clustered_policies_experienced_learning.pdf", bbox_inches="tight")
plt.show()
###################################################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsfonts}",
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 0.6,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.4,
})

c_main = "#2C5F9E"
c_alt  = "#B33030"
c_tert = "#1A7A56"

H = 0.001  # adjust to your calibration
kappa_grid = sorted([0.01,0.0025, 0.0005, 0.0001])

t_axis = np.arange(df['t'].max() + 1)

# ── Cross-sectional stats per period ─────────────────────────
def period_stats(df, col):
    g = df.groupby('t')[col]
    return g.mean().values, g.median().values, g.quantile(0.25).values, g.quantile(0.75).values

alpha_mean, alpha_med, alpha_p25, alpha_p75 = period_stats(df, 'alpha_gain')
tr_mean, tr_med, tr_p25, tr_p75 = period_stats(df, 'eig_tr')
delta_mean, delta_med, delta_p25, delta_p75 = period_stats(df, 'delta_E')
eig_max_mean, _, eig_max_p25, eig_max_p75 = period_stats(df, 'eig_max')

# Entropy floor = H * Tr(Sigma)
entropy_floor_mean = H * tr_mean
entropy_floor_p25 = H * tr_p25
entropy_floor_p75 = H * tr_p75

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# ── (a) Alpha gain over time ────────────────────────────────
ax = axes[0, 0]
ax.fill_between(t_axis, alpha_p25, alpha_p75, color=c_main, alpha=0.12)
ax.plot(t_axis, alpha_mean, color=c_main, lw=1.6, label="Mean")
ax.plot(t_axis, alpha_med, color=c_main, lw=1.0, ls="--", alpha=0.7, label="Median")
ax.set_xlabel("Period")
ax.set_ylabel("$\\alpha_{\\mathrm{gain}}$")
ax.set_title("(a) Information gain per observation", fontweight="medium")
ax.legend(frameon=False)
ax.grid(True)

# ── (b) Trace and entropy floor over time ────────────────────
ax = axes[0, 1]
ax.fill_between(t_axis, tr_p25, tr_p75, color=c_main, alpha=0.12)
ax.plot(t_axis, tr_mean, color=c_main, lw=1.6, label="Mean $\\mathrm{Tr}(\\Sigma_E)$")
ax.plot(t_axis, tr_med, color=c_main, lw=1.0, ls="--", alpha=0.7, label="Median")

# Prior trace as reference
n_actions = df.groupby('t')['eig_n_active'].mean().iloc[0]
prior_tr = n_actions * 25.0
ax.axhline(prior_tr, color="0.5", ls=":", lw=0.8, label=f"Prior $\\mathrm{{Tr}} = {prior_tr:.0f}$")

ax.set_xlabel("Period")
ax.set_ylabel("$\\mathrm{Tr}(\\Sigma_E)$", color=c_main)
ax.tick_params(axis='y', labelcolor=c_main)
ax.set_title("(b) Posterior uncertainty and entropy floor", fontweight="medium")

# Entropy floor on twin axis
ax2 = ax.twinx()
ax2.fill_between(t_axis, entropy_floor_p25, entropy_floor_p75, color=c_alt, alpha=0.08)
ax2.plot(t_axis, entropy_floor_mean, color=c_alt, lw=1.4, ls="--",
         label=f"Entropy floor $H \\cdot \\mathrm{{Tr}}(\\Sigma_E)$")
ax2.set_ylabel("$H \\cdot \\mathrm{Tr}(\\Sigma_E)$", color=c_alt)
ax2.tick_params(axis='y', labelcolor=c_alt)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=7)
ax.grid(True)

# ── (c) Delta_E over time ───────────────────────────────────
ax = axes[0, 2]
ax.fill_between(t_axis, delta_p25, delta_p75, color=c_main, alpha=0.12)
ax.plot(t_axis, delta_mean, color=c_main, lw=1.6, label="Mean")
ax.plot(t_axis, delta_med, color=c_main, lw=1.0, ls="--", alpha=0.7, label="Median")
ax.set_xlabel("Period")
ax.set_ylabel("$\\delta_E$")
ax.set_title("(c) Shadow price of entropy constraint", fontweight="medium")
ax.legend(frameon=False)
ax.grid(True)

# ── (d) Reasoning trigger fraction ──────────────────────────
# At t=0, prior variance is high -> all eigenvalues above water -> starts at 1
# As GP learns, eigenvalues shrink -> fraction drops
ax = axes[1, 0]
colors_kappa = ["#2C5F9E", "#1A7A56", "#E8871E", "#B33030"]

for kappa_R, ck in zip(kappa_grid, colors_kappa):
    water = kappa_R / (H * df['delta_E'].clip(lower=1e-12))
    would_reason = (df['eig_max'] > water).astype(float)
    frac_by_t = would_reason.groupby(df['t']).mean().values
    ax.plot(t_axis, frac_by_t, lw=1.4, color=ck,
            label=f"$\\kappa_R = {kappa_R}$")

ax.set_xlabel("Period")
ax.set_ylabel("Fraction of firms reasoning")
ax.set_ylim(-0.02, 1.02)
ax.set_title("(d) Reasoning trigger rate", fontweight="medium")
ax.legend(frameon=False, fontsize=7)
ax.grid(True)

# ── (e) Mean number of eigenvalues above water line ──────────
ax = axes[1, 1]

for kappa_R, ck in zip(kappa_grid, colors_kappa):
    # For each firm-period, count eigenvalues above water level
    # We only have summary stats, so estimate:
    # n_above ≈ n_active * P(lambda > water) 
    # Approximate: if eig_max > water, at least 1 is above
    # Better: use eig_mean and eig_std to estimate
    water = kappa_R / (H * df['delta_E'].clip(lower=1e-12))

    # Fraction of variance above water: (eig_tr - n_active * min(eig_mean, water)) / eig_tr
    # Rough estimate: interpolate between 0 (water > eig_max) and n_active (water < eig_min)
    frac_above = np.clip((df['eig_max'] - water) / (df['eig_max'] - df['eig_min'].clip(lower=1e-12)), 0, 1)
    n_above = frac_above * df['eig_n_active']
    n_above_by_t = n_above.groupby(df['t']).mean().values

    ax.plot(t_axis, n_above_by_t, lw=1.4, color=ck,
            label=f"$\\kappa_R = {kappa_R}$")

# Reference: total active dimensions
n_active_mean = df.groupby('t')['eig_n_active'].mean().values
ax.plot(t_axis, n_active_mean, color="0.5", ls=":", lw=0.8, label="Total active dims")

ax.set_xlabel("Period")
ax.set_ylabel("Dims above water level")
ax.set_title("(e) Reasoning intensity (dims compressed)", fontweight="medium")
ax.legend(frameon=False, fontsize=7, loc="upper right")
ax.grid(True)

# ── (f) Water level vs eigenvalue spectrum over time ─────────
ax = axes[1, 2]

# Plot eigenvalue range (only max, skip min)
ax.fill_between(t_axis, eig_max_p25, eig_max_p75, color=c_main, alpha=0.12)
ax.plot(t_axis, eig_max_mean, color=c_main, lw=1.6, label="$\\lambda_{\\max}$ (mean)")

# Eigenvalue mean as lower reference
eig_mean_mean = df.groupby('t')['eig_mean'].mean().values
ax.plot(t_axis, eig_mean_mean, color=c_main, lw=1.0, ls=":",
        label="$\\bar{\\lambda}$ (mean)")

# Water levels for each kappa_R
for kappa_R, ck in zip(kappa_grid, colors_kappa):
    water_by_t = kappa_R / (H * df.groupby('t')['delta_E'].mean().clip(lower=1e-12)).values
    water_clipped = np.clip(water_by_t, 0, eig_max_mean.max() * 2)
    ax.plot(t_axis, water_clipped, color=ck, lw=1.2, ls="--",
            label=f"Water ($\\kappa_R={kappa_R}$)")

ax.set_xlabel("Period")
ax.set_ylabel("Eigenvalue / Water level")
ax.set_title("(f) Eigenvalue spectrum vs reasoning threshold", fontweight="medium")
ax.legend(frameon=False, fontsize=6.5, loc="upper right")
ax.set_yscale("log")
ax.grid(True)

fig.tight_layout(h_pad=2.5, w_pad=2.0)
fig.savefig("../figures/learning_dynamics_dashboard.pdf", bbox_inches="tight")
plt.show()

# ── Summary ──────────────────────────────────────────────────
burn_in = 100
erg = df[df['t'] >= burn_in]

print(f"\n{'='*70}")
print(f"  Learning Dynamics Summary (t >= {burn_in})")
print(f"{'='*70}")
print(f"  {'Metric':<30} {'Mean':>10} {'Median':>10} {'Std':>10}")
print(f"  {'-'*60}")
for col, label in [
    ('alpha_gain', 'Alpha gain'),
    ('eig_tr', 'Tr(Sigma)'),
    ('eig_max', 'Lambda_max'),
    ('delta_E', 'Delta_E'),
    ('eig_n_active', 'N active dims'),
]:
    if col in erg.columns:
        print(f"  {label:<30} {erg[col].mean():>10.4f} {erg[col].median():>10.4f} {erg[col].std():>10.4f}")

print(f"\n  Entropy floor (ergodic): H * Tr = {H * erg['eig_tr'].mean():.6f}")

print(f"\n  {'Reasoning trigger rates (ergodic)':}")
print(f"  {'kappa_R':<12} {'Frac reasoning':>16} {'Est. dims compressed':>22}")
for kappa_R in kappa_grid:
    water = kappa_R / (H * erg['delta_E'].clip(lower=1e-12))
    frac = (erg['eig_max'] > water).mean()
    frac_above = np.clip((erg['eig_max'] - water) / (erg['eig_max'] - erg['eig_min'].clip(lower=1e-12)), 0, 1)
    n_above = (frac_above * erg['eig_n_active']).mean()
    print(f"  {kappa_R:<12.4f} {100*frac:>15.1f}% {n_above:>21.1f}")

print(f"{'='*70}")

# ── Summary stats ────────────────────────────────────────────
burn_in = 100
erg = df[df['t'] >= burn_in]

print(f"\n{'='*65}")
print(f"  Learning Dynamics Summary (post burn-in, t >= {burn_in})")
print(f"{'='*65}")
print(f"  {'Metric':<30} {'Mean':>10} {'Median':>10} {'Std':>10}")
print(f"  {'-'*60}")
for col, label in [
    ('alpha_gain', 'Alpha gain'),
    ('eig_tr', 'Tr(Sigma)'),
    ('eig_max', 'Lambda_max'),
    ('delta_E', 'Delta_E'),
    ('eig_n_active', 'N active dims'),
]:
    if col in erg.columns:
        print(f"  {label:<30} {erg[col].mean():>10.4f} {erg[col].median():>10.4f} {erg[col].std():>10.4f}")

print(f"\n  {'Reasoning trigger rates (ergodic)':}")
for kappa_R in kappa_grid:
    water = kappa_R / (H * erg['delta_E'].clip(lower=1e-12))
    frac = (erg['eig_max'] > water).mean()
    print(f"    kappa_R = {kappa_R:.4f}: {100*frac:.1f}% of firm-periods would trigger reasoning")

print(f"{'='*65}")



# ── Summary ──────────────────────────────────────────────────
burn_in = 100
erg = df[df['t'] >= burn_in]

print(f"\n{'='*70}")
print(f"  Learning Dynamics Summary (t >= {burn_in})")
print(f"{'='*70}")
print(f"  {'Metric':<30} {'Mean':>10} {'Median':>10} {'Std':>10}")
print(f"  {'-'*60}")
for col, label in [
    ('alpha_gain', 'Alpha gain'),
    ('eig_tr', 'Tr(Sigma)'),
    ('eig_max', 'Lambda_max'),
    ('delta_E', 'Delta_E'),
    ('eig_n_active', 'N active dims'),
]:
    if col in erg.columns:
        print(f"  {label:<30} {erg[col].mean():>10.4f} {erg[col].median():>10.4f} {erg[col].std():>10.4f}")

print(f"\n  Entropy floor (ergodic): H * Tr = {H * erg['eig_tr'].mean():.6f}")

print(f"\n  {'Reasoning trigger rates (ergodic)':}")
print(f"  {'kappa_R':<12} {'Frac reasoning':>16} {'Est. dims compressed':>22}")
for kappa_R in kappa_grid:
    water = kappa_R / (H * erg['delta_E'].clip(lower=1e-12))
    frac = (erg['eig_max'] > water).mean()
    frac_above = np.clip((erg['eig_max'] - water) / (erg['eig_max'] - erg['eig_min'].clip(lower=1e-12)), 0, 1)
    n_above = (frac_above * erg['eig_n_active']).mean()
    print(f"  {kappa_R:<12.4f} {100*frac:>15.1f}% {n_above:>21.1f}")

print(f"{'='*70}")




######################################################################
from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import gaussian_kde

def smooth(x, y, s_factor=0.005):
    spl = UnivariateSpline(x, y, s=len(x) * s_factor)
    return spl(x)

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsfonts}",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 0.6,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.4,
})

# ── Coarse screen ────────────────────────────────────────────
print("\nScreening firms at k* ...")
kp_at_ss = np.zeros(N_FIRMS)
for j, agent in enumerate(firm_agents):
    kp, _ = agent.get_expected_action(z=1.0, k=k_ss, b=0.0)
    kp_at_ss[j] = kp

deviation = kp_at_ss - k_ss

# Bucket: below / near / above (exclude top/bottom 10% as outliers)
p10, p90 = np.percentile(deviation, [10, 90])
threshold = 0.15 # within this of k_ss counts as "near"

bucket_below = np.where((deviation < -(threshold+0.2)) & (deviation > p10))[0]
bucket_near  = np.where(np.abs(deviation) <= threshold)[0]
bucket_above = np.where((deviation > (threshold+0.2)) & (deviation < p90))[0]

rng = np.random.default_rng(42)
idx_below = rng.choice(bucket_below) if len(bucket_below) > 0 else np.argmin(deviation)
idx_near  = rng.choice(bucket_near)  if len(bucket_near) > 0  else np.argmin(np.abs(deviation))
idx_above = rng.choice(bucket_above) if len(bucket_above) > 0 else np.argmax(deviation)

selected = [
    (idx_below, f"Firm Below $k^*$", "#2C5F9E"),
    (idx_near,  f"Firm Near $k^*$",   "#1A7A56"),
    (idx_above, f"Firm Above $k^*$",  "#E8871E"),
]

print(f"  k* = {k_ss:.2f}")
for idx, label, _ in selected:
    print(f"  {label}: k'(k*) = {kp_at_ss[idx]:.2f}, dev = {deviation[idx]:+.2f}")

# ── Full policy for selected + all firms on coarse grid ──────
eval_k_grid = np.linspace(p.K_min, 25.0, 40)
rational_policy = np.array([rational_agent.policy(1.0, k, 0.0)[0] for k in eval_k_grid])

def get_policy_bands(agent, eval_k_grid, z=1.0):
    means = np.zeros(len(eval_k_grid))
    stds = np.zeros(len(eval_k_grid))
    for i_k, k_val in enumerate(eval_k_grid):
        k_cands, X_q, mean, std = agent.get_beliefs(z, k_val, 0.0)
        probs, _ = agent._entropy_policy(mean, std)
        means[i_k] = np.dot(probs, k_cands)
        stds[i_k] = np.sqrt(max(np.dot(probs, k_cands**2) - means[i_k]**2, 0))
    return means, stds

# All firm policies on coarser grid for panel (b)
print("Extracting all firm policies (coarse grid for envelope)...")
env_k_grid_coarse = np.linspace(p.K_min, 25.0, 20)
all_policies = np.zeros((N_FIRMS, len(env_k_grid_coarse)))
for j, agent in enumerate(firm_agents):
    for i_k, k_val in enumerate(env_k_grid_coarse):
        kp, _ = agent.get_expected_action(z=1.0, k=k_val, b=0.0)
        all_policies[j, i_k] = kp

cross_mean = np.mean(all_policies, axis=0)
cross_std  = np.std(all_policies, axis=0)
cross_p10  = np.percentile(all_policies, 10, axis=0)
cross_p25  = np.percentile(all_policies, 25, axis=0)
cross_p75  = np.percentile(all_policies, 75, axis=0)
cross_p90  = np.percentile(all_policies, 90, axis=0)

rational_policy_coarse = np.array([rational_agent.policy(1.0, k, 0.0)[0] for k in env_k_grid_coarse])

# ── Feasibility ceiling ──────────────────────────────────────
feas_kp = []
for k in eval_k_grid:
    b_now = env.optimal_b_next(k)
    def d_of_kp(kp, k=k, b_now=b_now):
        i = kp - (1 - p.DELTA) * k
        return env.dividend(1.0, k, i, b_now, env.optimal_b_next(kp))
    res = minimize_scalar(lambda kp: -d_of_kp(kp), bounds=(p.K_min, p.K_max), method="bounded")
    kp_peak = res.x
    d_peak = d_of_kp(kp_peak)
    if d_peak < 0:
        feas_kp.append(np.nan)
    elif d_of_kp(p.K_max) >= 0:
        feas_kp.append(p.K_max)
    else:
        feas_kp.append(brentq(d_of_kp, kp_peak, p.K_max))
feas_kp = np.array(feas_kp)

# ── Figure ───────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.2))

# ── Panel (a): Representative individual policies ────────────
ax = ax_a
ax.plot(eval_k_grid, eval_k_grid, color="0.45", ls="--", lw=1.0, label="$45°$ line")
ax.plot(eval_k_grid, feas_kp, color="0.55", ls="-.", lw=1.0, label="Feasibility ceiling")

rat_smooth = smooth(eval_k_grid, rational_policy, s_factor=0.005)
ax.plot(eval_k_grid, rat_smooth, color="#B33030", lw=2.0, ls="--",
        label=f"Rational $\\mathbb{{E}}[k_{{t+1}}|k_t]$")
ax.scatter([k_ss], [k_ss], color="#B33030", s=45, zorder=6)

all_crossings = [k_ss]
for idx, label, color in selected:
    means, stds = get_policy_bands(firm_agents[idx], eval_k_grid)
    means_s = smooth(eval_k_grid, means, s_factor=0.005)
    stds_s = smooth(eval_k_grid, stds, s_factor=0.01)

    ax.plot(eval_k_grid, means_s, color=color, lw=1.6, label=label)
    ax.fill_between(eval_k_grid, means_s - stds_s, means_s + stds_s,
                    color=color, alpha=0.1)

    diff = means_s - eval_k_grid
    crossings = np.where(np.diff(np.sign(diff)))[0]
    for cx in crossings:
        k_cross = eval_k_grid[cx] + (-diff[cx]) / (diff[cx+1] - diff[cx]) * (eval_k_grid[cx+1] - eval_k_grid[cx])
        ax.scatter([k_cross], [k_cross], color=color, s=35, zorder=6,
                   edgecolors="k", linewidths=0.4)
        all_crossings.append(k_cross)

k_lo = max(0, min(all_crossings) - 3)
k_hi = max(all_crossings) + 3
ax.set_xlim(k_lo, k_hi)
ax.set_ylim(k_lo, k_hi)
ax.set_xlabel("$k_t$")
ax.set_ylabel("$k_{t+1}$")
ax.set_title("(a) Representative firm policies", fontweight="medium")
ax.legend(frameon=False, loc="upper left", fontsize=7.5)
ax.grid(True)

# ── Panel (b): Cross-sectional policy envelope ───────────────
ax = ax_b
ax.plot(env_k_grid_coarse, env_k_grid_coarse, color="0.45", ls="--", lw=0.8, label="$45°$ line")

# 10-90 band
ax.fill_between(env_k_grid_coarse, cross_p10, cross_p90,
                color="#2C5F9E", alpha=0.08, label="10th--90th pctile")
# 25-75 band
ax.fill_between(env_k_grid_coarse, cross_p25, cross_p75,
                color="#2C5F9E", alpha=0.15, label="25th--75th pctile")
# Cross-sectional mean
mean_s = smooth(env_k_grid_coarse, cross_mean, s_factor=0.005)
ax.plot(env_k_grid_coarse, mean_s, color="#2C5F9E", lw=1.6,
        label=f"Experience mean ($N={N_FIRMS}$)")

# Rational
rat_s = smooth(env_k_grid_coarse, rational_policy_coarse, s_factor=0.005)
ax.plot(env_k_grid_coarse, rat_s, color="#B33030", lw=2.0, ls="--",
        label="Rational $\\mathbb{E}[k_{t+1}|k_t]$")
ax.scatter([k_ss], [k_ss], color="#B33030", s=45, zorder=6)

ax.set_xlim(k_lo, k_hi)
ax.set_ylim(k_lo, k_hi)
ax.set_xlabel("$k_t$")
ax.set_ylabel("$k_{t+1}$")
ax.set_title("(b) Cross-sectional policy distribution", fontweight="medium")
ax.legend(frameon=False, loc="upper left", fontsize=7.5)
ax.grid(True)

fig.tight_layout(w_pad=2.5)
fig.savefig("../figures/representative_policies_and_envelope.pdf", bbox_inches="tight")
plt.show()

# ── Summary ──────────────────────────────────────────────────
print(f"\n  Rational k* = {k_ss:.2f}")
print(f"  Cross-sectional mean k'(k*) = {kp_at_ss.mean():.2f}")
print(f"  Cross-sectional std  k'(k*) = {kp_at_ss.std():.2f}")
print(f"  Below k*: {(deviation < 0).sum()} ({100*(deviation < 0).mean():.0f}%)")
print(f"  Above k*: {(deviation > 0).sum()} ({100*(deviation > 0).mean():.0f}%)")
########################################################################################

from scipy.interpolate import UnivariateSpline
from scipy.optimize import brentq, minimize_scalar

def smooth(x, y, s_factor=0.005):
    spl = UnivariateSpline(x, y, s=len(x) * s_factor)
    return spl(x)

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 0.6,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.4,
})

# Pick 3 firms from each cluster closest to cluster centroid
n_per_cluster = 3
selected_firms = {}
for c in range(n_clusters):
    mask = (cluster_labels == c)
    cluster_indices = np.where(mask)[0]
    centroid = kmeans.cluster_centers_[c]
    dists = np.linalg.norm(firm_policies[cluster_indices] - centroid, axis=1)
    closest = cluster_indices[np.argsort(dists)[:n_per_cluster]]
    selected_firms[c] = closest

colors_cluster = ["#B33030", "#2C5F9E", "#1A7A56"]
cluster_names = ["Cluster A", "Cluster B", "Cluster C"]
line_styles = ["-", "--", ":"]

fig, ax = plt.subplots(figsize=(6.5, 5.2))

# 45-degree line
ax.plot(eval_k_grid, eval_k_grid, color="0.45", ls="--", lw=0.8, label="$45°$ line")

# Feasibility ceiling
feas_kp = []
for k in eval_k_grid:
    b_now = env.optimal_b_next(k)
    def d_of_kp(kp):
        i = kp - (1 - p.DELTA) * k
        return env.dividend(1.0, k, i, b_now, env.optimal_b_next(kp))
    res = minimize_scalar(lambda kp: -d_of_kp(kp), bounds=(p.K_min, p.K_max), method="bounded")
    kp_peak = res.x
    d_peak = d_of_kp(kp_peak)
    if d_peak < 0:
        feas_kp.append(np.nan)
    elif d_of_kp(p.K_max) >= 0:
        feas_kp.append(p.K_max)
    else:
        feas_kp.append(brentq(d_of_kp, kp_peak, p.K_max))
feas_kp = np.array(feas_kp)
ax.plot(eval_k_grid, feas_kp, color="0.55", ls="-.", lw=0.8, label="Feasibility ceiling ($d=0$)")

# Rational policy
rat_smooth = smooth(eval_k_grid, rational_policy, s_factor=0.005)
ax.plot(eval_k_grid, rat_smooth, color="k", lw=1.6, ls="--", label="Rational $\\mathbb{E}[k'|k]$")
ax.scatter([k_ss], [k_ss], color="k", s=35, zorder=6)

# Individual firm policies
all_crossings = [k_ss]
legend_done = set()

for c in range(n_clusters):
    for idx, j in enumerate(selected_firms[c]):
        pol = firm_policies[j]
        pol_s = smooth(eval_k_grid, pol, s_factor=0.005)

        label = f"{cluster_names[c]}" if c not in legend_done else None
        legend_done.add(c)

        ax.plot(eval_k_grid, pol_s, color=colors_cluster[c], lw=1.2,
                ls=line_styles[idx], alpha=0.85, label=label)

        # Equilibrium dots
        diff = pol_s - eval_k_grid
        crossings = np.where(np.diff(np.sign(diff)))[0]
        for cx in crossings:
            k_cross = eval_k_grid[cx] + (-diff[cx]) / (diff[cx+1] - diff[cx]) * (eval_k_grid[cx+1] - eval_k_grid[cx])
            ax.scatter([k_cross], [k_cross], color=colors_cluster[c], s=25, zorder=6,
                       edgecolors="k", linewidths=0.4)
            all_crossings.append(k_cross)

# Zoom
k_lo = max(0, min(all_crossings) - 10)
k_hi = max(all_crossings) + 10
ax.set_xlim(k_lo, k_hi)
ax.set_ylim(k_lo, k_hi)

ax.set_xlabel("$k_{t}$")
ax.set_ylabel("$k_{t+1}$")
ax.set_title("Individual Firm Policies (Experience-Only)", fontweight="medium")
ax.legend(frameon=False, loc="upper left")
ax.grid(True)

fig.tight_layout()
fig.savefig("../figures/individual_firm_policies.pdf", bbox_inches="tight")
plt.show()

# Print which firms were selected
for c in range(n_clusters):
    print(f"{cluster_names[c]}: firms {selected_firms[c].tolist()}")




def smooth_expected_policy(agent, z, k, b=0.0):
    """Softmax-weighted E[k'] — the actual policy the firm uses."""
    k_cands, X_q, mean, std = agent.get_beliefs(z, k, b)
    probs, _ = agent._entropy_policy(mean, std)
    return np.dot(probs, k_cands)

eval_k = np.linspace(0.5, 20, 300)  # finer grid

fig, ax = plt.subplots(figsize=(6.5, 5.2))
ax.plot(eval_k, eval_k, color="0.45", ls="--", lw=0.8, label="$45°$ line")

rat_kp = np.array([rational_agent.policy(1.0, k)[0] for k in eval_k])
ax.plot(eval_k, smooth(eval_k, rat_kp), color="k", lw=1.6, ls="--",
        label="Rational $\\mathbb{E}[k'|k]$")
ax.scatter([k_ss], [k_ss], color="k", s=35, zorder=6)

colors_firm = ["#B33030", "#2C5F9E", "#1A7A56"]
all_crossings = [k_ss]

for idx, j in enumerate(firm_ids):
    kp = np.array([smooth_expected_policy(firm_agents[j], 1.0, k) for k in eval_k])
    kp_s = smooth(eval_k, kp, s_factor=0.003)

    ax.plot(eval_k, kp_s, color=colors_firm[idx], lw=1.4, label=f"Firm {j}")

    diff = kp_s - eval_k
    crossings = np.where(np.diff(np.sign(diff)))[0]
    for cx in crossings:
        k_cross = eval_k[cx] + (-diff[cx]) / (diff[cx+1] - diff[cx]) * (eval_k[cx+1] - eval_k[cx])
        ax.scatter([k_cross], [k_cross], color=colors_firm[idx], s=30, zorder=6,
                   edgecolors="k", linewidths=0.4)
        all_crossings.append(k_cross)

k_lo = max(0, min(all_crossings) - 2)
k_hi = max(all_crossings) + 2
ax.set_xlim(k_lo, k_hi)
ax.set_ylim(k_lo, k_hi)
ax.set_xlabel("$k_{t}$")
ax.set_ylabel("$k_{t+1}$")
ax.set_title("Individual Learned Policies", fontweight="medium")
ax.legend(frameon=False, loc="upper left")
ax.grid(True)
fig.tight_layout()

df


def smooth(x, y, s_factor=0.005):
    spl = UnivariateSpline(x, y, s=len(x) * s_factor)
    return spl(x)

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 0.6,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.4,
})

# Pick 3 random firms
rng = np.random.default_rng(0)
firm_ids = rng.choice(N_FIRMS, 3, replace=False)

colors_firm = ["#B33030", "#2C5F9E", "#1A7A56"]

fig, ax = plt.subplots(figsize=(6.5, 5.2))

ax.plot(eval_k_grid, eval_k_grid, color="0.45", ls="--", lw=0.8, label="$45°$ line")

# Rational
rat_smooth = smooth(eval_k_grid, rational_policy, s_factor=0.005)
ax.plot(eval_k_grid, rat_smooth, color="k", lw=1.6, ls="--", label="Rational $\\mathbb{E}[k'|k]$")
ax.scatter([k_ss], [k_ss], color="k", s=35, zorder=6)

all_crossings = [k_ss]

for idx, j in enumerate(firm_ids):
    pol_s = smooth(eval_k_grid, firm_policies[j], s_factor=0.005)
    ax.plot(eval_k_grid, pol_s, color=colors_firm[idx], lw=1.4,
            label=f"Firm {j}")

    diff = pol_s - eval_k_grid
    crossings = np.where(np.diff(np.sign(diff)))[0]
    for cx in crossings:
        k_cross = eval_k_grid[cx] + (-diff[cx]) / (diff[cx+1] - diff[cx]) * (eval_k_grid[cx+1] - eval_k_grid[cx])
        ax.scatter([k_cross], [k_cross], color=colors_firm[idx], s=30, zorder=6,
                   edgecolors="k", linewidths=0.4)
        all_crossings.append(k_cross)

k_lo = max(0, min(all_crossings) - 2)
k_hi = max(all_crossings) + 5
ax.set_xlim(k_lo, k_hi)
ax.set_ylim(k_lo, k_hi)

ax.set_xlabel("$k_{t}$")
ax.set_ylabel("$k_{t+1}$")
ax.set_title("Individual Learned Policies", fontweight="medium")
ax.legend(frameon=False, loc="upper left")
ax.grid(True)

fig.tight_layout()
fig.savefig("../figures/individual_firm_policies.pdf", bbox_inches="tight")
plt.show()

print(f"Selected firms: {firm_ids.tolist()}")


for j in firm_ids:
    firm_data = df[df['agent_id'] == j]
    n_high_k = (firm_data['k'] > 15).sum()
    print(f"Firm {j}: {n_high_k}/{len(firm_data)} obs with k > 15")






from scipy.interpolate import UnivariateSpline

def smooth(x, y, s_factor=0.005):
    spl = UnivariateSpline(x, y, s=len(x) * s_factor)
    return spl(x)

def smooth_expected_policy(agent, z, k, b=0.0):
    k_cands, X_q, mean, std = agent.get_beliefs(z, k, b)
    probs, _ = agent._entropy_policy(mean, std)
    return np.dot(probs, k_cands)

eval_k = np.linspace(0.5, 20, 300)

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot(eval_k, eval_k, color="0.45", ls="--", lw=0.8, label="$45°$ line")

rat_kp = np.array([rational_agent.policy(1.0, k)[0] for k in eval_k])
ax.plot(eval_k, smooth(eval_k, rat_kp), color="k", lw=2, ls="--", label="Rational")

# Plot all 20 firms, light lines
for j in range(20):
    kp = np.array([smooth_expected_policy(firm_agents[j], 1.0, k) for k in eval_k])
    kp_s = smooth(eval_k, kp, s_factor=0.003)
    ax.plot(eval_k, kp_s, color="#2C5F9E", lw=0.7, alpha=0.4)

# Highlight 3 specific firms
highlight = [0, 50, 75]
colors_h = ["#B33030", "#1A7A56", "#D4820C"]
for idx, j in enumerate(highlight):
    kp = np.array([smooth_expected_policy(firm_agents[j], 1.0, k) for k in eval_k])
    kp_s = smooth(eval_k, kp, s_factor=0.003)
    ax.plot(eval_k, kp_s, color=colors_h[idx], lw=1.6, label=f"Firm {j}")

ax.scatter([k_ss], [k_ss], color="k", s=35, zorder=6)

ax.set_xlim(0, 20)
ax.set_ylim(0, 20)
ax.set_xlabel("$k_{t}$")
ax.set_ylabel("$k_{t+1}$")
ax.set_title(f"Learned Policies ($T={T}$, no exit, $N={N_FIRMS}$)", fontweight="medium")
ax.legend(frameon=False, loc="upper left")
ax.grid(True)

fig.tight_layout()

from matplotlib import cm

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), subplot_kw={'projection': '3d'})

z_fixed = 1.0
k_grid = np.linspace(1, 20, 50)
i_grid = np.linspace(-3, 5, 50)
KK, II = np.meshgrid(k_grid, i_grid)

for ax, j in zip(axes, [0, 5, 12]):
    agent_j = firm_agents[j]
    X_eval = np.column_stack([
        np.full(KK.size, z_fixed),
        KK.ravel(),
        II.ravel()
    ])
    Q_hat = agent_j.gp.predict(X_eval, return_std=False).reshape(KK.shape)

    ax.plot_surface(KK, II, Q_hat, cmap=cm.viridis, alpha=0.85,
                    edgecolor='none', antialiased=True)
    ax.set_xlabel("$k$", fontsize=9)
    ax.set_ylabel("$i$", fontsize=9)
    ax.set_zlabel("$\\hat{Q}(k, i)$", fontsize=9)
    ax.set_title(f"Firm {j}", fontweight="medium")
    ax.view_init(elev=25, azim=-45)

fig.suptitle(f"GP Posterior Mean $\\hat{{Q}}^*(z=1, k, i)$ after $T={T}$",
             fontsize=13, fontweight="medium")
fig.tight_layout(rect=[0, 0, 1, 0.94])










fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

z0, k0 = 1.0, k_ss
i0 = rational_agent.policy(z0, k0)[0] - (1 - p.DELTA) * k0
q0 = q_at(z0, k0, i0)
sigma0 = np.sqrt(kernel.sigma0_sq)

def sigma_hat(dx, ell, sigma0):
    rho = np.exp(-dx**2 / (2 * ell**2))
    return sigma0 * np.sqrt(2 * (1 - rho))

# --- (a) z dimension ---
ax = axes[0]
dz_range = np.linspace(0.001, 1.0, 200)
dq_z = np.array([abs(q_at(z0 + dz, k0, i0) - q0) for dz in dz_range])

ax.plot(dz_range, dq_z, color="k", lw=2, label="True $|\\Delta Q^*|$")
ax.axhline(sigma0, color="0.5", ls=":", lw=0.8, label=f"$\\sigma_0 = {sigma0}$")

for ell, col, lab in [(0.08, "#B33030", "Current $\\ell_z=0.08$"),
                       (0.35, "#D4820C", "$\\ell_z=0.35$"),
                       (2.0, "#1A7A56", "$\\ell_z=2.0$"),
                       (3.6, "#2C5F9E", "Belief-rational $\\ell_z=3.6$")]:
    sig = np.array([sigma_hat(dz, ell, sigma0) for dz in dz_range])
    ax.plot(dz_range, sig, color=col, lw=1.4, ls="--", label=lab)

ax.set_xlabel("$\\Delta z$")
ax.set_ylabel("Variation in $Q^*$")
ax.set_title("(a) $z$ dimension", fontweight="medium")
ax.legend(frameon=False, fontsize=7)
ax.grid(True)

# --- (b) k dimension ---
ax = axes[1]
dk_range = np.linspace(0.01, 15, 200)
dq_k = np.array([abs(q_at(z0, k0 + dk, i0) - q0) for dk in dk_range])

ax.plot(dk_range, dq_k, color="k", lw=2, label="True $|\\Delta Q^*|$")
ax.axhline(sigma0, color="0.5", ls=":", lw=0.8, label=f"$\\sigma_0 = {sigma0}$")

for ell, col, lab in [(4.0, "#B33030", "Current $\\ell_k=4.0$"),
                       (7.7, "#2C5F9E", "Belief-rational $\\ell_k=7.7$")]:
    sig = np.array([sigma_hat(dk, ell, sigma0) for dk in dk_range])
    ax.plot(dk_range, sig, color=col, lw=1.4, ls="--", label=lab)

ax.set_xlabel("$\\Delta k$")
ax.set_ylabel("Variation in $Q^*$")
ax.set_title("(b) $k$ dimension", fontweight="medium")
ax.legend(frameon=False, fontsize=7)
ax.grid(True)

# --- (c) i dimension ---
ax = axes[2]
di_range = np.linspace(0.01, 10, 200)
dq_i = np.array([abs(q_at(z0, k0, i0 + di) - q0) for di in di_range])

ax.plot(di_range, dq_i, color="k", lw=2, label="True $|\\Delta Q^*|$")
ax.axhline(sigma0, color="0.5", ls=":", lw=0.8, label=f"$\\sigma_0 = {sigma0}$")

for ell, col, lab in [(1.5, "#B33030", "Current $\\ell_i=1.5$"),
                       (8.7, "#2C5F9E", "Belief-rational $\\ell_i=8.7$")]:
    sig = np.array([sigma_hat(di, ell, sigma0) for di in di_range])
    ax.plot(di_range, sig, color=col, lw=1.4, ls="--", label=lab)

ax.set_xlabel("$\\Delta i$")
ax.set_ylabel("Variation in $Q^*$")
ax.set_title("(c) $i$ dimension", fontweight="medium")
ax.legend(frameon=False, fontsize=7)
ax.grid(True)

fig.suptitle("Believed vs true variation in $Q^*$ from steady state", fontweight="medium")
fig.tight_layout(rect=[0, 0, 1, 0.94])




########################################################################################

from scipy.stats import gaussian_kde, skew, kurtosis

firm_df = df.sort_values(["agent_id", "t"])
T = firm_df['t'].nunique()

# Behavioral: (N_FIRMS, T)
K = firm_df.pivot(index="agent_id", columns="t", values="k").values
I = firm_df.pivot(index="agent_id", columns="t", values="i").values
D = firm_df.pivot(index="agent_id", columns="t", values="d").values

# Rational: also (N_FIRMS, T)
K_RAT = firm_df.pivot(index="agent_id", columns="t", values="k_rat").values
I_RAT = firm_df.pivot(index="agent_id", columns="t", values="i_rat").values
D_RAT = firm_df.pivot(index="agent_id", columns="t", values="d_rat").values
Z     = firm_df.pivot(index="agent_id", columns="t", values="z").values


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy import stats

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 0.6,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.4,
})

c_exp, c_rat = "#2C5F9E", "k"
burn_in = 100

# ── Helpers ──────────────────────────────────────────────────
def lorenz(arr):
    s = np.sort(arr)
    return np.insert(np.cumsum(s) / np.sum(s), 0, 0)

def gini(arr):
    a = np.sort(arr.flatten()) - arr.min() + 1e-8
    n = len(a)
    return np.sum((2 * np.arange(1, n + 1) - n - 1) * a) / (n * np.sum(a))

def binned_means(x, y, n_bins=20):
    edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    xm = np.array([np.mean(x[idx == b]) for b in range(n_bins)])
    ym = np.array([np.mean(y[idx == b]) for b in range(n_bins)])
    se = np.array([np.std(y[idx == b]) / np.sqrt((idx == b).sum()) for b in range(n_bins)])
    return xm, ym, se

# ── Compute ──────────────────────────────────────────────────
k_avg_gp  = K[:, burn_in:].mean(axis=1)
k_avg_rat = K_RAT[:, burn_in:].mean(axis=1)

Y     = Z * (K ** p.ALPHA)
Y_RAT = Z * (K_RAT ** p.ALPHA)

cfk_gp  = (Y[:, burn_in:]     / np.maximum(K[:, burn_in:], 1e-8)).flatten()
cfk_rat = (Y_RAT[:, burn_in:] / np.maximum(K_RAT[:, burn_in:], 1e-8)).flatten()
ik_gp   = (I[:, burn_in:]     / np.maximum(K[:, burn_in:], 1e-8)).flatten()
ik_rat  = (I_RAT[:, burn_in:] / np.maximum(K_RAT[:, burn_in:], 1e-8)).flatten()

# ── Figure ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# ── (a) Lorenz ───────────────────────────────────────────────
l_gp  = lorenz(k_avg_gp)
l_rat = lorenz(k_avg_rat)
x_lor = np.linspace(0, 1, len(l_gp))
g_gp, g_rat = gini(k_avg_gp), gini(k_avg_rat)

ax1.plot(x_lor, x_lor, color="0.45", ls="--", lw=0.8, label="Perfect equality")
ax1.plot(x_lor, l_gp, color=c_exp, lw=1.6,
         label=f"Experience (Gini = {g_gp:.3f})")
ax1.plot(x_lor, l_rat, color=c_rat, lw=1.6, ls="--",
         label=f"Rational (Gini = {g_rat:.3f})")
ax1.fill_between(x_lor, l_gp, x_lor, color=c_exp, alpha=0.06)
ax1.set_xlabel("Cumulative share of firms")
ax1.set_ylabel("Cumulative share of $\\bar{k}_j$")
ax1.set_title("(a) Ergodic capital Lorenz curve", fontweight="medium")
ax1.legend(frameon=False, loc="upper left")
ax1.grid(True)

# ── (b) Cash-flow sensitivity ────────────────────────────────
xm_gp, ym_gp, se_gp   = binned_means(cfk_gp, ik_gp)
xm_rat, ym_rat, se_rat = binned_means(cfk_rat, ik_rat)

ax2.scatter(xm_gp, ym_gp, s=25, color=c_exp, zorder=5)
ax2.errorbar(xm_gp, ym_gp, yerr=1.96 * se_gp, fmt="none", color=c_exp, alpha=0.4, lw=0.8)
ax2.scatter(xm_rat, ym_rat, s=25, color=c_rat, marker="s", zorder=5)
ax2.errorbar(xm_rat, ym_rat, yerr=1.96 * se_rat, fmt="none", color=c_rat, alpha=0.4, lw=0.8)

b_gp, a_gp, _, _, _   = stats.linregress(cfk_gp, ik_gp)
b_rat, a_rat, _, _, _  = stats.linregress(cfk_rat, ik_rat)
xl = np.linspace(min(xm_gp.min(), xm_rat.min()), max(xm_gp.max(), xm_rat.max()), 100)

ax2.plot(xl, a_gp + b_gp * xl, color=c_exp, lw=1.6,
         label=f"Experience ($\\beta = {b_gp:.3f}$)")
ax2.plot(xl, a_rat + b_rat * xl, color=c_rat, lw=1.6, ls="--",
         label=f"Rational ($\\beta = {b_rat:.3f}$)")
ax2.set_xlabel("$y_t / k_t$")
ax2.set_ylabel("$i_t / k_t$")
ax2.set_title("(b) Investment\u2013cash flow sensitivity", fontweight="medium")
ax2.legend(frameon=False, fontsize=8)
ax2.grid(True)

fig.tight_layout(w_pad=2.5)
fig.savefig("../figures/lorenz_and_cashflow_sensitivity.pdf", bbox_inches="tight")
plt.show()

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 0.6,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.4,
})

c_exp, c_rat = "#2C5F9E", "k"
burn_in = 100

# ── Helpers ──────────────────────────────────────────────────
def lorenz(arr):
    s = np.sort(arr)
    return np.insert(np.cumsum(s) / np.sum(s), 0, 0)

def gini(arr):
    a = np.sort(arr.flatten()) - arr.min() + 1e-8
    n = len(a)
    return np.sum((2 * np.arange(1, n + 1) - n - 1) * a) / (n * np.sum(a))

def binned_means(x, y, n_bins=20):
    edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    xm = np.array([np.mean(x[idx == b]) for b in range(n_bins)])
    ym = np.array([np.mean(y[idx == b]) for b in range(n_bins)])
    se = np.array([np.std(y[idx == b]) / np.sqrt((idx == b).sum()) for b in range(n_bins)])
    return xm, ym, se

# ── Compute ──────────────────────────────────────────────────
k_avg_gp  = K[:, burn_in:].mean(axis=1)
k_avg_rat = K_RAT[:, burn_in:].mean(axis=1)

Y     = Z * (K ** p.ALPHA)
Y_RAT = Z * (K_RAT ** p.ALPHA)

cfk_gp  = (Y[:, burn_in:]     / np.maximum(K[:, burn_in:], 1e-8)).flatten()
cfk_rat = (Y_RAT[:, burn_in:] / np.maximum(K_RAT[:, burn_in:], 1e-8)).flatten()
ik_gp   = (I[:, burn_in:]     / np.maximum(K[:, burn_in:], 1e-8)).flatten()
ik_rat  = (I_RAT[:, burn_in:] / np.maximum(K_RAT[:, burn_in:], 1e-8)).flatten()

# ── Figure ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# ── (a) Lorenz ───────────────────────────────────────────────
l_gp  = lorenz(k_avg_gp)
l_rat = lorenz(k_avg_rat)
x_lor = np.linspace(0, 1, len(l_gp))
g_gp, g_rat = gini(k_avg_gp), gini(k_avg_rat)

ax1.plot(x_lor, x_lor, color="0.45", ls="--", lw=0.8, label="Perfect equality")
ax1.plot(x_lor, l_gp, color=c_exp, lw=1.6,
         label=f"Experience (Gini = {g_gp:.3f})")
ax1.plot(x_lor, l_rat, color=c_rat, lw=1.6, ls="--",
         label=f"Rational (Gini = {g_rat:.3f})")
ax1.fill_between(x_lor, l_gp, x_lor, color=c_exp, alpha=0.06)
ax1.set_xlabel("Cumulative share of firms")
ax1.set_ylabel("Cumulative share of $\\bar{k}_j$")
ax1.set_title("(a) Ergodic capital Lorenz curve", fontweight="medium")
ax1.legend(frameon=False, loc="upper left")
ax1.grid(True)

# ── (b) Cash-flow sensitivity ────────────────────────────────
xm_gp, ym_gp, se_gp   = binned_means(cfk_gp, ik_gp)
xm_rat, ym_rat, se_rat = binned_means(cfk_rat, ik_rat)

ax2.scatter(xm_gp, ym_gp, s=25, color=c_exp, zorder=5)
ax2.errorbar(xm_gp, ym_gp, yerr=1.96 * se_gp, fmt="none", color=c_exp, alpha=0.4, lw=0.8)
ax2.scatter(xm_rat, ym_rat, s=25, color=c_rat, marker="s", zorder=5)
ax2.errorbar(xm_rat, ym_rat, yerr=1.96 * se_rat, fmt="none", color=c_rat, alpha=0.4, lw=0.8)

b_gp, a_gp, _, _, _   = stats.linregress(cfk_gp, ik_gp)
b_rat, a_rat, _, _, _  = stats.linregress(cfk_rat, ik_rat)
xl = np.linspace(min(xm_gp.min(), xm_rat.min()), max(xm_gp.max(), xm_rat.max()), 100)

ax2.plot(xl, a_gp + b_gp * xl, color=c_exp, lw=1.6,
         label=f"Experience ($\\beta = {b_gp:.3f}$)")
ax2.plot(xl, a_rat + b_rat * xl, color=c_rat, lw=1.6, ls="--",
         label=f"Rational ($\\beta = {b_rat:.3f}$)")
ax2.set_xlabel("$y_t / k_t$")
ax2.set_ylabel("$i_t / k_t$")
ax2.set_title("(b) Investment\u2013cash flow sensitivity", fontweight="medium")
ax2.legend(frameon=False, fontsize=8)
ax2.grid(True)

fig.tight_layout(w_pad=2.5)
fig.savefig("../figures/lorenz_and_cashflow_sensitivity.pdf", bbox_inches="tight")
plt.show()

# ── Print moments ────────────────────────────────────────────
print(f"{'Ratio':<12} {'':>5} {'Mean':>8} {'Std':>8} {'Skew':>8} {'Kurt':>8}")
print("-" * 50)
for name, gp, rat in [("i/k", ik_gp, ik_rat), ("y/k", yk_gp, yk_rat), ("d/k", dk_gp, dk_rat)]:
    print(f"{name:<12} {'GP':>5} {np.mean(gp):>8.4f} {np.std(gp):>8.4f} {skew(gp):>8.3f} {kurtosis(gp):>8.3f}")
    print(f"{'':12} {'Rat':>5} {np.mean(rat):>8.4f} {np.std(rat):>8.4f} {skew(rat):>8.3f} {kurtosis(rat):>8.3f}")

fig, ax = plt.subplots(figsize=(6.5, 5.2))

# Classify firms by average z over ergodic period
z_avg_by_firm = Z[:, burn_in:].mean(axis=1)
z_firm_mean = np.mean(z_avg_by_firm)
z_firm_std = np.std(z_avg_by_firm)

# Time-averaged capital per firm over ergodic window
k_avg_gp = K[:, burn_in:].mean(axis=1)      # (N_FIRMS,)
k_avg_rat = K_RAT[:, burn_in:].mean(axis=1)

regimes = [
    (z_avg_by_firm < z_firm_mean - z_firm_std,                                          "#B33030", "Unlucky"),
    ((z_avg_by_firm >= z_firm_mean - z_firm_std) & (z_avg_by_firm <= z_firm_mean + z_firm_std), "#2C5F9E", "Typical"),
    (z_avg_by_firm > z_firm_mean + z_firm_std,                                          "#1A7A56", "Lucky"),
]

ax.plot([0, 1], [0, 1], color="0.45", ls="--", lw=0.8, label="Perfect equality")

for mask, col, lab in regimes:
    n = mask.sum()
    if n < 10:
        continue
    l_gp = lorenz(k_avg_gp[mask])
    x_gp = np.linspace(0, 1, len(l_gp))
    g_gp = gini_coefficient(k_avg_gp[mask])
    ax.plot(x_gp, l_gp, color=col, lw=1.6,
            label=f"{lab} — Exp (Gini = {g_gp:.3f}, N={n})")

    l_rat = lorenz(k_avg_rat[mask])
    x_rat = np.linspace(0, 1, len(l_rat))
    g_rat = gini_coefficient(k_avg_rat[mask])
    ax.plot(x_rat, l_rat, color=col, lw=1.2, ls="--",
            label=f"{lab} — Rat (Gini = {g_rat:.3f})")

ax.set_xlabel("Cumulative share of firms")
ax.set_ylabel("Cumulative share of capital")
ax.set_title("Ergodic average capital Lorenz curves by shock history", fontweight="medium")
ax.legend(frameon=False, loc="upper left", fontsize=7.5)
ax.grid(True)

fig.tight_layout()
fig.savefig("../figures/lorenz_by_shock_history_ergodic.pdf", bbox_inches="tight")
plt.show()

print(f"{'Group':<12} {'N':>5} {'GP Gini':>10} {'Rat Gini':>10} {'GP mean k':>10} {'Rat mean k':>10}")
print("-" * 60)
for mask, _, lab in regimes:
    n = mask.sum()
    if n < 10:
        print(f"{lab:<12} {n:>5} {gini_coefficient(k_avg_gp[mask]):>10.4f} "
              f"{gini_coefficient(k_avg_rat[mask]):>10.4f} "
              f"{k_avg_gp[mask].mean():>10.2f} {k_avg_rat[mask].mean():>10.2f}")








plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 0.6,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.4,
})

colors_main = ["#2C5F9E", "#B33030"]
burn_in = int(T * 0.3)

# ── Compute i/k and z panels (N_FIRMS, T_erg) ───────────────
IK = I[:, burn_in:] / np.maximum(K[:, burn_in:], 1e-8)
IK_RAT = I_RAT[:, burn_in:] / np.maximum(K_RAT[:, burn_in:], 1e-8)
Z_erg = Z[:, burn_in:]

# ── Estimate E[i/k | z] nonparametrically via binned means ──
z_flat = Z_erg.flatten()
ik_flat = IK.flatten()
ik_rat_flat = IK_RAT.flatten()

n_bins = 50
z_bin_edges = np.percentile(z_flat, np.linspace(0, 100, n_bins + 1))
z_bin_centers = 0.5 * (z_bin_edges[:-1] + z_bin_edges[1:])
z_bin_idx = np.digitize(z_flat, z_bin_edges) - 1
z_bin_idx = np.clip(z_bin_idx, 0, n_bins - 1)

# Conditional mean of i/k in each z-bin
ik_cond_mean = np.zeros(n_bins)
ik_rat_cond_mean = np.zeros(n_bins)
for b in range(n_bins):
    mask = z_bin_idx == b
    if mask.sum() > 0:
        ik_cond_mean[b] = np.mean(ik_flat[mask])
        ik_rat_cond_mean[b] = np.mean(ik_rat_flat[mask])

# ── Build residual panels ────────────────────────────────────
z_bin_panel = np.digitize(Z_erg, z_bin_edges) - 1
z_bin_panel = np.clip(z_bin_panel, 0, n_bins - 1)

IK_resid = IK - ik_cond_mean[z_bin_panel]
IK_RAT_resid = IK_RAT - ik_rat_cond_mean[z_bin_panel]

# ── ACF helpers ──────────────────────────────────────────────
def panel_acf(panel, lags=10):
    acf = np.zeros(lags)
    count = 0
    for i in range(panel.shape[0]):
        s = panel[i]
        v = np.var(s)
        if v < 1e-12:
            continue
        sd = s - np.mean(s)
        for lag in range(lags):
            if lag == 0:
                acf[lag] += 1.0
            else:
                acf[lag] += np.sum(sd[lag:] * sd[:-lag]) / (len(sd) * v)
        count += 1
    return acf / max(count, 1)

def panel_acf_conditional(panel, z_panel, mask_fn, lags=10):
    acf = np.zeros(lags)
    count = 0
    for i in range(panel.shape[0]):
        mask = mask_fn(z_panel[i])
        if mask.sum() < lags + 5:
            continue
        s = panel[i][mask]
        v = np.var(s)
        if v < 1e-12:
            continue
        sd = s - np.mean(s)
        for lag in range(lags):
            if lag == 0:
                acf[lag] += 1.0
            else:
                n = len(sd) - lag
                if n < 1:
                    continue
                acf[lag] += np.sum(sd[lag:] * sd[:-lag]) / (len(sd) * v)
        count += 1
    return acf / max(count, 1)

# ── z regime cutoffs ─────────────────────────────────────────
z_mean = np.mean(z_flat)
z_std = np.std(z_flat)

lags_arr = np.arange(10)

# ── Figure: 1x2, raw ACF vs residual ACF ────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

regimes = [
    (lambda z: z < z_mean - z_std,                            "#B33030", "Low $z$"),
    (lambda z: (z >= z_mean - z_std) & (z <= z_mean + z_std), "#2C5F9E", "Mid $z$"),
    (lambda z: z > z_mean + z_std,                            "#1A7A56", "High $z$"),
]
markers = ["o", "D", "s"]

# ── (a) Raw ACF of i/k by z regime ──────────────────────────
for (mask_fn, col, lab), mk in zip(regimes, markers):
    acf_c = panel_acf_conditional(IK, Z_erg, mask_fn)
    ax1.plot(lags_arr, acf_c, marker=mk, ms=4, color=col, lw=1.6, label=f"Exp: {lab}")

acf_rat_pooled = panel_acf(IK_RAT)
ax1.plot(lags_arr, acf_rat_pooled, marker="x", ms=4, color="k", lw=1.2, ls="--", label="Rational (pooled)")

ax1.axhline(0, color="k", lw=0.6)
ax1.set_xlabel("Lag")
ax1.set_ylabel("Autocorrelation of $i/k$")
ax1.set_title("(a) Raw $i/k$ persistence by $z$ regime", fontweight="medium")
ax1.set_xticks(lags_arr)
ax1.legend(frameon=False, fontsize=8)
ax1.grid(True)

# ── (b) Residual ACF of i/k by z regime ─────────────────────
for (mask_fn, col, lab), mk in zip(regimes, markers):
    acf_c = panel_acf_conditional(IK_resid, Z_erg, mask_fn)
    ax2.plot(lags_arr, acf_c, marker=mk, ms=4, color=col, lw=1.6, label=f"Exp: {lab}")

acf_rat_resid_pooled = panel_acf(IK_RAT_resid)
ax2.plot(lags_arr, acf_rat_resid_pooled, marker="x", ms=4, color="k", lw=1.2, ls="--", label="Rational (pooled)")

ax2.axhline(0, color="k", lw=0.6)
ax2.set_xlabel("Lag")
ax2.set_ylabel("Autocorrelation of residual $i/k$")
ax2.set_title("(b) Residual $i/k$ persistence by $z$ regime", fontweight="medium")
ax2.set_xticks(lags_arr)
ax2.legend(frameon=False, fontsize=8)
ax2.grid(True)

fig.tight_layout(w_pad=2.5)
fig.savefig("../figures/acf_raw_vs_residual.pdf", bbox_inches="tight")
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))

regimes = [
    (lambda z: z < z_mean - z_std,                            "#B33030", "Low $z$"),
    (lambda z: (z >= z_mean - z_std) & (z <= z_mean + z_std), "#2C5F9E", "Mid $z$"),
    (lambda z: z > z_mean + z_std,                            "#1A7A56", "High $z$"),
]
markers = ["o", "D", "s"]

# ── (a) Raw ACF of i/k by z regime ──────────────────────────
for (mask_fn, col, lab), mk in zip(regimes, markers):
    acf_c = panel_acf_conditional(IK, Z_erg, mask_fn)
    ax1.plot(lags_arr, acf_c, marker=mk, ms=4, color=col, lw=1.6, label=f"Exp: {lab}")

acf_rat_pooled = panel_acf(IK_RAT)
ax1.plot(lags_arr, acf_rat_pooled, marker="x", ms=4, color="k", lw=1.2, ls="--", label="Rational (pooled)")

ax1.axhline(0, color="k", lw=0.6)
ax1.set_xlabel("Lag")
ax1.set_ylabel("Autocorrelation of $i/k$")
ax1.set_title("(a) Raw $i/k$ persistence", fontweight="medium")
ax1.set_xticks(lags_arr)
ax1.legend(frameon=False, fontsize=8)
ax1.grid(True)

# ── (b) Residual ACF of i/k by z regime ─────────────────────
resid_acfs_gp = {}
for (mask_fn, col, lab), mk in zip(regimes, markers):
    acf_c = panel_acf_conditional(IK_resid, Z_erg, mask_fn)
    resid_acfs_gp[lab] = acf_c
    ax2.plot(lags_arr, acf_c, marker=mk, ms=4, color=col, lw=1.6, label=f"Exp: {lab}")

acf_rat_resid_pooled = panel_acf(IK_RAT_resid)
ax2.plot(lags_arr, acf_rat_resid_pooled, marker="x", ms=4, color="k", lw=1.2, ls="--", label="Rational (pooled)")

ax2.axhline(0, color="k", lw=0.6)
ax2.set_xlabel("Lag")
ax2.set_ylabel("Autocorrelation of residual $i/k$")
ax2.set_title("(b) Residual $i/k$ persistence", fontweight="medium")
ax2.set_xticks(lags_arr)
ax2.legend(frameon=False, fontsize=8)
ax2.grid(True)

# ── (c) Belief persistence wedge: GP residual ACF − rational ─
for (mask_fn, col, lab), mk in zip(regimes, markers):
    wedge = resid_acfs_gp[lab] - acf_rat_resid_pooled
    ax3.plot(lags_arr, wedge, marker=mk, ms=4, color=col, lw=1.6, label=lab)

ax3.axhline(0, color="k", lw=0.6)
ax3.set_xlabel("Lag")
ax3.set_ylabel("$\\Delta$ ACF (Experience $-$ Rational)")
ax3.set_title("(c) Belief persistence wedge", fontweight="medium")
ax3.set_xticks(lags_arr)
ax3.legend(frameon=False, fontsize=8)
ax3.grid(True)

fig.tight_layout(w_pad=2.5)
fig.savefig("../figures/acf_raw_residual_wedge.pdf", bbox_inches="tight")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 0.6,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.4,
})

c_exp, c_rat = "#2C5F9E", "#B33030"
burn_in = 100

# ── Pivot panels ─────────────────────────────────────────────
firm_df = df.sort_values(["agent_id", "t"])
T_sim = firm_df['t'].nunique()

K     = firm_df.pivot(index="agent_id", columns="t", values="k").values
I     = firm_df.pivot(index="agent_id", columns="t", values="i").values
D     = firm_df.pivot(index="agent_id", columns="t", values="d").values
K_RAT = firm_df.pivot(index="agent_id", columns="t", values="k_rat").values
I_RAT = firm_df.pivot(index="agent_id", columns="t", values="i_rat").values
D_RAT = firm_df.pivot(index="agent_id", columns="t", values="d_rat").values
Z     = firm_df.pivot(index="agent_id", columns="t", values="z").values
Q_GP  = firm_df.pivot(index="agent_id", columns="t", values="q_chosen").values

# ── Compute rational Q at each (z, k) ───────────────────────
# Marginal Q = beta * dV/dk, approximated from VFI grid
from scipy.interpolate import interp1d

Q_RAT = np.zeros_like(K_RAT)
for iz in range(rational_agent.env.actual_nz):
    dv_dk = np.gradient(rational_agent.v[iz, :], rational_agent.env.k_grid)
    q_func = interp1d(rational_agent.env.k_grid, p.BETA * dv_dk,
                       kind="linear", fill_value="extrapolate")
    for j in range(K_RAT.shape[0]):
        for t in range(K_RAT.shape[1]):
            z_idx = rational_agent._z_to_idx(Z[j, t])
            if z_idx == iz:
                Q_RAT[j, t] = q_func(K_RAT[j, t])

# ── Ergodic flattened arrays ─────────────────────────────────
Y     = Z * (K ** p.ALPHA)
Y_RAT = Z * (K_RAT ** p.ALPHA)

ik_gp   = (I[:, burn_in:]     / np.maximum(K[:, burn_in:], 1e-8)).flatten()
ik_rat  = (I_RAT[:, burn_in:] / np.maximum(K_RAT[:, burn_in:], 1e-8)).flatten()
cfk_gp  = (Y[:, burn_in:]     / np.maximum(K[:, burn_in:], 1e-8)).flatten()
cfk_rat = (Y_RAT[:, burn_in:] / np.maximum(K_RAT[:, burn_in:], 1e-8)).flatten()
q_gp    = Q_GP[:, burn_in:].flatten()
q_rat   = Q_RAT[:, burn_in:].flatten()

# ── FHP Regressions: i/k = a + b1*Q + b2*(CF/k) ────────────
X_gp  = sm.add_constant(np.column_stack([q_gp, cfk_gp]))
X_rat = sm.add_constant(np.column_stack([q_rat, cfk_rat]))

reg_gp  = sm.OLS(ik_gp, X_gp).fit()
reg_rat = sm.OLS(ik_rat, X_rat).fit()

print("=" * 60)
print("FHP Regression: i/k = a + b1*Q + b2*(CF/k)")
print("=" * 60)
print(f"\n{'Experience-Learning Firms':}")
print(f"  b1 (Q)    = {reg_gp.params[1]:.4f}  (t = {reg_gp.tvalues[1]:.2f})")
print(f"  b2 (CF/k) = {reg_gp.params[2]:.4f}  (t = {reg_gp.tvalues[2]:.2f})")
print(f"  R²        = {reg_gp.rsquared:.4f}")

print(f"\n{'Rational Firms':}")
print(f"  b1 (Q)    = {reg_rat.params[1]:.4f}  (t = {reg_rat.tvalues[1]:.2f})")
print(f"  b2 (CF/k) = {reg_rat.params[2]:.4f}  (t = {reg_rat.tvalues[2]:.2f})")
print(f"  R²        = {reg_rat.rsquared:.4f}")

print(f"\nExcess sensitivity (b2_GP - b2_Rat) = {reg_gp.params[2] - reg_rat.params[2]:.4f}")

# ── Binned scatter helper ────────────────────────────────────
def binned_means(x, y, n_bins=20):
    edges = np.percentile(x, np.linspace(0, 100, n_bins + 1))
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)
    xm = np.array([np.mean(x[idx == b]) for b in range(n_bins)])
    ym = np.array([np.mean(y[idx == b]) for b in range(n_bins)])
    se = np.array([np.std(y[idx == b]) / np.sqrt((idx == b).sum()) for b in range(n_bins)])
    return xm, ym, se

# ── Partial residual: i/k - b1*Q vs CF/k ────────────────────
resid_gp  = ik_gp  - reg_gp.params[1] * q_gp
resid_rat = ik_rat  - reg_rat.params[1] * q_rat

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 0.6,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.4,
})

c_exp, c_rat = "#2C5F9E", "#B33030"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# ── (a) Naive: i/k on CF/k ──────────────────────────────────
xm_gp, ym_gp, se_gp   = binned_means(cfk_gp, ik_gp)
xm_rat, ym_rat, se_rat = binned_means(cfk_rat, ik_rat)

ax1.scatter(xm_gp, ym_gp, s=25, color=c_exp, zorder=5)
ax1.errorbar(xm_gp, ym_gp, yerr=1.96 * se_gp, fmt="none", color=c_exp, alpha=0.4, lw=0.8)
ax1.scatter(xm_rat, ym_rat, s=25, color=c_rat, marker="s", zorder=5)
ax1.errorbar(xm_rat, ym_rat, yerr=1.96 * se_rat, fmt="none", color=c_rat, alpha=0.4, lw=0.8)

b_naive_gp,  a_naive_gp,  _, _, _ = stats.linregress(cfk_gp, ik_gp)
b_naive_rat, a_naive_rat, _, _, _ = stats.linregress(cfk_rat, ik_rat)
xl = np.linspace(min(xm_gp.min(), xm_rat.min()), max(xm_gp.max(), xm_rat.max()), 100)

ax1.plot(xl, a_naive_gp + b_naive_gp * xl, color=c_exp, lw=1.6,
         label=r"Experience ($\hat{\beta} = " + f"{b_naive_gp:.3f}" + r"$)")
ax1.plot(xl, a_naive_rat + b_naive_rat * xl, color=c_rat, lw=1.6, ls="--",
         label=r"Rational ($\hat{\beta} = " + f"{b_naive_rat:.3f}" + r"$)")
ax1.set_xlabel(r"$CF_t \, / \, k_t$")
ax1.set_ylabel(r"$i_t \, / \, k_t$")
ax1.set_title(r"(a)\; $i/k$ on $CF/k$", fontweight="medium")
ax1.legend(frameon=False, fontsize=8)
ax1.grid(True)

# ── (b) Added-variable: partial out Q ────────────────────────
X_q_gp = sm.add_constant(q_gp)
cf_resid_gp = cfk_gp - sm.OLS(cfk_gp, X_q_gp).fit().predict(X_q_gp)
ik_resid_gp = ik_gp  - sm.OLS(ik_gp,  X_q_gp).fit().predict(X_q_gp)

X_q_rat = sm.add_constant(q_rat)
cf_resid_rat = cfk_rat - sm.OLS(cfk_rat, X_q_rat).fit().predict(X_q_rat)
ik_resid_rat = ik_rat  - sm.OLS(ik_rat,  X_q_rat).fit().predict(X_q_rat)

xm_gp2, ym_gp2, se_gp2   = binned_means(cf_resid_gp, ik_resid_gp)
xm_rat2, ym_rat2, se_rat2 = binned_means(cf_resid_rat, ik_resid_rat)

ax2.scatter(xm_gp2, ym_gp2, s=25, color=c_exp, zorder=5)
ax2.errorbar(xm_gp2, ym_gp2, yerr=1.96 * se_gp2, fmt="none", color=c_exp, alpha=0.4, lw=0.8)
ax2.scatter(xm_rat2, ym_rat2, s=25, color=c_rat, marker="s", zorder=5)
ax2.errorbar(xm_rat2, ym_rat2, yerr=1.96 * se_rat2, fmt="none", color=c_rat, alpha=0.4, lw=0.8)

b2_gp,  a2_gp,  _, _, _ = stats.linregress(cf_resid_gp, ik_resid_gp)
b2_rat, a2_rat, _, _, _ = stats.linregress(cf_resid_rat, ik_resid_rat)
xl2 = np.linspace(min(xm_gp2.min(), xm_rat2.min()), max(xm_gp2.max(), xm_rat2.max()), 100)

ax2.plot(xl2, a2_gp + b2_gp * xl2, color=c_exp, lw=1.6,
         label=r"Experience ($\hat{\beta}_2 = " + f"{b2_gp:.3f}" + r"$)")
ax2.plot(xl2, a2_rat + b2_rat * xl2, color=c_rat, lw=1.6, ls="--",
         label=r"Rational ($\hat{\beta}_2 = " + f"{b2_rat:.3f}" + r"$)")
ax2.set_xlabel(r"$\widetilde{CF/k} \;\perp\; Q$")
ax2.set_ylabel(r"$\widetilde{i/k} \;\perp\; Q$")
ax2.set_title(r"(b)\; Added-variable: $\hat{\beta}_2$ controlling for $Q$", fontweight="medium")
ax2.legend(frameon=False, fontsize=8)
ax2.grid(True)

fig.tight_layout(w_pad=2.5)
fig.savefig("../figures/fhp_regression_experience.pdf", bbox_inches="tight")

plt.show()

burn_in =100
# ── Compute ratios (ergodic) ─────────────────────────────────
Y = Z * (K ** p.ALPHA)
Y_RAT = Z * (K_RAT ** p.ALPHA)

ik_gp  = (I[:, burn_in:]     / np.maximum(K[:, burn_in:], 1e-8)).flatten()
ik_rat = (I_RAT[:, burn_in:] / np.maximum(K_RAT[:, burn_in:], 1e-8)).flatten()
yk_gp  = (Y[:, burn_in:]     / np.maximum(K[:, burn_in:], 1e-8)).flatten()
yk_rat = (Y_RAT[:, burn_in:] / np.maximum(K_RAT[:, burn_in:], 1e-8)).flatten()
dk_gp  = (D[:, burn_in:]     / np.maximum(K[:, burn_in:], 1e-8)).flatten()
dk_rat = (D_RAT[:, burn_in:] / np.maximum(K_RAT[:, burn_in:], 1e-8)).flatten()

c_exp, c_rat = "#2C5F9E", "#B33030"

# ── Figure ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

panels = [
    (ik_gp, ik_rat, r"$i_t / k_t$", r"(a) Investment rate $i/k$"),
    (yk_gp, yk_rat, r"$y_t / k_t$", r"(b) Output--capital ratio $y/k$"),
    (dk_gp, dk_rat, r"$d_t / k_t$", r"(c) Dividend yield $d/k$"),
]

for ax, (gp, rat, xlabel, title) in zip(axes, panels):
    lo = min(np.percentile(gp, 1), np.percentile(rat, 1))
    hi = max(np.percentile(gp, 99), np.percentile(rat, 99))
    x_eval = np.linspace(lo, hi, 400)

    kde_gp  = gaussian_kde(gp)(x_eval)
    kde_rat = gaussian_kde(rat)(x_eval)

    ax.plot(x_eval, kde_gp, color=c_exp, lw=1.6, label="Experience")
    ax.fill_between(x_eval, kde_gp, alpha=0.12, color=c_exp)
    ax.plot(x_eval, kde_rat, color=c_rat, lw=1.6, ls="--", label="Rational")
    ax.fill_between(x_eval, kde_rat, alpha=0.04, color=c_rat)

    ax.axvline(np.mean(gp), color=c_exp, ls=":", lw=0.8,
               label=rf"Experience $\mu = {np.mean(gp):.3f}$")
    ax.axvline(np.mean(rat), color=c_rat, ls=":", lw=0.8,
               label=rf"Rational $\mu = {np.mean(rat):.3f}$")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(title, fontweight="medium")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True)

fig.tight_layout(w_pad=2.5)
fig.savefig("../figures/ergodic_ratios.pdf", bbox_inches="tight")
plt.show()

# ── Print moments ────────────────────────────────────────────
print(f"{'Ratio':<12} {'':>5} {'Mean':>8} {'Std':>8} {'Skew':>8} {'Kurt':>8}")
print("-" * 50)
for name, gp, rat in [("i/k", ik_gp, ik_rat), ("y/k", yk_gp, yk_rat), ("d/k", dk_gp, dk_rat)]:
    print(f"{name:<12} {'GP':>5} {np.mean(gp):>8.4f} {np.std(gp):>8.4f} {skew(gp):>8.3f} {kurtosis(gp):>8.3f}")
    print(f"{'':12} {'Rat':>5} {np.mean(rat):>8.4f} {np.std(rat):>8.4f} {skew(rat):>8.3f} {kurtosis(rat):>8.3f}")


c_exp, c_rat = "#2C5F9E", "#B33030"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

# ── (a) Ergodic-averaged Lorenz ──────────────────────────────
k_avg_gp  = K[:, burn_in:].mean(axis=1)
k_avg_rat = K_RAT[:, burn_in:].mean(axis=1)

l_gp  = lorenz(k_avg_gp)
l_rat = lorenz(k_avg_rat)
x_lor = np.linspace(0, 1, len(l_gp))
g_gp, g_rat = gini(k_avg_gp), gini(k_avg_rat)

ax1.plot(x_lor, x_lor, color="0.45", ls="--", lw=0.8, label="Perfect equality")
ax1.plot(x_lor, l_gp, color=c_exp, lw=1.6,
         label=rf"Experience (Gini $= {g_gp:.3f}$)")
ax1.plot(x_lor, l_rat, color=c_rat, lw=1.6, ls="--",
         label=rf"Rational (Gini $= {g_rat:.3f}$)")
ax1.fill_between(x_lor, l_gp, x_lor, color=c_exp, alpha=0.06)
ax1.set_xlabel("Cumulative share of firms")
ax1.set_ylabel(r"Cumulative share of $\bar{k}_j$")
ax1.set_title(r"(a) Ergodic capital Lorenz curve", fontweight="medium")
ax1.legend(frameon=False, loc="upper left")
ax1.grid(True)

# ── (b) Gini coefficient over time ──────────────────────────
t_axis = np.arange(K.shape[1])
gini_gp_t  = np.array([gini(K[:, t]) for t in range(K.shape[1])])
gini_rat_t = np.array([gini(K_RAT[:, t]) for t in range(K_RAT.shape[1])])

ax2.plot(t_axis, gini_gp_t, color=c_exp, lw=1.6, label="Experience")
ax2.plot(t_axis, gini_rat_t, color=c_rat, lw=1.2, ls="--", label="Rational")

ax2.axvline(burn_in, color="0.5", ls=":", lw=0.6, label="Burn-in")

erg_gini_gp  = np.mean(gini_gp_t[burn_in:])
erg_gini_rat = np.mean(gini_rat_t[burn_in:])
ax2.axhline(erg_gini_gp, color=c_exp, ls=":", lw=0.8,
            label=rf"Ergodic mean $= {erg_gini_gp:.3f}$")
ax2.axhline(erg_gini_rat, color=c_rat, ls=":", lw=0.8,
            label=rf"Ergodic mean $= {erg_gini_rat:.3f}$")

ax2.set_xlabel(r"Period ($t$)")
ax2.set_ylabel("Gini coefficient of $k_t$")
ax2.set_title(r"(b) Capital inequality over time", fontweight="medium")
ax2.legend(frameon=False, fontsize=8)
ax2.grid(True)

fig.tight_layout(w_pad=2.5)
fig.savefig("../figures/lorenz_and_gini_dynamics.pdf", bbox_inches="tight")
plt.show()