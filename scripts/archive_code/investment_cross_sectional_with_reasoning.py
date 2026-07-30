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
    LaplaceanKernel,
    TrueValueFunctionPrior,
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

kernel = Laplace(sigma0=5.0, length_scales=[0.08, 4.0, 1.5])
gp_params = GPBeliefParameters(kernel=kernel, sigma_n=0.5)
true_value_prior = TrueValueFunctionPrior(rational_agent)
H_cal = 0.001

N_FIRMS = 2000
T = 100

print(f"Rational steady state: k*={k_ss:.4f}, b*={b_ss:.4f}")

print(f"Creating {N_FIRMS} firms...")
firm_agents = []
for j in range(N_FIRMS):
    gp_j = GPBelief(env_params=p, gp_params=gp_params, prior_mean_fn=true_value_prior)
    agent_j = ExperienceReasoningAgent(
        env=env, gp=gp_j,
        agent_params=InvestmentAgentParameters(H=H_cal, KAPPA_R=0.0001),
        experience_only=False,
        seed=j,
    )
    agent_j.name = f"Firm {j}"
    firm_agents.append(agent_j)



print(f"Simulating {N_FIRMS} firms x {T} periods...")
print(f"  k* = {k_ss:.4f},  b* = {b_ss:.4f},  H = {H_cal}")
print(f"  GP: Laplacian, sigma0^2={gp_params.kernel.sigma0_sq}, "
      f"sigma_n={gp_params.sigma_n}, "
      f"l_z={kernel.length_scales[0]}, l_k={kernel.length_scales[1]}, "
      f"l_i={kernel.length_scales[2]}")

df = run_simulation(env, firm_agents, T=T, z0=1.0, seed=42)



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
    "text.usetex": True,
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
ax.set_title("Clustered Firm Policies (Experience+Reasoning) in Expectation ($E[k_{t+1}|k_t]$)", fontweight="medium")
ax.legend(frameon=False, loc="upper left")
ax.grid(True)

fig.tight_layout()


from scipy.stats import gaussian_kde

c_exp, c_rat = "#2C5F9E", "k"
burn_in = int(T * 0.3)


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
# ── Compute ratios (ergodic) ─────────────────────────────────
Y = Z * (K ** p.ALPHA)
Y_RAT = Z * (K_RAT ** p.ALPHA)

ik_gp  = (I[:, burn_in:]     / np.maximum(K[:, burn_in:], 1e-8)).flatten()
ik_rat = (I_RAT[:, burn_in:] / np.maximum(K_RAT[:, burn_in:], 1e-8)).flatten()
yk_gp  = (Y[:, burn_in:]     / np.maximum(K[:, burn_in:], 1e-8)).flatten()
yk_rat = (Y_RAT[:, burn_in:] / np.maximum(K_RAT[:, burn_in:], 1e-8)).flatten()
dk_gp  = (D[:, burn_in:]     / np.maximum(K[:, burn_in:], 1e-8)).flatten()
dk_rat = (D_RAT[:, burn_in:] / np.maximum(K_RAT[:, burn_in:], 1e-8)).flatten()

c_exp, c_rat = "#2C5F9E", "k"

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




import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

def simulate_gp_reasoning(n_dims=20, T=50, kappa=0.5, H=0.01,
                           sigma0=2.0, experience_rate=0.05,
                           store_reasoning=False):
    """
    Simplified 1-step eigenvalue dynamics.
    Each period:
      1. Experience shrinks eigenvalues slightly (random, state-dependent)
      2. Reasoning compresses eigenvalues above water level
      3. Policy entropy = H * tr(Sigma_R)
    """
    lambdas = np.full(n_dims, sigma0**2)
    
    trace_R_history = np.zeros(T)
    entropy_floor_history = np.zeros(T)
    n_above_history = np.zeros(T)
    delta_E_history = np.zeros(T)

    for t in range(T):
        # Experience: each eigenvalue shrinks by a random amount
        shrinkage = experience_rate * (1 + 0.5 * np.random.randn(n_dims))
        shrinkage = np.clip(shrinkage, 0.01, 0.2)
        lambdas = lambdas * (1 - shrinkage)
        lambdas = np.maximum(lambdas, 1e-6)

        # Draft policy to get delta_E
        tr_E = np.sum(lambdas)
        entropy_floor_E = H * tr_E
        # Simplified: delta_E proportional to how binding the constraint is
        delta_E = max(entropy_floor_E * 10, 1e-6)
        delta_E_history[t] = delta_E

        # Water level
        w = kappa / (H * delta_E)

        # Reasoning: compress eigenvalues above water level
        lambdas_R = np.minimum(lambdas, w)

        # Record
        trace_R_history[t] = np.sum(lambdas_R)
        entropy_floor_history[t] = H * np.sum(lambdas_R)
        n_above_history[t] = np.sum(lambdas > w)

        # If storing reasoning, permanently shrink eigenvalues
        if store_reasoning:
            lambdas = lambdas_R.copy()

    return {
        'trace_R': trace_R_history,
        'entropy_floor': entropy_floor_history,
        'n_above': n_above_history,
        'delta_E': delta_E_history,
    }

# Run both versions
res_no_store = simulate_gp_reasoning(store_reasoning=False)
res_store = simulate_gp_reasoning(store_reasoning=True)

# Plot
plt.rcParams.update({
    "text.usetex": False, "font.family": "serif",
    "mathtext.fontset": "cm", "font.size": 10,
    "axes.labelsize": 11, "axes.titlesize": 12,
    "legend.fontsize": 8.5, "xtick.direction": "in",
    "ytick.direction": "in", "axes.linewidth": 0.6,
    "grid.alpha": 0.15, "grid.linewidth": 0.4,
})

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

ax1.plot(res_no_store['trace_R'], color="#2C5F9E", lw=1.6, label="Ephemeral (yours)")
ax1.plot(res_store['trace_R'], color="#B33030", lw=1.6, ls="--", label="Persistent (IV)")
ax1.set_xlabel("Period")
ax1.set_ylabel("$\\mathrm{Tr}(\\Sigma_R)$")
ax1.set_title("(a) Post-reasoning total variance", fontweight="medium")
ax1.legend(frameon=False)
ax1.grid(True)

ax2.plot(res_no_store['entropy_floor'], color="#2C5F9E", lw=1.6, label="Ephemeral")
ax2.plot(res_store['entropy_floor'], color="#B33030", lw=1.6, ls="--", label="Persistent")
ax2.set_xlabel("Period")
ax2.set_ylabel("$H \\cdot \\mathrm{Tr}(\\Sigma_R)$")
ax2.set_title("(b) Entropy floor", fontweight="medium")
ax2.legend(frameon=False)
ax2.grid(True)

diff = res_no_store['entropy_floor'] - res_store['entropy_floor']
ax3.plot(diff, color="k", lw=1.6)
ax3.axhline(0, color="0.5", lw=0.6)
ax3.set_xlabel("Period")
ax3.set_ylabel("$\\Delta$ entropy floor")
ax3.set_title("(c) Gap (ephemeral $-$ persistent)", fontweight="medium")
ax3.grid(True)

fig.tight_layout(w_pad=2.5)
plt.show()

# Print summary
print(f"\nTerminal Tr(Sigma_R):")
print(f"  Ephemeral:  {res_no_store['trace_R'][-1]:.4f}")
print(f"  Persistent: {res_store['trace_R'][-1]:.4f}")
print(f"  Gap:        {res_no_store['trace_R'][-1] - res_store['trace_R'][-1]:.4f}")
print(f"\nTerminal entropy floor:")
print(f"  Ephemeral:  {res_no_store['entropy_floor'][-1]:.6f}")
print(f"  Persistent: {res_store['entropy_floor'][-1]:.6f}")