"""
Two-panel phase diagram: (a) k' vs k, (b) Tobin's q vs k
Publication-quality with spline-smoothed policy/value functions.
"""
import sys
sys.path.append("../")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import UnivariateSpline

from src.simulation.environment import QuadraticAdjustmentCosts
from src.simulation.firm import *
from src.simulation.gaussian_process import *

# ── Solve model ──────────────────────────────────────────────
p = InvestmentParameters(
    KAPPA=2.0, SIGMA_EPS=0.05, RHO=0.9, DELTA=0.04,
    R=0.01, N_z=5, BETA=0.97, THETA=0.3,
    K_min=0.0, K_max=30, N_k=200,
)

env = InvestmentEnvironment(p, QuadraticAdjustmentCosts(p.KAPPA), seed=42)
rational_agent = RationalInvestmentAgent(env)
rational_agent.fit()

k_ss = rational_agent.fixed_point()
k_grid = env.k_grid
z_indices = [0, p.N_z // 2, p.N_z - 1]
labels = [r"Low $z$ (recession)", r"Normal $z$", r"High $z$ (boom)"]
colors = ["#B33030", "k", "#1A7A56"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

# ── Panel (a): k' vs k ──────────────────────────────────────
ax1.plot(k_grid, k_grid, color="0.45", ls="--", lw=0.8, label=r"$45°$ line")

k_ss_interp = k_ss

for idx, col, lab in zip(z_indices, colors, labels):
    z_val = env.z_grid[idx]
    kp_raw = np.array([rational_agent.policy(z_val, k)[0] for k in k_grid])
    kp = smooth(k_grid, kp_raw)
    ax1.plot(k_grid, kp, color=col, lw=1.6, label=f"{lab} ($z={z_val:.2f}$)")

    # Only compute crossing for black (normal z) to get SS
    if col == "k":
        diff = kp - k_grid
        crossings = np.where(np.diff(np.sign(diff)))[0]
        if len(crossings) > 0:
            c = crossings[-1]
            k_ss_interp = k_grid[c] + (-diff[c]) / (diff[c+1] - diff[c]) * (k_grid[c+1] - k_grid[c])

ax1.scatter([k_ss_interp], [k_ss_interp], color="k", s=50, zorder=7, marker="o",
            edgecolors="k", linewidths=0.5, label=f"SS ($k^*={k_ss_interp:.2f}$)")

ax1.set_xlabel(r"$k_{t}$")
ax1.set_ylabel(r"$k_{t+1}$")
ax1.set_title(r"(a) Policy function: $k_{t+1}$ vs $k_{t}$", fontweight="medium")
ax1.set_xlim(k_ss_interp - 6, k_ss_interp + 6)
ax1.set_ylim(k_ss_interp - 6, k_ss_interp + 6)
ax1.legend(frameon=False, loc="upper left")
ax1.grid(True)

# ── Panel (b): q vs k ───────────────────────────────────────
ax2.axhline(1.0, color="0.3", ls="--", lw=0.8, label=r"$q = 1$")

# Widen search range for crossings
xlim_lo = k_ss_interp - 6
xlim_hi = k_ss_interp + 6

for idx, col, lab in zip(z_indices, colors, labels):
    q_raw = p.BETA * np.gradient(rational_agent.v[idx, :], k_grid)
    q = smooth(k_grid, q_raw, s_factor=0.01)
    ax2.plot(k_grid, q, color=col, lw=1.6, label=f"{lab}")

    # All crossings with q = 1 over full grid
    diff_q = q - 1.0
    crossings = np.where(np.diff(np.sign(diff_q)))[0]
    for c in crossings:
        k_cross = k_grid[c] + (-diff_q[c]) / (diff_q[c+1] - diff_q[c]) * (k_grid[c+1] - k_grid[c])
        # Plot if in visible range
        if xlim_lo <= k_cross <= xlim_hi:
            ax2.scatter([k_cross], [1.0], color=col, s=35, zorder=6,
                        edgecolors="k", linewidths=0.4)

    # If no crossing found in visible range, check if q is always above or below
    visible_mask = (k_grid >= xlim_lo) & (k_grid <= xlim_hi)
    q_visible = q[visible_mask]
    if len(crossings) == 0 or not any(xlim_lo <= k_grid[c] + (-diff_q[c]) / (diff_q[c+1] - diff_q[c]) * (k_grid[c+1] - k_grid[c]) <= xlim_hi for c in crossings):
        if q_visible.min() > 1.0:
            print(f"Warning: q > 1 throughout for {lab} in visible range. Consider widening xlim.")
        elif q_visible.max() < 1.0:
            print(f"Warning: q < 1 throughout for {lab} in visible range. Consider widening xlim.")

ax2.set_xlabel(r"$k_{t}$")
ax2.set_ylabel(r"$q_{t}$")
ax2.set_xlim(xlim_lo, xlim_hi)
ax2.set_ylim(0.5, 1.6)
ax2.set_title(r"(b) Tobin's $q$ vs $k$", fontweight="medium")
ax2.legend(frameon=False, loc="upper right")
ax2.grid(True)

fig.tight_layout(w_pad=2.5)
fig.savefig("../figures/rational_firm_policy_phase_diagrams.pdf", bbox_inches="tight")
plt.show()

print(f"\n  SS from interpolation: k* = {k_ss_interp:.4f}")