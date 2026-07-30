#!/usr/bin/env python
"""
Kernel rationality diagnostic: believed vs true variation in Q*
for each dimension (z, k, i).
Top row: absolute curves. Bottom row: generalization gap G_j(dx).
"""
import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import UnivariateSpline

from src.simulation.environment import (
    InvestmentParameters,
    InvestmentEnvironment,
    QuadraticAdjustmentCosts,
)
from src.simulation.firm import RationalInvestmentAgent
from src.simulation.gaussian_process import (
    GPBelief, GPBeliefParameters, RBFKernel, TrueValueFunctionPrior,
)

# ── Environment ──────────────────────────────────────────────
p = InvestmentParameters(
    KAPPA=2.0, SIGMA_EPS=0.1, RHO=0.9, DELTA=0.04,
    R=0.01, N_z=5, BETA=0.97, THETA=0.3,
    K_min=0.0, K_max=30
)
env = InvestmentEnvironment(p, QuadraticAdjustmentCosts(p.KAPPA), seed=42)
rational_agent = RationalInvestmentAgent(env).fit()
k_ss = rational_agent.fixed_point()
true_value_prior = TrueValueFunctionPrior(rational_agent)

# ── Steady state ─────────────────────────────────────────────
z_ss = 1.0
i_ss = rational_agent.policy(z_ss, k_ss)[0] - (1 - p.DELTA) * k_ss
q0 = true_value_prior(np.array([[z_ss, k_ss, i_ss]]))[0]

def q_at(z, k, i):
    return true_value_prior(np.array([[z, k, i]]))[0]

def sigma_hat_rbf(dx, ell, sigma0):
    rho = np.exp(-dx**2 / (2 * ell**2))
    return sigma0 * np.sqrt(2 * (1 - rho))

# ── Kernel configurations ────────────────────────────────────
sigma0 = 5.0

def belief_rational_ell(dim_fn, sigma0):
    try:
        d_j = brentq(lambda d: abs(dim_fn(d) - q0) - sigma0, 0.001, 50)
    except:
        d_j = np.nan
    return d_j / np.sqrt(2 * np.log(2)), d_j

ell_z_rat, d_z = belief_rational_ell(lambda d: q_at(z_ss + d, k_ss, i_ss), sigma0)
ell_k_rat, d_k = belief_rational_ell(lambda d: q_at(z_ss, k_ss + d, i_ss), sigma0)

try:
    d_i = brentq(lambda d: q0 - q_at(z_ss, k_ss, i_ss + d) - sigma0, 0.001, 30)
    ell_i_rat = d_i / np.sqrt(2 * np.log(2))
except:
    d_i = np.nan
    ell_i_rat = np.nan

scale_under = 0.5
scale_over = 3.0

kernels = {
    "Rational": {
        "ell_z": ell_z_rat, "ell_k": ell_k_rat, "ell_i": ell_i_rat,
        "color": "0.45", "ls": "--", "lw": 1.4,
    },
    "Under-generalizing": {
        "ell_z": ell_z_rat * scale_under,
        "ell_k": ell_k_rat * scale_under,
        "ell_i": ell_i_rat * scale_under,
        "color": "#B33030", "ls": "--", "lw": 1.4,
    },
    "Over-generalizing": {
        "ell_z": ell_z_rat * scale_over,
        "ell_k": ell_k_rat * scale_over,
        "ell_i": ell_i_rat * scale_over,
        "color": "#2C5F9E", "ls": "--", "lw": 1.4,
    },
}

def find_crossing(dx_range, curve_a, curve_b):
    crossings = []
    diff = curve_b - curve_a
    for idx in range(len(dx_range) - 1):
        if diff[idx] * diff[idx + 1] < 0:
            frac = -diff[idx] / (diff[idx + 1] - diff[idx])
            cx = dx_range[idx] + frac * (dx_range[idx + 1] - dx_range[idx])
            crossings.append(cx)
    return crossings

# ── Style ────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsfonts}",
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 0.6,
    "grid.alpha": 0.15,
    "grid.linewidth": 0.4,
})

# ── Dimension configs ────────────────────────────────────────
dims = [
    {
        "name": "z", "label": "$\\Delta z$",
        "title_top": "(a) Productivity $z$",
        "title_bot": "(d) $G_z(\\Delta z)$",
        "range": np.linspace(0.001, 2.5, 300),
        "true_fn": lambda d: abs(q_at(z_ss + d, k_ss, i_ss) - q0),
        "ell_key": "ell_z",
    },
    {
        "name": "k", "label": "$\\Delta k$",
        "title_top": "(b) Capital $k$",
        "title_bot": "(e) $G_k(\\Delta k)$",
        "range": np.linspace(0.01, 10, 300),
        "true_fn": lambda d: abs(q_at(z_ss, k_ss + d, i_ss) - q0),
        "ell_key": "ell_k",
    },
    {
        "name": "i", "label": "$\\Delta i$",
        "title_top": "(c) Investment $i$",
        "title_bot": "(f) $G_i(\\Delta i)$",
        "range": np.linspace(0.01, 10, 300),
        "true_fn": lambda d: abs(q_at(z_ss, k_ss, i_ss + d) - q0),
        "ell_key": "ell_i",
    },
]

# ── Precompute all gaps to find shared y-limits ──────────────
all_gaps = []
for dim in dims:
    dx = dim["range"]
    true_curve = np.array([dim["true_fn"](d) for d in dx])
    for name, cfg in kernels.items():
        ell = cfg[dim["ell_key"]]
        if np.isnan(ell):
            continue
        kernel_curve = np.array([sigma_hat_rbf(d, ell, sigma0) for d in dx])
        all_gaps.extend(kernel_curve - true_curve)

gap_max = max(abs(np.nanmin(all_gaps)), abs(np.nanmax(all_gaps)))
gap_lim = np.ceil(gap_max)  # symmetric integer limits
gap_lim = 5.5

# ── Figure ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))

for col_idx, dim in enumerate(dims):
    ax_top = axes[0, col_idx]
    ax_bot = axes[1, col_idx]
    dx = dim["range"]
    true_curve = np.array([dim["true_fn"](d) for d in dx])

    # ── Top row: absolute curves ─────────────────────────────
    ax_top.plot(dx, true_curve, color="k", lw=2, label="True $|\\Delta Q^*|$")
    ax_top.axhline(sigma0, color="0.7", ls=":", lw=0.8, label=f"$\\sigma_0 = {sigma0}$")

    for name, cfg in kernels.items():
        ell = cfg[dim["ell_key"]]
        if np.isnan(ell):
            continue
        kernel_curve = np.array([sigma_hat_rbf(d, ell, sigma0) for d in dx])
        ax_top.plot(dx, kernel_curve, color=cfg["color"], ls=cfg["ls"],
                    lw=cfg["lw"], label=f"{name} ($\\ell={ell:.2f}$)")

        crossings = find_crossing(dx, true_curve, kernel_curve)
        for cx in crossings:
            cy = np.interp(cx, dx, true_curve)
            ax_top.scatter([cx], [cy], color=cfg["color"], s=40, zorder=6,
                           edgecolors="k", linewidths=0.4)

    ax_top.set_xlabel(dim["label"])
    # Vertical reference lines for rational kernel
    ell_br = kernels["Rational"][dim["ell_key"]]
    if not np.isnan(ell_br):
        d_br = ell_br * np.sqrt(2 * np.log(2))
        ax_top.axvline(ell_br, color="0.45", ls="--", lw=0.6, alpha=0.5)
        ax_top.axvline(d_br, color="0.45", ls="-", lw=0.6, alpha=0.5)
        ax_top.text(ell_br, ax_top.get_ylim()[1] * 0.85, "$\\ell_j$",
                    color="0.45", fontsize=8, ha="right")
        ax_top.text(d_br, ax_top.get_ylim()[1] * 0.85, "$d_j$",
                    color="0.45", fontsize=8, ha="right")
    ax_top.set_ylabel("$|\\Delta Q^*|$ or $\\hat{\\sigma}(\\Delta x)$")
    ax_top.set_title(dim["title_top"], fontweight="medium")
    ax_top.legend(frameon=False, fontsize=7)
    if col_idx == 0:
        ax_top.legend(frameon=False, fontsize=7, loc="upper left")
    ax_top.grid(True)

    # ── Bottom row: G_j(dx) ──────────────────────────────────
    ax_bot.axhline(0, color="k", lw=0.8)

    for name, cfg in kernels.items():
        ell = cfg[dim["ell_key"]]
        if np.isnan(ell):
            continue
        kernel_curve = np.array([sigma_hat_rbf(d, ell, sigma0) for d in dx])
        gap = kernel_curve - true_curve

        ax_bot.plot(dx, gap, color=cfg["color"], ls="-", lw=1.6, label=name)
        ax_bot.fill_between(dx, 0, gap, where=gap > 0, color=cfg["color"],
                            alpha=0.06, interpolate=True)
        ax_bot.fill_between(dx, 0, gap, where=gap < 0, color=cfg["color"],
                            alpha=0.06, interpolate=True)

        crossings = find_crossing(dx, np.zeros_like(dx), gap)
        for cx in crossings:
            ax_bot.scatter([cx], [0], color=cfg["color"], s=35, zorder=6,
                           edgecolors="k", linewidths=0.4)

    # Shared symmetric y-limits
    ax_bot.set_ylim(-gap_lim, gap_lim)

    # Region labels near zero line
    ax_bot.text(dx[len(dx) // 2], gap_lim * 0.85,
                "Under-generalizing ($G > 0$)",
                color="0.35", fontsize=7.5, style="italic",
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax_bot.text(dx[len(dx) // 2], -gap_lim * 0.85,
                "Over-generalizing ($G < 0$)",
                color="0.35", fontsize=7.5, style="italic",
                ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    ax_bot.set_xlabel(dim["label"])
    ax_bot.set_ylabel("$G_j(\\Delta x)$")
    ax_bot.set_title(dim["title_bot"], fontweight="medium")
    ax_bot.legend(frameon=False, fontsize=7)
    ax_bot.grid(True)

fig.tight_layout(h_pad=2.5, w_pad=2.0)
fig.savefig("../figures/kernel_rationality_diagnostic.pdf", bbox_inches="tight")
plt.show()

# ── Summary ──────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  Kernel Rationality Calibration Summary")
print(f"{'='*65}")
print(f"  sigma_0 = {sigma0}")
print(f"  Q*(x*) = {q0:.4f}")
print(f"  x* = (z={z_ss}, k={k_ss:.2f}, i={i_ss:.2f})")
print(f"  Gap y-limits: [-{gap_lim}, +{gap_lim}]")
print(f"")
print(f"  {'Dim':<6} {'d_j':>8} {'ell_rat':>10} {'ell_under':>10} {'ell_over':>10}")
print(f"  {'-'*46}")
print(f"  {'z':<6} {d_z:>8.4f} {ell_z_rat:>10.4f} {ell_z_rat*scale_under:>10.4f} {ell_z_rat*scale_over:>10.4f}")
print(f"  {'k':<6} {d_k:>8.4f} {ell_k_rat:>10.4f} {ell_k_rat*scale_under:>10.4f} {ell_k_rat*scale_over:>10.4f}")
print(f"  {'i':<6} {d_i:>8.4f} {ell_i_rat:>10.4f} {ell_i_rat*scale_under:>10.4f} {ell_i_rat*scale_over:>10.4f}")
print(f"{'='*65}")