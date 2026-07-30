#!/usr/bin/env python
"""
H sweep: dispersion statistics for experience-only learning.
"""
import sys
sys.path.append("..")
import numpy as np
import pandas as pd

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
    RBFKernel,
    TrueValueFunctionPrior,
)

# ── Environment ──────────────────────────────────────────────
p = InvestmentParameters(
    KAPPA=2.0, SIGMA_EPS=0.05, RHO=0.9, DELTA=0.04,
    R=0.01, N_z=5, BETA=0.97, THETA=0.3,
    K_min=0.0, K_max=30
)
env = InvestmentEnvironment(p, QuadraticAdjustmentCosts(p.KAPPA), seed=42)
rational_agent = RationalInvestmentAgent(env).fit()
k_ss = rational_agent.fixed_point()
b_ss = env.optimal_b_next(k_ss)
true_value_prior = TrueValueFunctionPrior(rational_agent)

N_FIRMS = 500
T = 300
BURN_IN = 100

kernel_args = dict(sigma0=5.0, length_scales=[2.29, 4.50, 5.88])
sigma_n = 0.3

H_grid = [0.00005, 0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0025]

# ── Run and collect ──────────────────────────────────────────
rows = []

for H in H_grid:
    print(f"\nRunning H = {H} ...")

    agents = []
    for j in range(N_FIRMS):
        kernel = RBFKernel(**kernel_args)
        gp_params = GPBeliefParameters(kernel=kernel, sigma_n=sigma_n)
        gp_j = GPBelief(env_params=p, gp_params=gp_params, prior_mean_fn=true_value_prior)
        agents.append(ExperienceReasoningAgent(
            env=env, gp=gp_j,
            agent_params=InvestmentAgentParameters(H=H),
            experience_only=True,
            seed=j,
        ))

    df = run_simulation(env, agents, T=T, z0=1.0, firm_exit_rate=0.02, seed=42)

    # Ergodic sample
    erg = df[df['t'] >= BURN_IN]
    k = erg['k'].values
    k_rat = erg['k_rat'].values
    i_v = erg['i'].values
    i_rat = erg['i_rat'].values
    z = erg['z'].values
    d = erg['d'].values
    d_rat = erg['d_rat'].values

    y = z * (k ** p.ALPHA)
    y_rat = z * (k_rat ** p.ALPHA)

    ik = i_v / np.maximum(k, 1e-8)
    ik_rat = i_rat / np.maximum(k_rat, 1e-8)
    yk = y / np.maximum(k, 1e-8)
    yk_rat = y_rat / np.maximum(k_rat, 1e-8)
    dk = d / np.maximum(k, 1e-8)
    dk_rat = d_rat / np.maximum(k_rat, 1e-8)

    # Policy dispersion: cross-sectional std at each t, then average
    ik_panel = erg.pivot(index="agent_id", columns="t", values="i")
    k_panel = erg.pivot(index="agent_id", columns="t", values="k")
    ik_ratio = (ik_panel / k_panel.clip(lower=1e-8))
    cross_std_ik = ik_ratio.std(axis=0).mean()

    rows.append({
        'H': H,
        # i/k
        'ik_mean': ik.mean(),
        'ik_std': ik.std(),
        'ik_cross_std': cross_std_ik,
        'ik_iqr': np.percentile(ik, 75) - np.percentile(ik, 25),
        'ik_rat_mean': ik_rat.mean(),
        'ik_rat_std': ik_rat.std(),
        'ik_gap': abs(ik.mean() - ik_rat.mean()),
        # y/k
        'yk_mean': yk.mean(),
        'yk_std': yk.std(),
        'yk_rat_mean': yk_rat.mean(),
        'yk_rat_std': yk_rat.std(),
        'yk_gap': abs(yk.mean() - yk_rat.mean()),
        # d/k
        'dk_mean': dk.mean(),
        'dk_std': dk.std(),
        'dk_rat_mean': dk_rat.mean(),
        'dk_rat_std': dk_rat.std(),
        'dk_gap': abs(dk.mean() - dk_rat.mean()),
        'dk_loss_pct': 100 * (1 - dk.mean() / dk_rat.mean()),
        # Eigenvalues
        'eig_tr': erg['eig_tr'].mean() if 'eig_tr' in erg else np.nan,
        'alpha_gain': erg['alpha_gain'].mean() if 'alpha_gain' in erg else np.nan,
    })

    del df, agents
    print(f"  Done. ik_std={rows[-1]['ik_std']:.4f}, dk_loss={rows[-1]['dk_loss_pct']:.2f}%")

results = pd.DataFrame(rows)

# ── Print table ──────────────────────────────────────────────
print(f"\n{'='*90}")
print(f"  Dispersion Statistics by H (experience-only, N={N_FIRMS}, T={T}, burn-in={BURN_IN})")
print(f"  Rational benchmarks: ik={rows[0]['ik_rat_mean']:.4f}, "
      f"yk={rows[0]['yk_rat_mean']:.4f}, dk={rows[0]['dk_rat_mean']:.4f}")
print(f"{'='*90}")

print(f"\n  {'H':<10} {'ik_mean':>8} {'ik_std':>8} {'ik_xstd':>8} {'ik_iqr':>8} "
      f"{'yk_mean':>8} {'yk_std':>8} {'dk_mean':>8} {'dk_std':>8} {'dk_loss%':>9}")
print(f"  {'-'*85}")

for _, r in results.iterrows():
    print(f"  {r['H']:<10.4f} {r['ik_mean']:>8.4f} {r['ik_std']:>8.4f} "
          f"{r['ik_cross_std']:>8.4f} {r['ik_iqr']:>8.4f} "
          f"{r['yk_mean']:>8.4f} {r['yk_std']:>8.4f} "
          f"{r['dk_mean']:>8.4f} {r['dk_std']:>8.4f} {r['dk_loss_pct']:>8.2f}%")

# Rational row
print(f"  {'Rational':<10} {rows[0]['ik_rat_mean']:>8.4f} {rows[0]['ik_rat_std']:>8.4f} "
      f"{'--':>8} {'--':>8} "
      f"{rows[0]['yk_rat_mean']:>8.4f} {rows[0]['yk_rat_std']:>8.4f} "
      f"{rows[0]['dk_rat_mean']:>8.4f} {rows[0]['dk_rat_std']:>8.4f} {'0.00':>8}%")
print(f"{'='*90}")

# ── Interpretation ───────────────────────────────────────────
print(f"\n  ik_std:    total std of i/k (time + cross-section)")
print(f"  ik_xstd:   avg cross-sectional std of i/k at each t (pure policy dispersion)")
print(f"  ik_iqr:    interquartile range of i/k")
print(f"  dk_loss%:  welfare loss relative to rational (in d/k)")