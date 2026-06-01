"""
Experience-only ablation sweep.
Varies H, sigma_n, sigma0, and length scales.
Each run saved as a separate parquet to avoid memory blowup.
Parallelized across firms.
"""
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from tqdm import tqdm

from src.simulation.environment import (
    InvestmentParameters,
    InvestmentEnvironment,
    QuadraticAdjustmentCosts,
    run_simulation_parallel,
)

# ── Fixed environment ────────────────────────────────────────
p = InvestmentParameters(
    KAPPA=2.0, SIGMA_EPS=0.05, RHO=0.9, DELTA=0.04,
    R=0.01, N_z=5, BETA=0.97, THETA=0.3,
    K_min=0.0, K_max=30
)
env = InvestmentEnvironment(p, QuadraticAdjustmentCosts(p.KAPPA), seed=42)

N_FIRMS = 1000
T = 250
N_WORKERS = 8
FIRM_EXIT_RATE = 0.02

# ── Output directory ─────────────────────────────────────────
out_dir = Path("../data/ablations/experience_only_sweep")
out_dir.mkdir(parents=True, exist_ok=True)

# ── Ablation grid ────────────────────────────────────────────
H_grid = [0.00025, 0.0005, 0.001, 0.002]
sigma_n_grid = [0.1, 0.3, 0.5, 1.0]
sigma0_grid = [5.0, 7.5, 15.0]
length_scales_configs = {
    "baseline": [0.08, 4.0, 1.5],
    "medium":   [0.12, 6.0, 2.25],
    "wide":     [0.16, 8.0, 3.0],
}

# ── Sweep ────────────────────────────────────────────────────
if __name__ == "__main__":
    configs = list(product(
        H_grid, sigma_n_grid, length_scales_configs.items(), sigma0_grid
    ))

    for H, sigma_n, (ls_name, ls), sigma0 in tqdm(configs):
        tag = f"H={H}_sn={sigma_n}_s0={sigma0}_ls={ls_name}"
        fpath = out_dir / f"{tag}.parquet"

        if fpath.exists():
            print(f"Skipping {tag}")
            continue

        print(f"Running {tag} ...")
        df = run_simulation_parallel(
            env,
            N_FIRMS=N_FIRMS,
            T=T,
            z0=1.0,
            firm_exit_rate=FIRM_EXIT_RATE,
            seed=0,
            n_workers=N_WORKERS,
            gp_kernel_args=(sigma0, ls),
            gp_sigma_n=sigma_n,
            agent_H=H,
            experience_only=True,
        )
        df.to_parquet(fpath)
        print(f"  Saved → {fpath}")
        del df

    print("\nDone.")