"""
Ablation summary: reads all parquet files, computes ergodic moments,
outputs a single CSV table for analysis.
"""
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis

from src.simulation.environment import InvestmentParameters

# ── Environment (for p.ALPHA) ────────────────────────────────
p = InvestmentParameters(
    KAPPA=2.0, SIGMA_EPS=0.05, RHO=0.9, DELTA=0.04,
    R=0.01, N_z=5, BETA=0.97, THETA=0.3,
    K_min=0.0, K_max=30
)
s
out_dir = Path("../data/ablations/experience_only")

def parse_tag(stem):
    """Extract parameters from filename like H=0.001_sn=0.5_s0=5.0_ls=baseline"""
    parts = stem.split("_")
    params = {}
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                params[k] = float(v)
            except ValueError:
                params[k] = v
    return params

def summarize_run(df, p, burn_in=None):
    T = df['t'].max() + 1
    if burn_in is None:
        burn_in = int(T * 0.3)

    erg = df[df['t'] >= burn_in]

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

    s = {
        # i/k
        "ik_mean": ik.mean(),
        "ik_std": ik.std(),
        "ik_skew": float(skew(ik)),
        "ik_kurt": float(kurtosis(ik)),
        "ik_rat_mean": ik_rat.mean(),
        "ik_rat_std": ik_rat.std(),

        # y/k
        "yk_mean": yk.mean(),
        "yk_std": yk.std(),
        "yk_rat_mean": yk_rat.mean(),
        "yk_rat_std": yk_rat.std(),

        # d/k
        "dk_mean": dk.mean(),
        "dk_std": dk.std(),
        "dk_rat_mean": dk_rat.mean(),
        "dk_rat_std": dk_rat.std(),
        "dk_loss_pct": 100 * (1 - dk.mean() / dk_rat.mean()),

        # Eigenvalues
        "eig_tr_mean": erg["eig_tr"].mean() if "eig_tr" in erg else np.nan,
        "eig_max_mean": erg["eig_max"].mean() if "eig_max" in erg else np.nan,
        "eig_n_active_mean": erg["eig_n_active"].mean() if "eig_n_active" in erg else np.nan,

        # Learning dynamics
        "delta_E_mean": erg["delta_E"].mean() if "delta_E" in erg else np.nan,
        "delta_E_median": erg["delta_E"].median() if "delta_E" in erg else np.nan,
        "alpha_gain_mean": erg["alpha_gain"].mean() if "alpha_gain" in erg else np.nan,

        # Resets
        "n_resets": int(erg["reset"].sum()) if "reset" in erg else 0,

        # Meta
        "T": int(T),
        "burn_in": burn_in,
        "n_firms": df["agent_id"].nunique(),
        "n_obs_ergodic": len(erg),
    }
    return s

# ── Build table ──────────────────────────────────────────────
files = sorted(out_dir.glob("H=*.parquet"))
print(f"Found {len(files)} ablation files.\n")

rows = []
for fpath in files:
    tag = fpath.stem
    params = parse_tag(tag)

    print(f"Processing {tag} ...", end=" ")
    df = pd.read_parquet(fpath)
    s = summarize_run(df, p, burn_in=100)
    rows.append({**params, **s})
    del df
    print("done.")

summary = pd.DataFrame(rows)

summary

def plot_ablation_results(df):
    """Full ablation diagnostic from summary CSV dataframe."""
    
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
        "axes.linewidth": 0.6,
        "grid.alpha": 0.15,
        "grid.linewidth": 0.4,
    })

    # ── 1. Heatmap: H vs sigma_n at each kernel ─────────────
    metrics = [
        ("ik_std", "Std$(i/k)$"),
        ("dk_loss_pct", "Welfare loss (%)"),
        ("alpha_gain_mean", "Mean $\\alpha_{gain}$"),
    ]

    for ls_name in df['ls'].unique():
        for s0 in df['s0'].unique():
            sub = df[(df['ls'] == ls_name) & (df['s0'] == s0)]
            if len(sub) < 4:
                continue

            fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
            for ax, (col, label) in zip(axes, metrics):
                if col not in sub.columns:
                    continue
                pivot = sub.pivot(index="sn", columns="H", values=col)
                im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", origin="lower")
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels([f"{v:.4f}" for v in pivot.columns], rotation=45, fontsize=8)
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels([f"{v}" for v in pivot.index], fontsize=8)
                ax.set_xlabel("$H$")
                ax.set_ylabel("$\\sigma_n$")
                ax.set_title(label)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(f"$\\sigma_0={s0}$, kernel={ls_name}", fontweight="medium")
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            plt.show()

    # ── 2. Marginal sensitivity ──────────────────────────────
    # Find baseline (median of each param)
    H_base = df['H'].median()
    sn_base = df['sn'].median()
    s0_base = df['s0'].median()
    ls_base = "baseline" if "baseline" in df['ls'].values else df['ls'].mode()[0]

    outcomes = [
        ("ik_std", "Std$(i/k)$"),
        ("dk_loss_pct", "Welfare loss (%)"),
        ("eig_tr_mean", "Tr$(\\Sigma)$"),
    ]
    sweeps = [
        ("H", "$H$", H_base),
        ("sn", "$\\sigma_n$", sn_base),
        ("s0", "$\\sigma_0$", s0_base),
    ]

    fig, axes = plt.subplots(len(outcomes), len(sweeps),
                             figsize=(4 * len(sweeps), 3.5 * len(outcomes)))

    for row, (out_col, out_label) in enumerate(outcomes):
        for col_idx, (param, param_label, bl_val) in enumerate(sweeps):
            ax = axes[row, col_idx]
            mask = pd.Series(True, index=df.index)
            for k, v, _ in sweeps:
                if k == param:
                    continue
                if isinstance(v, str):
                    mask &= df[k] == v
                else:
                    mask &= np.isclose(df[k], v)
            mask &= df['ls'] == ls_base
            sub = df[mask].sort_values(param)

            if out_col in sub.columns and len(sub) > 0:
                ax.plot(sub[param], sub[out_col], "o-", color="#2C5F9E", lw=1.6, ms=5)
                bl_row = sub[np.isclose(sub[param], bl_val)]
                if len(bl_row) > 0:
                    ax.scatter([bl_val], [bl_row[out_col].values[0]],
                              color="#B33030", s=60, zorder=5)
            ax.set_xlabel(param_label)
            if col_idx == 0:
                ax.set_ylabel(out_label)
            ax.grid(True)
            if row == 0:
                ax.set_title(f"Varying {param_label}", fontweight="medium")

    fig.tight_layout(h_pad=2.5, w_pad=2.0)
    plt.show()

    # ── 3. Kernel comparison bars ────────────────────────────
    sub = df[(np.isclose(df['H'], H_base)) & (np.isclose(df['sn'], sn_base))]
    if len(sub) > 3:
        ls_order = [l for l in ["baseline", "medium", "wide"] if l in sub['ls'].values]
        s0_vals = sorted(sub['s0'].unique())
        colors = {"baseline": "#2C5F9E", "medium": "#1A7A56", "wide": "#B33030"}

        bar_metrics = [
            ("ik_std", "Std$(i/k)$"),
            ("dk_loss_pct", "Welfare loss (%)"),
        ]

        fig, axes = plt.subplots(1, len(bar_metrics), figsize=(5 * len(bar_metrics), 4.5))
        if len(bar_metrics) == 1:
            axes = [axes]

        for ax, (col, label) in zip(axes, bar_metrics):
            x = np.arange(len(s0_vals))
            w = 0.8 / len(ls_order)
            for i, ls_name in enumerate(ls_order):
                vals = []
                for s0 in s0_vals:
                    row = sub[(sub['ls'] == ls_name) & (np.isclose(sub['s0'], s0))]
                    vals.append(row[col].values[0] if len(row) > 0 else np.nan)
                ax.bar(x + i * w, vals, width=w,
                       color=colors.get(ls_name, "gray"), alpha=0.85, label=ls_name)
            ax.set_xticks(x + w)
            ax.set_xticklabels([f"$\\sigma_0={v}$" for v in s0_vals])
            ax.set_ylabel(label)
            ax.legend(frameon=False, fontsize=8)
            ax.grid(True)

        fig.suptitle(f"Kernel comparison ($H={H_base}$, $\\sigma_n={sn_base}$)",
                     fontweight="medium")
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()

    # ── 4. Top/worst configs ─────────────────────────────────
    cols = ["H", "sn", "s0", "ls", "dk_loss_pct", "ik_std"]
    if "alpha_gain_mean" in df.columns:
        cols.append("alpha_gain_mean")

    print("=" * 70)
    print("  BEST 5 (lowest welfare loss)")
    print(df.nsmallest(5, "dk_loss_pct")[cols].to_string(index=False))
    print()
    print("  WORST 5 (highest welfare loss)")
    print(df.nlargest(5, "dk_loss_pct")[cols].to_string(index=False))
    print("=" * 70)


plot_ablation_results(summary)


import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

# ── Baseline calibration ─────────────────────────────────────
BASELINE = {"H": 0.001, "sn": 0.3, "s0": 7.5, "ls": "baseline"}

OUTCOMES = [
    ("ik_mean", "Mean $i/k$"),
    ("ik_std", "Std $i/k$"),
    ("yk_mean", "Mean $y/k$"),
    ("yk_std", "Std $y/k$"),
    ("dk_mean", "Mean $d/k$"),
    ("dk_std", "Std $d/k$"),
]

def match_baseline(df, exclude_param):
    """Select rows matching baseline on all params except exclude_param."""
    mask = pd.Series(True, index=df.index)
    for k, v in BASELINE.items():
        if k == exclude_param:
            continue
        if isinstance(v, str):
            mask &= df[k] == v
        else:
            mask &= np.isclose(df[k], v)
    return df[mask].copy()


# ── 1. Parameter tables ──────────────────────────────────────
def build_parameter_table(df, param, param_label):
    sub = match_baseline(df, param).sort_values(param)
    if len(sub) == 0:
        print(f"No data for {param} sweep at baseline")
        return None

    # Rational benchmarks (same across runs, take first)
    rat_ik = sub['ik_rat_mean'].iloc[0]
    rat_yk = sub['yk_rat_mean'].iloc[0]
    rat_dk = sub['dk_rat_mean'].iloc[0]

    rows = []
    for _, r in sub.iterrows():
        rows.append({
            param_label: r[param],
            "Mean $i/k$": f"{r['ik_mean']:.4f}",
            "Std $i/k$": f"{r['ik_std']:.4f}",
            "$\\Delta$ $i/k$": f"{r['ik_mean'] - rat_ik:+.4f}",
            "Mean $y/k$": f"{r['yk_mean']:.4f}",
            "Std $y/k$": f"{r['yk_std']:.4f}",
            "$\\Delta$ $y/k$": f"{r['yk_mean'] - rat_yk:+.4f}",
            "Mean $d/k$": f"{r['dk_mean']:.4f}",
            "Std $d/k$": f"{r['dk_std']:.4f}",
            "$\\Delta$ $d/k$": f"{r['dk_mean'] - rat_dk:+.4f}",
            "Welfare loss (\\%)": f"{r['dk_loss_pct']:.2f}",
        })

    table = pd.DataFrame(rows)
    return table


def print_all_tables(df):
    params = [
        ("H", "$H$"),
        ("sn", "$\\sigma_n$"),
        ("s0", "$\\sigma_0$"),
        ("ls", "Kernel"),
    ]

    for param, label in params:
        print(f"\n{'='*80}")
        print(f"  Table: Varying {label}")
        print(f"  Baseline: {', '.join(f'{k}={v}' for k, v in BASELINE.items() if k != param)}")
        print(f"{'='*80}")
        table = build_parameter_table(df, param, label)
        if table is not None:
            print(table.to_string(index=False))


def sensitivity_regressions(df):
    reg_df = df.copy()

    # Encode kernel as numeric
    ls_map = {name: i for i, name in enumerate(sorted(reg_df['ls'].unique()))}
    reg_df['ls_num'] = reg_df['ls'].map(ls_map).astype(float)

    # Standardize all continuous params
    continuous = ['H', 'sn', 's0', 'ls_num']
    for c in continuous:
        mu, sd = reg_df[c].mean(), reg_df[c].std()
        reg_df[f'{c}_z'] = ((reg_df[c] - mu) / sd).astype(float) if sd > 0 else 0.0

    x_cols = [f'{c}_z' for c in continuous]

    outcomes = [
        ("ik_mean", "Mean i/k"),
        ("ik_std", "Std i/k"),
        ("dk_loss_pct", "Welfare loss (%)"),
        ("eig_tr_mean", "Tr(Sigma)"),
        ("alpha_gain_mean", "Mean alpha_gain"),
    ]

    print(f"\n{'='*80}")
    print("  Sensitivity Regressions (standardized coefficients)")
    print(f"{'='*80}\n")

    coef_table = {}
    for out_col, out_label in outcomes:
        if out_col not in reg_df.columns:
            continue
        valid = reg_df[[out_col] + x_cols].dropna()
        if len(valid) < len(x_cols) + 2:
            print(f"  Not enough data for {out_label}")
            continue
        y = valid[out_col].astype(float)
        X = sm.add_constant(valid[x_cols].astype(float))
        try:
            res = sm.OLS(y, X).fit()
            coef_table[out_label] = {}
            for param, z_col in [("H", "H_z"), ("sigma_n", "sn_z"), 
                                  ("sigma_0", "s0_z"), ("kernel", "ls_num_z")]:
                b = res.params.get(z_col, np.nan)
                p = res.pvalues.get(z_col, np.nan)
                coef_table[out_label][param] = f"{b:+.4f} (p={p:.3f})"
            coef_table[out_label]["R2"] = f"{res.rsquared:.3f}"
            coef_table[out_label]["N"] = f"{int(res.nobs)}"
        except Exception as e:
            print(f"  Failed for {out_label}: {e}")

    coef_df = pd.DataFrame(coef_table)
    print(coef_df.to_string())

    # Variance decomposition
    print(f"\n{'='*80}")
    print("  Partial R-squared (variance explained by each parameter)")
    print(f"{'='*80}\n")

    decomp_rows = []
    for out_col, out_label in outcomes:
        if out_col not in reg_df.columns:
            continue
        valid = reg_df[[out_col] + x_cols].dropna()
        y = valid[out_col].astype(float)
        X_full = sm.add_constant(valid[x_cols].astype(float))
        r2_full = sm.OLS(y, X_full).fit().rsquared

        row = {"Outcome": out_label}
        for param, z_col in [("H", "H_z"), ("sigma_n", "sn_z"), 
                              ("sigma_0", "s0_z"), ("kernel", "ls_num_z")]:
            x_red = [c for c in x_cols if c != z_col]
            X_red = sm.add_constant(valid[x_red].astype(float))
            r2_red = sm.OLS(y, X_red).fit().rsquared
            row[param] = f"{r2_full - r2_red:.4f}"
        row["Full R2"] = f"{r2_full:.4f}"
        decomp_rows.append(row)

    print(pd.DataFrame(decomp_rows).set_index("Outcome").to_string())


# ── Run ──────────────────────────────────────────────────────
print_all_tables(summary)
sensitivity_regressions(summary)



# Isolate the behavioral firm data (long format for k, i, d)
# Ensure data is sorted by agent then time for proper reshaping

firm_df = pd.read_parquet("../data/ablations/experience_only/H=0.001_sn=0.3_s0=7.5_ls=baseline.parquet")
firm_df = firm_df.sort_values(["agent_id", "t"])

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

N_FIRMS = 1000

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
ax.set_title("Clustered Firm Policies (Experience-Only) in Expectation ($E[k_{t+1}|k_t]$)", fontweight="medium")
ax.legend(frameon=False, loc="upper left")
ax.grid(True)

fig.tight_layout()
fig.savefig("../figures/clustered_policies_experienced_learning.pdf", bbox_inches="tight")
plt.show()

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
burn_in = int(T * 0.3)

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
burn_in = int(T * 0.3)

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

c_exp, c_rat = "#2C5F9E", "k"

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
fig.savefig("../figures/ergodic_ratios.pdf", bbox_inches="tight")
plt.show()

# ── Print moments ────────────────────────────────────────────
print(f"{'Ratio':<12} {'':>5} {'Mean':>8} {'Std':>8} {'Skew':>8} {'Kurt':>8}")
print("-" * 50)
for name, gp, rat in [("i/k", ik_gp, ik_rat), ("y/k", yk_gp, yk_rat), ("d/k", dk_gp, dk_rat)]:
    print(f"{name:<12} {'GP':>5} {np.mean(gp):>8.4f} {np.std(gp):>8.4f} {skew(gp):>8.3f} {kurtosis(gp):>8.3f}")
    print(f"{'':12} {'Rat':>5} {np.mean(rat):>8.4f} {np.std(rat):>8.4f} {skew(rat):>8.3f} {kurtosis(rat):>8.3f}")


c_exp, c_rat = "#2C5F9E", "k"

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




