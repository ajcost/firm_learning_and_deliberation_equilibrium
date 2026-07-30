"""Visualization: Economist-style theme + entrepreneur model plot functions."""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

ECON = {
    "red": "#E3120B", "blue": "#006BA2", "cyan": "#3EBCD2", "green": "#379A8B",
    "yellow": "#EBB434", "olive": "#B4BA39", "purple": "#9A607F", "grey": "#758D99",
    "dark": "#333333", "bg": "#FFFFFF", "grid": "#B7C6CF",
}
CYCLE = [ECON["red"], ECON["blue"], ECON["green"], ECON["yellow"],
         ECON["purple"], ECON["cyan"], ECON["olive"], ECON["grey"]]
TYPE_COLORS = [ECON["red"], ECON["blue"], ECON["green"]]      # low / mid / high
TYPE_LABELS = ["low", "mid", "high"]


def apply_style(*, latex: bool = True, base_size: int = 11):
    r"""Economist-style rcParams: serif/LaTeX fonts, horizontal grid, no top/right
    spines, muted palette, embedded-font PDF output. Call once per session."""
    rc = {
        "font.family": "serif", "font.size": base_size,
        "axes.titlesize": base_size + 1, "axes.labelsize": base_size,
        "legend.fontsize": base_size - 2,
        "xtick.labelsize": base_size - 1, "ytick.labelsize": base_size - 1,
        "axes.facecolor": ECON["bg"], "figure.facecolor": ECON["bg"],
        "axes.edgecolor": ECON["dark"], "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": ECON["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.prop_cycle": cycler(color=CYCLE),
        "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.dpi": 300,
    }
    if latex:
        rc.update({"text.usetex": True,
                   "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}"})
    else:
        rc.update({"text.usetex": False, "mathtext.fontset": "cm"})
    mpl.rcParams.update(rc)


def save_fig(fig, path: str):
    r"""Write vector PDF (paper) + PNG preview."""
    base = path.rsplit(".", 1)[0]
    fig.savefig(f"{base}.pdf")
    fig.savefig(f"{base}.png", dpi=150)
    print(f"saved: {base}.pdf / .png")

def landmarks(env):
    r"""(Z, KS, WB) per z-node: productivity, k*(z), omega_bar(z)."""
    Z = [float(z) for z in env.z_grid]
    sc = lambda x: float(np.ravel(x)[0])
    return Z, [sc(env.k_star(z=z)) for z in Z], [sc(env.omega_bar(z=z)) for z in Z]

def plot_phase_diagrams(vfi, *, w0=5.0, T=80, path="entrepreneur_phase_diagrams.pdf"):
    r"""Transition map + cobweb, policies, velocity w/ rest points, MPK wedge,
    capital-debt locus, simulated path. Uses the raw (continuous-b) grid policy."""
    env, p = vfi.env, vfi.env.p
    Z, KS, WB = landmarks(env)
    W = env.omega_grid
    Wd = np.linspace(W[0], W[-1], 2000)
    slope = (1 + p.R) / (p.THETA_K * (1 - p.DELTA))

    def curves(iz):
        z = Z[iz]
        kp, bp = vfi.kp_pol[iz], vfi.bp_pol[iz]
        wn = env.production(z=z, k=kp) + kp * (1 - p.DELTA) - bp * (1 + p.R)
        kp_d = np.interp(Wd, W, kp); bp_d = np.interp(Wd, W, bp)
        wn_d = env.production(z=z, k=kp_d) + kp_d * (1 - p.DELTA) - bp_d * (1 + p.R)
        return dict(kp=kp, bp=bp, wn=wn, kp_d=kp_d, bp_d=bp_d, wn_d=wn_d)

    C = [curves(iz) for iz in range(len(Z))]

    def rest_point(wn, wb):
        g = wn - W
        s = np.where((g[:-1] > 0) & (g[1:] <= 0))[0]
        return W[s[np.argmin(np.abs(W[s] - wb))]] if len(s) else np.nan

    WSS = [rest_point(c["wn"], wb) for c, wb in zip(C, WB)]

    im = 1 if len(Z) > 1 else 0                                   # display type (mid)
    z_m = Z[im]
    om_p = np.empty(T); kp_p = np.empty(T); bp_p = np.empty(T); om_p[0] = w0
    for t in range(T):
        kp_p[t] = np.interp(om_p[t], W, C[im]["kp"]); bp_p[t] = np.interp(om_p[t], W, C[im]["bp"])
        if t < T - 1:
            om_p[t + 1] = float(np.ravel(env.next_worth(kp=kp_p[t], bp=bp_p[t], z=z_m))[0])

    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    a = ax[0, 0]
    a.plot(Wd, Wd, color=ECON["dark"], ls="--", lw=.8, label=r"$45°$")
    for c, z, c_ in zip(C, Z, TYPE_COLORS):
        a.plot(Wd, c["wn_d"], color=c_, lw=2, label=rf"$z={z:.2f}$")
    x = w0; xs, ys = [x], [x]
    for _ in range(80):
        y = float(np.interp(x, W, C[im]["wn"])); xs += [x, y]; ys += [y, y]
        if abs(y - x) < 1e-3: break
        x = y
    a.plot(xs, ys, color=ECON["purple"], lw=.6, label="cobweb")
    for wb, c_ in zip(WB, TYPE_COLORS): a.axvline(wb, color=c_, ls=":", lw=.8)
    a.set_title(r"Transition map $\omega'=\Phi(\omega;z)$")
    a.set_xlabel(r"$\omega$"); a.set_ylabel(r"$\omega'$")
    a.legend(); a.set_xlim(0, 35); a.set_ylim(0, 35)

    a = ax[0, 1]
    a.plot(Wd, C[im]["kp_d"], color=ECON["blue"], lw=2, label=r"$k'$ mid")
    if len(Z) > 2:
        a.plot(Wd, C[2]["kp_d"], color=ECON["green"], lw=1.4, ls="--", label=r"$k'$ high")
        a.plot(Wd, C[0]["kp_d"], color=ECON["red"], lw=1.4, ls="--", label=r"$k'$ low")
    a.plot(Wd, C[im]["bp_d"], color=ECON["yellow"], lw=2, label=r"$b'$ mid")
    for ks, c_ in zip(KS, TYPE_COLORS): a.axhline(ks, color=c_, ls=":", lw=.6)
    a.axhline(0, color=ECON["dark"], lw=.6); a.axvline(WB[im], color=ECON["grey"], ls=":", lw=.8)
    a.set_title("Capital \\& debt policy"); a.set_xlabel(r"$\omega$"); a.legend()
    a.set_xlim(-5, 60); a.set_ylim(-5, 85)

    a = ax[0, 2]
    for c, c_, l in zip(C, TYPE_COLORS, TYPE_LABELS):
        a.plot(Wd, c["wn_d"] - Wd, color=c_, lw=2, label=l)
    a.axhline(0, color=ECON["dark"], lw=.6)
    for ws, c_ in zip(WSS, TYPE_COLORS):
        if np.isfinite(ws): a.axvline(ws, color=c_, lw=1.1)
    for wb, c_ in zip(WB, TYPE_COLORS): a.axvline(wb, color=c_, ls=":", lw=.8)
    a.set_title(r"Velocity $\omega'-\omega$ (solid=rest, dotted=$\bar\omega$)")
    a.set_xlabel(r"$\omega$"); a.set_ylabel(r"$\omega'-\omega$"); a.legend(); a.set_xlim(0, 60)

    a = ax[1, 0]
    mp = env.production.marginal_product
    for c, z_x, c_, l in zip(C, Z, TYPE_COLORS, TYPE_LABELS):
        a.plot(Wd, mp(z=z_x, k=np.maximum(c["kp_d"], 1e-9)) - (p.R + p.DELTA),
               color=c_, lw=2, label=l)
    a.axhline(0, color=ECON["dark"], lw=.6)
    for wb, c_ in zip(WB, TYPE_COLORS): a.axvline(wb, color=c_, ls=":", lw=.8)
    a.set_title(r"MPK wedge $zf'(k')-(r+\delta)$")
    a.set_xlabel(r"$\omega$"); a.set_ylabel("wedge"); a.legend()
    a.set_ylim(-.05, .8); a.set_xlim(0, 30)

    a = ax[1, 1]
    bb = np.linspace(C[im]["bp"].min(), C[im]["bp"].max(), 50)
    a.plot(bb, slope * bb, color=ECON["dark"], ls="--", lw=.8, label="collateral line")
    a.plot(C[im]["bp_d"], C[im]["kp_d"], color=ECON["blue"], lw=2, label="policy locus")
    a.axhline(KS[im], color=ECON["blue"], ls=":", lw=.8); a.axvline(0, color=ECON["dark"], lw=.6)
    a.set_title("Capital--debt locus"); a.set_xlabel(r"$b'$"); a.set_ylabel(r"$k'$")
    a.legend(); a.set_xlim(-25, 25); a.set_ylim(0, 60)

    a = ax[1, 2]
    t = np.arange(T)
    a.plot(t, om_p, color=ECON["blue"], lw=2, label=r"$\omega$")
    a.plot(t, kp_p, color=ECON["green"], lw=1.5, label=r"$k'$")
    a.plot(t, bp_p, color=ECON["yellow"], lw=1.5, label=r"$b'$")
    a.plot(t, kp_p - bp_p, color=ECON["purple"], lw=1.5, label=r"$a'=k'-b'$")
    a.axhline(WB[im], color=ECON["blue"], ls=":", lw=.8)
    a.axhline(KS[im], color=ECON["green"], ls=":", lw=.6)
    a.set_title(rf"Simulated path (mid) from $\omega_0={w0}$"); a.set_xlabel(r"$t$"); a.legend()

    fig.tight_layout()
    save_fig(fig, path)
    return fig, WSS


def plot_wealth_hist_lorenz(pop, env, *, burn=300, path="wealth_hist_lorenz.pdf"):
    r"""Overlaid per-type wealth densities (alpha=0.3 + step outline) and wealth Lorenz."""
    Z, KS, WB = landmarks(env)
    om = pop["omega"][burn:].ravel(); zz = pop["z"][burn:].ravel()

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    a = ax[0]
    bins = np.linspace(om.min(), np.percentile(om, 99.5), 50)
    for z, c_, l in zip(Z, TYPE_COLORS, TYPE_LABELS):
        s = om[np.isclose(zz, z)]
        a.hist(s, bins=bins, density=True, color=c_, alpha=.3, histtype="stepfilled")
        a.hist(s, bins=bins, density=True, color=c_, histtype="step", lw=1.2, label=l)
    for wb, c_ in zip(WB, TYPE_COLORS): a.axvline(wb, color=c_, ls=":", lw=1.0)
    a.set_title(r"Stationary wealth distribution by type (dotted $=\bar\omega$)")
    a.set_xlabel(r"$\omega$"); a.set_ylabel("density"); a.legend()

    a = ax[1]
    ws = np.sort(om); cum = np.cumsum(ws) / ws.sum()
    pc = np.arange(1, len(ws) + 1) / len(ws)
    gini = 1 - 2 * np.trapz(cum, pc)
    a.plot([0, 1], [0, 1], color=ECON["dark"], ls="--", lw=.8, label="equality")
    a.plot(pc, cum, color=ECON["purple"], lw=2, label=rf"wealth (Gini$={gini:.3f}$)")
    a.fill_between(pc, cum, pc, alpha=.15, color=ECON["purple"])
    a.set_title("Lorenz curve --- wealth")
    a.set_xlabel("cumulative share of firms"); a.set_ylabel(r"cumulative share of $\omega$")
    a.legend()

    fig.tight_layout()
    save_fig(fig, path)
    return fig, gini