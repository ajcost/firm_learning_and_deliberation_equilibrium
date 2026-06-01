from __future__ import annotations

from matplotlib.axes import Axes
import numpy as np


# ---------------------------------------------------------------------------
# Economist palette & shared styling
# ---------------------------------------------------------------------------
PALETTE = ["#014d64", "#ad2624", "#01a2d9", "#6794a7", "#76c0c1",
           "#7a0177", "#d95f02", "#1b9e77"]
GRAY = "#595959"
LIGHT_GRAY = "#d4d4d4"
BG_COLOR = "#f0f0f0"


def style_ax(ax: Axes, title: str | None = None,
             xlabel: str | None = None, ylabel: str | None = None) -> Axes:
    """Apply Economist house style to an axes and return it."""
    ax.set_facecolor(BG_COLOR)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRAY)
    ax.spines["bottom"].set_color(GRAY)

    ax.grid(axis="y", color="gray", alpha=0.3, lw=0.6, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)

    ax.tick_params(axis="both", which="both", length=0, labelsize=9, colors=GRAY)

    if title is not None:
        ax.set_title(title, fontsize=13, fontweight="bold", color="#1a1a1a",
                     loc="left", pad=10)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=11, color=GRAY, labelpad=8)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=11, color=GRAY, labelpad=10)

    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", frameon=False, fontsize=9, labelcolor=GRAY)

    return ax


# ---------------------------------------------------------------------------
# Generic line plot
# ---------------------------------------------------------------------------

def plot_lines(
    ax: Axes,
    x: np.ndarray,
    lines: list[dict],
    hlines: list[dict] | None = None,
    band: dict | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> Axes:
    """Generic multi-series line plot.

    Parameters
    ----------
    ax : Axes
        Target axes.
    x : np.ndarray
        Shared x-axis values.
    lines : list[dict]
        Each dict must have ``'y'``.  Optional keys:
        ``'label'``, ``'color'``, ``'lw'``, ``'ls'``, ``'alpha'``, ``'zorder'``.
    hlines : list[dict] | None
        Horizontal reference lines.  Keys: ``'y'``, ``'label'``, ``'color'``,
        ``'ls'``, ``'alpha'``.
    band : dict | None
        Shaded band.  Keys: ``'upper'``, ``'lower'``, ``'color'``, ``'alpha'``.
    """
    for i, line in enumerate(lines):
        ax.plot(
            x, line["y"],
            color=line.get("color", PALETTE[i % len(PALETTE)]),
            label=line.get("label"),
            lw=line.get("lw", 1.8),
            ls=line.get("ls", "-"),
            alpha=line.get("alpha", 1.0),
            zorder=line.get("zorder", 2),
        )

    if hlines:
        for h in hlines:
            ax.axhline(
                h["y"], color=h.get("color", GRAY), ls=h.get("ls", "--"),
                alpha=h.get("alpha", 0.7), lw=h.get("lw", 1.2),
                label=h.get("label"), zorder=1,
            )

    if band:
        ax.fill_between(
            x, band["lower"], band["upper"],
            color=band.get("color", PALETTE[0]),
            alpha=band.get("alpha", 0.18),
        )

    return style_ax(ax, title=title, xlabel=xlabel, ylabel=ylabel)


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------

def plot_histogram(
    ax: Axes,
    data: np.ndarray,
    vlines: list[dict] | None = None,
    bins: int = 40,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> Axes:
    """Histogram with optional vertical reference lines.

    Parameters
    ----------
    vlines : list[dict] | None
        Each dict: ``'x'`` (required), plus optional ``'label'``, ``'color'``,
        ``'lw'``, ``'ls'``, ``'alpha'``.
    """
    ax.hist(data, bins=bins, density=True, color=LIGHT_GRAY,
            edgecolor="white", alpha=0.85, zorder=2)

    if vlines:
        for i, v in enumerate(vlines):
            ax.axvline(
                v["x"], color=v.get("color", PALETTE[i % len(PALETTE)]),
                lw=v.get("lw", 2), ls=v.get("ls", "-"),
                alpha=v.get("alpha", 0.85),
                label=v.get("label"), zorder=3,
            )

    return style_ax(ax, title=title, xlabel=xlabel, ylabel=ylabel)


# ---------------------------------------------------------------------------
# Phase diagram
# ---------------------------------------------------------------------------

def plot_phase(
    ax: Axes,
    k_grid: np.ndarray,
    policies: list[np.ndarray] | np.ndarray,
    fixed_points: list[float] | None = None,
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    title: str | None = None,
) -> Axes:
    """Economist-style k vs k' phase diagram on a provided axes."""
    if not isinstance(policies, list):
        policies = [policies]
    labels = labels or [f"Policy {i+1}" for i in range(len(policies))]
    colors = colors or PALETTE

    ax.plot(k_grid, k_grid, color=LIGHT_GRAY, ls="--", lw=1,
            label=r"$k' = k$", zorder=1)

    for i, pol in enumerate(policies):
        ax.plot(k_grid, pol, color=colors[i % len(colors)], lw=2.2,
                label=labels[i], zorder=2 + i)

    if fixed_points is not None:
        for i, fp in enumerate(fixed_points):
            ax.scatter(fp, fp, color=colors[i % len(colors)], s=40,
                       edgecolor="white", linewidth=0.8,
                       zorder=3 + len(policies) + i)

    return style_ax(ax, title=title, xlabel=r"$k_t$", ylabel=r"$k_{t+1}$")


# ---------------------------------------------------------------------------
# Clustered policy plot
# ---------------------------------------------------------------------------

def plot_clustered_policies(
    ax: Axes,
    eval_k_grid: np.ndarray,
    firm_policies: np.ndarray,
    cluster_labels: np.ndarray,
    rational_policy: np.ndarray | None = None,
    k_ss: float | None = None,
    title: str = "Clustered Average Policies",
    colors: list[str] | None = None,
) -> Axes:
    """Plot cluster-averaged policy curves with std bands on a provided axes."""
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)
    colors = colors or PALETTE[:n_clusters]

    if rational_policy is not None:
        ax.plot(eval_k_grid, rational_policy, color=GRAY, ls="--", lw=2,
                label=r"Rational $\mathbb{E}[k'|k]$", zorder=3)

    ax.plot(eval_k_grid, eval_k_grid, color=LIGHT_GRAY, ls=":", lw=1,
            label=r"$k' = k$", zorder=1)

    for idx, c in enumerate(unique_clusters):
        mask = cluster_labels == c
        avg = firm_policies[mask].mean(axis=0)
        std = firm_policies[mask].std(axis=0)
        n_in = int(mask.sum())
        color = colors[idx % len(colors)]

        ax.plot(eval_k_grid, avg, color=color, lw=2.2,
                label=f"Group {chr(65 + idx)} (N={n_in})", zorder=2)
        ax.fill_between(eval_k_grid, avg - std, avg + std,
                        color=color, alpha=0.15)

    if k_ss is not None:
        ax.scatter([k_ss], [k_ss], color="#1a1a1a", s=50, edgecolor="white",
                   linewidth=0.8, zorder=5, label=f"$k^*={k_ss:.1f}$")

    return style_ax(ax, title=title, xlabel=r"$k_t$", ylabel=r"$k_{t+1}$")
