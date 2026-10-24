"""
Matplotlib charts for the EDA notebooks (tables live in ``eda.py``).

Every function takes already-aggregated data or a frame, returns the Figure and never
calls ``plt.show()``, so the same charts can be saved to ``reports/figures``.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Categorical slots, used in this order; at most three series share a chart
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MUTED = "#898781"  # missing / unknown groups
INK, INK_SECONDARY = "#0b0b0b", "#52514e"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "sequential_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "diverging_blue_red", ["#184f95", "#3987e5", "#f0efec", "#e66767", "#a32a2a"]
)

# In order of preference. Segoe UI only ships with Windows; DejaVu Sans ships with matplotlib.
FONTS = ("Segoe UI", "DejaVu Sans")


def _installed_fonts() -> list[str]:
    """``FONTS`` that exist here: a missing one logs a "findfont" warning for every text."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    return [font for font in FONTS if font in installed]


def use_style() -> None:
    """Light surface, hairline grid and recessive axes for every following chart."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "font.family": [*_installed_fonts(), "sans-serif"],
        }
    )


def _colors(labels: Sequence) -> list[str]:
    """Series slots by sorted label, so a group keeps its color whatever the row order.

    Groups named "<NA>" are drawn muted.
    """
    known = sorted({str(label) for label in labels} - {"<NA>"})
    return [MUTED if str(label) == "<NA>" else SERIES[known.index(str(label))] for label in labels]


def _thousands(ax) -> None:
    if ax.get_ylim()[1] >= 10_000:
        ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter("{x:,.0f}"))


def plot_hist_box(df: pd.DataFrame, upper_quantile: float = 0.995, bins: int = 60) -> plt.Figure:
    """Histogram + boxplot per column; the histogram is clipped at ``upper_quantile``.

    Integer-valued columns get one bin per value, so binning does not draw gaps.
    """
    fig, axes = plt.subplots(
        len(df.columns), 2, figsize=(12, 2.8 * len(df.columns)), squeeze=False
    )
    for (ax_hist, ax_box), col in zip(axes, df.columns):
        values = df[col].dropna().to_numpy()
        top = np.quantile(values, upper_quantile)
        edges = bins
        if np.all(np.mod(values, 1) == 0):
            edges = np.arange(values.min(), top + 2) - 0.5
        ax_hist.hist(values, bins=edges, range=(values.min(), top), color=SERIES[0])
        _thousands(ax_hist)
        ax_hist.set_title(f"Distribution of {col} (up to p{upper_quantile * 100:g})")
        ax_hist.set_ylabel("trips")
        ax_box.boxplot(
            values,
            orientation="horizontal",
            showfliers=False,
            widths=0.5,
            medianprops={"color": SERIES[1], "linewidth": 2},
        )
        ax_box.set_yticks([])
        ax_box.set_title(f"Boxplot of {col} (whiskers 1.5 IQR, outliers hidden)")
    fig.tight_layout()
    return fig


def plot_bars(
    counts: pd.Series, title: str, xlabel: str = "", ylabel: str = "trips", labels: bool = True
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * len(counts)), 4))
    bars = ax.bar(counts.index.astype(str), counts.to_numpy(), color=_colors(counts.index))
    if labels:
        ax.bar_label(bars, labels=[f"{v:,.0f}" for v in counts], color=INK_SECONDARY, fontsize=8)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(axis="x", visible=False)
    _thousands(ax)
    fig.tight_layout()
    return fig


def plot_lines(
    df: pd.DataFrame, title: str, ylabel: str, xlabel: str = "", figsize=(12, 4)
) -> plt.Figure:
    """One line per column; a legend names them when there is more than one."""
    fig, ax = plt.subplots(figsize=figsize)
    for col, color in zip(df.columns, _colors(df.columns)):
        ax.plot(df.index, df[col], color=color, label=str(col))
    if len(df.columns) > 1:
        ax.legend()
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    _thousands(ax)
    fig.tight_layout()
    return fig


def plot_box_by(df: pd.DataFrame, value: str, by: str, title: str | None = None) -> plt.Figure:
    """Boxplot of ``value`` per group of ``by`` (outliers hidden: millions of points)."""
    groups = df.groupby(by, observed=True)[value]
    names = list(groups.groups)
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(names)), 4))
    parts = ax.boxplot(
        [g.dropna().to_numpy() for _, g in groups],
        tick_labels=[str(n) for n in names],
        showfliers=False,
        patch_artist=True,
        medianprops={"color": INK, "linewidth": 1.5},
    )
    for patch, color in zip(parts["boxes"], _colors(names)):
        patch.set(facecolor=color, alpha=0.75, edgecolor=SURFACE)
    ax.set(title=title or f"{value} by {by}", xlabel=by, ylabel=value)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def plot_heatmap(
    table: pd.DataFrame,
    title: str,
    diverging: bool = False,
    fmt: str = "{:.2f}",
    annotate: bool = True,
    vmax: float | None = None,
    figsize=None,
) -> plt.Figure:
    """Sequential (magnitude) or diverging around 0 (correlations, standardized values)."""
    values = table.to_numpy(dtype=float)
    if diverging:
        limit = vmax or np.nanmax(np.abs(values))
        norm = {"vmin": -limit, "vmax": limit, "cmap": DIVERGING}
    else:
        norm = {"vmin": 0, "vmax": vmax or np.nanmax(values), "cmap": SEQUENTIAL}
    fig, ax = plt.subplots(figsize=figsize or (1 + 0.8 * table.shape[1], 1 + 0.5 * table.shape[0]))
    image = ax.imshow(values, aspect="auto", **norm)
    ax.set_xticks(range(table.shape[1]), table.columns.astype(str), rotation=45, ha="right")
    ax.set_yticks(range(table.shape[0]), table.index.astype(str))
    ax.grid(False)
    if annotate:
        for (i, j), v in np.ndenumerate(values):
            if np.isnan(v):
                continue
            dark = abs(v) / norm["vmax"] > 0.6 if diverging else v / norm["vmax"] > 0.55
            ax.text(
                j,
                i,
                fmt.format(v),
                ha="center",
                va="center",
                fontsize=8,
                color=SURFACE if dark else INK,
            )
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_grouped_means(means: pd.DataFrame, title: str, ncols: int = 3) -> plt.Figure:
    """Small multiples: mean of each column (one panel each) across the index groups."""
    nrows = -(-means.shape[1] // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3 * nrows), squeeze=False)
    for ax, col in zip(axes.flat, means.columns):
        ax.plot(means.index, means[col], marker="o", markersize=4, color=SERIES[0])
        ax.set_title(col)
        ax.set_xlabel(means.index.name or "")
    for ax in axes.flat[means.shape[1] :]:
        ax.axis("off")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_scatter_matrix(
    df: pd.DataFrame, hue: pd.Series, title: str, upper_quantile: float = 0.995
) -> plt.Figure:
    """Pairwise scatter (lower triangle) and per-group histograms (diagonal).

    Axes stop at ``upper_quantile`` so a long tail does not squeeze the bulk.
    """
    cols = list(df.columns)
    labels = hue.astype("object").fillna("<NA>")
    groups = sorted(pd.unique(labels), key=lambda g: (g == "<NA>", str(g)))
    colors = dict(zip(groups, _colors(groups)))
    limits = {c: (df[c].min(), df[c].quantile(upper_quantile)) for c in cols}
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n))
    for i, row in enumerate(cols):
        for j, col in enumerate(cols):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            for group in groups:
                part = df[labels == group]
                if i == j:
                    ax.hist(
                        part[col].dropna(),
                        bins=40,
                        range=limits[col],
                        alpha=0.55,
                        density=True,
                        color=colors[group],
                        label=str(group),
                    )
                else:
                    ax.scatter(
                        part[col], part[row], s=3, alpha=0.25, color=colors[group], linewidths=0
                    )
                    ax.set_ylim(limits[row])
            ax.set_xlim(limits[col])
            if i == n - 1:
                ax.set_xlabel(col)
            if j == 0:
                ax.set_ylabel(row)
    axes[0, 0].legend(title=hue.name, fontsize=8)
    fig.suptitle(f"{title} (axes up to p{upper_quantile * 100:g})", fontweight="bold")
    fig.tight_layout()
    return fig
