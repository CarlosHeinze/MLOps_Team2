"""
Matplotlib charts for the notebooks (the EDA tables live in ``eda.py``) and for the figures the
backtest logs to MLflow.

Every function takes already-aggregated data or a frame, returns the Figure and never
calls ``plt.show()``, so the same charts can be saved to ``reports/figures`` or logged with
``mlflow.log_figure``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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


def plot_grouped_bars(
    df: pd.DataFrame,
    title: str,
    ylabel: str,
    references: dict | None = None,
    fmt: str = "{:.3f}",
) -> plt.Figure:
    """One group of bars per row and one bar per column (at most three columns), with an
    optional dashed reference line per column, such as a target to beat."""
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(df) * len(df.columns)), 4))
    width = 0.8 / len(df.columns)
    x = np.arange(len(df))
    for i, (col, color) in enumerate(zip(df.columns, _colors(df.columns))):
        bars = ax.bar(x + (i - (len(df.columns) - 1) / 2) * width, df[col], width, color=color)
        bars.set_label(str(col))
        ax.bar_label(
            bars, labels=[fmt.format(v) for v in df[col]], color=INK_SECONDARY, fontsize=8
        )
        if references and col in references:
            ax.axhline(references[col], color=color, linestyle="--", linewidth=1)
    ax.set_xticks(x, df.index.astype(str))
    ax.set(title=title, ylabel=ylabel)
    ax.grid(axis="x", visible=False)
    if len(df.columns) > 1:
        ax.legend()
    fig.tight_layout()
    return fig


def plot_lines(
    df: pd.DataFrame,
    title: str,
    ylabel: str,
    xlabel: str = "",
    figsize=(12, 4),
    muted: Sequence = (),
) -> plt.Figure:
    """One line per column; a legend names them when there is more than one.

    The ``muted`` columns, such as the baselines next to the models, are drawn dashed and grey
    and do not take one of the three series colours.
    """
    fig, ax = plt.subplots(figsize=figsize)
    shown = [c for c in df.columns if c not in muted]
    for col, color in zip(shown, _colors(shown)):
        ax.plot(df.index, df[col], color=color, label=str(col))
    for col, style in zip([c for c in df.columns if c in muted], ("--", ":", "-.")):
        ax.plot(df.index, df[col], color=MUTED, linestyle=style, linewidth=1.5, label=str(col))
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


def plot_maps(
    longitude: pd.Series, latitude: pd.Series, values: pd.DataFrame, title: str
) -> plt.Figure:
    """One map per column of ``values``: points colored diverging around 0 (e.g. PCA scores)."""
    n = values.shape[1]
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), squeeze=False)
    for ax, col in zip(axes[0], values.columns):
        limit = np.nanmax(np.abs(values[col]))
        points = ax.scatter(
            longitude,
            latitude,
            c=values[col],
            cmap=DIVERGING,
            vmin=-limit,
            vmax=limit,
            s=14,
            linewidths=0,
        )
        # A degree of longitude is shorter than one of latitude away from the equator
        ax.set_aspect(1 / np.cos(np.radians(np.nanmean(latitude))))
        ax.set(title=str(col), xticks=[], yticks=[])
        ax.grid(False)
        fig.colorbar(points, ax=ax, shrink=0.7)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


# --- hyperparameter search --------------------------------------------------------


def _log_scaled(values: pd.Series) -> bool:
    """Whether a param reads better on a log scale: its positive values span 10x or more."""
    positive = values[values > 0]
    return len(positive) > 1 and positive.max() / positive.min() >= 10


def _axis_values(values: pd.Series) -> pd.Series:
    """``values`` as drawn: log10 when ``_log_scaled``, with zeros at the smallest positive one."""
    if not _log_scaled(values):
        return values.astype("float64")
    return np.log10(values.clip(lower=values[values > 0].min()))


def plot_parallel_coordinates(
    trials: pd.DataFrame, metric: str, title: str, highlight=None
) -> plt.Figure:
    """One vertical axis per column of ``trials`` (``metric`` last) and one line per trial,
    darkest where ``metric`` is lowest, as in MLflow's parallel coordinates plot.

    Each axis spans its own min to max, on a log scale when it covers 10x or more. The row
    labelled ``highlight``, such as the defaults, is drawn dashed.
    """
    cols = [c for c in trials.columns if c != metric] + [metric]
    scaled = {}
    for col in cols:
        values = _axis_values(trials[col])
        span = values.max() - values.min()
        scaled[col] = (values - values.min()) / span if span else values * 0 + 0.5
    scaled = pd.DataFrame(scaled)
    cmap = SEQUENTIAL.reversed()
    norm = mpl.colors.Normalize(trials[metric].min(), trials[metric].max())
    fig, ax = plt.subplots(figsize=(1.5 * len(cols) + 2, 4.5))
    x = np.arange(len(cols))
    # The best trials are drawn last, on top
    for label in trials[metric].sort_values(ascending=False).index:
        if label != highlight:
            color = cmap(norm(trials.loc[label, metric]))
            ax.plot(x, scaled.loc[label], color=color, linewidth=1.5)
    if highlight is not None:
        ax.plot(x, scaled.loc[highlight], color=INK, linestyle="--", linewidth=1.5)
        ax.plot([], [], color=INK, linestyle="--", label=str(highlight))
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=8)
    for i, col in enumerate(cols):
        ax.axvline(i, color=AXIS, linewidth=1)
        for y, value, va, dy in (
            (1, trials[col].max(), "bottom", 4),
            (0, trials[col].min(), "top", -4),
        ):
            ax.annotate(
                f"{value:.3g}",
                (i, y),
                xytext=(0, dy),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=8,
                color=INK_SECONDARY,
            )
    labels = [f"{c}\n(log)" if _log_scaled(trials[c]) else c for c in cols]
    ax.set_xticks(x, labels, fontsize=8)
    ax.set(yticks=[], xlim=(-0.3, len(cols) - 0.7), ylim=(-0.12, 1.12))
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    fig.colorbar(mpl.cm.ScalarMappable(norm, cmap), ax=ax, label=metric, shrink=0.8)
    ax.set_title(title, pad=24 if highlight is not None else 6)
    fig.tight_layout()
    return fig


def plot_contour(
    trials: pd.DataFrame, x: str, y: str, metric: str, title: str, highlight=None
) -> plt.Figure:
    """``metric`` interpolated over the params ``x`` and ``y`` of ``trials``, darkest where it
    is lowest, as in MLflow's contour plot. Every trial is a point; ``highlight`` is a cross."""
    xs, ys = _axis_values(trials[x]), _axis_values(trials[y])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    filled = ax.tricontourf(xs, ys, trials[metric], levels=12, cmap=SEQUENTIAL.reversed())
    ax.scatter(xs, ys, s=24, color=SURFACE, edgecolors=INK, linewidths=0.8, zorder=3)
    if highlight is not None:
        ax.scatter(
            xs[highlight],
            ys[highlight],
            s=90,
            marker="X",
            color=INK,
            label=str(highlight),
            zorder=4,
        )
        ax.legend(fontsize=8)
    ax.set(
        title=title,
        xlabel=f"{x} (log10)" if _log_scaled(trials[x]) else x,
        ylabel=f"{y} (log10)" if _log_scaled(trials[y]) else y,
    )
    ax.grid(False)
    fig.colorbar(filled, ax=ax, label=metric)
    fig.tight_layout()
    return fig


def plot_param_effects(
    trials: pd.DataFrame, metric: str, title: str, highlight=None, ncols: int = 4
) -> plt.Figure:
    """``metric`` against each param (every other column of ``trials``), one panel per param
    on a shared y axis (log x when it spans 10x or more); ``highlight`` is a cross."""
    params = [c for c in trials.columns if c != metric]
    nrows = -(-len(params) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), sharey=True, squeeze=False
    )
    for ax, param in zip(axes.flat, params):
        ax.scatter(trials[param], trials[metric], s=24, color=SERIES[0], linewidths=0)
        if highlight is not None:
            ax.scatter(
                trials.loc[highlight, param],
                trials.loc[highlight, metric],
                s=70,
                marker="X",
                color=INK,
                label=str(highlight),
            )
        values = trials[param]
        if _log_scaled(values) and (values <= 0).any():  # a default of 0, such as no penalty
            ax.set_xscale("symlog", linthresh=values[values > 0].min())
        elif _log_scaled(values):
            ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        ax.set_xlabel(param)
    for ax in axes[:, 0]:
        ax.set_ylabel(metric)
    for ax in axes.flat[len(params) :]:
        ax.axis("off")
    if highlight is not None:
        axes[0, 0].legend(fontsize=8)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


# --- backtest -----------------------------------------------------------------------


def plot_station_forecasts(
    panels: Mapping[str, pd.DataFrame], title: str, ylabel: str, actual: str = "actual"
) -> plt.Figure:
    """One panel per station, stacked, each with the ``actual`` column in black and up to three
    forecast columns in the series colours, against its index (the hours)."""
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(12, 2.8 * len(panels)), squeeze=False, sharex=False
    )
    for ax, (label, frame) in zip(axes[:, 0], panels.items()):
        ax.plot(frame.index, frame[actual], color=INK, linewidth=1.2, label=actual)
        forecasts = [c for c in frame.columns if c != actual]
        for col, color in zip(forecasts, _colors(forecasts)):
            ax.plot(frame.index, frame[col], color=color, linewidth=1.5, label=str(col))
        ax.set(title=str(label), ylabel=ylabel)
    axes[0, 0].legend(fontsize=8, ncols=1 + len(panels[next(iter(panels))].columns))
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_residuals(
    residuals: pd.DataFrame, title: str, quantiles: tuple[float, float] = (0.001, 0.999)
) -> plt.Figure:
    """Histogram of each column of ``residuals`` (actual - forecast), one panel each, clipped
    to the ``quantiles``, with the mean as a dashed line. The count axis is logarithmic, since
    the residuals of the many quiet hours sit around 0."""
    fig, axes = plt.subplots(
        1, len(residuals.columns), figsize=(5.5 * len(residuals.columns), 3.8), squeeze=False
    )
    for ax, col in zip(axes[0], residuals.columns):
        values = residuals[col].dropna().to_numpy()
        low, high = np.quantile(values, quantiles)
        bins = 80
        if np.all(np.mod(values, 1) == 0):  # integer residuals: one bin per value, no gaps
            bins = np.arange(np.floor(low), np.ceil(high) + 2) - 0.5
        ax.hist(values, bins=bins, range=(low - 0.5, high + 0.5), color=SERIES[0], log=True)
        ax.axvline(values.mean(), color=SERIES[1], linestyle="--", linewidth=1.2)
        ax.set(
            title=f"{col} (mean {values.mean():+.3f})",
            xlabel="actual - forecast",
            ylabel="station-hours",
        )
    fig.suptitle(
        f"{title} (p{quantiles[0] * 100:g} to p{quantiles[1] * 100:g})", fontweight="bold"
    )
    fig.tight_layout()
    return fig


def plot_importance(importance: pd.DataFrame, title: str, top: int = 15) -> plt.Figure:
    """Horizontal bars of the ``top`` features by the ``mean`` column, the largest on top, with
    the ``std`` column as error bars (a permutation importance)."""
    shown = importance.sort_values("mean", ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(shown) + 1.5))
    ax.barh(
        shown.index.astype(str),
        shown["mean"],
        xerr=shown["std"],
        color=SERIES[0],
        error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1},
    )
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set(title=title, xlabel="increase in Poisson deviance when shuffled")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig
