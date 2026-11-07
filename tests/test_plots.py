import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from team2_mlops2026 import plots  # noqa: E402


def test_grouped_bars_draw_a_bar_per_cell_and_the_references():
    wape = pd.DataFrame(
        {"departures": [0.65, 0.53], "arrivals": [0.60, 0.50]}, index=["baseline", "model"]
    )
    fig = plots.plot_grouped_bars(wape, "Peak WAPE", "WAPE", {"departures": 0.584})
    (ax,) = fig.axes
    assert len(ax.patches) == 4
    assert [t.get_text() for t in ax.get_xticklabels()] == ["baseline", "model"]
    assert [line.get_ydata()[0] for line in ax.get_lines()] == [0.584]
    assert ax.containers[0].datavalues.tolist() == [0.65, 0.53]
    plt.close(fig)


TRIALS = pd.DataFrame(
    {
        "learning_rate": [0.1, 0.01, 0.3, 0.05],
        "num_leaves": [31, 60, 15, 63],
        "reg_lambda": [0.0, 1e-3, 5.0, 0.1],  # the default of 0 on a log-scaled param
        "deviance": [0.90, 0.85, 0.95, 0.80],
    },
    index=["defaults", "trial-01", "trial-02", "trial-03"],
)


def test_parallel_coordinates_draw_a_line_per_trial_and_an_axis_per_column():
    fig = plots.plot_parallel_coordinates(TRIALS, "deviance", "Search", highlight="defaults")
    ax = fig.axes[0]
    trial_lines = [line for line in ax.get_lines() if len(line.get_xdata()) == len(TRIALS.columns)]
    assert len(trial_lines) == len(TRIALS)
    # The metric is the last axis, and the log-scaled params say so
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["learning_rate\n(log)", "num_leaves", "reg_lambda\n(log)", "deviance"]
    assert all(0 <= y <= 1 for line in trial_lines for y in line.get_ydata())
    plt.close(fig)


def test_contour_and_param_effects_show_every_trial():
    fig = plots.plot_contour(
        TRIALS, "learning_rate", "num_leaves", "deviance", "Contour", highlight="defaults"
    )
    assert fig.axes[0].get_xlabel() == "learning_rate (log10)"
    assert len(fig.axes[0].collections[-2].get_offsets()) == len(TRIALS)
    plt.close(fig)
    fig = plots.plot_param_effects(TRIALS, "deviance", "Effects", highlight="defaults", ncols=2)
    assert [ax.get_xlabel() for ax in fig.axes[:3]] == [
        "learning_rate",
        "num_leaves",
        "reg_lambda",
    ]
    assert fig.axes[2].get_xscale() == "symlog"
    plt.close(fig)


def test_lines_draw_the_muted_columns_dashed_and_grey():
    folds = pd.DataFrame(
        {"lgbm_tweedie": [0.50, 0.52], "seasonal_naive": [0.65, 0.66]},
        index=["2025-10", "2025-11"],
    )
    fig = plots.plot_lines(folds, "Peak WAPE", "WAPE", muted=["seasonal_naive"])
    model, baseline = fig.axes[0].get_lines()
    assert model.get_linestyle() == "-" and model.get_color() == plots.SERIES[0]
    assert baseline.get_linestyle() == "--" and baseline.get_color() == plots.MUTED
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == list(folds.columns)
    plt.close(fig)


def test_station_forecasts_draw_one_panel_per_station():
    hours = pd.date_range("2026-09-07", periods=48, freq="h")
    week = pd.DataFrame({"actual": range(48), "lgbm_tweedie": range(48)}, index=hours)
    fig = plots.plot_station_forecasts(
        {"station 1": week, "station 2": week}, "Week", "departures"
    )
    assert [ax.get_title() for ax in fig.axes] == ["station 1", "station 2"]
    actual, forecast = fig.axes[0].get_lines()
    assert actual.get_color() == plots.INK and forecast.get_color() == plots.SERIES[0]
    plt.close(fig)


def test_residuals_draw_a_histogram_per_column_with_its_mean():
    residuals = pd.DataFrame(
        {"departures": [-1.0, 0.0, 0.0, 3.0], "arrivals": [0.0, 1.0, 1.0, 2.0]}
    )
    fig = plots.plot_residuals(residuals, "Residuals", quantiles=(0, 1))
    assert [ax.get_title() for ax in fig.axes] == [
        "departures (mean +0.500)",
        "arrivals (mean +1.000)",
    ]
    assert fig.axes[0].get_yscale() == "log"
    assert fig.axes[0].get_lines()[0].get_xdata()[0] == 0.5
    plt.close(fig)


def test_importance_draws_the_top_features_largest_on_top():
    importance = pd.DataFrame(
        {"mean": [0.01, 0.30, 0.05], "std": [0.0, 0.02, 0.01]}, index=["pc1", "lag_7d", "hour_sin"]
    )
    fig = plots.plot_importance(importance, "Importance", top=2)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == ["hour_sin", "lag_7d"]
    assert [p.get_width() for p in ax.patches] == [0.05, 0.30]
    plt.close(fig)
