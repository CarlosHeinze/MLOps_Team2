import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from team2_mlops2026 import eda, plots


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    x = rng.normal(size=2_000)
    return pd.DataFrame(
        {
            "x": x,
            "y": 2 * x + rng.normal(scale=0.1, size=x.size),
            "z": rng.exponential(size=x.size),
        }
    )


# --- tables ------------------------------------------------------------------


def test_data_quality_counts_missing_and_unique():
    df = pd.DataFrame({"a": [1, None, 1, 2], "b": ["x", "x", None, None]})
    out = eda.data_quality(df)
    assert out.loc["a", ["missing", "n_unique"]].tolist() == [1, 2]
    assert out.loc["b", "missing_pct"] == 50


def test_univariate_summary_sorted_by_absolute_skew(frame):
    out = eda.univariate_summary(frame)
    assert out.index[0] == "z"  # the exponential column is the skewed one
    assert {"median", "skewness", "kurtosis", "mean", "50%"} <= set(out.columns)


def test_iqr_outliers_bounds_and_counts():
    df = pd.DataFrame({"a": [1, 2, 3, 4, 100]})
    out = eda.iqr_outliers(df)
    assert out.loc["a", ["lower_bound", "upper_bound"]].tolist() == [-1, 7]
    assert out.loc["a", "n_potential_outliers"] == 1
    assert out.loc["a", "outlier_pct"] == 20


def test_top_correlated_pairs_upper_triangle_only(frame):
    out = eda.top_correlated_pairs(frame.corr(), n=10)
    assert len(out) == 3  # 3 columns -> 3 distinct pairs, no diagonal
    assert out.loc[0, ["feature_1", "feature_2"]].tolist() == ["x", "y"]


def test_cramers_v_extremes():
    a = pd.Series(list("ab") * 500)
    assert eda.cramers_v(a, a) == pytest.approx(1.0)
    rng = np.random.default_rng(1)
    noise = pd.Series(rng.integers(0, 2, size=a.size))
    assert eda.cramers_v(a, noise) < 0.1
    assert eda.cramers_v(a, pd.Series(np.zeros(a.size))) == 0.0


def test_missingness_association_shape():
    keys = pd.Series(list("ab") * 50)
    missing = pd.DataFrame({"m1": keys == "a", "m2": np.arange(100) % 7 == 0})
    out = eda.missingness_association(missing, {"key": keys})
    assert out.loc["m1", "key"] == pytest.approx(1.0)
    assert out.loc["m2", "key"] < 0.2


def test_group_loss_percentage():
    keys = pd.Series(["a", "a", "b", "b"])
    kept = pd.Series([True, False, True, True])
    assert eda.group_loss(keys, kept).to_dict() == {"a": 50.0, "b": 0.0}


def test_share_by_group_counts_missing_as_value():
    values, keys = pd.Series(["M", None, "F", "F"]), pd.Series([1, 1, 2, 2])
    out = eda.share_by_group(values, keys)
    assert out.loc[1, "<NA>"] == 50 and out.loc[2, "F"] == 100
    known = eda.share_by_group(values, keys, include_missing=False)
    assert "<NA>" not in known and known.loc[1, "M"] == 100


def test_standardized_profiles_are_in_std_units():
    df = pd.DataFrame({"v": [0.0, 0.0, 2.0, 2.0]})
    out = eda.standardized_profiles(df, pd.Series(["a", "a", "b", "b"]))
    std = df["v"].std()
    assert out.loc["a", "v"] == pytest.approx(-1 / std)
    assert out.loc["b", "v"] == pytest.approx(1 / std)


def test_distribution_tests_equivalent_vs_shifted(frame):
    same = eda.distribution_tests(frame, frame.sample(frac=0.5, random_state=0))
    assert same["equivalent"].all()
    shifted = eda.distribution_tests(frame, frame + 1, ["x"])
    assert not shifted.loc["x", "equivalent"]
    assert shifted.loc["x", "smd"] == pytest.approx(1, abs=0.05)


def test_distribution_tests_ignore_missing_values(frame):
    holes = frame.copy()
    holes.loc[::10, "x"] = np.nan
    assert eda.distribution_tests(frame, holes, ["x"]).loc["x", "equivalent"]


# --- charts (smoke) ----------------------------------------------------------


def test_charts_render(frame):
    plots.use_style()
    hue = pd.Series(np.where(frame["x"] > 0, "M", None), name="Genero")
    figures = [
        plots.plot_hist_box(frame),
        plots.plot_bars(pd.Series([3, 1], index=["M", "<NA>"]), "counts"),
        plots.plot_lines(frame.iloc[:50], "lines", "value"),
        plots.plot_box_by(frame.assign(g=hue.fillna("<NA>")), "z", "g"),
        plots.plot_heatmap(frame.corr(), "corr", diverging=True),
        plots.plot_heatmap(frame.abs().iloc[:5], "magnitude", fmt="{:.0f}"),
        plots.plot_grouped_means(frame.groupby(frame["x"].round()).mean(), "means"),
        plots.plot_scatter_matrix(frame.iloc[:200], hue.iloc[:200], "pairs"),
    ]
    assert all(isinstance(f, matplotlib.figure.Figure) for f in figures)
    matplotlib.pyplot.close("all")
