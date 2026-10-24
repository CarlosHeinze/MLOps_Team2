"""
Dataset-agnostic EDA tables used by the notebooks (charts live in ``plots.py``).

With tens of millions of rows every hypothesis test rejects equality on negligible
differences, so the comparisons here report effect sizes (KS statistic, standardized mean
difference, variance ratio, Cramér's V) instead of p-values.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

# Thresholds for "the two distributions are equivalent"
KS_TOLERANCE = 0.05
SMD_TOLERANCE = 0.1
VARIANCE_RATIO_TOLERANCE = 0.1


def data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """dtype, missing count / percentage and distinct values per column."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_pct": df.isna().mean() * 100,
            "n_unique": df.nunique(),
        }
    )


def univariate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """describe() plus median, skewness and kurtosis, most skewed first."""
    summary = df.describe().T
    summary["median"] = df.median()
    summary["skewness"] = df.skew()
    summary["kurtosis"] = df.kurt()
    return summary.sort_values("skewness", key=lambda s: s.abs(), ascending=False)


def iqr_outliers(df: pd.DataFrame, k: float = 1.5) -> pd.DataFrame:
    """Potential outliers outside [Q1 - k*IQR, Q3 + k*IQR] per column (diagnostic only)."""
    q1, q3 = df.quantile(0.25), df.quantile(0.75)
    lower, upper = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    flagged = df.lt(lower) | df.gt(upper)
    return pd.DataFrame(
        {
            "lower_bound": lower,
            "upper_bound": upper,
            "n_potential_outliers": flagged.sum(),
            "outlier_pct": flagged.mean() * 100,
        }
    ).sort_values("outlier_pct", ascending=False)


def top_correlated_pairs(corr: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The ``n`` feature pairs with the largest absolute correlation."""
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().dropna().rename("correlation").rename_axis(["feature_1", "feature_2"])
    pairs = pairs.reset_index()
    pairs["abs_correlation"] = pairs["correlation"].abs()
    return pairs.nlargest(n, "abs_correlation").reset_index(drop=True)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Association between two categorical series: 0 = independent, 1 = determined."""
    table = pd.crosstab(x, y)
    if min(table.shape) < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    return float(np.sqrt(chi2 / (table.to_numpy().sum() * (min(table.shape) - 1))))


def missingness_association(missing: pd.DataFrame, by: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Cramér's V between each missing-indicator column and each grouping.

    Values near 0 are consistent with MCAR for that grouping; larger values mean the
    missingness depends on it (MAR), so dropping incomplete rows biases it.
    """
    return pd.DataFrame(
        {name: {col: cramers_v(missing[col], key) for col in missing} for name, key in by.items()}
    )


def group_loss(keys: pd.Series, kept: pd.Series) -> pd.Series:
    """Percentage of rows a filter removes within each group."""
    return (1 - kept.groupby(keys, observed=True).mean()) * 100


def share_by_group(
    values: pd.Series, keys: pd.Series, include_missing: bool = True
) -> pd.DataFrame:
    """Row-normalized percentage of each value within each group.

    With ``include_missing`` the missing values count as their own value "<NA>";
    otherwise shares are computed over the known values only.
    """
    if include_missing:
        values = values.astype("object").fillna("<NA>")
    return pd.crosstab(keys, values, normalize="index") * 100


def standardized_profiles(df: pd.DataFrame, by: pd.Series) -> pd.DataFrame:
    """Group means expressed in standard deviations from the overall mean."""
    return (df.groupby(by, observed=True).mean() - df.mean()) / df.std()


def distribution_tests(
    reference: pd.DataFrame, other: pd.DataFrame, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """KS statistic, standardized mean difference and variance ratio per column.

    ``equivalent`` holds when the three are within the module tolerances.
    """
    rows = []
    for col in columns or reference.columns:
        ref, oth = reference[col].dropna().to_numpy(), other[col].dropna().to_numpy()
        rows.append(
            {
                "feature": col,
                "ks_stat": stats.ks_2samp(ref, oth).statistic,
                "smd": (oth.mean() - ref.mean()) / ref.std(),
                "variance_ratio": oth.var() / ref.var(),
            }
        )
    out = pd.DataFrame(rows).set_index("feature")
    out["equivalent"] = (
        (out["ks_stat"] < KS_TOLERANCE)
        & (out["smd"].abs() < SMD_TOLERANCE)
        & ((out["variance_ratio"] - 1).abs() < VARIANCE_RATIO_TOLERANCE)
    )
    return out
