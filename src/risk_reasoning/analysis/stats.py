"""Statistical significance testing: bootstrap CIs, Welch t-tests, summary tables."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Canonical ordering used throughout the project
CONDITION_ORDER = ["bare_baseline", "surface_stakes", "persistent_stakes", "cot_only"]

# helpers

def bootstrap_ci(
    values: Sequence[float],
    *,
    stat: str = "mean",
    n_bootstrap: int = 2000,
    ci: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval.

    Returns ``(lower, upper)`` for ``stat`` of *values*.  Supported stats:
    ``"mean"``, ``"median"``, ``"std"``.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if rng is None:
        rng = np.random.default_rng(42)
    fn = {"mean": np.mean, "median": np.median, "std": np.std}[stat]
    boot = np.array(
        [fn(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_bootstrap)]
    )
    alpha = (1 - ci) / 2
    return float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha))

# diff in means divided by pooled stddev, where pooled stddev is sqrt((var_a + var_b) / 2).
def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Cohen's d: (mean_b − mean_a) / pooled SD."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    a_arr, b_arr = a_arr[~np.isnan(a_arr)], b_arr[~np.isnan(b_arr)]
    if a_arr.size < 2 or b_arr.size < 2:
        return float("nan")
    pooled = np.sqrt((a_arr.var(ddof=1) + b_arr.var(ddof=1)) / 2)
    if pooled == 0:
        return 0.0 if a_arr.mean() == b_arr.mean() else float("nan")
    return float((b_arr.mean() - a_arr.mean()) / pooled)


# ---------------------------------------------------------------------------
# Per-pair test
# ---------------------------------------------------------------------------

def ttest_conditions(
    df: pd.DataFrame,
    condition_a: str,
    condition_b: str,
    *,
    metric: str = "correct",
    group_by: list[str] | None = None,
) -> dict:
    """Welch's t-test comparing *metric* means for two conditions.

    When *group_by* is set (e.g. ``["question_id"]``), values are first
    aggregated to item-level means, removing within-item variance so that
    each observation is one problem rather than one sample.
    """
    def _extract(cond: str) -> pd.Series:
        sub = df[df["condition"] == cond]
        if group_by:
            return sub.groupby(group_by)[metric].mean().dropna()
        return sub[metric].dropna()

    a, b = _extract(condition_a), _extract(condition_b)
    mean_a = float(a.mean()) if len(a) else float("nan")
    mean_b = float(b.mean()) if len(b) else float("nan")
    delta = mean_b - mean_a

    if len(a) < 2 or len(b) < 2:
        return dict(
            condition_a=condition_a, condition_b=condition_b, metric=metric,
            mean_a=mean_a, mean_b=mean_b, delta=delta,
            t_stat=float("nan"), p_value=float("nan"),
            cohens_d=float("nan"), significant=False,
        )

    t_stat, p_value = scipy_stats.ttest_ind(a, b, equal_var=False)
    d = cohens_d(a.tolist(), b.tolist())
    return dict(
        condition_a=condition_a, condition_b=condition_b, metric=metric,
        mean_a=mean_a, mean_b=mean_b, delta=delta,
        t_stat=float(t_stat), p_value=float(p_value),
        cohens_d=d, significant=bool(p_value < 0.05),
    )


# ---------------------------------------------------------------------------
# Aggregate tables
# ---------------------------------------------------------------------------

def significance_table(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    *,
    group_by: list[str] | None = None,
) -> pd.DataFrame:
    """All pairwise Welch t-tests across every condition pair for each metric.

    Returns a DataFrame with columns: condition_a, condition_b, metric,
    mean_a, mean_b, delta, t_stat, p_value, cohens_d, significant.
    """
    if metrics is None:
        metrics = ["correct"]
    conditions = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    rows = []
    for m in metrics:
        if m not in df.columns:
            continue
        for i, ca in enumerate(conditions):
            for cb in conditions[i + 1:]:
                rows.append(ttest_conditions(df, ca, cb, metric=m, group_by=group_by))
    return pd.DataFrame(rows)


def summary_stats(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    *,
    group_cols: list[str] | None = None,
    n_bootstrap: int = 2000,
) -> pd.DataFrame:
    """Mean ± bootstrap 95 % CI for each metric, grouped by *group_cols*.

    Default grouping is ``["condition"]``; pass e.g.
    ``["condition", "dataset", "model"]`` for a full breakdown.
    """
    if metrics is None:
        metrics = ["correct"]
    if group_cols is None:
        group_cols = ["condition"]

    rng = np.random.default_rng(42)
    rows = []
    for keys, grp in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict = dict(zip(group_cols, keys))
        for m in metrics:
            if m not in grp.columns:
                continue
            vals = grp[m].dropna().tolist()
            if not vals:
                row[f"{m}_mean"] = float("nan")
                row[f"{m}_ci_lo"] = float("nan")
                row[f"{m}_ci_hi"] = float("nan")
            else:
                lo, hi = bootstrap_ci(vals, n_bootstrap=n_bootstrap, rng=rng)
                row[f"{m}_mean"] = float(np.nanmean(vals))
                row[f"{m}_ci_lo"] = lo
                row[f"{m}_ci_hi"] = hi
        row["n"] = len(grp)
        rows.append(row)
    return pd.DataFrame(rows)
