"""Ablation: surface-stakes vs persistent-stakes comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from risk_reasoning.analysis.stats import bootstrap_ci


def persistent_vs_surface_analysis(
    df: pd.DataFrame,
    metrics: list[str] | None = None,
    *,
    n_bootstrap: int = 2000,
) -> pd.DataFrame:
    """Bootstrap comparison of persistent vs surface stakes per dataset × model.

    Returns one row per (dataset, model) with, for each metric:
    - ``<m>_surface`` / ``<m>_persistent``   mean accuracy under each condition
    - ``<m>_delta``                           persistent − surface
    - ``<m>_ci_lo`` / ``<m>_ci_hi``          95 % bootstrap CI on the delta
    - ``<m>_p_persistent_better``            fraction of bootstrap samples where persistent > surface
    """
    if metrics is None:
        metrics = ["correct"]

    rng = np.random.default_rng(42)
    rows = []
    for dataset, dset_grp in df.groupby("dataset"):
        for model, model_grp in dset_grp.groupby("model"):
            row: dict = {"dataset": dataset, "model": model}
            surf = model_grp[model_grp["condition"] == "surface_stakes"]
            pers = model_grp[model_grp["condition"] == "persistent_stakes"]
            for m in metrics:
                if m not in df.columns:
                    continue
                vs = surf[m].dropna().to_numpy(dtype=float)
                vp = pers[m].dropna().to_numpy(dtype=float)
                if not vs.size or not vp.size:
                    for suffix in ("_surface", "_persistent", "_delta", "_ci_lo", "_ci_hi", "_p_persistent_better"):
                        row[f"{m}{suffix}"] = float("nan")
                    continue
                row[f"{m}_surface"] = float(vs.mean())
                row[f"{m}_persistent"] = float(vp.mean())
                row[f"{m}_delta"] = float(vp.mean() - vs.mean())
                n_s, n_p = len(vs), len(vp)
                boot_deltas = np.array([
                    rng.choice(vp, n_p, replace=True).mean()
                    - rng.choice(vs, n_s, replace=True).mean()
                    for _ in range(n_bootstrap)
                ])
                row[f"{m}_ci_lo"] = float(np.quantile(boot_deltas, 0.025))
                row[f"{m}_ci_hi"] = float(np.quantile(boot_deltas, 0.975))
                row[f"{m}_p_persistent_better"] = float((boot_deltas > 0).mean())
            rows.append(row)
    return pd.DataFrame(rows)
