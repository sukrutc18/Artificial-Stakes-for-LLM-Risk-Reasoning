"""Visualization: accuracy bars, calibration curves, regret curves, reasoning proxy bars.

Calibration curve plotting delegates to the existing
``risk_reasoning.eval.calibration.plot_calibration_curves`` to avoid duplication.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PLT: Any = None


def _plt() -> Any:
    global _PLT
    if _PLT is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _PLT = plt
    return _PLT


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

CONDITION_ORDER = ["bare_baseline", "surface_stakes", "persistent_stakes", "cot_only"]

CONDITION_LABELS = {
    "bare_baseline": "Baseline",
    "surface_stakes": "Surface Stakes",
    "persistent_stakes": "Persistent Stakes",
    "cot_only": "CoT Only",
}

CONDITION_COLORS = {
    "bare_baseline": "#4477AA",
    "surface_stakes": "#EE6677",
    "persistent_stakes": "#228833",
    "cot_only": "#CCBB44",
}

DATASET_LABELS = {
    "synthetic_ev": "Synthetic EV",
    "convfinqa": "ConvFinQA",
    "arc_challenge": "ARC-Challenge",
    "medqa": "MedQA",
    "sequential_budget": "Sequential",
}


def _save(fig: Any, path: str | Path | None) -> None:
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight", dpi=150)


def _apply_style() -> None:
    _plt().rcParams.update({
        "figure.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.4,
        "font.size": 10,
    })


# ---------------------------------------------------------------------------
# Accuracy bar chart
# ---------------------------------------------------------------------------

def accuracy_bar_chart(
    df: pd.DataFrame,
    *,
    metric: str = "correct",
    save_path: str | Path | None = None,
    title: str | None = None,
    models: list[str] | None = None,
    n_bootstrap: int = 1000,
) -> Any:
    """Grouped bar chart: one group per dataset, one bar per condition, error bars = 95 % CI.

    Pass a pre-filtered DataFrame (e.g. only surface/persistent rows) to produce
    a surface-vs-persistent comparison plot.  Returns the matplotlib Figure.
    """
    from risk_reasoning.analysis.stats import bootstrap_ci

    plt = _plt()
    _apply_style()

    if models is not None:
        df = df[df["model"].isin(models)]

    datasets = sorted(
        df["dataset"].unique(),
        key=lambda d: list(DATASET_LABELS).index(d) if d in DATASET_LABELS else 99,
    )
    conditions = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    n_datasets, n_conds = len(datasets), len(conditions)
    bar_width = 0.7 / n_conds
    x = np.arange(n_datasets)

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(max(6, 1.8 * n_datasets), 4))

    for ci, cond in enumerate(conditions):
        offset = (ci - n_conds / 2 + 0.5) * bar_width
        means, lo_errs, hi_errs = [], [], []
        for dset in datasets:
            vals = (
                df[(df["condition"] == cond) & (df["dataset"] == dset)][metric]
                .dropna().tolist()
            )
            if vals:
                m = float(np.mean(vals))
                lo, hi = bootstrap_ci(vals, n_bootstrap=n_bootstrap, rng=rng)
            else:
                m, lo, hi = float("nan"), float("nan"), float("nan")
            means.append(m)
            lo_errs.append(m - lo if not np.isnan(lo) else 0)
            hi_errs.append(hi - m if not np.isnan(hi) else 0)

        ax.bar(
            x + offset, means,
            width=bar_width,
            color=CONDITION_COLORS.get(cond, "#888888"),
            label=CONDITION_LABELS.get(cond, cond),
            yerr=[lo_errs, hi_errs],
            capsize=3,
            error_kw={"elinewidth": 1, "ecolor": "black", "alpha": 0.7},
            zorder=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [DATASET_LABELS.get(d, d) for d in datasets], rotation=15, ha="right"
    )
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title or "Accuracy by Dataset and Condition")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Calibration curves — delegates to eval.calibration.plot_calibration_curves
# ---------------------------------------------------------------------------

def calibration_curves_grid(
    df: pd.DataFrame,
    *,
    save_dir: str | Path | None = None,
    n_bins: int = 10,
) -> list[Any]:
    """One calibration-reliability figure per dataset, all conditions overlaid.

    Delegates to :func:`risk_reasoning.eval.calibration.plot_calibration_curves`
    so there is no duplication of the binning / plotting logic.
    Returns a list of Figures (one per dataset).
    """
    from risk_reasoning.eval.calibration import plot_calibration_curves

    if "confidence" not in df.columns or "correct" not in df.columns:
        return []

    figs = []
    for dset in df["dataset"].unique():
        sub = df[df["dataset"] == dset].dropna(subset=["confidence", "correct"])
        curves = {}
        for cond in CONDITION_ORDER:
            cond_sub = sub[sub["condition"] == cond]
            if cond_sub.empty:
                continue
            curves[CONDITION_LABELS.get(cond, cond)] = (
                cond_sub["confidence"].tolist(),
                cond_sub["correct"].astype(bool).tolist(),
            )
        if not curves:
            continue
        save_path = (
            Path(save_dir) / f"calibration_{dset}.png"
            if save_dir is not None else None
        )
        fig = plot_calibration_curves(
            curves, n_bins=n_bins, save_path=save_path,
            title=f"Calibration — {DATASET_LABELS.get(dset, dset)}",
        )
        figs.append(fig)
    return figs


# ---------------------------------------------------------------------------
# Regret curves (sequential dataset only)
# ---------------------------------------------------------------------------

def regret_curves(
    df: pd.DataFrame,
    *,
    save_path: str | Path | None = None,
    title: str | None = None,
) -> Any | None:
    """Cumulative regret over steps for the sequential budget dataset.

    One line per condition; shaded band = ±1 SD across sessions.
    Returns ``None`` if no sequential data is present.
    """
    plt = _plt()
    _apply_style()

    seq = df[df["dataset"] == "sequential_budget"].copy()
    if seq.empty or "budget_step" not in seq.columns or "correct" not in seq.columns:
        return None

    conditions = [c for c in CONDITION_ORDER if c in seq["condition"].unique()]
    fig, ax = plt.subplots(figsize=(7, 4))

    for cond in conditions:
        cond_df = seq[seq["condition"] == cond]
        if cond_df.empty:
            continue
        session_col = "question_id" if "question_id" in cond_df.columns else "sample_idx"
        all_cumregret = []
        max_steps = 0
        for _, sess in cond_df.groupby(session_col):
            sess = sess.sort_values("budget_step")
            regret = (1.0 - sess["correct"].to_numpy(dtype=float)).cumsum()
            all_cumregret.append(regret)
            max_steps = max(max_steps, len(regret))

        if not all_cumregret:
            continue

        padded = np.full((len(all_cumregret), max_steps), np.nan)
        for i, arr in enumerate(all_cumregret):
            padded[i, : len(arr)] = arr

        mean = np.nanmean(padded, axis=0)
        std = np.nanstd(padded, axis=0)
        steps = np.arange(1, max_steps + 1)
        color = CONDITION_COLORS.get(cond, "#888888")
        ax.plot(steps, mean, label=CONDITION_LABELS.get(cond, cond), color=color, linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Regret")
    ax.set_title(title or "Cumulative Regret by Condition (Sequential Budget)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Reasoning-proxy bar charts
# ---------------------------------------------------------------------------

PROXY_METRICS = {
    "judge_alternatives_considered": "Alternatives Considered",
    "judge_risk_factors_enumerated": "Risk Factors",
    "judge_coherence": "Coherence Score",
    "judge_reasoning_coverage": "Reasoning Coverage",
}


def reasoning_proxy_bars(
    df: pd.DataFrame,
    *,
    save_path: str | Path | None = None,
    title: str | None = None,
    n_bootstrap: int = 1000,
) -> Any | None:
    """Bar chart of judge-derived reasoning proxies, one sub-plot per metric.

    Returns ``None`` if no judge columns are present in *df*.
    """
    from risk_reasoning.analysis.stats import bootstrap_ci

    available = {k: v for k, v in PROXY_METRICS.items() if k in df.columns}
    if not available:
        return None

    plt = _plt()
    _apply_style()

    conditions = [c for c in CONDITION_ORDER if c in df["condition"].unique()]
    n_metrics = len(available)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    rng = np.random.default_rng(42)
    for ax, (col, label) in zip(axes, available.items()):
        for ci, cond in enumerate(conditions):
            vals = df[df["condition"] == cond][col].dropna().tolist()
            if vals:
                m = float(np.mean(vals))
                lo, hi = bootstrap_ci(vals, n_bootstrap=n_bootstrap, rng=rng)
                err = [[m - lo], [hi - m]]
            else:
                m, err = float("nan"), [[0], [0]]
            ax.bar(
                ci, m,
                color=CONDITION_COLORS.get(cond, "#888888"),
                label=CONDITION_LABELS.get(cond, cond),
                yerr=err, capsize=4,
                error_kw={"elinewidth": 1},
                width=0.6, zorder=3,
            )
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(
            [CONDITION_LABELS.get(c, c) for c in conditions],
            rotation=20, ha="right", fontsize=8,
        )
        ax.set_title(label, fontsize=9)

    axes[0].set_ylabel("Score")
    if title:
        fig.suptitle(title, fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8, bbox_to_anchor=(1, 1))
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def make_all_figures(
    df: pd.DataFrame,
    save_dir: str | Path,
    *,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Produce and save every standard figure. Returns a ``{name: Figure}`` dict."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    figs: dict[str, Any] = {}

    figs["accuracy_bars"] = accuracy_bar_chart(
        df, save_path=save_dir / "accuracy_bars.png", n_bootstrap=n_bootstrap,
    )

    for fig in calibration_curves_grid(df, save_dir=save_dir / "calibration"):
        figs[f"calibration_{len(figs)}"] = fig

    f = regret_curves(df, save_path=save_dir / "regret_curves.png")
    if f is not None:
        figs["regret_curves"] = f

    f = reasoning_proxy_bars(
        df, save_path=save_dir / "reasoning_proxies.png", n_bootstrap=n_bootstrap,
    )
    if f is not None:
        figs["reasoning_proxies"] = f

    return figs
