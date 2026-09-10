"""Post-hoc analysis: stats, ablation, and visualization for experiment results."""

from risk_reasoning.analysis.ablation import persistent_vs_surface_analysis
from risk_reasoning.analysis.loader import load_multiple_runs, load_run_matrix, load_samples
from risk_reasoning.analysis.stats import (
    bootstrap_ci,
    significance_table,
    summary_stats,
    ttest_conditions,
)
from risk_reasoning.analysis.visualization import (
    accuracy_bar_chart,
    calibration_curves_grid,
    make_all_figures,
    reasoning_proxy_bars,
    regret_curves,
)

__all__ = [
    "load_samples",
    "load_multiple_runs",
    "load_run_matrix",
    "bootstrap_ci",
    "ttest_conditions",
    "significance_table",
    "summary_stats",
    "persistent_vs_surface_analysis",
    "accuracy_bar_chart",
    "calibration_curves_grid",
    "regret_curves",
    "reasoning_proxy_bars",
    "make_all_figures",
]
