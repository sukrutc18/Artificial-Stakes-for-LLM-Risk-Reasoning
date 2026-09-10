"""Evaluation metrics: accuracy, calibration, regret, reasoning proxies, judge."""

from risk_reasoning.eval.accuracy import (
    accuracy,
    best_of_n_accuracy,
    majority_vote_accuracy,
)
from risk_reasoning.eval.calibration import (
    brier_score,
    ece,
    plot_calibration_curves,
    reliability_curve,
)
from risk_reasoning.eval.propagation import mistake_propagation
from risk_reasoning.eval.reasoning_proxies import (
    count_alternatives,
    count_risk_factors,
    logical_error_rate,
)
from risk_reasoning.eval.regret import (
    budget_efficiency,
    cumulative_regret,
    cumulative_reward,
    recovery_rate,
)

__all__ = [
    "accuracy",
    "budget_efficiency",
    "best_of_n_accuracy",
    "brier_score",
    "count_alternatives",
    "count_risk_factors",
    "cumulative_regret",
    "cumulative_reward",
    "ece",
    "logical_error_rate",
    "majority_vote_accuracy",
    "mistake_propagation",
    "plot_calibration_curves",
    "recovery_rate",
    "reliability_curve",
]
