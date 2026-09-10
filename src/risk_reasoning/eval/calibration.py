"""Calibration metrics: Expected Calibration Error, Brier score, reliability curve."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def ece(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error with equal-width bins."""
    if len(confidences) == 0:
        return 0.0
    bin_confs, bin_accs, bin_counts = reliability_curve(confidences, correct, n_bins=n_bins)
    n = sum(bin_counts)
    if n == 0:
        return 0.0
    return float(
        sum(
            (count / n) * abs(conf - acc)
            for conf, acc, count in zip(bin_confs, bin_accs, bin_counts, strict=True)
            if count > 0
        )
    )


def brier_score(
    confidences: Sequence[float], correct: Sequence[bool]
) -> float:
    """Mean squared error between confidence and correctness."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have equal length")
    if len(confidences) == 0:
        return 0.0
    confs = np.asarray(confidences, dtype=float)
    targets = np.asarray(correct, dtype=float)
    return float(np.mean((confs - targets) ** 2))


def reliability_curve(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> tuple[list[float], list[float], list[int]]:
    """Return ``(bin_confidences, bin_accuracies, bin_counts)`` for plotting.

    Each list has length ``n_bins``. Empty bins report their midpoint for
    confidence and ``0.0`` for accuracy so the arrays remain plottable; callers
    that care should filter on ``bin_counts > 0``.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must have equal length")

    confs = np.asarray(confidences, dtype=float)
    targets = np.asarray(correct, dtype=float)

    if confs.size and (confs.min() < 0.0 or confs.max() > 1.0):
        raise ValueError("confidences must lie in [0, 1]")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    midpoints = (edges[:-1] + edges[1:]) / 2.0
    # Interior cuts only, so the index range is exactly [0, n_bins-1].
    # Using all edges with np.digitize would push confidence==1.0 into a
    # phantom bin n_bins.
    bin_idx = np.digitize(confs, edges[1:-1]) if confs.size else np.empty(0, dtype=int)

    bin_confs: list[float] = []
    bin_accs: list[float] = []
    bin_counts: list[int] = []
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count > 0:
            bin_confs.append(float(confs[mask].mean()))
            bin_accs.append(float(targets[mask].mean()))
        else:
            bin_confs.append(float(midpoints[b]))
            bin_accs.append(0.0)
        bin_counts.append(count)
    return bin_confs, bin_accs, bin_counts


def plot_calibration_curves(
    curves: Mapping[str, tuple[Sequence[float], Sequence[bool]]],
    n_bins: int = 10,
    ax: Axes | None = None,
    save_path: str | os.PathLike[str] | None = None,
    title: str | None = None,
) -> Figure:
    """Plot one reliability curve per condition against the perfect-calibration diagonal.

    Args:
        curves: Mapping from condition label to ``(confidences, correct)``.
        n_bins: Bin count forwarded to :func:`reliability_curve`.
        ax: Optional pre-existing axes. A new figure is created when ``None``.
        save_path: If given, the figure is written here (parent dirs created).
        title: Optional axes title.

    Returns:
        The :class:`matplotlib.figure.Figure` containing the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure

    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", label="perfect")

    for label, (confs, correct) in curves.items():
        bin_confs, bin_accs, bin_counts = reliability_curve(confs, correct, n_bins=n_bins)
        xs = [c for c, n in zip(bin_confs, bin_counts, strict=True) if n > 0]
        ys = [a for a, n in zip(bin_accs, bin_counts, strict=True) if n > 0]
        ax.plot(xs, ys, marker="o", label=label)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="best")

    if save_path is not None:
        path = os.fspath(save_path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=150)

    return fig
