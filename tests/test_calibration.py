"""Tests for calibration metrics (ECE, Brier, reliability curve)."""

from __future__ import annotations

import pytest

from risk_reasoning.eval import brier_score, ece, reliability_curve


def test_ece_perfect_calibration_is_zero() -> None:
    confs = [0.5] * 100
    correct = [True] * 50 + [False] * 50
    assert ece(confs, correct, n_bins=10) == pytest.approx(0.0, abs=1e-6)


def test_brier_bounds() -> None:
    val = brier_score([1.0, 0.0, 0.5], [True, False, True])
    assert 0.0 <= val <= 1.0


def test_reliability_curve_shape() -> None:
    c, a, n = reliability_curve([0.1, 0.9], [False, True], n_bins=10)
    assert len(c) == len(a) == len(n) == 10
