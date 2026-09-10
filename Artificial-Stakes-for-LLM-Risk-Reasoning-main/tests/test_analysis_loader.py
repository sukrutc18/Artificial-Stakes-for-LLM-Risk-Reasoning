"""Tests for risk_reasoning.analysis — loader, stats, ablation, and visualization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from risk_reasoning.analysis.loader import load_multiple_runs, load_samples
from risk_reasoning.analysis.stats import (
    bootstrap_ci,
    significance_table,
    summary_stats,
    ttest_conditions,
)
from risk_reasoning.analysis.ablation import persistent_vs_surface_analysis
from risk_reasoning.analysis.visualization import (
    accuracy_bar_chart,
    calibration_curves_grid,
    make_all_figures,
    reasoning_proxy_bars,
    regret_curves,
)


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _minimal_record(**overrides) -> dict:
    base = {
        "question_id": "q1",
        "condition": "bare_baseline",
        "dataset": "synthetic_ev",
        "model": "llama_base",
        "sample_idx": 0,
        "answer": "A",
        "confidence": 0.7,
        "correct": True,
    }
    base.update(overrides)
    return base


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Multi-condition DataFrame with known accuracy values.

    base model accuracy by condition:
      bare_baseline=0.0, surface_stakes=0.6, persistent_stakes=1.0, cot_only=0.5
    instruct model: always correct (1.0) regardless of condition.
    """
    rng = np.random.default_rng(0)
    rows = []
    datasets = ["synthetic_ev", "arc_challenge"]
    base_acc = {
        "bare_baseline": 0.1,
        "surface_stakes": 0.55,
        "persistent_stakes": 0.9,
        "cot_only": 0.5,
    }
    for dataset in datasets:
        for cond, acc in base_acc.items():
            for qid in range(10):
                for sample_idx in range(3):
                    rows.append({
                        "question_id": f"{dataset}_q{qid}",
                        "condition": cond,
                        "dataset": dataset,
                        "model": "llama_base",
                        "sample_idx": sample_idx,
                        "correct": float(rng.random() < acc),
                        "confidence": float(rng.uniform(0.4, 0.9)),
                        "judge_coherence": float(rng.uniform(0.5, 1.0)),
                        "judge_alternatives_considered": int(rng.integers(1, 5)),
                    })
                    rows.append({
                        "question_id": f"{dataset}_q{qid}",
                        "condition": cond,
                        "dataset": dataset,
                        "model": "llama_instruct",
                        "sample_idx": sample_idx,
                        "correct": float(rng.random() < (acc + 0.2)),
                        "confidence": float(rng.uniform(0.6, 1.0)),
                        "judge_coherence": float(rng.uniform(0.7, 1.0)),
                        "judge_alternatives_considered": int(rng.integers(2, 6)),
                    })
    return pd.DataFrame(rows)


@pytest.fixture()
def seq_df() -> pd.DataFrame:
    """Sequential-budget DataFrame for regret-curve tests."""
    rows = []
    for cond in ["bare_baseline", "persistent_stakes"]:
        for qid in range(5):
            for step in range(8):
                rows.append({
                    "question_id": f"seq_q{qid}",
                    "condition": cond,
                    "dataset": "sequential_budget",
                    "model": "llama_base",
                    "sample_idx": 0,
                    "correct": float(cond == "persistent_stakes"),
                    "budget_step": step,
                })
    return pd.DataFrame(rows)


# ===========================================================================
# loader
# ===========================================================================

def test_load_samples_returns_dataframe(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "samples.jsonl", [_minimal_record()])
    assert isinstance(load_samples(tmp_path), pd.DataFrame)


def test_load_samples_row_count(tmp_path: Path) -> None:
    records = [_minimal_record(question_id=f"q{i}") for i in range(5)]
    _write_jsonl(tmp_path / "samples.jsonl", records)
    assert len(load_samples(tmp_path)) == 5


def test_load_samples_empty_file_returns_empty_df(tmp_path: Path) -> None:
    (tmp_path / "samples.jsonl").write_text("")
    assert load_samples(tmp_path).empty


def test_load_samples_correct_is_numeric(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "samples.jsonl", [_minimal_record(correct=True)])
    assert load_samples(tmp_path)["correct"].dtype.kind in "fiu"


def test_load_samples_correct_false_maps_to_zero(tmp_path: Path) -> None:
    _write_jsonl(tmp_path / "samples.jsonl", [_minimal_record(correct=False)])
    assert load_samples(tmp_path)["correct"].iloc[0] == pytest.approx(0.0)


def test_load_samples_flattens_judge_score(tmp_path: Path) -> None:
    rec = _minimal_record(judge_score={"coherence": 0.9, "risk_factors_enumerated": 3})
    _write_jsonl(tmp_path / "samples.jsonl", [rec])
    df = load_samples(tmp_path)
    assert "judge_coherence" in df.columns
    assert "judge_risk_factors_enumerated" in df.columns
    assert "judge_score" not in df.columns


def test_load_samples_judge_values_preserved(tmp_path: Path) -> None:
    rec = _minimal_record(judge_score={"coherence": 0.75})
    _write_jsonl(tmp_path / "samples.jsonl", [rec])
    assert load_samples(tmp_path)["judge_coherence"].iloc[0] == pytest.approx(0.75)


def test_load_samples_null_judge_score_handled(tmp_path: Path) -> None:
    records = [
        _minimal_record(question_id="q1", judge_score={"coherence": 0.8}),
        _minimal_record(question_id="q2", judge_score=None),
    ]
    _write_jsonl(tmp_path / "samples.jsonl", records)
    df = load_samples(tmp_path)
    assert len(df) == 2
    assert "judge_coherence" in df.columns


def test_load_samples_flattens_budget_states(tmp_path: Path) -> None:
    rec = _minimal_record(
        budget_state_before={"current": 1000.0, "step": 0},
        budget_state_after={"current": 900.0, "step": 1},
    )
    _write_jsonl(tmp_path / "samples.jsonl", [rec])
    df = load_samples(tmp_path)
    assert "budget_before" in df.columns
    assert "budget_after" in df.columns
    assert "budget_state_before" not in df.columns


def test_load_samples_budget_values_correct(tmp_path: Path) -> None:
    rec = _minimal_record(
        budget_state_before={"current": 1000.0, "step": 0},
        budget_state_after={"current": 900.0, "step": 1},
    )
    _write_jsonl(tmp_path / "samples.jsonl", [rec])
    df = load_samples(tmp_path)
    assert df["budget_before"].iloc[0] == pytest.approx(1000.0)
    assert df["budget_after"].iloc[0] == pytest.approx(900.0)


def test_load_multiple_runs_concatenates(tmp_path: Path) -> None:
    for name in ("run_a", "run_b"):
        d = tmp_path / name; d.mkdir()
        _write_jsonl(d / "samples.jsonl", [_minimal_record(question_id=name)])
    assert len(load_multiple_runs([tmp_path / "run_a", tmp_path / "run_b"])) == 2


def test_load_multiple_runs_tags_source_dir(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"; run_dir.mkdir()
    _write_jsonl(run_dir / "samples.jsonl", [_minimal_record()])
    assert "run_dir" in load_multiple_runs([run_dir]).columns


def test_load_multiple_runs_skips_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty"; empty.mkdir()
    (empty / "samples.jsonl").write_text("")
    full = tmp_path / "full"; full.mkdir()
    _write_jsonl(full / "samples.jsonl", [_minimal_record()])
    assert len(load_multiple_runs([empty, full])) == 1


# ===========================================================================
# stats — bootstrap_ci
# ===========================================================================

def test_bootstrap_ci_contains_true_mean() -> None:
    rng = np.random.default_rng(0)
    vals = rng.normal(loc=0.5, scale=0.1, size=200).tolist()
    lo, hi = bootstrap_ci(vals, n_bootstrap=2000)
    assert lo < 0.5 < hi


def test_bootstrap_ci_empty_returns_nan() -> None:
    lo, hi = bootstrap_ci([])
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ci_single_value() -> None:
    lo, hi = bootstrap_ci([0.7])
    assert lo == pytest.approx(0.7, abs=1e-6)
    assert hi == pytest.approx(0.7, abs=1e-6)


def test_bootstrap_ci_width_shrinks_with_larger_n() -> None:
    rng = np.random.default_rng(1)
    small = rng.normal(0.5, 0.1, 20).tolist()
    large = rng.normal(0.5, 0.1, 500).tolist()
    lo_s, hi_s = bootstrap_ci(small, n_bootstrap=500)
    lo_l, hi_l = bootstrap_ci(large, n_bootstrap=500)
    assert (hi_s - lo_s) > (hi_l - lo_l)



# ===========================================================================
# stats — ttest_conditions
# ===========================================================================

def test_ttest_significant_for_clearly_different_groups(sample_df: pd.DataFrame) -> None:
    result = ttest_conditions(sample_df, "bare_baseline", "persistent_stakes", metric="correct")
    assert result["significant"] is True
    assert result["delta"] > 0


def test_ttest_delta_equals_mean_difference(sample_df: pd.DataFrame) -> None:
    result = ttest_conditions(sample_df, "bare_baseline", "persistent_stakes", metric="correct")
    assert result["delta"] == pytest.approx(result["mean_b"] - result["mean_a"])


def test_ttest_returns_cohens_d(sample_df: pd.DataFrame) -> None:
    result = ttest_conditions(sample_df, "bare_baseline", "persistent_stakes", metric="correct")
    assert not np.isnan(result["cohens_d"])


def test_ttest_missing_condition_returns_nan(sample_df: pd.DataFrame) -> None:
    result = ttest_conditions(sample_df, "bare_baseline", "nonexistent", metric="correct")
    assert np.isnan(result["p_value"])


# ===========================================================================
# stats — significance_table
# ===========================================================================

def test_significance_table_all_pairs_present(sample_df: pd.DataFrame) -> None:
    table = significance_table(sample_df, ["correct"])
    n = sample_df["condition"].nunique()
    assert len(table) == n * (n - 1) // 2


def test_significance_table_has_required_columns(sample_df: pd.DataFrame) -> None:
    table = significance_table(sample_df, ["correct"])
    for col in ("condition_a", "condition_b", "delta", "p_value", "significant"):
        assert col in table.columns


def test_significance_table_baseline_vs_persistent_significant(sample_df: pd.DataFrame) -> None:
    table = significance_table(sample_df, ["correct"])
    row = table[
        (table["condition_a"] == "bare_baseline") &
        (table["condition_b"] == "persistent_stakes")
    ]
    assert not row.empty
    assert bool(row["significant"].iloc[0]) is True


# ===========================================================================
# stats — summary_stats
# ===========================================================================

def test_summary_stats_has_mean_and_ci_columns(sample_df: pd.DataFrame) -> None:
    ss = summary_stats(sample_df, ["correct"])
    for col in ("correct_mean", "correct_ci_lo", "correct_ci_hi", "n"):
        assert col in ss.columns


def test_summary_stats_ci_bounds_valid(sample_df: pd.DataFrame) -> None:
    ss = summary_stats(sample_df, ["correct"]).dropna(subset=["correct_ci_lo", "correct_ci_hi"])
    assert (ss["correct_ci_lo"] <= ss["correct_mean"]).all()
    assert (ss["correct_mean"] <= ss["correct_ci_hi"]).all()


def test_summary_stats_mean_matches_pandas(sample_df: pd.DataFrame) -> None:
    ss = summary_stats(sample_df, ["correct"], group_cols=["condition"])
    for _, row in ss.iterrows():
        expected = sample_df[sample_df["condition"] == row["condition"]]["correct"].mean()
        assert row["correct_mean"] == pytest.approx(expected, abs=1e-6)


# ===========================================================================
# ablation — persistent_vs_surface_analysis
# ===========================================================================

def test_pvs_has_delta_column(sample_df: pd.DataFrame) -> None:
    pvs = persistent_vs_surface_analysis(sample_df, ["correct"])
    assert "correct_delta" in pvs.columns


def test_pvs_ci_contains_delta(sample_df: pd.DataFrame) -> None:
    pvs = persistent_vs_surface_analysis(sample_df, ["correct"], n_bootstrap=500)
    valid = pvs.dropna(subset=["correct_delta", "correct_ci_lo", "correct_ci_hi"])
    assert (valid["correct_ci_lo"] <= valid["correct_delta"]).all()
    assert (valid["correct_delta"] <= valid["correct_ci_hi"]).all()


def test_pvs_p_proxy_in_unit_interval(sample_df: pd.DataFrame) -> None:
    pvs = persistent_vs_surface_analysis(sample_df, ["correct"], n_bootstrap=500)
    valid = pvs["correct_p_persistent_better"].dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


def test_pvs_p_high_for_base_model_when_persistent_clearly_better(sample_df: pd.DataFrame) -> None:
    pvs = persistent_vs_surface_analysis(sample_df, ["correct"], n_bootstrap=500)
    # base model: persistent≈0.9, surface≈0.55 → persistent clearly better
    base_pvs = pvs[pvs["model"] == "llama_base"]
    assert (base_pvs["correct_p_persistent_better"].dropna() > 0.7).all()


def test_pvs_groups_by_dataset_and_model(sample_df: pd.DataFrame) -> None:
    pvs = persistent_vs_surface_analysis(sample_df, ["correct"])
    assert "dataset" in pvs.columns and "model" in pvs.columns


# ===========================================================================
# visualization — accuracy_bar_chart
# ===========================================================================

def test_accuracy_bar_chart_returns_figure(sample_df: pd.DataFrame) -> None:
    import matplotlib.figure
    assert isinstance(accuracy_bar_chart(sample_df), matplotlib.figure.Figure)


def test_accuracy_bar_chart_saves_pdf(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    out = tmp_path / "acc.pdf"
    accuracy_bar_chart(sample_df, save_path=out)
    assert out.exists()


def test_accuracy_bar_chart_surface_vs_persistent_subset(sample_df: pd.DataFrame) -> None:
    import matplotlib.figure
    subset = sample_df[sample_df["condition"].isin(["surface_stakes", "persistent_stakes"])]
    assert isinstance(accuracy_bar_chart(subset), matplotlib.figure.Figure)


# ===========================================================================
# visualization — calibration_curves_grid
# ===========================================================================

def test_calibration_curves_grid_returns_list(sample_df: pd.DataFrame) -> None:
    figs = calibration_curves_grid(sample_df)
    assert isinstance(figs, list) and len(figs) > 0


def test_calibration_curves_grid_empty_without_confidence_column() -> None:
    df = pd.DataFrame({"condition": ["bare_baseline"], "dataset": ["synthetic_ev"]})
    assert calibration_curves_grid(df) == []


def test_calibration_curves_grid_saves_files(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    calibration_curves_grid(sample_df, save_dir=tmp_path / "calib")
    assert any((tmp_path / "calib").glob("*.pdf"))


# ===========================================================================
# visualization — regret_curves
# ===========================================================================

def test_regret_curves_none_without_sequential_data(sample_df: pd.DataFrame) -> None:
    assert regret_curves(sample_df) is None


def test_regret_curves_returns_figure(seq_df: pd.DataFrame) -> None:
    import matplotlib.figure
    assert isinstance(regret_curves(seq_df), matplotlib.figure.Figure)


def test_regret_curves_saves_file(seq_df: pd.DataFrame, tmp_path: Path) -> None:
    regret_curves(seq_df, save_path=tmp_path / "regret.pdf")
    assert (tmp_path / "regret.pdf").exists()


# ===========================================================================
# visualization — reasoning_proxy_bars
# ===========================================================================

def test_reasoning_proxy_bars_none_without_judge_columns() -> None:
    df = pd.DataFrame({"condition": ["bare_baseline"], "correct": [1.0]})
    assert reasoning_proxy_bars(df) is None


def test_reasoning_proxy_bars_returns_figure(sample_df: pd.DataFrame) -> None:
    import matplotlib.figure
    assert isinstance(reasoning_proxy_bars(sample_df), matplotlib.figure.Figure)


def test_reasoning_proxy_bars_saves_file(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    reasoning_proxy_bars(sample_df, save_path=tmp_path / "proxies.pdf")
    assert (tmp_path / "proxies.pdf").exists()


# ===========================================================================
# visualization — make_all_figures
# ===========================================================================

def test_make_all_figures_returns_dict(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    figs = make_all_figures(sample_df, tmp_path / "figs", n_bootstrap=100)
    assert isinstance(figs, dict) and len(figs) > 0


def test_make_all_figures_saves_accuracy_pdf(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    make_all_figures(sample_df, tmp_path / "figs", n_bootstrap=100)
    assert (tmp_path / "figs" / "accuracy_bars.pdf").exists()
