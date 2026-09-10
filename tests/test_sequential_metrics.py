"""Tests for Phase 5 sequential metrics from JSONL logs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from risk_reasoning.eval.sequential_metrics import (
    collect_samples_from_run_dirs,
    parse_cell_id,
    summarize_sequential_metrics,
)
from risk_reasoning.eval.regret import cumulative_regret


def test_parse_cell_id_roundtrip() -> None:
    cid = "persistent_stakes__sequential_budget__llama31_8b_base__seed42"
    m = parse_cell_id(cid)
    assert m["condition"] == "persistent_stakes"
    assert m["dataset"] == "sequential_budget"
    assert m["model"] == "llama31_8b_base"
    assert m["seed"] == 42


def test_collect_samples_from_run_dirs(tmp_path: Path) -> None:
    cell = tmp_path / "bare_baseline__medqa__llama__seed0"
    cell.mkdir(parents=True)
    rec = {
        "item_index": 0,
        "sample_idx": 0,
        "correct": True,
        "budget_before_current": 100.0,
        "budget_after_current": 95.0,
    }
    (cell / "samples.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    df = collect_samples_from_run_dirs([tmp_path])
    assert len(df) == 1
    assert df["_cell_id"].iloc[0] == cell.name


def test_sequential_env_two_episodes_regret(tmp_path: Path) -> None:
    """Env-style rows: regret uses best EV 0.2 per step when config is absent."""
    run = tmp_path / "run1"
    cell = run / "cot_only__sequential_budget__m__seed1"
    cell.mkdir(parents=True)
    rows = []
    for ep in (0, 1):
        for step, (rew, bud, act) in enumerate(
            [
                (0.1, 999.0, "allocate_low"),
                (0.1, 998.0, "allocate_low"),
            ]
        ):
            rows.append(
                {
                    "episode": ep,
                    "step": step,
                    "reward": rew,
                    "budget": bud,
                    "action": act,
                    "condition": "cot_only",
                    "dataset": "sequential_budget",
                    "model_id": "m",
                }
            )
    (cell / "samples.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    df = collect_samples_from_run_dirs([run])
    out = summarize_sequential_metrics(df, default_best_ev_per_step=0.2, default_initial_budget=1000.0)
    assert "by_cell" in out
    assert len(out["by_cell"]) == 1
    # per episode: two steps, regret (0.2-0.1)*2 = 0.2 each → mean 0.2
    assert out["by_cell"][0]["cumulative_regret_final_mean"] == pytest.approx(0.2)
    assert out["by_cell"][0]["n_sessions"] == 2


def test_sequential_env_reads_best_ev_from_snapshot(tmp_path: Path) -> None:
    """Custom mixture EV base = (0.5*2 + 0.5*0)/1 = 1.0 → best = 2.0 * 1.0 = 2.0 per step."""
    run = tmp_path / "run2"
    cell = run / "bare_baseline__sequential_budget__m__seed2"
    cell.mkdir(parents=True)
    snapshot = {
        "cell": {
            "extra": {
                "dataset_cfg": {
                    "env": {
                        "initial_budget": 500.0,
                        "reward_params": {
                            "components": [
                                {"weight": 0.5, "mean": 2.0, "std": 0.1},
                                {"weight": 0.5, "mean": 0.0, "std": 0.1},
                            ]
                        },
                    }
                }
            }
        }
    }
    (cell / "config_snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    rows = [
        {
            "episode": 0,
            "step": 0,
            "reward": 1.0,
            "budget": 499.0,
            "action": "allocate_high",
            "model_id": "m",
        }
    ]
    (cell / "samples.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    df = collect_samples_from_run_dirs([run])
    out = summarize_sequential_metrics(df)
    # best = 2.0 * 1.0 = 2.0, actual 1.0 → regret 1.0 for one step
    assert out["by_cell"][0]["cumulative_regret_final_mean"] == pytest.approx(1.0)


def test_mcq_trajectory_mistake_propagation(tmp_path: Path) -> None:
    run = tmp_path / "run3"
    cell = run / "persistent_stakes__medqa__m__seed0"
    cell.mkdir(parents=True)
    rows = []
    for i, corr in enumerate([True, False, False, True]):
        rows.append(
            {
                "item_index": i,
                "sample_idx": 0,
                "correct": corr,
                "budget_before_current": 100.0 - i,
                "budget_after_current": 100.0 - i - 1,
                "model_id": "m",
            }
        )
    (cell / "samples.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    df = collect_samples_from_run_dirs([run])
    out = summarize_sequential_metrics(df)
    assert out["by_cell"][0]["n_sessions"] == 1
    # mistakes F,T,T,F — use eval.propagation formula on one session
    mistakes = [not x for x in [True, False, False, True]]
    from risk_reasoning.eval.propagation import mistake_propagation

    assert out["by_cell"][0]["mistake_propagation_mean"] == pytest.approx(
        mistake_propagation(mistakes)
    )
    flat = [1.0 if c else 0.0 for c in [True, False, False, True]]
    cr = cumulative_regret(flat, [1.0] * len(flat))
    assert out["by_cell"][0]["cumulative_regret_final_mean"] == pytest.approx(cr[-1])


def test_rollup_aggregates_two_seeds(tmp_path: Path) -> None:
    run = tmp_path / "run4"
    for seed in (1, 2):
        cell = run / f"bare_baseline__d__model__seed{seed}"
        cell.mkdir(parents=True)
        (cell / "samples.jsonl").write_text(
            json.dumps(
                {
                    "item_index": 0,
                    "sample_idx": 0,
                    "correct": True,
                    "budget_before_current": 10.0,
                    "budget_after_current": 9.0,
                    "model_id": "model",
                }
            )
            + "\n",
            encoding="utf-8",
        )
    df = collect_samples_from_run_dirs([run])
    out = summarize_sequential_metrics(df)
    agg = out["aggregated_by_condition_dataset_model"]
    assert len(agg) == 1
    key = next(iter(agg))
    assert agg[key]["n_cells"] == 2
