"""Tests for ``BudgetTracker`` and ``SequentialBudgetEnv``."""

from __future__ import annotations

import pytest

from risk_reasoning.envs.budget_tracker import BudgetTracker
from risk_reasoning.envs.sequential_budget import SequentialBudgetEnv


# ---------------------------------------------------------------------------
# Shared config helper
# ---------------------------------------------------------------------------

def _cfg(**overrides: object) -> dict:
    base = {
        "horizon": 20,
        "env": {
            "initial_budget": 1000.0,
            "bankruptcy": True,
            "reward_params": {
                "components": [
                    {"weight": 0.7, "mean": 1.0, "std": 0.3},
                    {"weight": 0.3, "mean": -2.0, "std": 1.0},
                ]
            },
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# BudgetTracker  (Shreekar)
# ---------------------------------------------------------------------------

def test_initial_state() -> None:
    tracker = BudgetTracker(initial=1000.0, stake_per_item=100.0)
    s = tracker.state()
    assert s.initial == 1000.0
    assert s.current == 1000.0
    assert s.step == 0
    assert s.bankrupt is False


def test_correct_answer_increases_budget() -> None:
    tracker = BudgetTracker(initial=1000.0, stake_per_item=100.0)
    new_state = tracker.update(correct=True)
    assert new_state.current > 1000.0


def test_reset_restores_initial() -> None:
    tracker = BudgetTracker(initial=1000.0, stake_per_item=100.0)
    tracker.update(correct=False)
    tracker.reset()
    assert tracker.state().current == 1000.0
    assert tracker.state().step == 0


def test_symmetric_wrong_decreases_budget() -> None:
    tracker = BudgetTracker(initial=100.0, stake_per_item=10.0, reward_fn="symmetric")
    s = tracker.update(correct=False)
    assert s.current == 90.0


def test_symmetric_correct_increases_budget() -> None:
    tracker = BudgetTracker(initial=100.0, stake_per_item=10.0, reward_fn="symmetric")
    s = tracker.update(correct=True)
    assert s.current == 110.0


def test_floor_clamps_budget() -> None:
    tracker = BudgetTracker(initial=10.0, stake_per_item=100.0, floor=0.0)
    s = tracker.update(correct=False)
    assert s.current == 0.0
    assert s.bankrupt is True


def test_log_score_raises_without_confidence() -> None:
    tracker = BudgetTracker(initial=100.0, stake_per_item=10.0, reward_fn="log_score")
    with pytest.raises(ValueError):
        tracker.update(correct=True, confidence=None)


def test_history_tracks_across_steps() -> None:
    tracker = BudgetTracker(initial=100.0, stake_per_item=10.0, reward_fn="symmetric")
    tracker.update(correct=True)
    tracker.update(correct=False)
    s = tracker.state()
    assert s.step == 2
    assert len(s.history) == 3
    assert s.history[0] == 100.0


def test_asymmetric_loss_overconfident_wrong() -> None:
    tracker = BudgetTracker(initial=100.0, stake_per_item=10.0, reward_fn="asymmetric_loss")
    s_low = tracker.update(correct=False, confidence=0.1)
    tracker.reset()
    s_high = tracker.update(correct=False, confidence=0.9)
    assert s_high.current < s_low.current


# ---------------------------------------------------------------------------
# SequentialBudgetEnv  (Shrey)
# ---------------------------------------------------------------------------

def test_reset_returns_string() -> None:
    env = SequentialBudgetEnv(_cfg())
    obs = env.reset(seed=0)
    assert isinstance(obs, str)
    assert len(obs) > 0


def test_reset_contains_initial_budget() -> None:
    env = SequentialBudgetEnv(_cfg())
    obs = env.reset(seed=0)
    assert "1000" in obs


def test_invalid_action_raises() -> None:
    env = SequentialBudgetEnv(_cfg())
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step("yolo")


def test_step_returns_env_step() -> None:
    from risk_reasoning.envs.sequential_budget import EnvStep
    env = SequentialBudgetEnv(_cfg())
    env.reset(seed=0)
    result = env.step("allocate_med")
    assert isinstance(result, EnvStep)


def test_skip_gives_zero_reward() -> None:
    env = SequentialBudgetEnv(_cfg())
    env.reset(seed=0)
    result = env.step("skip")
    assert result.reward == 0.0


def test_skip_does_not_change_budget() -> None:
    env = SequentialBudgetEnv(_cfg())
    env.reset(seed=0)
    before = env.tracker.state().current
    env.step("skip")
    after = env.tracker.state().current
    assert after == before


def test_terminated_at_horizon() -> None:
    env = SequentialBudgetEnv(_cfg(horizon=3))
    env.reset(seed=42)
    for _ in range(2):
        result = env.step("allocate_med")
        assert not result.terminated
    result = env.step("allocate_med")
    assert result.terminated


def test_terminated_on_bankruptcy() -> None:
    cfg = _cfg()
    cfg["env"]["initial_budget"] = 1.0
    env = SequentialBudgetEnv(cfg)
    env.reset(seed=0)
    for _ in range(20):
        result = env.step("allocate_high")
        if result.terminated:
            break
    assert result.terminated


def test_log_grows_each_step() -> None:
    env = SequentialBudgetEnv(_cfg())
    env.reset(seed=0)
    env.step("allocate_low")
    env.step("allocate_med")
    state = env.get_state()
    assert len(state["log"]) == 2


def test_log_contains_expected_keys() -> None:
    env = SequentialBudgetEnv(_cfg())
    env.reset(seed=0)
    env.step("allocate_med")
    entry = env.get_state()["log"][0]
    assert "step" in entry
    assert "action" in entry
    assert "raw_reward" in entry
    assert "budget" in entry
    assert "bankrupt" in entry


def test_reset_clears_log() -> None:
    env = SequentialBudgetEnv(_cfg())
    env.reset(seed=0)
    env.step("allocate_med")
    env.reset(seed=1)
    assert env.get_state()["log"] == []


def test_same_seed_reproducible() -> None:
    env = SequentialBudgetEnv(_cfg())
    env.reset(seed=7)
    r1 = env.step("allocate_med").reward
    env.reset(seed=7)
    r2 = env.step("allocate_med").reward
    assert r1 == r2


def test_render_updates_after_step() -> None:
    env = SequentialBudgetEnv(_cfg())
    obs_before = env.reset(seed=0)
    env.step("allocate_med")
    obs_after = env.render()
    assert obs_before != obs_after


def test_get_state_budget_matches_render() -> None:
    env = SequentialBudgetEnv(_cfg())
    env.reset(seed=0)
    env.step("allocate_med")
    state = env.get_state()
    render = env.render()
    assert str(round(state["budget_current"], 2)) in render


def test_three_turn_mock_session() -> None:
    """Verify budget number in render() matches tracker state each turn."""
    env = SequentialBudgetEnv(_cfg())
    env.reset(seed=42)
    for _ in range(3):
        result = env.step("allocate_med")
        tracker_budget = env.tracker.state().current
        render = env.render()
        assert f"{tracker_budget:.2f}" in render
        if result.terminated:
            break
