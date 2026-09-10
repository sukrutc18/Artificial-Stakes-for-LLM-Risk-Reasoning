"""Regret and reward metrics for the sequential budget env."""

from __future__ import annotations

from collections.abc import Sequence


def cumulative_reward(rewards: Sequence[float]) -> list[float]:
    """Return the running sum of ``rewards``."""
    result: list[float] = []
    total = 0.0
    for r in rewards:
        total += r
        result.append(total)
    return result


def cumulative_regret(
    rewards: Sequence[float], best_rewards: Sequence[float]
) -> list[float]:
    """Running sum of ``best_rewards[t] - rewards[t]``."""
    if len(rewards) != len(best_rewards):
        raise ValueError("rewards and best_rewards must have equal length")
    result: list[float] = []
    total = 0.0
    for r, b in zip(rewards, best_rewards):
        total += b - r
        result.append(total)
    return result


def budget_efficiency(final_budget: float, initial_budget: float) -> float:
    """Fraction of the initial budget retained at episode end."""
    if initial_budget == 0.0:
        return 0.0
    return final_budget / initial_budget


def recovery_rate(rewards: Sequence[float], bankruptcy_floor: float = 0.0) -> float:
    """Fraction of episodes that recover above the floor after a dip below it.

    Interprets ``rewards`` as the budget history of one episode. Returns 1.0 if
    the budget dipped to or below ``bankruptcy_floor`` and later recovered above
    it; 0.0 otherwise. Average this over multiple episodes for the overall rate.
    """
    dipped = False
    for val in rewards:
        if val <= bankruptcy_floor:
            dipped = True
        elif dipped:
            return 1.0
    return 0.0
