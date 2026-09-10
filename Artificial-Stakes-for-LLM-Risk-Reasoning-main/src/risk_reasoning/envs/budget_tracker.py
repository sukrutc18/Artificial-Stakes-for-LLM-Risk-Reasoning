"""Persistent budget state carried across items in the ``persistent_stakes`` condition.

The tracker is deliberately thin: it owns only state transitions and exposes a
snapshot via ``state()``. Prompt rendering is handled by
``risk_reasoning.prompts.framings``.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Literal

RewardFn = Literal["symmetric", "asymmetric_loss", "log_score"]


@dataclass(frozen=True)
class BudgetState:
    """Immutable snapshot of the carried budget at a single point in time."""
    initial: float
    current: float
    step: int
    history: tuple[float, ...] = field(default_factory=tuple)
    bankrupt: bool = False


@dataclass
class BudgetTracker:
    """Mutable tracker for a persistent virtual budget.

    Args:
        initial: Starting budget.
        stake_per_item: Magnitude of per-item reward / penalty.
        reward_fn: How to map (correct?, confidence) to a budget delta.
        floor: If current <= floor, tracker is marked bankrupt.
        visible: Whether the framing should display the budget to the model.
    """
    initial: float
    stake_per_item: float
    reward_fn: RewardFn = "symmetric"
    floor: float = 0.0
    visible: bool = True

    _current: float = field(init=False)
    _step: int = field(init=False, default=0)
    _history: list[float] = field(init=False, default_factory=list)
    _bankrupt: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self._current = self.initial
        self._history = [self.initial]

    def update(self, correct: bool, confidence: float | None = None) -> BudgetState:
        """Apply a per-item outcome and return the post-update BudgetState.

        Args:
            correct: Whether the model's answer was correct.
            confidence: Model's stated confidence in [0, 1]. Required for log_score.

        Returns:
            BudgetState snapshot after this update.
        """
        delta = self._calculate_reward(correct, confidence)
        self._current = max(self.floor, self._current + delta) 
        self._bankrupt = self._current <= self.floor
        self._history.append(self._current)
        self._step += 1
        return self.state() 

    def _calculate_reward(self, correct: bool, confidence: float | None) -> float:
        """Map (correct, confidence) to a budget delta.

        Args:
            correct: Whether the model's answer was correct.
            confidence: Model's stated confidence in [0, 1].

        Returns:
            Float delta to apply to the current budget.
        """
        if self.reward_fn == "symmetric":
            # Confidence is irrelevant here — clean baseline
            return self.stake_per_item if correct else -self.stake_per_item

        elif self.reward_fn == "asymmetric_loss":
            # scale loss by confidence so overconfident wrong answers hurt more
            conf = confidence if confidence is not None else 0.5
            if correct:
                return self.stake_per_item * 0.5          # modest gain
            else:
                return -self.stake_per_item * (1.0 + conf)  # overconfidence amplifies loss

        elif self.reward_fn == "log_score":
            # actual proper scoring rule — only way to maximize is honest reporting
            if confidence is None:
                raise ValueError("log_score requires a confidence value")
            c = 1.0 if correct else 0.0
            p = max(1e-9, min(1 - 1e-9, confidence))      # clip to avoid log(0)
            raw = c * math.log(p) + (1 - c) * math.log(1 - p)  # in [-inf, 0]
            # Normalize: log(0.5) ≈ -0.693 is the worst-calibrated score at p=0.5
            # Shift so that a perfectly calibrated agent scores near +stake
            normalized = (raw - math.log(0.5)) / abs(math.log(0.5))  # [-inf, 1]
            return self.stake_per_item * max(-2.0, normalized)        # floor at -2x stake

        else:
            raise ValueError(f"Unknown reward_fn: {self.reward_fn!r}")

    def state(self) -> BudgetState:
        """Return the current immutable snapshot."""
        return BudgetState(
            initial=self.initial,
            current=self._current,
            step=self._step,
            history=tuple(self._history),
            bankrupt=self._bankrupt,
        )

    def reset(self) -> None:
        """Restore tracker to initial conditions."""
        self._current = self.initial
        self._step = 0
        self._history = [self.initial]
        self._bankrupt = False

