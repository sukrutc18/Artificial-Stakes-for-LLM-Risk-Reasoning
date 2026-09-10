"""Purpose-built multi-step budget-allocation environment.

Gym-style API (``reset`` / ``step``) so the same loop code that drives MCQ items
can drive sequential episodes. Outcomes are sampled from a known distribution
specified in ``configs/datasets/sequential_budget.yaml``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from risk_reasoning.envs.budget_tracker import BudgetTracker
from risk_reasoning.prompts.templates import STAKES_PERSISTENT


@dataclass
class EnvStep:
    """Result of a single environment step."""

    observation: str
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class SequentialBudgetEnv:
    """Sequential allocation env with a persistent budget.

    The model picks one of four actions each turn. A reward is sampled from a
    gaussian mixture whose parameters come from the config. The budget updates
    via ``BudgetTracker`` and the new state is injected into the next prompt
    via ``render()``.

    Actions:
        allocate_low  — 0.5x reward scale (conservative)
        allocate_med  — 1.0x reward scale
        allocate_high — 2.0x reward scale (aggressive)
        skip          — 0 reward, budget unchanged

    Methods:
        reset(seed): start a new episode, return initial observation.
        step(action): apply action, return EnvStep.
        render(): budget state string for prompt injection.
    """

    ACTIONS = ("allocate_low", "allocate_med", "allocate_high", "skip")
    ACTION_SCALE = {"allocate_low": 0.5, "allocate_med": 1.0,
                    "allocate_high": 2.0, "skip": 0.0}

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        env_cfg = cfg.get("env", {})

        self.horizon: int = cfg.get("horizon", 20)
        self.initial_budget: float = env_cfg.get("initial_budget", 1000.0)
        self.bankruptcy: bool = env_cfg.get("bankruptcy", True)

        components = env_cfg.get("reward_params", {}).get("components", [
            {"weight": 0.7, "mean": 1.0, "std": 0.3},
            {"weight": 0.3, "mean": -2.0, "std": 1.0},
        ])
        self._weights = [c["weight"] for c in components]
        self._means   = [c["mean"]   for c in components]
        self._stds    = [c["std"]    for c in components]

        self.tracker = BudgetTracker(
            initial=self.initial_budget,
            stake_per_item=self.initial_budget * 0.05,
            reward_fn="symmetric",
            floor=0.0 if self.bankruptcy else -float("inf"),
        )

        self._rng: random.Random = random.Random()
        self._step: int = 0
        self._log: list[dict[str, Any]] = []

    def reset(self, seed: int | None = None) -> str:
        """Start a new episode. Returns the initial observation string."""
        self.tracker.reset()
        self._rng = random.Random(seed)
        self._step = 0
        self._log = []
        return self.render()

    def step(self, action: str) -> EnvStep:
        """Apply action, update budget, log turn, return EnvStep.

        Args:
            action: One of ``allocate_low``, ``allocate_med``,
                    ``allocate_high``, ``skip``.

        Returns:
            EnvStep with the next observation, reward, and done flags.
        """
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action!r}. Must be one of {self.ACTIONS}")

        if action == "skip":
            # skip: zero reward, tracker NOT updated, budget unchanged
            raw_reward = 0.0
        else:
            raw_reward = self._sample_reward() * self.ACTION_SCALE[action]
            self.tracker.update(correct=raw_reward > 0)

        self._step += 1
        state = self.tracker.state()

        self._log.append({
            "step":       self._step,
            "action":     action,
            "raw_reward": raw_reward,
            "budget":     state.current,
            "bankrupt":   state.bankrupt,
        })

        terminated = (self.bankruptcy and state.bankrupt) or self._step >= self.horizon

        return EnvStep(
            observation=self.render(),
            reward=raw_reward,
            terminated=terminated,
            truncated=False,
            info={
                "step":     self._step,
                "budget":   state.current,
                "bankrupt": state.bankrupt,
                "action":   action,
                "log":      list(self._log),
            },
        )

    def render(self) -> str:
        """Return the budget state string for injection into the persistent prompt."""
        state = self.tracker.state()
        return STAKES_PERSISTENT.format(
            current=state.current,
            initial=state.initial,
            step=state.step,
        )

    def get_state(self) -> dict[str, Any]:
        """Return a snapshot of the current env state."""
        state = self.tracker.state()
        return {
            "step":           self._step,
            "budget_current": state.current,
            "budget_initial": state.initial,
            "bankrupt":       state.bankrupt,
            "log":            list(self._log),
        }

    def _sample_reward(self) -> float:
        """Sample a raw reward from the gaussian mixture."""
        r = self._rng.random()
        cumulative = 0.0
        chosen = 0
        for i, w in enumerate(self._weights):
            cumulative += w
            if r < cumulative:
                chosen = i
                break
        return self._rng.gauss(self._means[chosen], self._stds[chosen])


def make_env(cfg: dict[str, Any]) -> SequentialBudgetEnv:
    """Factory referenced by ``configs/datasets/sequential_budget.yaml::loader``."""
    return SequentialBudgetEnv(cfg)
