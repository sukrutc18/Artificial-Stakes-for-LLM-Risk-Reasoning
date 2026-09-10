"""Sequential-episode loop for the ``sequential_budget`` dataset.

Called by the experiment runner when ``dataset_cfg["task_type"] == "sequential"``.
After the MultiSampleandBudgetStateManagement and Sequential-Environment branches
merge, replace the ``raise NotImplementedError`` in ``runner._run_cell`` with:

    if dataset_cfg.get("task_type") == "sequential":
        return run_sequential_cell(cfg, cell, logger)

"""

from __future__ import annotations

import sys
from dataclasses import asdict
from typing import Any

from risk_reasoning.data.schemas import Answer, Choice, Question
from risk_reasoning.envs.budget_tracker import BudgetState, BudgetTracker
from risk_reasoning.eval.propagation import mistake_propagation
from risk_reasoning.eval.regret import budget_efficiency, cumulative_regret, recovery_rate
from risk_reasoning.experiments.conditions import ExperimentCell
from risk_reasoning.experiments.logger import RunLogger
from risk_reasoning.models.base import LLMClient
from risk_reasoning.prompts.framings import get_framing


# Expected value of the best action (allocate_high, scale=2.0) under the default
# gaussian mixture: 2.0 * (0.7*1.0 + 0.3*(-2.0)) = 2.0 * 0.1 = 0.2 per step.
_DEFAULT_BEST_EV_PER_STEP = 0.2


def run_sequential_cell(
    cfg: dict[str, Any],
    cell: ExperimentCell,
    logger: RunLogger,
    model: LLMClient,
) -> dict[str, Any]:
    """Run one sequential-environment cell and return aggregated metrics.

    Each episode starts with ``env.reset()``, runs until ``terminated`` or
    ``truncated``, and logs every step as a sample record.  Budget is tracked by
    the env's own ``BudgetTracker``; a separate tracker may optionally mirror it
    for the framing (persistent / surface conditions).
    """
    import importlib

    from risk_reasoning.utils.seeding import seed_all

    seed_all(cell.seed)

    dataset_cfg = cell.extra["dataset_cfg"]
    cond_cfg = cell.extra["condition_cfg"]

    # Load env via loader spec (e.g. "risk_reasoning.envs.sequential_budget:make_env")
    loader_spec: str = dataset_cfg["loader"]
    module_path, _, fn_name = loader_spec.partition(":")
    env = getattr(importlib.import_module(module_path), fn_name)(dataset_cfg)

    num_episodes: int = dataset_cfg.get("num_episodes", 10)
    actions: list[str] = dataset_cfg.get("env", {}).get(
        "actions", ["allocate_low", "allocate_med", "allocate_high", "skip"]
    )
    initial_budget: float = dataset_cfg.get("env", {}).get("initial_budget", 1000.0)
    best_ev_per_step: float = _best_ev(dataset_cfg)

    framing_name: str = cond_cfg.get("framing", "bare")
    framing = get_framing(framing_name, initial_budget=initial_budget)

    all_episode_rewards: list[list[float]] = []
    all_episode_budgets: list[list[float]] = []
    all_mistakes: list[bool] = []

    for ep in range(num_episodes):
        if ep in logger.completed_item_indices:
            continue
        obs = env.reset(seed=cell.seed + ep)
        done = False
        ep_rewards: list[float] = []
        ep_budgets: list[float] = [initial_budget]
        step_idx = 0

        while not done:
            state: BudgetState | None = None
            if hasattr(env, "tracker"):
                state = env.tracker.state()

            # Wrap observation as a pseudo-Question so the framing protocol works.
            pseudo_item = Question(
                id=f"ep{ep}_step{step_idx}",
                dataset=cell.dataset,
                task_type="sequential",
                prompt=obs,
                choices=[Choice(id=a, text=a) for a in actions],
                answer=Answer(action="allocate_high"),
                metadata={"episode": ep, "step": step_idx},
            )

            messages = framing.render(pseudo_item, state)

            try:
                completions = model.generate(
                    messages,
                    n=1,
                    temperature=cfg.get("generation", {}).get("temperature"),
                    top_p=cfg.get("generation", {}).get("top_p"),
                    max_new_tokens=cfg.get("generation", {}).get("max_new_tokens"),
                    seed=cell.seed,
                )
                raw_text = completions[0].text
            except Exception as exc:  # noqa: BLE001
                print(f"[sequential] generate failed ep={ep} step={step_idx}: {exc}", file=sys.stderr)
                raw_text = ""

            action = _parse_action(raw_text, actions)
            step_result = env.step(action)

            ep_rewards.append(step_result.reward)
            ep_budgets.append(step_result.info.get("budget", 0.0))
            is_mistake = action != "allocate_high"
            all_mistakes.append(is_mistake)

            logger.log_sample({
                "item_index": ep,
                "condition": cell.condition,
                "dataset": cell.dataset,
                "model_id": cell.model,
                "seed": cell.seed,
                "episode": ep,
                "step": step_idx,
                "action": action,
                "reward": step_result.reward,
                "budget": step_result.info.get("budget", 0.0),
                "bankrupt": step_result.info.get("bankrupt", False),
                "terminated": step_result.terminated,
                "raw_output": raw_text[:500],
            })

            obs = step_result.observation
            done = step_result.terminated or step_result.truncated
            step_idx += 1

        logger.commit_item(f"ep{ep}", ep, step_idx)
        all_episode_rewards.append(ep_rewards)
        all_episode_budgets.append(ep_budgets)

    flat_rewards = [r for ep in all_episode_rewards for r in ep]
    best_rewards = [best_ev_per_step] * len(flat_rewards)
    cum_regret = cumulative_regret(flat_rewards, best_rewards)

    ep_recovery = [recovery_rate(b, 0.0) for b in all_episode_budgets]

    final_budgets = [b[-1] for b in all_episode_budgets if b]
    mean_efficiency = (
        sum(budget_efficiency(b, initial_budget) for b in final_budgets) / len(final_budgets)
        if final_budgets
        else 0.0
    )

    return {
        "n_episodes": num_episodes,
        "mean_episode_reward": sum(flat_rewards) / max(num_episodes, 1),
        "cumulative_regret_final": cum_regret[-1] if cum_regret else 0.0,
        "budget_efficiency_mean": mean_efficiency,
        "recovery_rate": sum(ep_recovery) / max(len(ep_recovery), 1),
        "mistake_propagation": mistake_propagation(all_mistakes),
    }


# ─── helpers ──────────────────────────────────────────────────────────────────

def _parse_action(text: str, actions: list[str]) -> str:
    """Return the first action keyword found in ``text``, defaulting to first."""
    lower = text.lower()
    for action in actions:
        if action.lower() in lower:
            return action
    return actions[0]


def _best_ev(dataset_cfg: dict[str, Any]) -> float:
    """Expected reward of the best action (allocate_high, scale=2.0)."""
    components = dataset_cfg.get("env", {}).get("reward_params", {}).get("components", [])
    if not components:
        return _DEFAULT_BEST_EV_PER_STEP
    total_weight = sum(c["weight"] for c in components)
    ev_base = sum(c["weight"] * c["mean"] for c in components) / max(total_weight, 1e-9)
    return 2.0 * ev_base  # allocate_high scale = 2.0
