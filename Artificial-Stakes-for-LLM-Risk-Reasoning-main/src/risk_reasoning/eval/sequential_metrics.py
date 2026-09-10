"""Phase 5 — sequential metrics from experiment logs (JSONL → aggregates).

Computes, per experiment cell and optionally rolled up by condition × dataset × model:

- **Cumulative regret** (final value per session, then mean across sessions):
  sum_t (r*_t - r_t) with r* from oracle / best-action EV for env logs, or r*=1 for
  graded MCQ trajectories when only ``correct`` is present.
- **Budget efficiency** mean(final_budget / initial_budget) per session.
- **Recovery rate** mean over sessions (budget path dips to floor then recovers).
- **Mistake propagation** mean over sessions of
  P(mistake_{t+1}|mistake_t) - P(mistake_{t+1}) (see ``propagation``).

Sessions are **episodes** for ``sequential_budget``-style rows (``episode`` + ``step``),
or **one full cell trajectory** for standard MCQ runs (ordered ``item_index``, one row
per item via ``sample_idx == 0`` when that column exists).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from risk_reasoning.eval.propagation import mistake_propagation
from risk_reasoning.eval.regret import budget_efficiency, cumulative_regret, recovery_rate


def collect_samples_from_run_dirs(run_dirs: list[str | Path]) -> pd.DataFrame:
    """Load every ``samples.jsonl`` under *run_dirs* (rglob), tagging ``_cell_id``."""
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        rd = Path(run_dir)
        if not rd.is_dir():
            continue
        for sf in sorted(rd.rglob("samples.jsonl")):
            rel = sf.relative_to(rd)
            cell_id = rel.parts[0] if len(rel.parts) > 1 else "root"
            try:
                with sf.open(encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        rec["_cell_id"] = cell_id
                        rec["_run_dir"] = str(rd)
                        rows.append(rec)
            except OSError:
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "model_id" in df.columns and "model" not in df.columns:
        df["model"] = df["model_id"]
    return df


def parse_cell_id(cell_id: str) -> dict[str, Any]:
    """Parse ``{condition}__{dataset}__{model}__seed{N}`` from the runner layout."""
    m = re.match(r"^(.+?)__(.+?)__(.+?)__seed(\d+)$", cell_id)
    if not m:
        return {"condition": None, "dataset": None, "model": None, "seed": None}
    return {
        "condition": m.group(1),
        "dataset": m.group(2),
        "model": m.group(3),
        "seed": int(m.group(4)),
    }


def _dataset_cfg_from_cell_dir(cell_dir: Path) -> dict[str, Any] | None:
    """Return ``dataset_cfg`` from ``config_snapshot`` written by the runner."""
    for name in ("config_snapshot.yaml", "config_snapshot.json"):
        p = cell_dir / name
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            if p.suffix == ".yaml":
                import yaml  # type: ignore[import-untyped]

                cfg = yaml.safe_load(text)
            else:
                cfg = json.loads(text)
        except Exception:
            continue
        if not isinstance(cfg, dict):
            continue
        cell = cfg.get("cell")
        if not isinstance(cell, dict):
            continue
        extra = cell.get("extra")
        if not isinstance(extra, dict):
            continue
        dcfg = extra.get("dataset_cfg")
        if isinstance(dcfg, dict):
            return dcfg
    return None


def _best_ev_per_step(dataset_cfg: dict[str, Any]) -> float:
    """Expected reward of ``allocate_high`` (scale 2.0), matching ``sequential._best_ev``."""
    components = dataset_cfg.get("env", {}).get("reward_params", {}).get("components", [])
    if not components:
        return 0.2
    total_w = sum(float(c["weight"]) for c in components)
    ev_base = sum(float(c["weight"]) * float(c["mean"]) for c in components) / max(
        total_w, 1e-9
    )
    return 2.0 * ev_base


def _initial_budget_from_dataset_cfg(dataset_cfg: dict[str, Any], default: float) -> float:
    env = dataset_cfg.get("env")
    if isinstance(env, dict) and "initial_budget" in env:
        return float(env["initial_budget"])
    return default


def _session_metrics_env(
    rewards: list[float],
    budget_path: list[float],
    mistakes: list[bool],
    *,
    initial_budget: float,
    best_ev_per_step: float,
) -> dict[str, float]:
    best = [best_ev_per_step] * len(rewards)
    cr = cumulative_regret(rewards, best)
    final_budget = budget_path[-1] if budget_path else 0.0
    return {
        "cumulative_regret_final": float(cr[-1]) if cr else 0.0,
        "budget_efficiency": budget_efficiency(final_budget, initial_budget),
        "recovery_rate": recovery_rate(budget_path, 0.0),
        "mistake_propagation": mistake_propagation(mistakes),
    }


def _session_metrics_mcq(
    correct_flags: list[bool],
    budget_path: list[float],
    *,
    initial_budget: float,
) -> dict[str, float]:
    rewards = [1.0 if c else 0.0 for c in correct_flags]
    best = [1.0] * len(rewards)
    cr = cumulative_regret(rewards, best)
    mistakes = [not c for c in correct_flags]
    final_budget = budget_path[-1] if budget_path else 0.0
    return {
        "cumulative_regret_final": float(cr[-1]) if cr else 0.0,
        "budget_efficiency": budget_efficiency(final_budget, initial_budget),
        "recovery_rate": recovery_rate(budget_path, 0.0),
        "mistake_propagation": mistake_propagation(mistakes),
    }


def _compute_env_episode_rows(
    sub: pd.DataFrame,
    *,
    default_initial_budget: float,
    default_best_ev: float,
    cell_dir: Path | None,
) -> list[dict[str, float]]:
    dcfg: dict[str, Any] | None = None
    if cell_dir is not None and cell_dir.is_dir():
        dcfg = _dataset_cfg_from_cell_dir(cell_dir)
    best_ev = _best_ev_per_step(dcfg) if dcfg is not None else default_best_ev
    initial_budget = (
        _initial_budget_from_dataset_cfg(dcfg, default_initial_budget)
        if dcfg is not None
        else default_initial_budget
    )

    out: list[dict[str, float]] = []
    if "episode" not in sub.columns or "step" not in sub.columns:
        return out

    for _, ep_df in sub.groupby("episode", sort=True):
        ep_df = ep_df.sort_values("step")
        if "reward" not in ep_df.columns:
            continue
        rewards = ep_df["reward"].astype(float).tolist()
        if "budget" in ep_df.columns:
            bud = ep_df["budget"].astype(float).tolist()
        else:
            bud = []
        if bud:
            budget_path = [float(initial_budget)] + bud
        else:
            budget_path = [float(initial_budget)]

        gold = "allocate_high"
        if "action" in ep_df.columns:
            mistakes = [str(a).lower() != gold for a in ep_df["action"].tolist()]
        else:
            mistakes = [True] * len(rewards)

        out.append(
            _session_metrics_env(
                rewards,
                budget_path,
                mistakes,
                initial_budget=float(initial_budget),
                best_ev_per_step=best_ev,
            )
        )
    return out


def _compute_mcq_cell_sessions(
    sub: pd.DataFrame,
    *,
    default_initial_budget: float,
) -> list[dict[str, float]]:
    if "item_index" not in sub.columns or "correct" not in sub.columns:
        return []
    s = sub.copy()
    if "sample_idx" in s.columns:
        s = s[s["sample_idx"] == 0]
    if s.empty:
        return []
    s = s.sort_values("item_index")
    correct_flags = [bool(x) for x in s["correct"].tolist()]

    before_col = (
        "budget_before_current"
        if "budget_before_current" in s.columns
        else ("budget_before" if "budget_before" in s.columns else None)
    )
    after_col = (
        "budget_after_current"
        if "budget_after_current" in s.columns
        else ("budget_after" if "budget_after" in s.columns else None)
    )

    if after_col is not None and s[after_col].notna().any():
        bud_col = s[after_col].astype(float)
        initial_budget = float(default_initial_budget)
        if before_col is not None and pd.notna(s[before_col].iloc[0]):
            initial_budget = float(s[before_col].iloc[0])
        budget_path = [initial_budget] + bud_col.tolist()
    else:
        initial_budget = float(default_initial_budget)
        budget_path = [initial_budget, initial_budget]

    return [_session_metrics_mcq(correct_flags, budget_path, initial_budget=initial_budget)]


def summarize_sequential_metrics(
    df: pd.DataFrame,
    *,
    default_initial_budget: float = 1000.0,
    default_best_ev_per_step: float = 0.2,
) -> dict[str, Any]:
    """Return a JSON-serializable summary dict (by cell + rolled-up means)."""
    if df.empty:
        return {}

    by_cell: list[dict[str, Any]] = []
    if "_cell_id" not in df.columns:
        df = df.copy()
        df["_cell_id"] = "single_file"

    for cell_id, sub in df.groupby("_cell_id", sort=True):
        meta = parse_cell_id(str(cell_id))
        run_dir = sub["_run_dir"].iloc[0] if "_run_dir" in sub.columns else None
        cell_dir = Path(run_dir) / cell_id if run_dir and cell_id != "single_file" else None

        sessions: list[dict[str, float]] = []
        is_env = (
            "episode" in sub.columns
            and "step" in sub.columns
            and "reward" in sub.columns
        )
        if is_env:
            sessions = _compute_env_episode_rows(
                sub,
                default_initial_budget=default_initial_budget,
                default_best_ev=default_best_ev_per_step,
                cell_dir=cell_dir if cell_dir and cell_dir.is_dir() else None,
            )
        else:
            sessions = _compute_mcq_cell_sessions(
                sub, default_initial_budget=default_initial_budget
            )

        if not sessions:
            continue

        def mean_key(k: str) -> float:
            vals = [float(s[k]) for s in sessions]
            return sum(vals) / len(vals)

        row = {
            "cell_id": str(cell_id),
            "condition": meta.get("condition"),
            "dataset": meta.get("dataset"),
            "model": meta.get("model"),
            "seed": meta.get("seed"),
            "n_sessions": len(sessions),
            "cumulative_regret_final_mean": mean_key("cumulative_regret_final"),
            "budget_efficiency_mean": mean_key("budget_efficiency"),
            "recovery_rate_mean": mean_key("recovery_rate"),
            "mistake_propagation_mean": mean_key("mistake_propagation"),
        }
        by_cell.append(row)

    if not by_cell:
        return {}

    rollup: dict[str, dict[str, Any]] = {}
    for row in by_cell:
        key = f"{row.get('condition')}__{row.get('dataset')}__{row.get('model')}"
        rollup.setdefault(key, {"rows": []})
        rollup[key]["rows"].append(row)

    aggregated: dict[str, Any] = {}
    for key, pack in rollup.items():
        rs = pack["rows"]
        aggregated[key] = {
            "condition": rs[0].get("condition"),
            "dataset": rs[0].get("dataset"),
            "model": rs[0].get("model"),
            "n_cells": len(rs),
            "cumulative_regret_final_mean": sum(r["cumulative_regret_final_mean"] for r in rs)
            / len(rs),
            "budget_efficiency_mean": sum(r["budget_efficiency_mean"] for r in rs) / len(rs),
            "recovery_rate_mean": sum(r["recovery_rate_mean"] for r in rs) / len(rs),
            "mistake_propagation_mean": sum(r["mistake_propagation_mean"] for r in rs) / len(rs),
        }

    return {"by_cell": by_cell, "aggregated_by_condition_dataset_model": aggregated}


__all__ = [
    "collect_samples_from_run_dirs",
    "parse_cell_id",
    "summarize_sequential_metrics",
]
