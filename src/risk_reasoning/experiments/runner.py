"""Top-level experiment loop.

Consumes an OmegaConf / Hydra config, expands it into :class:`ExperimentCell`
instances, runs each cell, and streams results to an append-only JSONL log via
:class:`RunLogger`. Exposed as ``risk-run`` entry point in ``pyproject.toml``.

Per-cell semantics
------------------
For each cell we instantiate exactly one :class:`BudgetTracker` when the
condition declares ``uses_budget: true``. The tracker is mutated **once per
item**, after collecting K independent samples — the budget reflects the
majority-vote answer and the mean confidence across those K samples. This
keeps ``BudgetState.step`` aligned with the per-item ``step {step}`` shown to
the model in ``STAKES_PERSISTENT``.

Aggregation
-----------
For each cell we report three accuracies (``single``, ``majority_vote``,
``best_of_n``) computed over all items × K samples. Per-sample records are
streamed to ``samples.jsonl`` so any of the three views can be reconstructed
offline.
"""

from __future__ import annotations

import importlib
import json
import warnings
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from risk_reasoning.data.schemas import Question
from risk_reasoning.envs.budget_tracker import BudgetState, BudgetTracker
from risk_reasoning.experiments.conditions import ExperimentCell, expand_matrix
from risk_reasoning.experiments.logger import RunLogger
from risk_reasoning.experiments.sample_record import SampleRecord
from risk_reasoning.models.base import LLMClient
from risk_reasoning.prompts.framings import Framing, get_framing
from risk_reasoning.prompts.parsing import (
    extract_confidence,
    parse_mcq_answer,
    parse_numeric_answer,
)
from risk_reasoning.utils.cli import build_base_parser
from risk_reasoning.utils.io import load_yaml


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run a full experiment from a parsed config dict.

    Returns a summary dict of per-cell metrics and the resolved run directory.
    """
    cells = expand_matrix(cfg)
    run_dir = _resolve_run_dir(cfg)
    summary: dict[str, dict[str, Any]] = {}
    cell_bar = tqdm(cells, desc="cells", unit="cell", dynamic_ncols=True)
    for cell in cell_bar:
        cell_id = _cell_id(cell)
        cell_bar.set_postfix_str(f"{cell.condition}/{cell.dataset}/{cell.model}/s{cell.seed}")
        cell_dir = _resolve_cell_dir(run_dir, cell)
        logger = RunLogger(cell_dir)
        try:
            if logger.is_complete:
                summary[cell_id] = json.loads(
                    (cell_dir / "metrics.json").read_text(encoding="utf-8")
                )
                continue
            logger.snapshot_config({**cfg, "cell": asdict(cell)})
            metrics = _run_cell(cfg, cell, logger, cell_bar)
            logger.log_metrics(metrics)
            summary[cell_id] = metrics
        finally:
            logger.close()
    return {"run_dir": str(run_dir), "cells": summary}


def main() -> None:
    """CLI entry point for ``risk-run``."""
    parser = build_base_parser("Run a risk-reasoning experiment.")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    if args.run_name is not None:
        cfg.setdefault("logging", {})["run_name"] = args.run_name
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.dry_run:
        cfg["max_items"] = 1
    run(cfg)


def _run_cell(
    cfg: dict[str, Any],
    cell: ExperimentCell,
    logger: RunLogger,
    cell_bar: Any = None,
) -> dict[str, Any]:
    condition_cfg = _get_condition_cfg(cell)
    dataset_cfg = _get_dataset_cfg(cell)
    model_cfg = _get_model_cfg(cell)

    if dataset_cfg.get("task_type") == "sequential":
        from risk_reasoning.experiments.sequential import run_sequential_cell
        client = _build_client(model_cfg)
        return run_sequential_cell(cfg, cell, logger, client)

    framing = _build_framing(condition_cfg)
    tracker = _build_tracker(condition_cfg)
    client = _build_client(model_cfg)

    gen_cfg = cfg.get("generation", {})
    max_items = cfg.get("max_items")
    n_samples = cell.n_samples

    items = list(_load_items(dataset_cfg, max_items))

    per_item_parsed: list[list[tuple[str | None, float | None]]] = []
    per_item_correct: list[list[bool | None]] = []
    items_done: list[Question] = []

    # --- Resume: replay committed items to restore per-item arrays and tracker ---
    items_by_idx = {i: q for i, q in enumerate(items)}
    for idx, qid, rows in logger.replay_samples():
        if idx not in items_by_idx or items_by_idx[idx].id != qid:
            raise RuntimeError(
                f"Resume mismatch at item_index {idx}: progress says {qid!r} "
                f"but current items list has {items_by_idx.get(idx)!r}. "
                "Dataset config likely changed since the prior run."
            )
        parsed_row: list[tuple[str | None, float | None]] = [
            (r["answer"], r["confidence"]) for r in rows
        ]
        correct_row: list[bool | None] = [r["correct"] for r in rows]
        per_item_parsed.append(parsed_row)
        per_item_correct.append(correct_row)
        items_done.append(items_by_idx[idx])
        if tracker is not None:
            chosen_answer = _modal_or_first(a for a, _ in parsed_row)
            item_correct = (
                _grade(chosen_answer, items_by_idx[idx]) if chosen_answer is not None else False
            )
            confs_present = [c for _, c in parsed_row if c is not None]
            item_confidence = (
                sum(confs_present) / len(confs_present) if confs_present else None
            )
            tracker.update(item_correct, item_confidence)
    # --- End resume ---

    n_correct = 0
    n_graded = 0
    item_bar = tqdm(
        items,
        desc=f"  {cell.condition[:4]}/{cell.dataset[:8]}",
        unit="item",
        dynamic_ncols=True,
        leave=False,
        mininterval=2.0,
    )
    for item_index, item in enumerate(item_bar):
        if item_index in logger.completed_item_indices:
            continue
        if tracker is not None and tracker.state().bankrupt:
            break
        state_before: BudgetState | None = tracker.state() if tracker is not None else None
        messages = framing.render(item, state_before)

        completions = client.generate(
            messages,
            n=n_samples,
            temperature=gen_cfg.get("temperature"),
            top_p=gen_cfg.get("top_p"),
            max_new_tokens=gen_cfg.get("max_new_tokens"),
            stop=gen_cfg.get("stop") or None,
            seed=cell.seed,
        )

        parsed: list[tuple[str | None, float | None]] = [
            _parse_completion(c.text, item) for c in completions
        ]
        target = _target_str(item)
        per_sample_correct: list[bool | None] = [
            _grade(ans, item) if ans is not None else None for ans, _ in parsed
        ]

        state_after: BudgetState | None = None
        if tracker is not None:
            chosen_answer = _modal_or_first(ans for ans, _ in parsed)
            item_correct = (
                _grade(chosen_answer, item) if chosen_answer is not None else False
            )
            confs_present = [c for _, c in parsed if c is not None]
            item_confidence = (
                sum(confs_present) / len(confs_present) if confs_present else None
            )
            state_after = tracker.update(item_correct, item_confidence)

        for k, (comp, (ans, conf), corr) in enumerate(
            zip(completions, parsed, per_sample_correct, strict=True)
        ):
            record = SampleRecord(
                condition=cell.condition,
                dataset=cell.dataset,
                model_id=cell.model,
                seed=cell.seed,
                framing=framing.name,
                question_id=item.id,
                item_index=item_index,
                sample_idx=k,
                raw_output=comp.text,
                answer=ans,
                confidence=conf,
                target=target,
                correct=corr,
                budget_before_current=state_before.current if state_before else None,
                budget_after_current=state_after.current if state_after else None,
                budget_before_step=state_before.step if state_before else None,
                budget_after_step=state_after.step if state_after else None,
                reward_delta=(
                    state_after.current - state_before.current
                    if state_before is not None and state_after is not None
                    else None
                ),
                bankrupt_after=state_after.bankrupt if state_after else None,
                finish_reason=comp.finish_reason,
                prompt_tokens=comp.prompt_tokens,
                completion_tokens=comp.completion_tokens,
            )
            logger.log_sample(record.to_dict())

        logger.commit_item(item.id, item_index, n_samples)

        per_item_parsed.append(parsed)
        per_item_correct.append(per_sample_correct)
        items_done.append(item)

        n_graded += 1
        if any(c for c in per_sample_correct if c is not None):
            n_correct += 1
        acc = n_correct / n_graded if n_graded else 0.0
        budget_str = (
            f" budget={tracker.state().current:.0f}" if tracker is not None else ""
        )
        item_bar.set_postfix_str(f"acc={acc:.2f}{budget_str}")
        if cell_bar is not None:
            cell_bar.set_postfix_str(
                f"{cell.condition[:6]}/{cell.dataset[:8]} acc={acc:.2f} [{n_graded}/{len(items)}]"
            )

        if tracker is not None and tracker.state().bankrupt:
            break

    final_state = tracker.state() if tracker is not None else None
    metrics: dict[str, Any] = {
        "single": _single_accuracy(per_item_correct),
        "majority_vote": _majority_vote_accuracy(per_item_parsed, items_done),
        "best_of_n": _best_of_n_accuracy(per_item_parsed, items_done),
        "n_items": len(items_done),
        "n_samples": n_samples,
        "bankrupt": final_state.bankrupt if final_state is not None else False,
        "final_budget": final_state.current if final_state is not None else None,
    }
    return metrics


def _single_accuracy(per_item_correct: list[list[bool | None]]) -> float:
    total = 0
    hits = 0
    for row in per_item_correct:
        for c in row:
            total += 1
            if c:
                hits += 1
    return hits / total if total > 0 else 0.0


def _majority_vote_accuracy(
    per_item_parsed: list[list[tuple[str | None, float | None]]],
    items: list[Question],
) -> float:
    if not items:
        return 0.0
    hits = 0
    for parsed_row, item in zip(per_item_parsed, items, strict=True):
        chosen = _modal_or_first(ans for ans, _ in parsed_row)
        if chosen is not None and _grade(chosen, item):
            hits += 1
    return hits / len(items)


def _best_of_n_accuracy(
    per_item_parsed: list[list[tuple[str | None, float | None]]],
    items: list[Question],
) -> float:
    if not items:
        return 0.0
    hits = 0
    for parsed_row, item in zip(per_item_parsed, items, strict=True):
        if not parsed_row:
            continue
        best_idx = max(
            range(len(parsed_row)),
            key=lambda i: parsed_row[i][1] if parsed_row[i][1] is not None else -1.0,
        )
        chosen = parsed_row[best_idx][0]
        if chosen is not None and _grade(chosen, item):
            hits += 1
    return hits / len(items)


def _parse_completion(text: str, q: Question) -> tuple[str | None, float | None]:
    confidence = extract_confidence(text)
    if q.task_type == "mcq":
        choices = [c.id for c in q.choices] if q.choices else None
        return parse_mcq_answer(text, valid_choices=choices), confidence
    if q.task_type == "numeric":
        value = parse_numeric_answer(text)
        return (None if value is None else repr(value)), confidence
    return (None, confidence)


def _build_framing(condition_cfg: dict[str, Any]) -> Framing:
    framing_name = condition_cfg["framing"]
    kwargs: dict[str, Any] = {}
    if framing_name == "surface":
        kwargs["initial_budget"] = condition_cfg.get("budget", {}).get("initial", 0.0)
    return get_framing(framing_name, **kwargs)


def _build_tracker(condition_cfg: dict[str, Any]) -> BudgetTracker | None:
    if not condition_cfg.get("uses_budget", False):
        return None
    b = condition_cfg["budget"]
    return BudgetTracker(
        initial=float(b["initial"]),
        stake_per_item=float(b["stake_per_item"]),
        reward_fn=b.get("reward_fn", "symmetric"),
        floor=float(b.get("floor", 0.0)),
        visible=bool(b.get("visible", True)),
    )


def _build_client(model_cfg: dict[str, Any]) -> LLMClient:
    backend = model_cfg.get("backend") or "vllm"
    if backend == "vllm":
        from risk_reasoning.models.vllm_client import VLLMClient

        return VLLMClient(model_cfg)
    if backend == "ollama":
        warnings.warn(
            "backend 'ollama' is deprecated; routing to VLLMClient. "
            "Switch your model YAML to backend: vllm.",
            DeprecationWarning,
            stacklevel=2,
        )
        from risk_reasoning.models.vllm_client import VLLMClient

        return VLLMClient(model_cfg)
    if backend == "hf":
        from risk_reasoning.models.hf_client import HFClient

        return HFClient(model_cfg)
    if backend == "anthropic":
        from risk_reasoning.models.anthropic_client import AnthropicClient

        return AnthropicClient(model_cfg)
    raise ValueError(f"unknown model backend: {backend!r}")


def _load_items(dataset_cfg: dict[str, Any], max_items: int | None) -> Iterator[Question]:
    loader_path = dataset_cfg["loader"]
    module_path, _, attr = loader_path.partition(":")
    if not attr:
        raise ValueError(
            f"dataset loader {loader_path!r} must be of the form 'module:callable'"
        )
    module = importlib.import_module(module_path)
    loader = getattr(module, attr)
    items: Iterable[Question] = loader(dataset_cfg)
    cap = max_items if max_items is not None else dataset_cfg.get("max_items")
    for i, item in enumerate(items):
        if cap is not None and i >= cap:
            break
        yield item


def _target_str(q: Question) -> str:
    a = q.answer
    if q.task_type == "mcq":
        if a.choice_id is None:
            raise ValueError(f"mcq question {q.id!r} missing choice_id")
        return a.choice_id
    if q.task_type == "numeric":
        if a.numeric_value is None:
            raise ValueError(f"numeric question {q.id!r} missing numeric_value")
        return repr(a.numeric_value)
    if q.task_type == "sequential":
        if a.action is None:
            raise ValueError(f"sequential question {q.id!r} missing action")
        return a.action
    raise ValueError(f"unknown task_type: {q.task_type!r}")


def _grade(parsed: str | None, q: Question) -> bool:
    if parsed is None:
        return False
    if q.task_type == "mcq":
        target = q.answer.choice_id or ""
        return parsed.strip().upper() == target.strip().upper()
    if q.task_type == "numeric":
        try:
            value = float(parsed)
        except ValueError:
            return False
        target_value = q.answer.numeric_value
        if target_value is None:
            return False
        tol = q.answer.tolerance if q.answer.tolerance is not None else 0.0
        scale = max(abs(target_value), 1.0)
        return abs(value - target_value) <= tol * scale
    return False


def _modal_or_first(values: Iterable[str | None]) -> str | None:
    counts: Counter[str] = Counter()
    first_index: dict[str, int] = {}
    for i, v in enumerate(values):
        if v is None:
            continue
        counts[v] += 1
        first_index.setdefault(v, i)
    if not counts:
        return None
    return min(counts.items(), key=lambda kv: (-kv[1], first_index[kv[0]]))[0]


def _resolve_run_dir(cfg: dict[str, Any]) -> Path:
    logging_cfg = cfg.get("logging", {})
    base = Path(logging_cfg.get("results_dir", "./results"))
    run_name = logging_cfg.get("run_name") or "default"
    return base / "runs" / run_name


def _resolve_cell_dir(run_dir: Path, cell: ExperimentCell) -> Path:
    return run_dir / _cell_id(cell)


def _cell_id(cell: ExperimentCell) -> str:
    return f"{cell.condition}__{cell.dataset}__{cell.model}__seed{cell.seed}"


def _get_condition_cfg(cell: ExperimentCell) -> dict[str, Any]:
    return _require_subcfg(cell, "condition_cfg")


def _get_dataset_cfg(cell: ExperimentCell) -> dict[str, Any]:
    return _require_subcfg(cell, "dataset_cfg")


def _get_model_cfg(cell: ExperimentCell) -> dict[str, Any]:
    return _require_subcfg(cell, "model_cfg")


def _require_subcfg(cell: ExperimentCell, key: str) -> dict[str, Any]:
    sub = cell.extra.get(key)
    if not isinstance(sub, dict):
        raise KeyError(
            f"ExperimentCell.extra[{key!r}] must be a resolved config dict; "
            "expand_matrix is responsible for populating it"
        )
    return sub


__all__ = ["main", "run"]
