"""ConvFinQA loader (financial reasoning over conversational + tabular input)."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from risk_reasoning.data.schemas import Answer, Question


def preprocess(cfg: dict[str, Any]) -> Path:
    """Download ConvFinQA from HuggingFace, normalize, and write a JSONL cache.

    Writes one ``Question`` per conversation to
    ``${RISK_REASONING_DATA_DIR:-./data}/processed/<name>.jsonl``. Honors
    ``cfg['preprocessing']['flatten_turns']`` and
    ``cfg['preprocessing']['include_tables_as_markdown']`` when reconstructing a
    prompt from raw fields. Rows whose answer cannot be parsed as a number are
    skipped; the count is reported on stderr. Writes the full dataset —
    ``max_items`` is applied at load time.
    """
    from datasets import load_dataset

    ds = load_dataset(cfg["hf_path"], split=cfg["split"])

    pre = cfg.get("preprocessing") or {}
    flatten_turns = bool(pre.get("flatten_turns", True))
    tables_as_md = bool(pre.get("include_tables_as_markdown", True))
    tolerance = (cfg.get("metrics") or {}).get("tolerance")

    out_path = _processed_path(cfg["name"])
    skipped = 0
    kept = 0

    with out_path.open("w", encoding="utf-8") as fh:
        for idx, row in enumerate(ds):
            numeric = _parse_number(row.get("answer"))
            if numeric is None:
                skipped += 1
                continue
            prompt = _build_convfinqa_prompt(
                row, flatten_turns=flatten_turns, tables_as_md=tables_as_md
            )
            question = Question(
                id=str(row.get("id") or idx),
                dataset="convfinqa",
                task_type="numeric",
                prompt=prompt,
                choices=None,
                answer=Answer(numeric_value=numeric, tolerance=tolerance),
                metadata={
                    "source_id": row.get("id"),
                    "raw_answer": row.get("answer"),
                    "has_raw_table": isinstance(row.get("table"), list)
                    and bool(row.get("table")),
                },
            )
            fh.write(json.dumps(question.model_dump(), ensure_ascii=False) + "\n")
            kept += 1

    print(
        f"[convfinqa] wrote {kept} items to {out_path} (skipped {skipped} unparseable)",
        file=sys.stderr,
    )
    return out_path


def load(cfg: dict[str, Any]) -> Iterable[Question]:
    """Load ConvFinQA items from HuggingFace and normalize to ``Question``.

    Expects ``cfg`` to contain ``hf_path`` and ``split``. Flattens multi-turn
    conversations into a single prompt when ``preprocessing.flatten_turns`` is
    set. Reads from the on-disk JSONL cache produced by :func:`preprocess`,
    triggering preprocessing on the first call if the cache is missing. Yields
    up to ``cfg['max_items']`` items (``None`` = no cap), in the original HF
    dataset order.
    """
    path = _processed_path(cfg["name"])
    if not path.exists():
        preprocess(cfg)

    max_items = cfg.get("max_items")
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if max_items is not None and idx >= max_items:
                break
            if not line.strip():
                continue
            yield Question(**json.loads(line))


def _processed_path(name: str) -> Path:
    root = Path(os.environ.get("RISK_REASONING_DATA_DIR", "./data"))
    out = root / "processed" / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _build_convfinqa_prompt(
    row: dict[str, Any], *, flatten_turns: bool, tables_as_md: bool
) -> str:
    # FLARE-ConvFinQA ships a pre-flattened ``query`` with table + prior turns +
    # final question; prefer it verbatim to avoid duplicating context.
    query = row.get("query")
    if isinstance(query, str) and query.strip():
        return query.strip()

    parts: list[str] = []

    raw_table = row.get("table")
    if tables_as_md and isinstance(raw_table, list) and raw_table:
        parts.append(_table_to_markdown(raw_table))

    context = row.get("text") or row.get("context")
    if context:
        parts.append(str(context).strip())

    turns = row.get("conversation") or row.get("turns")
    if flatten_turns and isinstance(turns, list) and turns:
        flattened: list[str] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            q = turn.get("question") or turn.get("q")
            a = turn.get("answer") or turn.get("a")
            if q:
                flattened.append(f"Q: {q}")
            if a is not None:
                flattened.append(f"A: {a}")
        if flattened:
            parts.append("\n".join(flattened))

    last_q = row.get("question")
    if last_q:
        parts.append(f"Question: {str(last_q).strip()}")

    return "\n\n".join(p for p in parts if p)


def _table_to_markdown(rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    header = [str(c).strip() for c in rows[0]]
    width = len(header)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        cells = [str(c).strip() for c in r]
        if len(cells) < width:
            cells = cells + [""] * (width - len(cells))
        else:
            cells = cells[:width]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _parse_number(raw: Any) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in {"n/a", "na", "none", "null"}:
        return None
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    # Accounting-style negatives: "(1234)" means -1234.
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except ValueError:
        return None
