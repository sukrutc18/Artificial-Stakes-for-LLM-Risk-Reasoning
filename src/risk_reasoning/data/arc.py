"""ARC-Challenge loader (grade-school science MCQ requiring logical reasoning)."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from risk_reasoning.data.schemas import Answer, Choice, Question


def preprocess(cfg: dict[str, Any]) -> Path:
    """Download ARC-Challenge from HuggingFace, normalize, and write a JSONL cache.

    Writes one ``Question`` per item to
    ``${RISK_REASONING_DATA_DIR:-./data}/processed/<name>.jsonl``. Rows where
    ``answerKey`` is missing from the provided choice labels — or where the
    label / text arrays disagree — are skipped defensively and reported on
    stderr. Writes the full dataset; ``max_items`` is applied at load time.
    """
    from datasets import load_dataset

    ds = load_dataset(cfg["hf_path"], cfg["hf_config"], split=cfg["split"])

    out_path = _processed_path(cfg["name"])
    skipped = 0
    kept = 0

    with out_path.open("w", encoding="utf-8") as fh:
        for row in ds:
            choices_obj = row.get("choices") or {}
            labels = list(choices_obj.get("label", []) or [])
            texts = list(choices_obj.get("text", []) or [])
            answer_key = row.get("answerKey")
            if not labels or len(labels) != len(texts) or answer_key not in labels:
                skipped += 1
                continue
            question = Question(
                id=str(row["id"]),
                dataset="arc_challenge",
                task_type="mcq",
                prompt=str(row["question"]).strip(),
                choices=[
                    Choice(id=str(lbl), text=str(txt))
                    for lbl, txt in zip(labels, texts, strict=True)
                ],
                answer=Answer(choice_id=str(answer_key)),
                metadata={},
            )
            fh.write(json.dumps(question.model_dump(), ensure_ascii=False) + "\n")
            kept += 1

    print(
        f"[arc_challenge] wrote {kept} items to {out_path} (skipped {skipped} malformed)",
        file=sys.stderr,
    )
    return out_path


def load(cfg: dict[str, Any]) -> Iterable[Question]:
    """Load ARC-Challenge items and normalize to ``Question``.

    Expects ``cfg`` with ``hf_path='allenai/ai2_arc'`` and
    ``hf_config='ARC-Challenge'``. Reads from the on-disk JSONL cache produced
    by :func:`preprocess`, triggering preprocessing on the first call if the
    cache is missing. Yields up to ``cfg['max_items']`` items (``None`` = no
    cap), in the original HF dataset order.
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
