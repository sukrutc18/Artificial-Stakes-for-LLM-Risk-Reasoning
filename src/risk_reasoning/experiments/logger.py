"""Append-only JSONL / parquet run logger.

Writes one ``samples.jsonl`` per cell plus a ``metrics.json`` summary and a
``config_snapshot.yaml`` alongside raw CoT traces under ``traces/``.

Checkpoint protocol
-------------------
``samples.jsonl`` is the data file; ``progress.jsonl`` is the commit index.
After writing K sample rows for an item, the runner calls ``commit_item`` which
fsyncs ``samples.jsonl`` then appends + fsyncs a single progress row.  On
restart, ``__init__`` reads ``progress.jsonl``, truncates ``samples.jsonl`` past
the last committed item (discarding any partial-K write from a crash), and
exposes ``completed_item_indices`` so the runner can skip those items.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from risk_reasoning.utils.io import ensure_dir, read_jsonl


class RunLogger:
    """One logger per experiment cell."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = ensure_dir(run_dir)
        self._samples_path = self.run_dir / "samples.jsonl"
        self._progress_path = self.run_dir / "progress.jsonl"
        self._metrics_path = self.run_dir / "metrics.json"
        self._config_path = self.run_dir / "config_snapshot.yaml"

        self.is_complete: bool = self._metrics_path.exists()

        self._committed: list[dict[str, Any]] = list(read_jsonl(self._progress_path))
        self.completed_question_ids: frozenset[str] = frozenset(
            row["question_id"] for row in self._committed
        )
        self.completed_item_indices: frozenset[int] = frozenset(
            row["item_index"] for row in self._committed
        )

        self._truncate_samples_to_committed()

        self._samples_fh = self._samples_path.open("a", encoding="utf-8", buffering=1)
        self._progress_fh = self._progress_path.open("a", encoding="utf-8", buffering=1)

    def log_sample(self, record: dict[str, Any]) -> None:
        self._samples_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def commit_item(self, question_id: str, item_index: int, n_samples: int) -> None:
        """Mark all K samples for an item as durable.

        Must be called after exactly K ``log_sample`` calls for this item.
        """
        self._samples_fh.flush()
        os.fsync(self._samples_fh.fileno())
        row = {"item_index": item_index, "question_id": question_id, "n_samples": n_samples}
        self._progress_fh.write(json.dumps(row) + "\n")
        self._progress_fh.flush()
        os.fsync(self._progress_fh.fileno())

    def replay_samples(self) -> Iterator[tuple[int, str, list[dict[str, Any]]]]:
        """Yield ``(item_index, question_id, [K rows])`` for each committed item.

        Yields in the original commit order so the runner can replay
        ``BudgetTracker.update`` calls in the correct sequence.
        """
        groups: dict[int, list[dict[str, Any]]] = {}
        for row in read_jsonl(self._samples_path):
            groups.setdefault(row["item_index"], []).append(row)
        for committed in self._committed:
            idx = committed["item_index"]
            yield idx, committed["question_id"], groups.get(idx, [])

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        self._metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def snapshot_config(self, cfg: dict[str, Any]) -> None:
        if self._config_path.exists():
            return
        try:
            import yaml
            self._config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        except ImportError:
            self._config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def close(self) -> None:
        for fh in (self._samples_fh, self._progress_fh):
            try:
                fh.flush()
                fh.close()
            except Exception:
                pass

    def _truncate_samples_to_committed(self) -> None:
        if not self._samples_path.exists():
            return
        kept: list[str] = []
        with self._samples_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.rstrip("\n")
                if not stripped.strip():
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    break
                if row.get("item_index") in self.completed_item_indices:
                    kept.append(stripped)
        tmp = self._samples_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for line in kept:
                f.write(line + "\n")
        os.replace(tmp, self._samples_path)
