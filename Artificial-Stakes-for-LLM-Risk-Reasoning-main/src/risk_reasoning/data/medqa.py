"""MedQA loader (USMLE-style multiple choice medical decision making)."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from huggingface_hub import snapshot_download

if TYPE_CHECKING:
    from risk_reasoning.data.schemas import Question


def preprocess(cfg: dict[str, Any]) -> Path:
    """Download MedQA from HuggingFace, normalize, and write a JSONL cache.

    Downloads the requested MedQA config into
    ``${RISK_REASONING_DATA_DIR:-./data}/raw/medqa_hf`` via the Hub, then reads
    the requested split from the local Parquet shard(s). Writes one normalized
    ``Question`` per row to ``data/processed/<name>.jsonl``. Rows with missing
    question text, malformed options, or invalid ``answer_idx`` values are
    skipped defensively and reported on stderr. Writes the full split;
    ``max_items`` is applied at load time.
    """
    _validate_config(cfg)
    root = Path(os.environ.get("RISK_REASONING_DATA_DIR", "./data"))
    cache_dir = _configure_hf_cache(root)
    raw_dir = root / "raw" / "medqa_hf"
    config_name = str(cfg["hf_config"])
    split = str(cfg["split"])

    from datasets import load_dataset

    _download_config(cfg, raw_dir, cache_dir, config_name)
    split_files = _split_files(raw_dir / config_name, split)
    ds = load_dataset(
        "parquet",
        data_files={split: [str(path) for path in split_files]},
        split=split,
        cache_dir=str(cache_dir / "datasets"),
    )

    out_path = _processed_path(str(cfg["name"]))
    skipped = 0
    kept = 0

    with out_path.open("w", encoding="utf-8") as fh:
        for idx, row in enumerate(ds):
            prompt = str(row.get("question") or "").strip()
            answer_key = str(row.get("answer_idx") or "").strip()
            if not prompt or not answer_key:
                skipped += 1
                continue

            choices = _normalize_choices(row.get("options"))
            valid_keys = {choice["id"] for choice in choices}
            if not choices or answer_key not in valid_keys:
                skipped += 1
                continue

            source_id = f"{cfg['name']}-{split}-{idx}"
            record = {
                "id": source_id,
                "dataset": "medqa",
                "task_type": "mcq",
                "prompt": prompt,
                "choices": choices,
                "answer": {
                    "choice_id": answer_key,
                    "numeric_value": None,
                    "action": None,
                    "explanation": _optional_text(row.get("answer")),
                    "tolerance": None,
                },
                "metadata": {
                    "meta_info": row.get("meta_info"),
                    "answer_text": row.get("answer"),
                },
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(
        f"[medqa] wrote {kept} items to {out_path} (skipped {skipped} malformed)",
        file=sys.stderr,
    )
    return out_path


def load(cfg: dict[str, Any]) -> Iterable["Question"]:
    """Load MedQA items from the on-disk JSONL cache.

    Reads from the JSONL cache produced by :func:`preprocess`, triggering
    preprocessing on the first call if the cache is missing. Yields up to
    ``cfg['max_items']`` items (``None`` = no cap), in the original split
    order.
    """
    from risk_reasoning.data.schemas import Question

    _validate_config(cfg, required=("name",))
    path = _processed_path(str(cfg["name"]))
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


def _split_files(base: Path, split: str) -> list[Path]:
    patterns = [
        f"{split}-*.parquet",
        f"*{split}*.parquet",
        f"{split}/*.parquet",
    ]
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(base.glob(pattern)))
    if not matches:
        raise FileNotFoundError(f"No parquet file found for split={split!r} under {base}")
    return _dedupe_paths(matches)


def _download_config(
    cfg: dict[str, Any],
    raw_dir: Path,
    cache_dir: Path,
    config_name: str,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=str(cfg["hf_path"]),
        repo_type="dataset",
        revision=cfg.get("hf_revision"),
        cache_dir=str(cache_dir / "hub"),
        local_dir=str(raw_dir),
        allow_patterns=[f"{config_name}/*", f"{config_name}/**"],
    )


def _validate_config(
    cfg: dict[str, Any],
    *,
    required: tuple[str, ...] = ("name", "split", "hf_path", "hf_config"),
) -> None:
    for key in required:
        if key not in cfg:
            raise ValueError(f"[medqa] Missing required config key: {key!r}")


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _configure_hf_cache(data_root: Path) -> Path:
    cache_dir = data_root / "cache" / "huggingface"
    hub_cache = cache_dir / "hub"
    xet_cache = cache_dir / "xet"
    xdg_cache = data_root / "cache"

    hub_cache.mkdir(parents=True, exist_ok=True)
    xet_cache.mkdir(parents=True, exist_ok=True)

    # Keep preprocessing self-contained: sandboxed Windows environments often
    # cannot write to the default user-wide Hugging Face / Xet cache paths.
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_XET_CACHE"] = str(xet_cache)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return cache_dir


def _normalize_choices(raw: Any) -> list[dict[str, str]]:
    choices: list[dict[str, str]] = []
    if not isinstance(raw, list):
        return choices

    for option in raw:
        if not isinstance(option, dict):
            continue
        key = _optional_text(option.get("key"))
        value = _optional_text(option.get("value"))
        if key and value:
            choices.append({"id": key, "text": value})
    return choices


def _optional_text(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None
