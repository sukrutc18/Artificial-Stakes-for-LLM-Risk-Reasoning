#!/usr/bin/env python
"""Download + normalize ConvFinQA, ARC-Challenge, and MedQA into ``data/processed/``.

Usage:
    python scripts/prepare_real_data.py --datasets convfinqa arc_challenge medqa
"""

from __future__ import annotations

import argparse
import ast
import importlib
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main(argv: list[str] | None = None) -> int:
    """Run dataset preprocessing for one or more configured real datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset config names under configs/datasets/ without the .yaml suffix.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=REPO_ROOT / "configs" / "datasets",
        help="Directory containing dataset YAML configs.",
    )
    args = parser.parse_args(argv)

    for dataset_name in args.datasets:
        cfg = _load_cfg(args.config_dir / f"{dataset_name}.yaml")
        preprocess = _resolve_preprocess_fn(cfg["loader"])
        out_path = preprocess(cfg)
        print(f"[prepare_real_data] {dataset_name}: wrote {out_path}")

    return 0
def _load_cfg(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        return _load_simple_yaml(path)

    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(cfg, dict)
    return cfg


def _resolve_preprocess_fn(loader_path: str):
    module_path, _, _ = loader_path.partition(":")
    if not module_path:
        raise ValueError(f"Invalid loader path: {loader_path!r}")
    module = importlib.import_module(module_path)
    preprocess = getattr(module, "preprocess", None)
    if preprocess is None:
        raise AttributeError(f"{module_path} does not expose preprocess(cfg)")
    return preprocess


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_comment(raw_line)
        if not line.strip():
            continue

        indent = len(line) - len(line.lstrip(" "))
        key, sep, raw_value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"Unsupported config line in {path}: {raw_line!r}")

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        current = stack[-1][1]
        value = raw_value.strip()
        if not value:
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
            continue

        current[key] = _parse_scalar(value)

    return root


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    bracket_depth = 0
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "[" and not in_single and not in_double:
            bracket_depth += 1
        elif char == "]" and not in_single and not in_double and bracket_depth > 0:
            bracket_depth -= 1
        elif char == "#" and not in_single and not in_double and bracket_depth == 0:
            return line[:idx].rstrip()
    return line.rstrip()


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in _split_inline_list(inner)]

    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None

    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return ast.literal_eval(text)

    try:
        if any(ch in text for ch in (".", "e", "E")):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _split_inline_list(raw: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    in_single = False
    in_double = False
    bracket_depth = 0

    for char in raw:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "[" and not in_single and not in_double:
            bracket_depth += 1
        elif char == "]" and not in_single and not in_double and bracket_depth > 0:
            bracket_depth -= 1

        if char == "," and not in_single and not in_double and bracket_depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


if __name__ == "__main__":
    sys.exit(main())
