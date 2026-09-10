"""Shared CLI helpers (arg parsing, Rich-based progress bars)."""

from __future__ import annotations

import argparse


def build_base_parser(description: str) -> argparse.ArgumentParser:
    """Return an ``argparse`` parser preloaded with common experiment flags."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser
