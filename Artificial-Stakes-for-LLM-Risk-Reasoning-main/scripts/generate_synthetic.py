#!/usr/bin/env python
"""Generate the synthetic EV / probability dataset to ``data/synthetic/``.

Usage:
    python scripts/generate_synthetic.py --config configs/datasets/synthetic_ev.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic EV / probability questions."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a dataset YAML config (e.g. configs/datasets/synthetic_ev.yaml)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: ${RISK_REASONING_DATA_DIR:-./data}/synthetic/)",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    import yaml
    from risk_reasoning.data.synthetic import load

    with open(args.config, encoding="utf-8") as f:
        cfg: dict = yaml.safe_load(f) or {}

    out_root = Path(
        args.out_dir
        or os.environ.get("RISK_REASONING_DATA_DIR", "./data")
    )
    out_dir = out_root / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_name: str = cfg.get("name", "synthetic")
    out_path = out_dir / f"{dataset_name}.jsonl"

    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for item in load(cfg):
            fh.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")
            count += 1

    print(f"Wrote {count} items to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
