#!/usr/bin/env python
"""Primary entry point: run an experiment from a config.

Usage:
    python scripts/run_experiment.py --config configs/experiments/full_matrix.yaml
    python scripts/run_experiment.py --config configs/experiments/full_matrix.yaml --dry-run
"""

from __future__ import annotations

import sys

from risk_reasoning.experiments.runner import main as runner_main


def main() -> int:
    runner_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
