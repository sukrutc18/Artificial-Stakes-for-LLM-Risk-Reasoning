"""Seed every known RNG (python, numpy, torch) for reproducible runs."""

from __future__ import annotations

import random


def seed_all(seed: int) -> None:
    """Seed ``random``, ``numpy``, and ``torch`` (CPU + CUDA) with ``seed``."""
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
