"""Decision accuracy: single-sample, majority-vote, best-of-n by confidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence


def accuracy(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Plain single-sample accuracy."""
    if len(predictions) != len(targets):
        raise ValueError(
            f"predictions ({len(predictions)}) and targets ({len(targets)}) length mismatch"
        )
    if not predictions:
        return 0.0
    hits = sum(1 for p, t in zip(predictions, targets, strict=True) if _match(p, t))
    return hits / len(predictions)


def majority_vote_accuracy(
    samples: Sequence[Sequence[str]], targets: Sequence[str]
) -> float:
    """Accuracy when each item's prediction is the modal sample."""
    if len(samples) != len(targets):
        raise ValueError(
            f"samples ({len(samples)}) and targets ({len(targets)}) length mismatch"
        )
    if not samples:
        return 0.0
    hits = 0
    for item_samples, target in zip(samples, targets, strict=True):
        chosen = _modal(item_samples)
        if chosen is not None and _match(chosen, target):
            hits += 1
    return hits / len(samples)


def best_of_n_accuracy(
    samples: Sequence[Sequence[str]],
    confidences: Sequence[Sequence[float]],
    targets: Sequence[str],
) -> float:
    """Accuracy when each item's prediction is the most confident sample."""
    if not (len(samples) == len(confidences) == len(targets)):
        raise ValueError(
            "samples, confidences, and targets must all have the same length "
            f"(got {len(samples)}, {len(confidences)}, {len(targets)})"
        )
    if not samples:
        return 0.0
    hits = 0
    for item_samples, item_confs, target in zip(samples, confidences, targets, strict=True):
        if not item_samples:
            continue
        if len(item_samples) != len(item_confs):
            raise ValueError(
                "per-item samples and confidences length mismatch "
                f"({len(item_samples)} vs {len(item_confs)})"
            )
        best_idx = max(range(len(item_samples)), key=lambda i: item_confs[i])
        if _match(item_samples[best_idx], target):
            hits += 1
    return hits / len(samples)


def _match(prediction: str, target: str) -> bool:
    return prediction.strip().upper() == target.strip().upper()


def _modal(values: Sequence[str]) -> str | None:
    if not values:
        return None
    counts: Counter[str] = Counter()
    first_index: dict[str, int] = {}
    for i, v in enumerate(values):
        counts[v] += 1
        first_index.setdefault(v, i)
    return min(counts.items(), key=lambda kv: (-kv[1], first_index[kv[0]]))[0]
