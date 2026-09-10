"""Mistake-propagation metric: P(mistake_{t+1} | mistake_t) minus baseline."""

from __future__ import annotations

from collections.abc import Sequence


def mistake_propagation(
    mistakes: Sequence[bool],
) -> float:
    """P(mistake_{t+1} | mistake_t) - P(mistake_{t+1}).

    Positive values indicate that mistakes cluster / compound in a sequence.
    """
    n = len(mistakes)
    if n < 2:
        return 0.0

    p_mistake = sum(mistakes[1:]) / (n - 1)

    n_prior_mistakes = sum(1 for m in mistakes[:-1] if m)
    if n_prior_mistakes == 0:
        return 0.0

    n_consecutive = sum(1 for i in range(n - 1) if mistakes[i] and mistakes[i + 1])
    p_mistake_given_mistake = n_consecutive / n_prior_mistakes

    return p_mistake_given_mistake - p_mistake
