"""Reasoning proxies extracted from CoT traces via the judge rubric.

These functions intentionally stay thin: the LLM judge owns scoring, and this
module exposes the individual proxy fields used by downstream aggregation.
"""

from __future__ import annotations

from risk_reasoning.eval.judge import judge_trace
from risk_reasoning.models.base import LLMClient


def count_alternatives(prompt: str, trace: str, judge: LLMClient) -> int:
    """Count distinct alternatives considered in ``trace``."""
    score = judge_trace(judge, prompt, trace)
    return score.alternatives_considered


def count_risk_factors(prompt: str, trace: str, judge: LLMClient) -> int:
    """Count distinct risk factors mentioned in ``trace``."""
    score = judge_trace(judge, prompt, trace)
    return score.risk_factors_enumerated


def logical_error_rate(prompt: str, trace: str, judge: LLMClient) -> float:
    """Return the inverse of judge-rated coherence."""
    score = judge_trace(judge, prompt, trace)
    return 1.0 - score.coherence
