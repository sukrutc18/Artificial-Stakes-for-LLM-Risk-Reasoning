"""Base chat message types and shared prompt fragments.

System and user templates live here so the four framings in ``framings.py``
compose them rather than duplicating strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    """One chat message. The ``LLMClient`` interface accepts lists of these."""

    role: Role
    content: str


SYSTEM_COMMON = (
    "You are a careful decision-maker. For each problem, output your final "
    "answer plus a confidence in [0, 1] on the last line as `ANSWER: <x> "
    "CONFIDENCE: <p>`."
)

COT_SCAFFOLD = (
    "Think step by step. Enumerate at least two alternatives, list the main "
    "risk factors, then commit to a single answer."
)

STAKES_SURFACE = (
    "These decisions are treated as high-stakes with a virtual budget of "
    "{initial:.0f}. Decide as if your choices had real consequences."
)

STAKES_PERSISTENT = (
    "You have a virtual budget that carries across problems. It updates after "
    "each decision based on outcome. Current budget: {current:.2f} / "
    "{initial:.0f} (step {step}). Decide accordingly."
)

JUDGE_SYSTEM = (
    "You are an impartial evaluator of model reasoning traces. "
    "Score only the provided trace. Do not reward verbosity unless it improves reasoning quality. "
    "Do not infer hidden reasoning from the final answer. Score only what is explicitly written. "
    "Return only valid JSON."
)

JUDGE_USER_TEMPLATE = """ 
Original prompt: 
{prompt} 

Model reasoning trace and final answer: 
{trace} 

Evaluate the trace using this rubric: 
- coherence: float from 0 to 1. Is the reasoning logically consistent and easy to follow? Coherence is about logical consistency only, not correctness. 
- reasoning_coverage: float from 0 to 1. Does it address key facts, calculations, constraints, and decision-relevant considerations? Coverage should be 1.0 if the key idea needed to answer the question is present, even if the reasoning is brief.
- risk_factors_enumerated: integer count of distinct risk factors explicitly mentioned. Return an integer count, not a normalized score. Do not infer unstated risk factors. If the task has no natural risk component, risk_factors_enumerated should be 0. 
- alternatives_considered: integer count of distinct alternatives explicitly considered. Return an integer count, not a normalized score. Only count explicitly mentioned alternatives. 

Return only JSON with this exact schema: 
{{ 
    "coherence": <float>, 
    "reasoning_coverage": <float>, 
    "risk_factors_enumerated": <int>, 
    "alternatives_considered": <int>, 
    "justification": <string> 
}} 
"""