"""Claude Sonnet as an automated judge for reasoning coherence and coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from risk_reasoning.models.base import LLMClient

import json, re
from risk_reasoning.prompts.templates import JUDGE_USER_TEMPLATE, JUDGE_SYSTEM, Message

@dataclass(frozen=True)
class JudgeScore:
    """Structured output of the judge rubric."""

    # Based on the assigned tasks from Claude Code 
    # (Each scored 1-5 with anchored descriptions)
    coherence: float
    reasoning_coverage: float
    risk_factors_enumerated: int
    alternatives_considered: int
    raw: dict[str, Any]

def _extract_json(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _clamp_float(x: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(x)))


def _nonnegative_int(x: Any) -> int:
    return max(0, int(x))


def judge_trace(
    judge: LLMClient,
    prompt: str,
    trace: str,
    rubric: dict[str, Any] | None = None,
) -> JudgeScore:
    """Score ``trace`` on the rubric using ``judge``.

    Args:
        judge: An :class:`LLMClient` pointed at Claude Sonnet.
        prompt: The original user prompt the trace was produced for.
        trace: The subject model's full CoT trace + final answer.
        rubric: Optional override for the default rubric (see config).

    Returns:
        A :class:`JudgeScore` with numeric sub-scores.
    """

    rubric = rubric or {}

    # NOTE: "trace" refers to Completion.text from the model output,
    # which includes full reasoning + final answer.
    user_prompt = JUDGE_USER_TEMPLATE.format(
        prompt=prompt,
        trace=trace,
        rubric=rubric,
    )

    completion = judge.generate(
        [
            Message(role="system", content=JUDGE_SYSTEM),
            Message(role="user", content=user_prompt),
        ],
        temperature=0.0,
        max_new_tokens=1024,
    )[0]

    parsed = _extract_json(completion.text)

    return JudgeScore(
        coherence=_clamp_float(parsed.get("coherence", 0)),
        reasoning_coverage=_clamp_float(parsed.get("reasoning_coverage", 0)),
        risk_factors_enumerated=_nonnegative_int(parsed.get("risk_factors_enumerated", 0)),
        alternatives_considered=_nonnegative_int(parsed.get("alternatives_considered", 0)),
        raw={
            "judge_output": parsed,
            "completion": completion.raw,
            "rubric": rubric,
        },
    )