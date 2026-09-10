"""Small parsers for the model-output contract."""

from __future__ import annotations

import re
from collections.abc import Sequence

_ANSWER_RE = re.compile(
    r"(?:\bANSWER\s*:\s*|\bThe answer is\s*|\()(?P<answer>[A-Za-z]|\d{1,2})(?:\))?",
    re.IGNORECASE,
)
_BARE_ANSWER_RE = re.compile(r"^\s*(?P<answer>[A-Za-z]|\d{1,2})\.?\s*$")
_CONFIDENCE_DECIMAL_RE = re.compile(
    r"\bconfidence\s*:\s*(?P<confidence>0(?:\.\d+)?|1(?:\.0+)?)\b",
    re.IGNORECASE,
)
_CONFIDENCE_PERCENT_RE = re.compile(
    r"\bI am\s+(?P<confidence>\d+(?:\.\d+)?)%\s+confident\b",
    re.IGNORECASE,
)
_ANSWER_LINE_RE = re.compile(r"^\s*ANSWER\s*:", re.IGNORECASE)


def parse_mcq_answer(text: str, valid_choices: Sequence[str] | None = None) -> str | None:
    """Extract an MCQ answer from ``ANSWER: B``, ``The answer is B``, ``(B)``, or ``B``."""

    choices = _normalize_choices(valid_choices)
    if not text.strip() or not choices:
        return None

    for match in reversed(list(_ANSWER_RE.finditer(text))):
        answer = _normalize_answer(match.group("answer"), choices)
        if answer is not None:
            return answer

    match = _BARE_ANSWER_RE.match(text)
    if match is None:
        return None
    return _normalize_answer(match.group("answer"), choices)


def extract_confidence(text: str) -> float | None:
    """Extract confidence from ``confidence: 0.7`` or ``I am 70% confident``."""

    match = _CONFIDENCE_DECIMAL_RE.search(text)
    if match is not None:
        return float(match.group("confidence"))

    match = _CONFIDENCE_PERCENT_RE.search(text)
    if match is not None:
        confidence = float(match.group("confidence")) / 100.0
        if 0.0 <= confidence <= 1.0:
            return confidence

    return None


def extract_cot_trace(text: str) -> str | None:
    """Return everything before the final ``ANSWER:`` line, or the full text."""

    if not text.strip():
        return None

    lines = text.strip().splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        if _ANSWER_LINE_RE.match(lines[idx]):
            trace = "\n".join(lines[:idx]).strip()
            return trace or None
    return text.strip()


def _normalize_choices(valid_choices: Sequence[str] | None) -> set[str]:
    if valid_choices is None:
        return {"A", "B", "C", "D", "E", "1", "2", "3", "4", "5"}
    return {str(choice).strip().upper() for choice in valid_choices if str(choice).strip()}


def _normalize_answer(value: str, choices: set[str]) -> str | None:
    answer = value.strip().upper()
    return answer if answer in choices else None
