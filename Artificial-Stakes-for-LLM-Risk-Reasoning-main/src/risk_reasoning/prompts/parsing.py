"""Parse model completions into structured answers and confidence scores.

Re-exports the real implementations from risk_reasoning.parsers and adds
parse_numeric_answer, which the runner needs for numeric-answer tasks.
"""

from __future__ import annotations

import re

from risk_reasoning.parsers import extract_cot_trace, extract_confidence, parse_mcq_answer

_NUMERIC_LABELLED_RE = re.compile(
    r"(?:ANSWER\s*:\s*|The answer is\s*|=\s*)"
    r"(?P<sign>[+-])?"
    r"(?P<num>\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)
_NUMERIC_BARE_RE = re.compile(
    r"^\s*(?P<sign>[+-])?(?P<num>\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$"
)


def parse_numeric_answer(text: str) -> float | None:
    """Extract a numeric value from model output.

    Tries labelled patterns (``ANSWER: 3.14``, ``The answer is -2e5``) first,
    then falls back to a bare number that occupies its own line.
    """
    if not text or not text.strip():
        return None

    for match in reversed(list(_NUMERIC_LABELLED_RE.finditer(text))):
        try:
            raw = match.group("num").replace(",", "")
            value = float(raw)
            if match.group("sign") == "-":
                value = -value
            return value
        except ValueError:
            continue

    match = _NUMERIC_BARE_RE.match(text.strip())
    if match:
        try:
            raw = match.group("num").replace(",", "")
            value = float(raw)
            if match.group("sign") == "-":
                value = -value
            return value
        except ValueError:
            pass

    return None


__all__ = [
    "extract_cot_trace",
    "extract_confidence",
    "parse_mcq_answer",
    "parse_numeric_answer",
]
