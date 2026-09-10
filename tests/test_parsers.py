"""Tests for tolerant model-output parsers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from risk_reasoning.data import ModelOutput
from risk_reasoning.parsers import extract_confidence, extract_cot_trace, parse_mcq_answer


def test_model_output_record() -> None:
    record = ModelOutput(
        question_id="q1",
        model_id="llama31_8b_instruct",
        condition="persistent_stakes",
        dataset="synthetic_ev",
        raw_output="Reasoning\nANSWER: B CONFIDENCE: 0.72",
        answer="B",
        confidence=0.72,
        cot_trace="Reasoning",
        budget_state={"initial": 1000.0, "current": 1100.0, "step": 1},
        seed=42,
    )

    assert record.model_id == "llama31_8b_instruct"
    assert record.budget_state is not None


def test_model_output_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        ModelOutput(
            question_id="q1",
            model_id="m",
            condition="bare",
            dataset="synthetic",
            raw_output="ANSWER: A CONFIDENCE: 1.5",
            confidence=1.5,
        )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ANSWER: A CONFIDENCE: 0.81", "A"),
        ("answer: b confidence: .63", "B"),
        ("The answer is C. Confidence: 70%", "C"),
        ("After comparing them, the answer is E.", "E"),
        ("A", "A"),
        ("The expected value favors (C) under these numbers.", "C"),
        ("ANSWER: 2 confidence: 0.9", "2"),
        ("The answer is 3.", "3"),
        ("Rambling about apples and budgets.\nANSWER: Z", None),
        ("I cannot determine this from the prompt.", None),
    ],
)
def test_parse_mcq_answer_styles(text: str, expected: str | None) -> None:
    assert parse_mcq_answer(text) == expected


def test_parse_mcq_valid_choices() -> None:
    assert parse_mcq_answer("The answer is E.", valid_choices=["A", "B", "C"]) is None
    assert parse_mcq_answer("The answer is c.", valid_choices=["A", "B", "C"]) == "C"


def test_parse_mcq_last_answer() -> None:
    text = "At first I thought ANSWER: A.\nRechecking the arithmetic.\nANSWER: C CONFIDENCE: 0.8"
    assert parse_mcq_answer(text) == "C"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ANSWER: A CONFIDENCE: 0.81", 0.81),
        ("Confidence: .63", None),
        ("I am 70% confident.", 0.7),
        ("confidence: 1", 1.0),
        ("No confidence stated.", None),
        ("Confidence: 140%", None),
    ],
)
def test_extract_confidence_styles(text: str, expected: float | None) -> None:
    parsed = extract_confidence(text)
    if expected is None:
        assert parsed is None
    else:
        assert parsed == pytest.approx(expected)

def test_cot_trace_before_answer() -> None:
    text = "Compare A and B.\nA has higher EV.\nANSWER: A CONFIDENCE: 0.9"
    assert extract_cot_trace(text) == "Compare A and B.\nA has higher EV."


def test_cot_trace_without_answer_marker() -> None:
    text = "First I compute both expected values.\nTherefore, final answer is B."
    assert extract_cot_trace(text) == text


def test_cot_trace_rambling_text() -> None:
    text = "I need to compare options.\nA pays more in expectation.\nThis suggests A."
    assert extract_cot_trace(text) == text


def test_cot_trace_single_line() -> None:
    assert extract_cot_trace("Probably A") == "Probably A"


def test_parse_base_model_ramble() -> None:
    text = """
    The problem appears to be about expected utility, not just the largest payoff.
    Let us calculate: option A is risky but has higher upside. Option B is safer.
    I initially want B, but the expected value calculation points the other way.
    ANSWER: A
    I am 66% confident.
    """

    assert parse_mcq_answer(text, valid_choices=["A", "B"]) == "A"
    assert extract_confidence(text) == pytest.approx(0.66)
    assert "expected utility" in (extract_cot_trace(text) or "")
