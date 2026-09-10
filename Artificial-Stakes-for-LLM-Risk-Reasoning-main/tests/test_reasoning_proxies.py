"""Tests for judge-backed reasoning proxy helpers."""

from __future__ import annotations

from typing import Any

import pytest

from risk_reasoning.eval import reasoning_proxies
from risk_reasoning.eval.judge import JudgeScore
from risk_reasoning.models.base import Completion, LLMClient
from risk_reasoning.prompts.templates import Message


class FakeJudge(LLMClient):
    """Minimal judge client for tests; ``judge_trace`` is monkeypatched."""

    name = "fake-judge"

    def generate(
        self,
        messages: list[Message],
        *,
        n: int = 1,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
        return_logprobs: bool = False,
    ) -> list[Completion]:
        return []


@pytest.fixture
def fake_score() -> JudgeScore:
    return JudgeScore(
        coherence=0.75,
        reasoning_coverage=0.8,
        risk_factors_enumerated=3,
        alternatives_considered=2,
        raw={"source": "test"},
    )


def test_count_alternatives(monkeypatch: pytest.MonkeyPatch, fake_score: JudgeScore) -> None:
    calls: list[tuple[LLMClient, str, str, dict[str, Any] | None]] = []

    def fake_judge_trace(
        judge: LLMClient,
        prompt: str,
        trace: str,
        rubric: dict[str, Any] | None = None,
    ) -> JudgeScore:
        calls.append((judge, prompt, trace, rubric))
        return fake_score

    monkeypatch.setattr(reasoning_proxies, "judge_trace", fake_judge_trace)

    judge = FakeJudge()
    assert reasoning_proxies.count_alternatives("prompt", "trace", judge) == 2
    assert calls == [(judge, "prompt", "trace", None)]


def test_count_risk_factors(monkeypatch: pytest.MonkeyPatch, fake_score: JudgeScore) -> None:
    monkeypatch.setattr(reasoning_proxies, "judge_trace", lambda *_args, **_kwargs: fake_score)

    assert reasoning_proxies.count_risk_factors("prompt", "trace", FakeJudge()) == 3


def test_logical_error_rate(monkeypatch: pytest.MonkeyPatch, fake_score: JudgeScore) -> None:
    monkeypatch.setattr(reasoning_proxies, "judge_trace", lambda *_args, **_kwargs: fake_score)

    assert reasoning_proxies.logical_error_rate("prompt", "trace", FakeJudge()) == pytest.approx(0.25)
