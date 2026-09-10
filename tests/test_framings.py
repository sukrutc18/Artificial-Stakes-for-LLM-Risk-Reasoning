"""Tests for the four experimental framings."""

from __future__ import annotations

import pytest

from risk_reasoning.data.schemas import Answer, Choice, Question
from risk_reasoning.prompts.framings import (
    BareFraming,
    CotOnlyFraming,
    PersistentStakesFraming,
    SurfaceStakesFraming,
)


@pytest.fixture
def mcq_item() -> Question:
    return Question(
        id="t1",
        dataset="test",
        task_type="mcq",
        prompt="Which option has higher expected value?",
        choices=[Choice(id="A", text="50% $100"), Choice(id="B", text="$40 sure")],
        answer=Answer(choice_id="A"),
    )


@pytest.mark.xfail(reason="framings not yet implemented", strict=False)
def test_bare_has_no_stakes_language(mcq_item: Question) -> None:
    msgs = BareFraming().render(mcq_item)
    text = " ".join(m.content.lower() for m in msgs)
    assert "budget" not in text
    assert "stake" not in text


@pytest.mark.xfail(reason="framings not yet implemented", strict=False)
def test_surface_mentions_budget_once(mcq_item: Question) -> None:
    msgs = SurfaceStakesFraming(initial_budget=1000.0).render(mcq_item)
    combined = " ".join(m.content for m in msgs)
    assert "1000" in combined


@pytest.mark.xfail(reason="framings not yet implemented", strict=False)
def test_persistent_requires_state(mcq_item: Question) -> None:
    with pytest.raises((TypeError, AssertionError, ValueError)):
        PersistentStakesFraming().render(mcq_item, state=None)


@pytest.mark.xfail(reason="framings not yet implemented", strict=False)
def test_cot_only_has_scaffold_no_stakes(mcq_item: Question) -> None:
    msgs = CotOnlyFraming().render(mcq_item)
    text = " ".join(m.content.lower() for m in msgs)
    assert "step by step" in text
    assert "budget" not in text
