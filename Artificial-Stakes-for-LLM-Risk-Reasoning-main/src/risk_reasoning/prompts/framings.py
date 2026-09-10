"""The four experimental framings, implementing a single :class:`Framing` protocol.

Each framing renders the same ``Question`` into a list of ``Message`` objects,
optionally incorporating a :class:`BudgetState` for the ``persistent`` condition.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from risk_reasoning.data.schemas import Question
from risk_reasoning.envs.budget_tracker import BudgetState
from risk_reasoning.prompts.templates import (
    COT_SCAFFOLD,
    STAKES_PERSISTENT,
    STAKES_SURFACE,
    SYSTEM_COMMON,
    Message,
)

FramingName = Literal["bare", "surface", "persistent", "cot_only"]


@runtime_checkable
class Framing(Protocol):
    """Renders a ``Question`` (and optional ``BudgetState``) into chat messages."""

    name: FramingName

    def render(
        self,
        item: Question,
        state: BudgetState | None = None,
    ) -> list[Message]: ...


def _format_user_message(item: Question) -> str:
    """Render a Question into the user-turn text shown to the model."""
    parts = [item.prompt]
    if item.choices:
        parts.append("")
        for choice in item.choices:
            if choice.id != choice.text:
                parts.append(f"{choice.id}) {choice.text}")
            else:
                parts.append(f"- {choice.id}")
    return "\n".join(parts)


class BareFraming:
    """Condition 1: raw task, no framing, no CoT, no stakes."""

    name: FramingName = "bare"

    def render(self, item: Question, state: BudgetState | None = None) -> list[Message]:
        return [
            Message(role="system", content=SYSTEM_COMMON),
            Message(role="user", content=_format_user_message(item)),
        ]


class SurfaceStakesFraming:
    """Condition 2: one-shot stakes mention in the system prompt."""

    name: FramingName = "surface"

    def __init__(self, initial_budget: float) -> None:
        self.initial_budget = initial_budget

    def render(self, item: Question, state: BudgetState | None = None) -> list[Message]:
        system = (
            SYSTEM_COMMON
            + "\n\n"
            + STAKES_SURFACE.format(initial=self.initial_budget)
        )
        return [
            Message(role="system", content=system),
            Message(role="user", content=_format_user_message(item)),
        ]


class PersistentStakesFraming:
    """Condition 3: persistent, visible, updating budget across items."""

    name: FramingName = "persistent"

    def render(self, item: Question, state: BudgetState | None = None) -> list[Message]:
        if state is None:
            raise ValueError(
                "PersistentStakesFraming.render() requires a BudgetState; "
                "pass the tracker's current state() snapshot."
            )
        stakes = STAKES_PERSISTENT.format(
            current=state.current,
            initial=state.initial,
            step=state.step,
        )
        system = SYSTEM_COMMON + "\n\n" + stakes
        return [
            Message(role="system", content=system),
            Message(role="user", content=_format_user_message(item)),
        ]


class CotOnlyFraming:
    """Condition 4: CoT scaffold without any stakes language."""

    name: FramingName = "cot_only"

    def render(self, item: Question, state: BudgetState | None = None) -> list[Message]:
        system = SYSTEM_COMMON + "\n\n" + COT_SCAFFOLD
        return [
            Message(role="system", content=system),
            Message(role="user", content=_format_user_message(item)),
        ]


def get_framing(name: FramingName, **kwargs: object) -> Framing:
    """Instantiate the framing for the given condition name."""
    if name == "bare":
        return BareFraming()
    if name == "surface":
        initial_budget = float(kwargs.get("initial_budget", 1000.0))
        return SurfaceStakesFraming(initial_budget=initial_budget)
    if name == "persistent":
        return PersistentStakesFraming()
    if name == "cot_only":
        return CotOnlyFraming()
    raise ValueError(f"Unknown framing name: {name!r}")
