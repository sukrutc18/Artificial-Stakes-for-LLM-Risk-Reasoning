"""Pydantic schemas shared across dataset loaders and the experiment runner.

A single ``Question`` type is used for MCQ, numeric, and sequential tasks so the
runner and framings can treat items uniformly.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TaskType = Literal["mcq", "numeric", "sequential"]


class Choice(BaseModel):
    """One option in a multiple-choice question."""

    id: str = Field(..., description="Short label like 'A', 'B', '1'.")
    text: str = Field(..., description="Rendered option text.")


class Answer(BaseModel):
    """Ground-truth answer for a ``Question``.

    Exactly one of ``choice_id`` / ``numeric_value`` / ``action`` is expected to
    be populated depending on the task type.
    """

    choice_id: str | None = None
    numeric_value: float | None = None
    action: str | None = None
    explanation: str | None = None
    tolerance: float | None = Field(
        default=None,
        description="Relative tolerance for numeric tasks (e.g. 0.01).",
    )


class Question(BaseModel):
    """A single evaluation item (MCQ, numeric, or sequential step)."""

    id: str
    dataset: str
    task_type: TaskType
    prompt: str
    choices: list[Choice] | None = None
    answer: Answer
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelOutput(BaseModel):
    """Structured representation of one raw model completion.

    This is the logger-facing record produced after parsing a completion. It
    keeps the original model text intact while exposing the normalized fields
    the runner needs for scoring, aggregation, and budget updates.
    """

    model_config = ConfigDict(protected_namespaces=())

    question_id: str
    model_id: str
    condition: str
    dataset: str
    raw_output: str
    answer: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    cot_trace: str | None = None
    budget_state: dict[str, Any] | None = None
    seed: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
