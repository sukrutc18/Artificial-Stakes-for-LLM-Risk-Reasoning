"""One row of ``samples.jsonl``: per-(item, sample) record emitted by the runner.

Field names overlap with ``risk_reasoning.data.schemas.ModelOutput`` (landing
on the Output-Standardization branch) so JSONL rows can be consumed by the
same downstream tooling. Fields ``ModelOutput`` does not carry — the per-item
ordinal, sample index, grading, and budget snapshots — are added here.

Budget fields are identical for all K samples within an item: the runner
mutates the tracker once per item, after collecting all K samples. Filter on
``sample_idx == 0`` downstream to recover the per-item budget trajectory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SampleRecord:
    """One sample of one item within one experiment cell."""

    condition: str
    dataset: str
    model_id: str
    seed: int
    framing: str
    question_id: str
    item_index: int
    sample_idx: int
    raw_output: str
    answer: str | None
    confidence: float | None
    target: str
    correct: bool | None
    budget_before_current: float | None
    budget_after_current: float | None
    budget_before_step: int | None
    budget_after_step: int | None
    reward_delta: float | None
    bankrupt_after: bool | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
