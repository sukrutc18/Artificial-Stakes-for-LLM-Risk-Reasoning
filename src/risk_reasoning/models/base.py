"""Abstract LLM client interface.

All subject / judge models implement :class:`LLMClient` so the experiment runner
is agnostic to the backend (vLLM, HF, Anthropic; OllamaClient legacy).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from risk_reasoning.prompts.templates import Message


@dataclass(frozen=True)
class TokenLogprob:
    """A single token and its log-probability (backend-permitting)."""

    token: str
    logprob: float


@dataclass(frozen=True)
class Completion:
    """One sampled completion plus metadata."""

    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    logprobs: tuple[TokenLogprob, ...] | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class LLMClient(ABC):
    """Backend-agnostic chat-completion client."""

    name: str

    @abstractmethod
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
        """Return ``n`` completions for ``messages``."""

    def logprobs(self, messages: list[Message], continuation: str) -> float:
        """Optional: score ``continuation`` under the model. Raise if unsupported."""
        raise NotImplementedError(f"{self.__class__.__name__} does not support scoring")
