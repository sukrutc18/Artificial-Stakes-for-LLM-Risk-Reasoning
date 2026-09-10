"""Anthropic client used for the Claude-Sonnet LLM-as-judge."""

from __future__ import annotations

from typing import Any

from risk_reasoning.models.base import Completion, LLMClient
from risk_reasoning.prompts.templates import Message
import anthropic, os

class AnthropicClient(LLMClient):
    """Thin wrapper around the Anthropic Messages API."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.name = cfg.get("name", "anthropic")
        self.model = cfg.get("api_model_id", "claude-sonnet-4-5-20250929")

        serving = cfg.get("serving", {})
        api_key =  os.environ.get("ANTHROPIC_API_KEY") or serving.get("api_key")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        self.client = anthropic.Anthropic(api_key=api_key)

        defaults = cfg.get("generation_defaults", {})
        self.default_temperature = defaults.get("temperature", 0.0)
        self.default_max_tokens = defaults.get("max_new_tokens", 1024)
    
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
        if n != 1:
            raise NotImplementedError("AnthropicClient currently supports n=1 only.")
        if seed is not None:
            raise NotImplementedError("Anthropic API does not support seed here.")
        if return_logprobs:
            raise NotImplementedError("AnthropicClient does not support logprobs.")

        system_parts = [m.content for m in messages if m.role == "system"]
        api_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in {"user", "assistant"}
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_new_tokens or self.default_max_tokens,
            "temperature": self.default_temperature if temperature is None else temperature,
            "messages": api_messages,
        }

        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if top_p is not None:
            kwargs["top_p"] = top_p
        if stop is not None:
            kwargs["stop_sequences"] = stop

        msg = self.client.messages.create(**kwargs)

        text = "".join(
            block.text for block in msg.content if getattr(block, "type", None) == "text"
        )

        usage = getattr(msg, "usage", None)

        return [
            Completion(
                text=text,
                finish_reason=getattr(msg, "stop_reason", None),
                prompt_tokens=getattr(usage, "input_tokens", None) if usage else None,
                completion_tokens=getattr(usage, "output_tokens", None) if usage else None,
                raw=msg.model_dump() if hasattr(msg, "model_dump") else {},
            )
        ]