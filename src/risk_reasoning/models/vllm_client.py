"""vLLM OpenAI-compatible client.

Talks to a running vLLM server via its OpenAI-compatible HTTP API.
Uses ``httpx`` (available as a transitive dependency of ``anthropic``) so
the project does not need ``openai`` as a direct dependency.
"""

from __future__ import annotations

import json
import os
from typing import Any

from risk_reasoning.models.base import Completion, LLMClient, TokenLogprob
from risk_reasoning.prompts.templates import Message


class VLLMClient(LLMClient):
    """Talks to a vLLM server exposing the OpenAI-compatible API."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        import httpx

        self.cfg = cfg
        self.name = cfg.get("name", "vllm")
        self._model = cfg.get("hf_id") or cfg.get("model", "")
        serving = cfg.get("serving", {})
        base_url = serving.get("base_url") or os.environ.get(
            "VLLM_BASE_URL", "http://localhost:8000/v1"
        )
        api_key = serving.get("api_key") or os.environ.get("VLLM_API_KEY", "EMPTY")
        self._base_url = base_url.rstrip("/")
        self._headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self._gen_defaults: dict[str, Any] = cfg.get("generation_defaults", {})
        self._http = httpx.Client(timeout=120.0)
        # Base models have no chat template — use /v1/completions with raw text.
        tokenizer_cfg = cfg.get("tokenizer", {}) or {}
        self._use_completions = tokenizer_cfg.get("chat_template") is None

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
        max_tok = max_new_tokens or self._gen_defaults.get("max_new_tokens", 1024)
        t = temperature if temperature is not None else self._gen_defaults.get("temperature")
        tp = top_p if top_p is not None else self._gen_defaults.get("top_p")

        if self._use_completions:
            payload: dict[str, Any] = {
                "model": self._model,
                "prompt": _messages_to_prompt(messages),
                "n": n,
                "max_tokens": max_tok,
            }
            if t is not None:
                payload["temperature"] = t
            if tp is not None:
                payload["top_p"] = tp
            if stop:
                payload["stop"] = stop
            if seed is not None:
                payload["seed"] = seed
            if return_logprobs:
                payload["logprobs"] = 1

            response = self._http.post(
                f"{self._base_url}/completions",
                headers=self._headers,
                content=json.dumps(payload),
            )
            response.raise_for_status()
            data = response.json()
            usage = data.get("usage", {})
            return [
                Completion(
                    text=choice.get("text") or "",
                    finish_reason=choice.get("finish_reason"),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    logprobs=None,
                    raw={},
                )
                for choice in data.get("choices", [])
            ]

        payload = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "n": n,
            "max_tokens": max_tok,
        }
        if t is not None:
            payload["temperature"] = t
        if tp is not None:
            payload["top_p"] = tp
        if stop:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed
        if return_logprobs:
            payload["logprobs"] = True

        response = self._http.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers,
            content=json.dumps(payload),
        )
        response.raise_for_status()
        data = response.json()

        results: list[Completion] = []
        usage = data.get("usage", {})
        for choice in data.get("choices", []):
            lp: tuple[TokenLogprob, ...] | None = None
            if return_logprobs and choice.get("logprobs"):
                lp = tuple(
                    TokenLogprob(token=tok["token"], logprob=tok["logprob"])
                    for tok in (choice["logprobs"].get("content") or [])
                )
            results.append(
                Completion(
                    text=choice.get("message", {}).get("content") or "",
                    finish_reason=choice.get("finish_reason"),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    logprobs=lp,
                    raw={},
                )
            )
        return results


def _messages_to_prompt(messages: list) -> str:
    """Flatten chat messages into a raw text prompt for base models."""
    parts = []
    for m in messages:
        role = getattr(m, "role", m.get("role", "")) if not hasattr(m, "role") else m.role
        content = getattr(m, "content", m.get("content", "")) if not hasattr(m, "content") else m.content
        if role == "system":
            parts.append(content)
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)
