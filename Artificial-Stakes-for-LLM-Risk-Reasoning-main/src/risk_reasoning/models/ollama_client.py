"""Ollama-backed :class:`LLMClient` for local Base / Instruct serving.

Deprecated: prefer VLLMClient. Retained for reference; the runner no longer instantiates this class.
"""

from __future__ import annotations

from typing import Any

from risk_reasoning.models.base import Completion, LLMClient
from risk_reasoning.prompts.templates import Message


class OllamaClient(LLMClient):
    """Thin wrapper around the ``ollama`` Python client."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        import ollama as _ollama

        self.cfg = cfg
        self.name = cfg.get("name", "ollama")
        self._model: str = cfg.get("hf_id") or cfg.get("model", "llama3.1:8b")
        serving = cfg.get("serving", {})
        host = serving.get("base_url", "http://localhost:11434")
        self._client = _ollama.Client(host=host)
        self._gen_defaults: dict[str, Any] = cfg.get("generation_defaults", {})

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
        options: dict[str, Any] = {}
        t = temperature if temperature is not None else self._gen_defaults.get("temperature")
        if t is not None:
            options["temperature"] = t
        tp = top_p if top_p is not None else self._gen_defaults.get("top_p")
        if tp is not None:
            options["top_p"] = tp
        if max_new_tokens is not None:
            options["num_predict"] = max_new_tokens
        if stop:
            options["stop"] = stop
        if seed is not None:
            options["seed"] = seed

        ollama_messages = [{"role": m.role, "content": m.content} for m in messages]

        results: list[Completion] = []
        for _ in range(n):
            resp = self._client.chat(
                model=self._model,
                messages=ollama_messages,
                options=options or None,
            )
            text: str = ""
            if hasattr(resp, "message"):
                text = resp.message.content or ""
            elif isinstance(resp, dict):
                text = resp.get("message", {}).get("content", "")

            finish: str | None = None
            if hasattr(resp, "done_reason"):
                finish = resp.done_reason
            elif isinstance(resp, dict):
                finish = resp.get("done_reason")

            prompt_tokens: int | None = None
            completion_tokens: int | None = None
            if hasattr(resp, "prompt_eval_count"):
                prompt_tokens = resp.prompt_eval_count
                completion_tokens = resp.eval_count
            elif isinstance(resp, dict):
                prompt_tokens = resp.get("prompt_eval_count")
                completion_tokens = resp.get("eval_count")

            results.append(
                Completion(
                    text=text,
                    finish_reason=finish or "stop",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    raw={},
                )
            )
        return results
