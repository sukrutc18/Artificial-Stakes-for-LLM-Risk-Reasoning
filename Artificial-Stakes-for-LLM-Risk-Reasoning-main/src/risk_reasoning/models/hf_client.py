"""HuggingFace ``transformers`` client for ad-hoc / offline inference."""

from __future__ import annotations

from typing import Any

from risk_reasoning.models.base import Completion, LLMClient
from risk_reasoning.prompts.templates import Message


class HFClient(LLMClient):
    """Local transformers-backed generation. Primarily useful for logprob scoring."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        from transformers import pipeline as hf_pipeline

        self.cfg = cfg
        self.name = cfg.get("name", "hf")
        self._model_id: str = cfg.get("hf_id") or cfg.get("model", "")
        self._gen_defaults: dict[str, Any] = cfg.get("generation_defaults", {})

        # Build the pipeline once at construction time so repeated generate()
        # calls don't reload weights.
        self._pipe = hf_pipeline(
            "text-generation",
            model=self._model_id,
            device_map="auto",
        )

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
        prompt = self._build_prompt(messages)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens or self._gen_defaults.get("max_new_tokens", 1024),
            "num_return_sequences": n,
            "do_sample": True,
        }
        t = temperature if temperature is not None else self._gen_defaults.get("temperature", 0.7)
        if t is not None:
            gen_kwargs["temperature"] = t
        tp = top_p if top_p is not None else self._gen_defaults.get("top_p")
        if tp is not None:
            gen_kwargs["top_p"] = tp
        if stop:
            gen_kwargs["stop_sequences"] = stop

        outputs = self._pipe(prompt, **gen_kwargs)

        results: list[Completion] = []
        for out in outputs:
            full_text: str = out["generated_text"]
            # Strip the prompt prefix so only the new tokens are returned.
            new_text = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
            results.append(
                Completion(text=new_text, finish_reason="stop", raw={})
            )
        return results

    def logprobs(self, messages: list[Message], continuation: str) -> float:
        import torch

        prompt = self._build_prompt(messages)
        tokenizer = self._pipe.tokenizer
        model = self._pipe.model

        enc_prompt = tokenizer(prompt, return_tensors="pt")
        enc_full = tokenizer(prompt + continuation, return_tensors="pt")

        with torch.no_grad():
            out = model(**enc_full)

        logits = out.logits[0]
        prompt_len = enc_prompt["input_ids"].shape[1]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        ids = enc_full["input_ids"][0]

        total: float = 0.0
        for i in range(prompt_len, len(ids)):
            total += float(log_probs[i - 1, ids[i]])
        return total

    @staticmethod
    def _build_prompt(messages: list[Message]) -> str:
        parts: list[str] = []
        for m in messages:
            if m.role == "system":
                parts.append(f"[System]\n{m.content}")
            elif m.role == "user":
                parts.append(f"[User]\n{m.content}")
            elif m.role == "assistant":
                parts.append(f"[Assistant]\n{m.content}")
        parts.append("[Assistant]\n")
        return "\n\n".join(parts)
