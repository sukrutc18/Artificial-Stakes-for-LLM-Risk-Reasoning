"""risk_reasoning: Do artificial stakes improve LLM risk reasoning?

Top-level package for the CS639 final project. Submodules:

- ``data``: dataset loaders and synthetic problem generators.
- ``envs``: sequential budget environment and persistent budget tracker.
- ``prompts``: prompt templates and the four experimental framings.
- ``models``: a single ``LLMClient`` interface; vLLM is the default subject
  backend, with HF and Anthropic available; OllamaClient is retained as
  a legacy backend (deprecated, redirects to vLLM).
- ``eval``: accuracy, calibration, regret, reasoning proxies, LLM-as-judge.
- ``experiments``: config-driven run loop and run logging.
- ``utils``: seeding, I/O, CLI helpers.
"""

__version__ = "0.1.0"

from risk_reasoning.parsers import extract_confidence, extract_cot_trace, parse_mcq_answer

__all__ = [
    "__version__",
    "extract_confidence",
    "extract_cot_trace",
    "parse_mcq_answer",
]
