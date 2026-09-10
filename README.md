# Do Artificial Stakes Improve LLM Risk Reasoning?

**CS 639: Introduction to Foundation Models — Final Project (Spring 2026)**

Shrey Katyal, Dev Sinha, Aditya Dube, Shreekar Earanti, Atiksh Shah,
Sukrut Chikodikar, Taran Savoor

## Overview

LLMs are increasingly deployed for reasoning in high-stakes domains (finance,
medicine, risk assessment), yet they never face real consequences for bad
choices and are often overconfident. We test a simple alternative to
RL-on-reasoning-steps: give the model a **persistent virtual budget** that
depletes based on decision quality across a sequence of problems, and measure
whether this improves risk reasoning relative to matched baselines.

**Hypothesis.** Persistent resource framing — where the model sees a visible
budget that updates based on its decisions — leads to higher decision accuracy
and better probability calibration than matched sequential baselines receiving
the same feedback without stakes.

See [`report/FMProposal.pdf`](report/FMProposal.pdf) for the full proposal.

## Experimental matrix

| Axis | Values |
| --- | --- |
| Conditions | `bare_baseline`, `surface_stakes`, `persistent_stakes`, `cot_only` |
| Datasets | synthetic EV/probability, ConvFinQA, ARC-Challenge, MedQA, sequential budget env |
| Subject models | Llama-3.1-8B Base, Llama-3.1-8B Instruct |
| Judge | Claude Sonnet (coherence + reasoning coverage) |
| Metrics | accuracy, ECE, Brier, reliability curves, regret, mistake propagation, reasoning proxies |

## Repository layout

```text
configs/       YAML configs for conditions, datasets, models, and experiments
src/risk_reasoning/
  data/        dataset loaders + Pydantic schemas
  envs/        sequential budget env + persistent budget tracker
  prompts/     templates + four framings (bare/surface/persistent/cot_only)
  models/      LLMClient ABC + vLLM (default) / HF / Anthropic implementations; OllamaClient retained as legacy
  eval/        accuracy, calibration, regret, reasoning proxies, LLM-as-judge
  experiments/ config-driven runner + JSONL run logger
  utils/       seeding, I/O, CLI helpers
scripts/       entry points: generate_synthetic, prepare_real_data, run_experiment, run_judge, analyze_results
notebooks/     EDA, calibration curves, sequential trajectories, final figures
tests/         pytest skeletons for generator, budget env, framings, calibration
docker/        CUDA + vLLM stack
data/          (gitignored) raw/, processed/, synthetic/
results/       (gitignored) runs/<exp_id>/, figures/, tables/
report/        proposal + final writeup
```

## Quickstart

### Local (without Docker)

Requires Python 3.11 and, for GPU inference, a working CUDA 12.4 install.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[vllm,dev]"

cp .env.example .env
# Fill in ANTHROPIC_API_KEY and HF_TOKEN in .env

python scripts/generate_synthetic.py --config configs/datasets/synthetic_ev.yaml
python scripts/prepare_real_data.py --datasets convfinqa arc_challenge medqa

python scripts/run_experiment.py experiment=full_matrix
python scripts/run_judge.py results/runs/<exp_id>
python scripts/analyze_results.py results/runs/<exp_id>
```

Use `uv sync` instead of `pip install -e .` if you prefer the `uv` toolchain.

### Real-data preprocessing

`scripts/prepare_real_data.py` reads the YAMLs in `configs/datasets/` and can
write normalized JSONL caches to `data/processed/`. MedQA is fully wired:

```bash
python scripts/prepare_real_data.py --datasets medqa
```

Output: `data/processed/medqa.jsonl`.

`medqa` is downloaded through the Hugging Face Hub and normalized from a
known-good Parquet snapshot because newer `datasets` releases no longer execute
Hub dataset scripts such as `bigbio/med_qa` directly.

### Docker

```bash
docker compose -f docker/docker-compose.yml up --build
docker exec -it risk-experiments python scripts/run_experiment.py experiment=full_matrix
```

The compose file brings up a `vllm` service exposing the OpenAI-compatible API
on port 8000 and an `experiments` container with the package installed in
editable mode. Both share GPU access; an RTX 6000 is sufficient for 8B inference.

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy src
pytest -q
```

Tests are currently scaffolded as `xfail` stubs. They will turn green as each
module is implemented.

## License

MIT. See [`LICENSE`](LICENSE).
