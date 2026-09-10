#!/usr/bin/env python
"""Run the Claude-Sonnet judge over an existing run directory's CoT traces.

Usage:
    python scripts/run_judge.py results/runs/<exp_id>
"""

from __future__ import annotations
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import yaml
from tqdm import tqdm

from risk_reasoning.eval.judge import judge_trace
from risk_reasoning.models.anthropic_client import AnthropicClient

# Anthropic Sonnet tier-1 limit is ~50 RPM. 5 workers with a 0.5s delay between
# submissions stays well under that while giving real parallelism.
MAX_WORKERS = 5
RETRY_DELAYS = [5, 15, 45]  # seconds to wait on rate-limit errors


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _judge_item(item: dict, judge: AnthropicClient, rubric: dict) -> dict | None:
    prompt = item.get("prompt") or item.get("question") or ""
    trace = item.get("raw_output") or item.get("trace") or item.get("completion") or ""
    if not trace:
        return None

    for attempt, delay in enumerate([0] + RETRY_DELAYS):
        if delay:
            time.sleep(delay)
        try:
            score = judge_trace(judge=judge, prompt=prompt, trace=trace, rubric=rubric)
            return {
                "id": item.get("question_id") or item.get("id"),
                "sample_idx": item.get("sample_idx"),
                "condition": item.get("condition"),
                "dataset": item.get("dataset"),
                "model_id": item.get("model_id"),
                "coherence": score.coherence,
                "reasoning_coverage": score.reasoning_coverage,
                "risk_factors_enumerated": score.risk_factors_enumerated,
                "alternatives_considered": score.alternatives_considered,
                "raw": score.raw,
            }
        except Exception as e:
            if attempt == len(RETRY_DELAYS):
                print(f"\n[judge] giving up on item after retries: {e}")
                return None
            err = str(e).lower()
            if "429" in str(e) or "529" in str(e) or "rate" in err or "overload" in err:
                print(f"\n[judge] rate limited / overloaded, retrying in {RETRY_DELAYS[attempt]}s...")
            else:
                raise
    return None


def _judge_cell(samples_path: Path, judge: AnthropicClient, rubric: dict) -> str:
    cell_dir = samples_path.parent
    out_path = cell_dir / "judge_scores.jsonl"

    with samples_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    write_lock = Lock()
    completed = 0

    with out_path.open("w", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {}
            for item in records:
                time.sleep(0.5)  # stagger submissions to avoid burst
                futures[pool.submit(_judge_item, item, judge, rubric)] = item

            bar = tqdm(as_completed(futures), total=len(futures),
                       desc=cell_dir.name, unit="sample", dynamic_ncols=True, leave=False)
            for future in bar:
                row = future.result()
                if row is not None:
                    with write_lock:
                        out.write(json.dumps(row) + "\n")
                        completed += 1

    return f"Wrote {completed}/{len(records)} judge scores → {out_path}"


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_judge.py results/runs/<exp_id>")
        return 2

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    cfg = load_yaml(Path("configs/models/claude_sonnet_judge.yaml"))
    judge = AnthropicClient(cfg)
    rubric = cfg.get("rubric")

    sample_files = sorted(run_dir.glob("*/samples.jsonl"))
    if not sample_files:
        raise FileNotFoundError(f"No samples.jsonl files found under {run_dir}")

    pending = [p for p in sample_files if not (p.parent / "judge_scores.jsonl").exists()]
    skipped = len(sample_files) - len(pending)
    if skipped:
        print(f"Skipping {skipped} already-scored cells.")

    cell_bar = tqdm(pending, desc="cells", unit="cell", dynamic_ncols=True)
    for samples_path in cell_bar:
        cell_bar.set_postfix_str(samples_path.parent.name)
        msg = _judge_cell(samples_path, judge, rubric)
        print(msg)

    return 0


if __name__ == "__main__":
    sys.exit(main())