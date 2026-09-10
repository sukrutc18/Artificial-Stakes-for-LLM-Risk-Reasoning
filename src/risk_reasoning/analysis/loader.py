"""Load experiment samples.jsonl into a tidy pandas DataFrame."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

_JUDGE_SCORE_COLS = {
    "coherence",
    "reasoning_coverage",
    "risk_factors_enumerated",
    "alternatives_considered",
}


def load_samples(run_dir: str | Path) -> pd.DataFrame:
    """Read ``samples.jsonl`` from *run_dir* and return a tidy DataFrame.

    Flattening applied:
    - ``judge_score`` dict  → columns prefixed ``judge_``
    - ``budget_state_before/after`` dicts → ``budget_before``, ``budget_after``,
      ``budget_step`` scalar columns
    """
    path = Path(run_dir) / "samples.jsonl"
    records: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # --- flatten judge_score ---
    if "judge_score" in df.columns:
        non_null = df["judge_score"].notna()
        if non_null.any():
            flat = pd.json_normalize(df.loc[non_null, "judge_score"].tolist())
            flat.index = df.index[non_null]
            for col in flat.columns:
                df[f"judge_{col}"] = flat[col]
        df = df.drop(columns=["judge_score"])

    # --- flatten budget states ---
    for raw_col, scalar_col in [
        ("budget_state_before", "budget_before"),
        ("budget_state_after", "budget_after"),
    ]:
        if raw_col not in df.columns:
            continue
        non_null = df[raw_col].notna()
        if non_null.any():
            flat = pd.json_normalize(df.loc[non_null, raw_col].tolist())
            flat.index = df.index[non_null]
            if "current" in flat.columns:
                df[scalar_col] = flat["current"]
            if "step" in flat.columns and "budget_step" not in df.columns:
                df["budget_step"] = flat["step"]
        df = df.drop(columns=[raw_col])

    # ensure correct is bool/int for arithmetic
    if "correct" in df.columns:
        df["correct"] = df["correct"].astype(float)

    return df


def _load_judge_scores(cell_dir: Path) -> pd.DataFrame:
    """Read ``judge_scores.jsonl`` and return a flat DataFrame with ``judge_`` prefixed columns."""
    path = cell_dir / "judge_scores.jsonl"
    if not path.exists():
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    present = _JUDGE_SCORE_COLS & set(df.columns)
    keep = ["id", "sample_idx"] + [c for c in present if c in df.columns]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.rename(columns={c: f"judge_{c}" for c in present})
    return df


def load_run_matrix(run_dirs: str | Path | list[str | Path]) -> pd.DataFrame:
    """Load all experiment cells from one or more run matrix directories.

    Discovers every ``samples.jsonl`` under *run_dirs*, loads each cell with
    :func:`load_samples`, merges any ``judge_scores.jsonl`` from the same cell
    directory, and concatenates into a single tidy DataFrame.

    The resulting DataFrame has ``condition``, ``dataset``, ``model``,
    ``seed``, ``correct``, ``confidence``, and (when available)
    ``judge_coherence``, ``judge_reasoning_coverage``,
    ``judge_risk_factors_enumerated``, ``judge_alternatives_considered``.
    """
    if isinstance(run_dirs, (str, Path)):
        run_dirs = [run_dirs]

    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        rd = Path(run_dir)
        for sf in sorted(rd.rglob("samples.jsonl")):
            cell_dir = sf.parent
            frame = load_samples(cell_dir)
            if frame.empty:
                continue
            if "model" not in frame.columns and "model_id" in frame.columns:
                frame["model"] = frame["model_id"]
            judge_df = _load_judge_scores(cell_dir)
            if not judge_df.empty and "question_id" in frame.columns:
                frame = frame.merge(
                    judge_df,
                    left_on=["question_id", "sample_idx"],
                    right_on=["id", "sample_idx"],
                    how="left",
                ).drop(columns=["id"], errors="ignore")
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_multiple_runs(run_dirs: list[str | Path]) -> pd.DataFrame:
    """Concatenate samples from multiple run directories, tagging each with its path."""
    frames = []
    for d in run_dirs:
        frame = load_samples(d)
        if not frame.empty:
            frame["run_dir"] = str(d)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
