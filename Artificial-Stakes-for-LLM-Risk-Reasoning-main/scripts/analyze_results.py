#!/usr/bin/env python
"""Post-hoc analysis: aggregate metrics, significance tests, surface-vs-persistent ablation, figures.

Usage:
    python scripts/analyze_results.py results/runs/full_matrix_base
    python scripts/analyze_results.py results/runs/full_matrix_instruct

Each argument must be a run matrix directory that contains cell subdirectories.
Run once per model to produce independent figures and stats for each.
Multiple directories can be passed to concatenate across seeds:
    python scripts/analyze_results.py <seed1_dir> <seed2_dir>

Outputs written to ``<first_run_dir>/analysis/``:
    summary_stats.csv           mean ± CI per condition × dataset × model
    significance_table.csv      pairwise Welch t-tests with p-values
    persistent_vs_surface.csv   bootstrap comparison of the two stakes conditions
    figures/                    accuracy bars, calibration curves, regret, proxy bars
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_dirs", nargs="+", metavar="RUN_DIR")
    p.add_argument("--out-dir", default=None, help="Override output directory")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--no-figures", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from risk_reasoning.analysis import (
        load_run_matrix,
        make_all_figures,
        persistent_vs_surface_analysis,
        significance_table,
        summary_stats,
    )

    run_dirs = [Path(d) for d in args.run_dirs]
    for d in run_dirs:
        if not d.exists():
            print(f"ERROR: {d} does not exist", file=sys.stderr)
            return 1

    out_dir = Path(args.out_dir) if args.out_dir else run_dirs[0] / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(run_dirs)} run dir(s)…")
    df = load_run_matrix(run_dirs)
    if df.empty:
        print("ERROR: no samples loaded", file=sys.stderr)
        return 1

    print(
        f"  {len(df):,} samples | conditions: {sorted(df['condition'].unique())} | "
        f"datasets: {sorted(df['dataset'].unique())} | models: {sorted(df['model'].unique())}"
    )

    judge_cols = [c for c in df.columns if c.startswith("judge_") and df[c].dtype.kind in "fiu"]
    metrics = ["correct"] + judge_cols

    # --- Summary stats ---
    print("\nSummary statistics…")
    summ = summary_stats(df, metrics, group_cols=["condition", "dataset", "model"],
                         n_bootstrap=args.n_bootstrap)
    summ.to_csv(out_dir / "summary_stats.csv", index=False)
    print(summ[["condition", "dataset", "model", "correct_mean", "correct_ci_lo",
                "correct_ci_hi", "n"]].sort_values(["dataset", "condition"]).to_string(index=False))

    # --- Significance table ---
    print("\nPairwise significance tests…")
    sig = significance_table(
        df, ["correct"],
        group_by=["question_id"] if "question_id" in df.columns else None,
    )
    sig.to_csv(out_dir / "significance_table.csv", index=False)
    print(sig[["condition_a", "condition_b", "delta", "p_value", "cohens_d", "significant"]]
          .to_string(index=False))

    # --- Surface vs persistent ablation ---
    print("\nSurface vs persistent bootstrap comparison…")
    pvs = persistent_vs_surface_analysis(df, ["correct"], n_bootstrap=args.n_bootstrap)
    pvs.to_csv(out_dir / "persistent_vs_surface.csv", index=False)
    if not pvs.empty and "correct_delta" in pvs.columns:
        print(pvs[["dataset", "model", "correct_surface", "correct_persistent",
                    "correct_delta", "correct_ci_lo", "correct_ci_hi",
                    "correct_p_persistent_better"]].to_string(index=False))

    # --- Figures ---
    if not args.no_figures:
        print("\nGenerating figures…")
        figs = make_all_figures(df, out_dir / "figures", n_bootstrap=args.n_bootstrap)
        print(f"  {len(figs)} figure(s) → {out_dir / 'figures'}")

    print(f"\nDone. Outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
