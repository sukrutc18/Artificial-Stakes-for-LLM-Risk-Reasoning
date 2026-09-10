#!/usr/bin/env bash
# Runs judge + analysis only (phases 6-7).
# Use after both instruct and base experiments have already completed.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# ── Timing helpers ─────────────────────────────────────────────────────────────
RUN_START=$SECONDS
declare -A PHASE_TIMES
phase_start() { _PHASE_T=$SECONDS; }
phase_end() {
  local name="$1"
  local elapsed=$(( SECONDS - _PHASE_T ))
  PHASE_TIMES["$name"]=$elapsed
  printf "[TIMER] %s: %dm %ds\n" "$name" $(( elapsed / 60 )) $(( elapsed % 60 ))
}

# ── Environment ────────────────────────────────────────────────────────────────
set -a; source .env; set +a
: "${ANTHROPIC_API_KEY:?ERROR: ANTHROPIC_API_KEY not set}"
source .venv/bin/activate

# ── Verify instruct results ────────────────────────────────────────────────────
echo ""
echo "=== Verifying instruct results ==="
INSTRUCT_DIR="results/runs/full_matrix_instruct"

if [[ ! -d "$INSTRUCT_DIR" ]]; then
  echo "ERROR: $INSTRUCT_DIR does not exist."
  exit 1
fi

N_CELLS=$(ls "$INSTRUCT_DIR" | grep -v analysis | wc -l)
N_SAMPLES=$(find "$INSTRUCT_DIR" -name "samples.jsonl" | wc -l)
N_METRICS=$(find "$INSTRUCT_DIR" -name "metrics.json" | wc -l)
N_EXPECTED=20

echo "  Cells found:   $N_CELLS / $N_EXPECTED"
echo "  samples.jsonl: $N_SAMPLES / $N_EXPECTED"
echo "  metrics.json:  $N_METRICS / $N_EXPECTED"

if [[ "$N_SAMPLES" -lt "$N_EXPECTED" || "$N_METRICS" -lt "$N_EXPECTED" ]]; then
  echo "ERROR: Instruct results incomplete. Aborting."
  exit 1
fi

BAD=0
while IFS= read -r f; do
  [[ "$f" == *sequential_budget* ]] && continue
  count=$(wc -l < "$f")
  if [[ "$count" -lt 4 ]]; then
    echo "  WARNING: $f has only $count samples"
    BAD=1
  fi
done < <(find "$INSTRUCT_DIR" -name "samples.jsonl")
[[ "$BAD" -eq 1 ]] && { echo "ERROR: Some instruct cells have too few samples. Aborting."; exit 1; }
echo "  Instruct looks complete."

# ── Verify base results ────────────────────────────────────────────────────────
echo ""
echo "=== Verifying base results ==="
BASE_DIR="results/runs/full_matrix_base"

if [[ ! -d "$BASE_DIR" ]]; then
  echo "ERROR: $BASE_DIR does not exist."
  exit 1
fi

N_CELLS=$(ls "$BASE_DIR" | grep -v analysis | wc -l)
N_METRICS=$(find "$BASE_DIR" -name "metrics.json" | wc -l)

echo "  Cells found:   $N_CELLS / $N_EXPECTED"
echo "  metrics.json:  $N_METRICS / $N_EXPECTED"

if [[ "$N_METRICS" -lt "$N_EXPECTED" ]]; then
  echo "ERROR: Base results incomplete ($N_METRICS/20 metrics.json). Aborting."
  exit 1
fi
echo "  Base looks complete."

# ── Phase 6: Judge ─────────────────────────────────────────────────────────────
echo ""
echo "=== Phase 6: LLM-as-Judge scoring ==="
phase_start
python scripts/run_judge.py results/runs/full_matrix_instruct
python scripts/run_judge.py results/runs/full_matrix_base
phase_end "judge"

# ── Phase 7: Analysis ──────────────────────────────────────────────────────────
echo ""
echo "=== Phase 7: Analyze results ==="
phase_start
python scripts/analyze_results.py results/runs/full_matrix_instruct
python scripts/analyze_results.py results/runs/full_matrix_base
phase_end "analysis"

# ── Summary ────────────────────────────────────────────────────────────────────
TOTAL=$(( SECONDS - RUN_START ))
echo ""
echo "════════════════════════════════════════"
echo " TIMING SUMMARY"
echo "════════════════════════════════════════"
for phase in judge analysis; do
  t=${PHASE_TIMES[$phase]:-0}
  printf "  %-28s %dm %ds\n" "$phase" $(( t / 60 )) $(( t % 60 ))
done
printf "  %-28s %dm %ds\n" "TOTAL" $(( TOTAL / 60 )) $(( TOTAL % 60 ))
echo "════════════════════════════════════════"
echo ""
echo "Results in results/runs/"
echo "  Instruct: results/runs/full_matrix_instruct/analysis/"
echo "  Base:     results/runs/full_matrix_base/analysis/"
