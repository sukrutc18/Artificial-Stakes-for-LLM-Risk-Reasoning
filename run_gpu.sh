#!/usr/bin/env bash
# Full experiment pipeline for the RTX 6000 GPU.
#
# Usage:
#   chmod +x run_gpu.sh
#   ./run_gpu.sh          # full run
#   ./run_gpu.sh --dry-run # 1 item per cell, quick smoke test
#
# Expects a .env file in the repo root with ANTHROPIC_API_KEY and HF_TOKEN set.
# Run this inside a tmux session so it survives SSH disconnects:
#
#   tmux new -s stakes
#   ./run_gpu.sh 2>&1 | tee run_gpu.log
#
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

DRY_RUN_FLAG=""
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN_FLAG="--dry-run"
  echo "[run_gpu] DRY RUN mode: 1 item per cell"
fi

# ── Timing helpers ────────────────────────────────────────────────────────────
RUN_START=$SECONDS
declare -A PHASE_TIMES

phase_start() { _PHASE_T=$SECONDS; }
phase_end() {
  local name="$1"
  local elapsed=$(( SECONDS - _PHASE_T ))
  PHASE_TIMES["$name"]=$elapsed
  printf "[TIMER] %s: %dm %ds\n" "$name" $(( elapsed / 60 )) $(( elapsed % 60 ))
}

# ── 0. Environment ────────────────────────────────────────────────────────────
if [[ ! -f .env ]]; then
  echo "ERROR: .env not found. Copy .env.example and fill in ANTHROPIC_API_KEY + HF_TOKEN."
  exit 1
fi
set -a; source .env; set +a

: "${ANTHROPIC_API_KEY:?ERROR: ANTHROPIC_API_KEY not set in .env}"
: "${HF_TOKEN:?ERROR: HF_TOKEN not set in .env}"

source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124 --quiet
pip install -e ".[vllm]" --quiet

# ── 1. Data preparation ───────────────────────────────────────────────────────
echo ""
echo "=== Phase 1: Data preparation ==="
phase_start
python scripts/generate_synthetic.py --config configs/datasets/synthetic_ev.yaml
python scripts/prepare_real_data.py --datasets convfinqa arc_challenge medqa
phase_end "data_prep"

# ── 2. Instruct model run ─────────────────────────────────────────────────────
echo ""
echo "=== Phase 2: Start vLLM (Instruct) ==="
phase_start
echo "Clearing port 8000..."
kill $(lsof -t -i:8000) 2>/dev/null || true
sleep 5
echo "Starting vLLM server for meta-llama/Meta-Llama-3.1-8B-Instruct..."
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.45 \
  --enforce-eager \
  > vllm_instruct.log 2>&1 \
  &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

echo "Waiting for vLLM to be ready..."
until curl -sf -H "Authorization: Bearer ${VLLM_API_KEY:-EMPTY}" http://localhost:8000/v1/models > /dev/null 2>&1; do
  sleep 5
done
phase_end "vllm_instruct_startup"
echo "vLLM is ready."

echo ""
echo "=== Phase 3: Run instruct experiments ==="
phase_start
PYTHONUNBUFFERED=1 python scripts/run_experiment.py \
  --config configs/experiments/full_matrix_instruct.yaml \
  $DRY_RUN_FLAG
phase_end "instruct_experiments"

echo "Stopping vLLM (Instruct)..."
kill "$VLLM_PID" || true
wait "$VLLM_PID" 2>/dev/null || true

echo "Waiting for GPU CUDA context to fully release..."
for _i in $(seq 1 24); do
  _live=0
  for _pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$'); do
    kill -0 "$_pid" 2>/dev/null && { _live=1; break; }
  done
  [[ "$_live" -eq 0 ]] && break
  echo "  GPU still in use, waiting 5s..."
  sleep 5
done
sleep 5
echo "GPU clear: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader | head -1) free."

# ── 3. Base model run ─────────────────────────────────────────────────────────
echo ""
echo "=== Phase 4: Start vLLM (Base) ==="
phase_start
echo "Clearing port 8000..."
kill $(lsof -t -i:8000) 2>/dev/null || true
sleep 5
echo "Starting vLLM server for meta-llama/Meta-Llama-3.1-8B (base)..."
CUDA_VISIBLE_DEVICES=0 vllm serve meta-llama/Meta-Llama-3.1-8B \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.45 \
  --enforce-eager \
  > vllm_base.log 2>&1 \
  &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

echo "Waiting for vLLM to be ready..."
until curl -sf -H "Authorization: Bearer ${VLLM_API_KEY:-EMPTY}" http://localhost:8000/v1/models > /dev/null 2>&1; do
  sleep 5
done
phase_end "vllm_base_startup"
echo "vLLM is ready."

echo ""
echo "=== Phase 5: Run base experiments ==="
phase_start
PYTHONUNBUFFERED=1 python scripts/run_experiment.py \
  --config configs/experiments/full_matrix_base.yaml \
  $DRY_RUN_FLAG
phase_end "base_experiments"

echo "Stopping vLLM (Base)..."
kill "$VLLM_PID" || true
wait "$VLLM_PID" 2>/dev/null || true

# ── 4. Judge + analysis ───────────────────────────────────────────────────────
echo ""
echo "=== Phase 6: LLM-as-Judge scoring ==="
phase_start
python scripts/run_judge.py results/runs/full_matrix_instruct
python scripts/run_judge.py results/runs/full_matrix_base
phase_end "judge"

echo ""
echo "=== Phase 7: Analyze results ==="
phase_start
python scripts/analyze_results.py results/runs/full_matrix_instruct
python scripts/analyze_results.py results/runs/full_matrix_base
phase_end "analysis"

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL=$(( SECONDS - RUN_START ))
echo ""
echo "════════════════════════════════════════"
echo " TIMING SUMMARY"
echo "════════════════════════════════════════"
for phase in data_prep vllm_instruct_startup instruct_experiments vllm_base_startup base_experiments judge analysis; do
  t=${PHASE_TIMES[$phase]:-0}
  printf "  %-28s %dm %ds\n" "$phase" $(( t / 60 )) $(( t % 60 ))
done
printf "  %-28s %dm %ds\n" "TOTAL" $(( TOTAL / 60 )) $(( TOTAL % 60 ))
echo "════════════════════════════════════════"
echo ""
echo "Results in results/runs/"
echo "  Instruct: results/runs/full_matrix_instruct/analysis/"
echo "  Base:     results/runs/full_matrix_base/analysis/"
