#!/bin/bash
#SBATCH --job-name=layer_ablation_cka
#SBATCH --output=outputs/EXP_tsfm_full_run_all206_20260530_172932/layer_ablation_cka/logs/slurm_%j.out
#SBATCH --error=outputs/EXP_tsfm_full_run_all206_20260530_172932/layer_ablation_cka/logs/slurm_%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yaelsn@post.bgu.ac.il

set -e   # fail fast

echo "=== CKA Layer Ablation [6, 8, 10] ==="
echo "Host:  $(hostname)"
echo "Start: $(date)"
echo "Job:   ${SLURM_JOB_ID}"
echo ""

# GPU info
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

# Create log directory before SLURM writes to it
mkdir -p outputs/EXP_tsfm_full_run_all206_20260530_172932/layer_ablation_cka/logs

echo ""
echo "--- Step 1: run_layer_ablation_cka.py ---"
conda run --no-capture-output -n yael_env \
  python scripts/run_layer_ablation_cka.py \
    --cache_dir  outputs/EXP_tsfm_full_run_all206_20260530_172932/cache \
    --orig_jsonl outputs/EXP_tsfm_full_run_all206_20260530_172932/results/results_incremental.jsonl \
    --fixed_jsonl outputs/EXP_tsfm_full_run_all206_20260530_172932/lagged_cka_fixed_rerun/results_incremental_fixed.jsonl \
    --out_dir    outputs/EXP_tsfm_full_run_all206_20260530_172932/layer_ablation_cka

echo ""
echo "--- Step 2: analyze_layer_ablation.py ---"
conda run --no-capture-output -n yael_env \
  python scripts/analyze_layer_ablation.py \
    --exp_dir outputs/EXP_tsfm_full_run_all206_20260530_172932

echo ""
echo "Done: $(date)"
echo "Results: outputs/EXP_tsfm_full_run_all206_20260530_172932/layer_ablation_cka/"
