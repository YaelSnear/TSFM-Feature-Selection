#!/bin/bash
#SBATCH --partition main
#SBATCH --time 1-00:00:00
#SBATCH --job-name=tsfm_full_run
#SBATCH --gpus=1
#SBATCH --constraint="rtx_6000|rtx_pro_6000|rtx_4090"
#SBATCH --output logs/%x_%j.out
#SBATCH --error logs/%x_%j.err
#SBATCH --mail-user=yaelsn@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mem=60G
#SBATCH --cpus-per-task=8

# -----------------------------------------------------------------------
# TSFM Feature Selection — FULL RUN: tsfm_full_run_all206
#
# Research question: Can a model-aware FS method select a small Top-K
# subset from all 206 non-target sensors and improve or compete with
# using all features in a frozen TSFM?
#
# Config:   configs/config_tsfm_full_run_all206.yaml
# Methods:  target_only, all_features_206, random_k,
#           Pearson, RandomForest (50 trees), SparseLinear_L1,
#           Mean_CKA_L8, Lagged_CKA_L8
#
# Targets:  10 role-selected sensors (2 per role)
# Layer:    [8]  — pragmatic default; not optimized in this run
#           Layer ablation [6, 8, 10] is a future step.
# K:        [5, 10, 20]
# Universe: all 206 non-target sensors (max_candidates=null)
# Split:    80/20 exploratory — do not select K or layer from test results
# Windows:  max_scoring_windows=1000 (first N, temporal order)
#           test windows uncapped (~547 per target)
#           Window overlap 91.7% — not independent statistical samples
#
# Excluded: Soft_DTW, LassoCV, whitening, structured_50, layers != 8
#
# Incremental saving: results_incremental.jsonl written after every
# forecasting condition. Resume from any interruption point.
#
# Output:   outputs/EXP_tsfm_full_run_all206_YYYYMMDD_HHMMSS/
# Estimated runtime: 30–45 min
# -----------------------------------------------------------------------

set -euo pipefail

CONFIG=configs/config_tsfm_full_run_all206.yaml

echo "======================================================"
echo "FULL RUN: tsfm_full_run_all206"
echo "Job:       $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Start:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "Config:    $CONFIG"
echo "======================================================"

# GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    && echo "" || echo "[warn] nvidia-smi not available"
nvidia-smi 2>/dev/null | head -20 || true

module load anaconda
source activate yael_env

echo "Python:    $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo ""

cd /home/yaelsn/Thesis/tsfm_feature_selection_framework

echo "------------------------------------------------------"
echo "Launching: python -u run_experiment_tsfm.py --config $CONFIG"
echo "------------------------------------------------------"

python -u run_experiment_tsfm.py --config "$CONFIG"
EXIT_CODE=$?

echo "======================================================"
echo "End:       $(date '+%Y-%m-%d %H:%M:%S')"
echo "Exit code: $EXIT_CODE"
echo "======================================================"

exit $EXIT_CODE
