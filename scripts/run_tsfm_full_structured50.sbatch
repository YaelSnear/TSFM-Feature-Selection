#!/bin/bash
#SBATCH --partition main
#SBATCH --time 1-00:00:00
#SBATCH --job-name=tsfm_full_s50
#SBATCH --gpus=1
#SBATCH --constraint="rtx_6000|rtx_pro_6000|rtx_4090"
#SBATCH --output logs/%x_%j.out
#SBATCH --error logs/%x_%j.err
#SBATCH --mail-user=yaelsn@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mem=60G
#SBATCH --cpus-per-task=6

# -----------------------------------------------------------------------
# TSFM Feature Selection — Full Experiment (structured_50)
#
# Config:  configs/config_tsfm_full_structured50.yaml
# Methods: Pearson, Lasso, RandomForest, Mean_CKA, Lagged_CKA
#          + target_only, all_candidates, random_k (k_repeats=3)
# Soft_DTW: EXCLUDED (benchmark: ~60 min for 10 targets × 3 layers)
#
# Targets: 10 role-selected sensors (2 per role: central, peripheral,
#          bridge, dense, high_variance)
# Layers:  [6, 8, 10]  — reported as separate exploratory ablations
# K:       [5, 10, 20, 30]
# Pool:    structured_50 (20 near + 10 far + 10 high_corr + 10 random)
# Split:   80/20 exploratory — do not select K/layer from test results
#
# Estimated runtime: 55–65 min
# -----------------------------------------------------------------------

CONFIG=configs/config_tsfm_full_structured50.yaml

echo "======================================================"
echo "Job:       $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "Start:     $(date '+%Y-%m-%d %H:%M:%S')"
echo "Config:    $CONFIG"
echo "======================================================"

# GPU info
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    && echo "" || echo "[warn] nvidia-smi not available"

module load anaconda
source activate yael_env

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
