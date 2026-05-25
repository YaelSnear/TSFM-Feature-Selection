#!/bin/bash
#SBATCH --partition main
#SBATCH --time 1-00:00:00
#SBATCH --job-name=tsfm_exp
#SBATCH --gpus=1
#SBATCH --constraint="rtx_6000|rtx_pro_6000"
#SBATCH --output logs/%x_%j.out
#SBATCH --error logs/%x_%j.err
#SBATCH --mail-user=yaelsn@post.bgu.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mem=60G
#SBATCH --cpus-per-task=6

module load anaconda
source activate yael_env

python run_experiment.py --config configs/config_lag_2h.yaml
