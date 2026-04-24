#!/bin/bash
#SBATCH --job-name=brats-h100-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Customize these two lines for your cluster environment.
module purge
module load cuda/12.2

# Activate your Python environment.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate brats-tf

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs saved_models

python train_brats_h100.py \
  --train-h5 h5_dataset/train_data.h5 \
  --valid-h5 h5_dataset/valid_data.h5 \
  --test-h5 h5_dataset/test_data.h5 \
  --output-dir saved_models \
  --plots-dir saved_models/plots \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-4
