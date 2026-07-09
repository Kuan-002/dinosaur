#!/usr/bin/env bash
#SBATCH --job-name=settr_evi
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus24
#SBATCH --time=24:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs

PYTHON="${PYTHON:-.venv/bin/python}"
RUN_NAME="${RUN_NAME:-settransformer_evidence_gain_contrast_$(date +%Y%m%d_%H%M%S)}"

"${PYTHON}" -m settransformer.train_evidence_gain_contrast \
  --output_dir "checkpoints/settransformer/${RUN_NAME}" \
  --bs "${BS:-32}" \
  --epochs "${EPOCHS:-80}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --gamma_comp 0.0 \
  --extra_weight "${EXTRA_WEIGHT:-0.6}"
