#!/usr/bin/env bash
#SBATCH --job-name=settr_anchor_evidence
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus24
#SBATCH --time=24:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs

RUN_NAME="${RUN_NAME:-settransformer_anchor_evidence_$(date +%Y%m%d_%H%M%S)}"

PYTHON="${PYTHON:-.venv/bin/python}"

"${PYTHON}" -m settransformer.train \
  --output_dir "checkpoints/settransformer/${RUN_NAME}" \
  --bs "${BS:-32}" \
  --epochs "${EPOCHS:-80}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --gamma_comp "${GAMMA_COMP:-0.0}" \
  --delta_marginal "${DELTA_MARGINAL:-0.7}" \
  --epsilon_consistency "${EPSILON_CONSISTENCY:-0.3}"
