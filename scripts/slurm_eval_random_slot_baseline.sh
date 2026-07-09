#!/usr/bin/env bash
#SBATCH --job-name=random_slot_base
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs

RUN_NAME="${RUN_NAME:-random_slot_baseline_$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-.venv/bin/python}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
"${PYTHON}" analysis/set_transformer_diagnostics/evaluate_random_slot_baseline.py \
  --out_dir "analysis/bbox_settransformer_eval/${RUN_NAME}" \
  --split "${SPLIT:-test}" \
  --top_k "${TOP_K:-3}" \
  --thresholds "${THRESHOLDS:-0.20,0.50}" \
  --bs "${BS:-16}" \
  --num_workers "${NUM_WORKERS:-0}" \
  --device "${DEVICE:-auto}"
