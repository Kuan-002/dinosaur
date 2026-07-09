#!/usr/bin/env bash
#SBATCH --job-name=settr_bbox_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs

RUN_NAME="${RUN_NAME:-settransformer_anchor_evidence_174127_test_top3_$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-.venv/bin/python}"
PROBE_CHECKPOINT="${PROBE_CHECKPOINT:-checkpoints/settransformer/settransformer_anchor_evidence_20260705_174127/settransformer_discriminative_best.pt}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
"${PYTHON}" analysis/set_transformer_diagnostics/evaluate_set_transformer_bbox.py \
  --probe_checkpoint "${PROBE_CHECKPOINT}" \
  --out_dir "analysis/slot_coverage_compare/${RUN_NAME}" \
  --split "${SPLIT:-test}" \
  --top_k "${TOP_K:-3}" \
  --threshold_rel "${THRESHOLD_REL:-0.5}" \
  --contact_sheets "${CONTACT_SHEETS:-0}" \
  --bs "${BS:-16}" \
  --num_workers "${NUM_WORKERS:-0}" \
  --device "${DEVICE:-auto}"
