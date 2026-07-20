#!/usr/bin/env bash
#SBATCH --job-name=grpo80bbox
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs

PYTHON="${PYTHON:-.venv/bin/python}"
RUN_DIR="${RUN_DIR:-/vol/biomedic3/kw1025/dinosaur/GRPO80/checkpoints/grpo80_slothead_u_m3_4_lightcount_20260710_114710}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/visualizations/test_grpo80_bbox_at3_at4_thr02}"

"${PYTHON}" GRPO80/evaluate_grpo80_bbox.py \
  --run_dir "${RUN_DIR}" \
  --split "${SPLIT:-test}" \
  --out_dir "${OUT_DIR}" \
  --top_ks "${TOP_KS:-3,4}" \
  --hit_threshold "${HIT_THRESHOLD:-0.20}" \
  --threshold_rel "${THRESHOLD_REL:-0.5}" \
  --contact_sheets "${CONTACT_SHEETS:-80}" \
  --contact_sheets_per_class "${CONTACT_SHEETS_PER_CLASS:-0}" \
  --bs "${BS:-32}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --device "${DEVICE:-auto}"

echo "GRPO80 bbox eval: ${OUT_DIR}/summary.json"
