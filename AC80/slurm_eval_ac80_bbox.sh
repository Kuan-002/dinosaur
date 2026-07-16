#!/usr/bin/env bash
#SBATCH --job-name=ac80bbox
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs

PYTHON="${PYTHON:-.venv/bin/python}"
if [[ -z "${RUN_DIR:-}" ]]; then
  RUN_DIR="$(find AC80/checkpoints -maxdepth 1 -type d -name 'ac80_*' | sort | tail -n 1)"
fi
if [[ -z "${RUN_DIR}" ]]; then
  echo "RUN_DIR must point to an AC80 checkpoint directory." >&2
  exit 2
fi
OUT_DIR="${OUT_DIR:-${RUN_DIR}/visualizations/test_ac80_bbox_at3_at4_thr02}"

"${PYTHON}" AC80/evaluate_ac80_bbox.py \
  --run_dir "${RUN_DIR}" \
  --checkpoint "${CHECKPOINT:-selector_ac_best.pt}" \
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

echo "AC80 bbox eval: ${OUT_DIR}/summary.json"
