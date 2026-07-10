#!/usr/bin/env bash
#SBATCH --job-name=eval_set_bbox
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs

PYTHON="${PYTHON:-.venv/bin/python}"
DATA="${DATA:-/vol/biomedic3/kw1025/dinosaur/dataset/coco_top2_clean10_area006_004_600_200_200/classification_dataset}"
COCO_ROOT="${COCO_ROOT:-/vol/biomedic3/kw1025/dinosaur/dataset/coco2017}"
SA_CHECKPOINT="${SA_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt}"
VARIANT="${VARIANT:?VARIANT is required: full, 56, 80, or 112}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
OUT_DIR="${OUT_DIR:?OUT_DIR is required}"
SLOTHEAD_CHECKPOINT="${SLOTHEAD_CHECKPOINT:-}"
SLOTHEAD_MODE="${SLOTHEAD_MODE:-u}"

cmd=(
  "${PYTHON}" scripts/evaluate_set_bbox.py
  --variant "${VARIANT}"
  --checkpoint "${CHECKPOINT}"
  --data "${DATA}"
  --split "${SPLIT:-test}"
  --coco_root "${COCO_ROOT}"
  --sa_checkpoint "${SA_CHECKPOINT}"
  --out_dir "${OUT_DIR}"
  --top_ks "${TOP_KS:-3,4}"
  --hit_threshold "${HIT_THRESHOLD:-0.4}"
  --threshold_rel "${THRESHOLD_REL:-0.5}"
  --bs "${BS:-32}"
  --num_workers "${NUM_WORKERS:-4}"
)

if [[ "${VARIANT}" != "full" ]]; then
  if [[ -z "${SLOTHEAD_CHECKPOINT}" ]]; then
    echo "SLOTHEAD_CHECKPOINT is required for VARIANT=${VARIANT}" >&2
    exit 2
  fi
  cmd+=(--slothead_checkpoint "${SLOTHEAD_CHECKPOINT}" --slothead_mode "${SLOTHEAD_MODE}")
fi

echo "Evaluating SET${VARIANT} bbox @${TOP_KS:-3,4} threshold=${HIT_THRESHOLD:-0.4}"
printf '%q ' "${cmd[@]}"
echo
"${cmd[@]}"

