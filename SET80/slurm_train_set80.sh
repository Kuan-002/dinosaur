#!/usr/bin/env bash
#SBATCH --job-name=set80
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs SET80/checkpoints

SLOTHEAD80_CHECKPOINT="${SLOTHEAD80_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/SET80/checkpoints/slothead80_obj16_geo16_res48_20260709_192454/slothead_best.pt}"

if [[ -z "${SLOTHEAD80_CHECKPOINT}" ]]; then
  echo "SLOTHEAD80_CHECKPOINT must point to a fresh object-mode slothead checkpoint for the current dataset." >&2
  exit 2
fi

RUN_NAME="${RUN_NAME:-set80_u_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-SET80/checkpoints/${RUN_NAME}}"
PYTHON="${PYTHON:-.venv/bin/python}"

"${PYTHON}" SET80/train_set80.py \
  --output_dir "${OUT_DIR}" \
  --data "${DATA:-/vol/biomedic3/kw1025/dinosaur/dataset/coco_top2_clean10_area006_004_600_200_200/classification_dataset}" \
  --sa_checkpoint "${SA_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt}" \
  --slothead_checkpoint "${SLOTHEAD80_CHECKPOINT}" \
  --slothead_mode "${SLOTHEAD_MODE:-u}" \
  --epochs "${EPOCHS:-80}" \
  --bs "${BS:-32}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --lr "${LR:-3e-4}" \
  --wd "${WD:-1e-4}" \
  --seed "${SEED:-8}"

echo "SET80 checkpoint: ${OUT_DIR}/set80_best.pt"
