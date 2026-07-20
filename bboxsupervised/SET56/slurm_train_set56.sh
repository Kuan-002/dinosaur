#!/usr/bin/env bash
#SBATCH --job-name=set56
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs SET56/checkpoints

RUN_NAME="${RUN_NAME:-set56_u_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-SET56/checkpoints/${RUN_NAME}}"
PYTHON="${PYTHON:-.venv/bin/python}"
SLOTHEAD56_CHECKPOINT="${SLOTHEAD56_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/SET56/checkpoints/slothead56_obj16_geo16_res24_20260709_192411/slothead_best.pt}"

if [[ -z "${SLOTHEAD56_CHECKPOINT}" ]]; then
  echo "SLOTHEAD56_CHECKPOINT must point to a fresh object-mode slothead checkpoint for the current dataset." >&2
  exit 2
fi

"${PYTHON}" SET56/train_set56.py \
  --output_dir "${OUT_DIR}" \
  --data "${DATA:-/vol/biomedic3/kw1025/dinosaur/dataset/coco_top2_clean10_area006_004_600_200_200/classification_dataset}" \
  --sa_checkpoint "${SA_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt}" \
  --slothead_checkpoint "${SLOTHEAD56_CHECKPOINT}" \
  --slothead_mode "${SLOTHEAD_MODE:-u}" \
  --epochs "${EPOCHS:-80}" \
  --bs "${BS:-32}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --lr "${LR:-3e-4}" \
  --wd "${WD:-1e-4}" \
  --seed "${SEED:-8}"

echo "SET56 checkpoint: ${OUT_DIR}/set56_best.pt"
