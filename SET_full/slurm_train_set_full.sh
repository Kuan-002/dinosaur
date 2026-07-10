#!/usr/bin/env bash
#SBATCH --job-name=setfull
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs SET_full/checkpoints

RUN_NAME="${RUN_NAME:-set_full_raw_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-SET_full/checkpoints/${RUN_NAME}}"
PYTHON="${PYTHON:-.venv/bin/python}"

"${PYTHON}" SET_full/train_set_full.py \
  --output_dir "${OUT_DIR}" \
  --data "${DATA:-/vol/biomedic3/kw1025/dinosaur/dataset/coco_top2_clean10_area006_004_600_200_200/classification_dataset}" \
  --sa_checkpoint "${SA_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt}" \
  --epochs "${EPOCHS:-80}" \
  --bs "${BS:-32}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --lr "${LR:-3e-4}" \
  --wd "${WD:-1e-4}" \
  --seed "${SEED:-8}"

echo "SET_full checkpoint: ${OUT_DIR}/set_full_best.pt"
