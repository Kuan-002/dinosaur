#!/usr/bin/env bash
#SBATCH --job-name=set112
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs SET112/checkpoints

STRUCTURED112_CHECKPOINT="${STRUCTURED112_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/SET112/checkpoints/structured112_obj16_geo32_res64_20260708_185741/structured_slot_bottleneck_best.pt}"

RUN_NAME="${RUN_NAME:-set112_u_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-SET112/checkpoints/${RUN_NAME}}"
PYTHON="${PYTHON:-.venv/bin/python}"

"${PYTHON}" SET112/train_set112.py \
  --output_dir "${OUT_DIR}" \
  --data "${DATA:-/vol/biomedic3/kw1025/dinosaur/analysis/coco_top2_clean_scenes_anchor009_evidence005_10cls_450_150_150/classification_dataset}" \
  --sa_checkpoint "${SA_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt}" \
  --structured_checkpoint "${STRUCTURED112_CHECKPOINT}" \
  --structured_mode "${STRUCTURED_MODE:-u}" \
  --epochs "${EPOCHS:-80}" \
  --bs "${BS:-32}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --lr "${LR:-3e-4}" \
  --wd "${WD:-1e-4}" \
  --seed "${SEED:-8}"

echo "SET112 checkpoint: ${OUT_DIR}/set112_best.pt"
