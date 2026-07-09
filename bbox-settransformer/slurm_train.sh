#!/usr/bin/env bash
#SBATCH --job-name=bbox_settr
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus24
#SBATCH --time=24:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs

RUN_NAME="${RUN_NAME:-bbox_settransformer_$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-.venv/bin/python}"

"${PYTHON}" bbox-settransformer/train.py \
  --output_dir "checkpoints/bbox-settransformer/${RUN_NAME}" \
  --data "${DATA:-/vol/biomedic3/kw1025/dinosaur/analysis/coco_top2_clean_scenes_anchor009_evidence005_10cls_450_150_150/classification_dataset}" \
  --coco_root "${COCO_ROOT:-/vol/biomedic3/kw1025/dinosaur/dataset/coco2017}" \
  --checkpoint "${SA_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt}" \
  --bs "${BS:-32}" \
  --epochs "${EPOCHS:-40}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --hit_threshold "${HIT_THRESHOLD:-0.20}" \
  --role_loss_weight "${ROLE_LOSS_WEIGHT:-1.0}" \
  --subset_loss_weight "${SUBSET_LOSS_WEIGHT:-1.0}" \
  --quick_limit_train "${QUICK_LIMIT_TRAIN:-0}" \
  --quick_limit_val "${QUICK_LIMIT_VAL:-0}"
