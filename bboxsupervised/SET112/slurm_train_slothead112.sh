#!/usr/bin/env bash
#SBATCH --job-name=slothead112
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs SET112/checkpoints

RUN_NAME="${RUN_NAME:-slothead112_obj16_geo32_res64_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-SET112/checkpoints/${RUN_NAME}}"
PYTHON="${PYTHON:-.venv/bin/python}"

COCO_ROOT="${COCO_ROOT:-/vol/biomedic3/kw1025/dinosaur/dataset/coco2017}"
CLASSIFICATION_DATASET="${CLASSIFICATION_DATASET:-/vol/biomedic3/kw1025/dinosaur/dataset/coco_top2_clean10_area006_004_600_200_200/classification_dataset}"
SA_CHECKPOINT="${SA_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt}"

mkdir -p "${OUT_DIR}"

echo "SET112 slothead"
echo "out_dir=${OUT_DIR}"
echo "split=z -> u_obj:${OBJ_DIM:-16} u_geo:${GEO_DIM:-32} u_res:${RES_DIM:-64}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
"${PYTHON}" SET112/train_slothead112.py \
  --coco_root "${COCO_ROOT}" \
  --classification_dataset "${CLASSIFICATION_DATASET}" \
  --sa_checkpoint "${SA_CHECKPOINT}" \
  --out_dir "${OUT_DIR}" \
  --input_res "${INPUT_RES:-224}" \
  --bs "${BS:-32}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --epochs "${EPOCHS:-20}" \
  --lr "${LR:-3e-4}" \
  --wd "${WD:-1e-4}" \
  --hidden_dim "${HIDDEN_DIM:-256}" \
  --obj_dim "${OBJ_DIM:-16}" \
  --geo_dim "${GEO_DIM:-32}" \
  --res_dim "${RES_DIM:-64}" \
  --dropout "${DROPOUT:-0.1}" \
  --lambda_obj "${LAMBDA_OBJ:-1.0}" \
  --lambda_geo "${LAMBDA_GEO:-2.0}" \
  --lambda_cat "${LAMBDA_CAT:-0.5}" \
  --category_mode "${CATEGORY_MODE:-object}" \
  --lambda_rec "${LAMBDA_REC:-0.1}" \
  --lambda_orth "${LAMBDA_ORTH:-0.02}" \
  --obj_pos_weight "${OBJ_POS_WEIGHT:-4.0}" \
  --threshold_rel "${THRESHOLD_REL:-0.5}" \
  --pos_coverage "${POS_COVERAGE:-0.25}" \
  --pos_purity "${POS_PURITY:-0.20}" \
  --ignore_coverage "${IGNORE_COVERAGE:-0.12}" \
  --ignore_purity "${IGNORE_PURITY:-0.10}" \
  --bbox_shrink "${BBOX_SHRINK:-0.70}" \
  --quick_limit_train "${QUICK_LIMIT_TRAIN:-0}" \
  --quick_limit_val "${QUICK_LIMIT_VAL:-0}" \
  --diagnostic_max_train_slots "${DIAGNOSTIC_MAX_TRAIN_SLOTS:-300000}" \
  --diagnostic_max_val_slots "${DIAGNOSTIC_MAX_VAL_SLOTS:-0}" \
  --diagnostic_epochs "${DIAGNOSTIC_EPOCHS:-5}" \
  --device "${DEVICE:-auto}" \
  --seed "${SEED:-8}"

echo "SET112 slothead checkpoint: ${OUT_DIR}/slothead_best.pt"
