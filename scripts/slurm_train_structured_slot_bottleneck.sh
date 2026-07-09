#!/usr/bin/env bash
#SBATCH --job-name=slot_bottleneck
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs

RUN_NAME="${RUN_NAME:-structured_slot_bottleneck_10cls450_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-analysis/structured_slot_bottleneck/${RUN_NAME}}"
PYTHON="${PYTHON:-.venv/bin/python}"

COCO_ROOT="${COCO_ROOT:-/vol/biomedic3/kw1025/dinosaur/dataset/coco2017}"
CLASSIFICATION_DATASET="${CLASSIFICATION_DATASET:-/vol/biomedic3/kw1025/dinosaur/analysis/coco_top2_clean_scenes_anchor009_evidence005_10cls_450_150_150/classification_dataset}"
SA_CHECKPOINT="${SA_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt}"

INPUT_RES="${INPUT_RES:-224}"
BS="${BS:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-3e-4}"
WD="${WD:-1e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
OBJ_DIM="${OBJ_DIM:-8}"
GEO_DIM="${GEO_DIM:-16}"
RES_DIM="${RES_DIM:-32}"
DROPOUT="${DROPOUT:-0.1}"

LAMBDA_OBJ="${LAMBDA_OBJ:-1.0}"
LAMBDA_GEO="${LAMBDA_GEO:-2.0}"
LAMBDA_CAT="${LAMBDA_CAT:-0.5}"
LAMBDA_REC="${LAMBDA_REC:-0.1}"
LAMBDA_ORTH="${LAMBDA_ORTH:-0.02}"
OBJ_POS_WEIGHT="${OBJ_POS_WEIGHT:-4.0}"

THRESHOLD_REL="${THRESHOLD_REL:-0.5}"
POS_COVERAGE="${POS_COVERAGE:-0.25}"
POS_PURITY="${POS_PURITY:-0.20}"
IGNORE_COVERAGE="${IGNORE_COVERAGE:-0.12}"
IGNORE_PURITY="${IGNORE_PURITY:-0.10}"
BBOX_SHRINK="${BBOX_SHRINK:-0.70}"

QUICK_LIMIT_TRAIN="${QUICK_LIMIT_TRAIN:-0}"
QUICK_LIMIT_VAL="${QUICK_LIMIT_VAL:-0}"
DIAGNOSTIC_MAX_TRAIN_SLOTS="${DIAGNOSTIC_MAX_TRAIN_SLOTS:-300000}"
DIAGNOSTIC_MAX_VAL_SLOTS="${DIAGNOSTIC_MAX_VAL_SLOTS:-0}"
DIAGNOSTIC_EPOCHS="${DIAGNOSTIC_EPOCHS:-5}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-8}"

mkdir -p "${OUT_DIR}"

echo "Structured slot bottleneck"
echo "out_dir=${OUT_DIR}"
echo "classification_dataset=${CLASSIFICATION_DATASET}"
echo "split=z -> u_obj:${OBJ_DIM} u_geo:${GEO_DIM} u_res:${RES_DIM}"
echo "bbox compensation=polygon masks when available, otherwise shrunken bbox factor ${BBOX_SHRINK}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
"${PYTHON}" analysis/structured_slot_bottleneck/train_structured_slot_bottleneck.py \
  --coco_root "${COCO_ROOT}" \
  --classification_dataset "${CLASSIFICATION_DATASET}" \
  --sa_checkpoint "${SA_CHECKPOINT}" \
  --out_dir "${OUT_DIR}" \
  --input_res "${INPUT_RES}" \
  --bs "${BS}" \
  --num_workers "${NUM_WORKERS}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --wd "${WD}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --obj_dim "${OBJ_DIM}" \
  --geo_dim "${GEO_DIM}" \
  --res_dim "${RES_DIM}" \
  --dropout "${DROPOUT}" \
  --lambda_obj "${LAMBDA_OBJ}" \
  --lambda_geo "${LAMBDA_GEO}" \
  --lambda_cat "${LAMBDA_CAT}" \
  --lambda_rec "${LAMBDA_REC}" \
  --lambda_orth "${LAMBDA_ORTH}" \
  --obj_pos_weight "${OBJ_POS_WEIGHT}" \
  --threshold_rel "${THRESHOLD_REL}" \
  --pos_coverage "${POS_COVERAGE}" \
  --pos_purity "${POS_PURITY}" \
  --ignore_coverage "${IGNORE_COVERAGE}" \
  --ignore_purity "${IGNORE_PURITY}" \
  --bbox_shrink "${BBOX_SHRINK}" \
  --quick_limit_train "${QUICK_LIMIT_TRAIN}" \
  --quick_limit_val "${QUICK_LIMIT_VAL}" \
  --diagnostic_max_train_slots "${DIAGNOSTIC_MAX_TRAIN_SLOTS}" \
  --diagnostic_max_val_slots "${DIAGNOSTIC_MAX_VAL_SLOTS}" \
  --diagnostic_epochs "${DIAGNOSTIC_EPOCHS}" \
  --device "${DEVICE}" \
  --seed "${SEED}"

echo "Final metrics: ${OUT_DIR}/final_metrics.json"
echo "Projection checkpoint: ${OUT_DIR}/structured_slot_bottleneck_best.pt"
