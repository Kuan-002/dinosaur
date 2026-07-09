#!/usr/bin/env bash
#SBATCH --job-name=slothead
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs

RUN_NAME="${RUN_NAME:-slothead_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-analysis/slothead/${RUN_NAME}}"
PYTHON="${PYTHON:-.venv/bin/python}"

DATA="${DATA:-/vol/biomedic3/kw1025/dinosaur/analysis/coco_top2_clean_scenes_anchor009_evidence005_10cls_450_150_150/classification_dataset}"
COCO_ROOT="${COCO_ROOT:-/vol/biomedic3/kw1025/dinosaur/dataset/coco2017}"
SA_CHECKPOINT="${SA_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt}"

SPLIT="${SPLIT:-test}"
INPUT_RES="${INPUT_RES:-224}"
BS="${BS:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_ITEMS="${MAX_ITEMS:-0}"
TOP_KS="${TOP_KS:-1,2,3,5}"
THRESHOLD_REL="${THRESHOLD_REL:-0.5}"
COVERAGE_THRESHOLD="${COVERAGE_THRESHOLD:-0.50}"
PURITY_THRESHOLD="${PURITY_THRESHOLD:-0.30}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.30}"
DEVICE="${DEVICE:-auto}"
EXPORT_EMBEDDINGS="${EXPORT_EMBEDDINGS:-0}"

mkdir -p "${OUT_DIR}"

echo "SlotHead analysis"
echo "out_dir=${OUT_DIR}"
echo "split=${SPLIT} top_ks=${TOP_KS}"
echo "hit: coverage>=${COVERAGE_THRESHOLD} purity>=${PURITY_THRESHOLD}; binary threshold_rel=${THRESHOLD_REL}"

extra_args=()
if [[ "${EXPORT_EMBEDDINGS}" == "1" ]]; then
  extra_args+=(--export_embeddings)
fi

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
"${PYTHON}" analysis/slothead/run_slothead_analysis.py \
  --data "${DATA}" \
  --split "${SPLIT}" \
  --coco_root "${COCO_ROOT}" \
  --sa_checkpoint "${SA_CHECKPOINT}" \
  --out_dir "${OUT_DIR}" \
  --input_res "${INPUT_RES}" \
  --bs "${BS}" \
  --num_workers "${NUM_WORKERS}" \
  --max_items "${MAX_ITEMS}" \
  --top_ks "${TOP_KS}" \
  --threshold_rel "${THRESHOLD_REL}" \
  --coverage_threshold "${COVERAGE_THRESHOLD}" \
  --purity_threshold "${PURITY_THRESHOLD}" \
  --iou_threshold "${IOU_THRESHOLD}" \
  --device "${DEVICE}" \
  "${extra_args[@]}"

echo "Summary:"
echo "${OUT_DIR}/summary.json"
echo "Per-class CSV:"
echo "${OUT_DIR}/slothead_per_class_summary.csv"
sed -n '1,20p' "${OUT_DIR}/slothead_per_class_summary.csv"
