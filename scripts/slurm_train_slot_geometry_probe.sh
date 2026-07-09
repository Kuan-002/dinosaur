#!/usr/bin/env bash
#SBATCH --job-name=slot_geo_probe
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs

RUN_NAME="${RUN_NAME:-slot_geometry_probe_cov030_pur020_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-analysis/slothead/${RUN_NAME}}"
PYTHON="${PYTHON:-.venv/bin/python}"

DATA="${DATA:-/vol/biomedic3/kw1025/dinosaur/analysis/coco_top2_clean_scenes_anchor009_evidence005_10cls_450_150_150/classification_dataset}"
COCO_ROOT="${COCO_ROOT:-/vol/biomedic3/kw1025/dinosaur/dataset/coco2017}"
SA_CHECKPOINT="${SA_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt}"

INPUT_RES="${INPUT_RES:-224}"
BS="${BS:-32}"
PROBE_BS="${PROBE_BS:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EPOCHS="${EPOCHS:-40}"
LR="${LR:-5e-4}"
WD="${WD:-1e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
DROPOUT="${DROPOUT:-0.1}"
TOP_KS="${TOP_KS:-1,2,3,5}"
THRESHOLD_REL="${THRESHOLD_REL:-0.5}"
COVERAGE_THRESHOLD="${COVERAGE_THRESHOLD:-0.30}"
PURITY_THRESHOLD="${PURITY_THRESHOLD:-0.20}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-8}"
REFRESH_CACHE="${REFRESH_CACHE:-0}"

mkdir -p "${OUT_DIR}"

echo "Slot geometry probe"
echo "out_dir=${OUT_DIR}"
echo "features=geometry-only"
echo "hit: coverage>=${COVERAGE_THRESHOLD} purity>=${PURITY_THRESHOLD}; binary threshold_rel=${THRESHOLD_REL}"

extra_args=()
if [[ "${REFRESH_CACHE}" == "1" ]]; then
  extra_args+=(--refresh_cache)
fi

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
"${PYTHON}" analysis/slothead/train_slot_geometry_probe.py \
  --data "${DATA}" \
  --coco_root "${COCO_ROOT}" \
  --sa_checkpoint "${SA_CHECKPOINT}" \
  --out_dir "${OUT_DIR}" \
  --input_res "${INPUT_RES}" \
  --bs "${BS}" \
  --probe_bs "${PROBE_BS}" \
  --num_workers "${NUM_WORKERS}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --wd "${WD}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --dropout "${DROPOUT}" \
  --top_ks "${TOP_KS}" \
  --threshold_rel "${THRESHOLD_REL}" \
  --coverage_threshold "${COVERAGE_THRESHOLD}" \
  --purity_threshold "${PURITY_THRESHOLD}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  "${extra_args[@]}"

echo "Final metrics:"
echo "${OUT_DIR}/final_metrics.json"
echo "Checkpoint:"
echo "${OUT_DIR}/slot_geometry_probe_best.pt"
