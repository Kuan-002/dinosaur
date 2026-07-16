#!/usr/bin/env bash
#SBATCH --job-name=ac80
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs AC80/checkpoints

RUN_NAME="${RUN_NAME:-ac80_lr3e4_e60_m3_5_ce12_pc020_vc025_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-AC80/checkpoints/${RUN_NAME}}"
PYTHON="${PYTHON:-.venv/bin/python}"

SLOTHEAD80_CHECKPOINT="${SLOTHEAD80_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/SET80/checkpoints/slothead80_obj16_geo16_res48_20260709_192454/slothead_best.pt}"

if [[ -z "${SLOTHEAD80_CHECKPOINT}" ]]; then
  echo "SLOTHEAD80_CHECKPOINT must point to a fresh object-mode slothead checkpoint for the current dataset." >&2
  exit 2
fi

"${PYTHON}" AC80/train_ac80.py \
  --output_dir "${OUT_DIR}" \
  --data "${DATA:-/vol/biomedic3/kw1025/dinosaur/dataset/coco_top2_clean10_area006_004_600_200_200/classification_dataset}" \
  --sa_checkpoint "${SA_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt}" \
  --slothead_checkpoint "${SLOTHEAD80_CHECKPOINT}" \
  --slothead_mode "${SLOTHEAD_MODE:-u}" \
  --epochs "${EPOCHS:-60}" \
  --warmup_epochs "${WARMUP_EPOCHS:-3}" \
  --warmup_steps "${WARMUP_STEPS:-3}" \
  --bs "${BS:-16}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --lr "${LR:-3e-4}" \
  --wd "${WD:-1e-4}" \
  --max_steps "${MAX_STEPS:-8}" \
  --min_steps "${MIN_STEPS:-2}" \
  --free_slots "${FREE_SLOTS:-5}" \
  --min_free_slots "${MIN_FREE_SLOTS:-3}" \
  --max_free_slots "${MAX_FREE_SLOTS:-5}" \
  --classification_coef "${CLASSIFICATION_COEF:-1.20}" \
  --objectness_coef "${OBJECTNESS_COEF:-0.6}" \
  --geometry_coef "${GEOMETRY_COEF:-0.2}" \
  --residual_coef "${RESIDUAL_COEF:-0.25}" \
  --selected_count_coef "${SELECTED_COUNT_COEF:-0.01}" \
  --over_select_coef "${OVER_SELECT_COEF:-0.005}" \
  --good_stop_bonus "${GOOD_STOP_BONUS:-0.18}" \
  --good_stop_obj_threshold "${GOOD_STOP_OBJ_THRESHOLD:-0.45}" \
  --good_stop_res_threshold "${GOOD_STOP_RES_THRESHOLD:-0.25}" \
  --geometry_grid_size "${GEOMETRY_GRID_SIZE:-14}" \
  --policy_coef "${POLICY_COEF:-0.20}" \
  --value_coef "${VALUE_COEF:-0.25}" \
  --entropy_coef "${ENTROPY_COEF:-0.0}" \
  --label_smoothing "${LABEL_SMOOTHING:-0.01}" \
  --confidence_penalty "${CONFIDENCE_PENALTY:-0.03}" \
  --confidence_penalty_threshold "${CONFIDENCE_PENALTY_THRESHOLD:-0.85}" \
  --early_confidence_penalty_until_slots "${EARLY_CONFIDENCE_PENALTY_UNTIL_SLOTS:-3}" \
  --seed "${SEED:-8}"

echo "AC80 checkpoint: ${OUT_DIR}/selector_ac_best.pt"
