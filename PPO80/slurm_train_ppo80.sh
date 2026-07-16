#!/usr/bin/env bash
#SBATCH --job-name=ppo80
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs PPO80/checkpoints

RUN_NAME="${RUN_NAME:-ppo80_slothead_u_rewardv3_acc_steps_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-PPO80/checkpoints/${RUN_NAME}}"
PYTHON="${PYTHON:-.venv/bin/python}"

SLOTHEAD80_CHECKPOINT="${SLOTHEAD80_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/SET80/checkpoints/slothead80_obj16_geo16_res48_20260709_192454/slothead_best.pt}"

if [[ -z "${SLOTHEAD80_CHECKPOINT}" ]]; then
  echo "SLOTHEAD80_CHECKPOINT must point to a fresh object-mode slothead checkpoint for the current dataset." >&2
  exit 2
fi

"${PYTHON}" PPO80/train_ppo80.py \
  --output_dir "${OUT_DIR}" \
  --data "${DATA:-/vol/biomedic3/kw1025/dinosaur/dataset/coco_top2_clean10_area006_004_600_200_200/classification_dataset}" \
  --sa_checkpoint "${SA_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt}" \
  --slothead_checkpoint "${SLOTHEAD80_CHECKPOINT}" \
  --slothead_mode "${SLOTHEAD_MODE:-u}" \
  --epochs "${EPOCHS:-40}" \
  --warmup_epochs "${WARMUP_EPOCHS:-3}" \
  --warmup_steps "${WARMUP_STEPS:-3}" \
  --bs "${BS:-16}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --lr "${LR:-5e-4}" \
  --wd "${WD:-1e-4}" \
  --max_steps "${MAX_STEPS:-8}" \
  --min_steps "${MIN_STEPS:-2}" \
  --ppo_epochs "${PPO_EPOCHS:-4}" \
  --ppo_clip_eps "${PPO_CLIP_EPS:-0.20}" \
  --ppo_log_ratio_clip "${PPO_LOG_RATIO_CLIP:-10.0}" \
  --value_coef "${VALUE_COEF:-0.50}" \
  --free_slots "${FREE_SLOTS:-4}" \
  --min_free_slots "${MIN_FREE_SLOTS:-3}" \
  --max_free_slots "${MAX_FREE_SLOTS:-4}" \
  --classification_coef "${CLASSIFICATION_COEF:-1.20}" \
  --objectness_coef "${OBJECTNESS_COEF:-0.6}" \
  --geometry_coef "${GEOMETRY_COEF:-0.2}" \
  --residual_coef "${RESIDUAL_COEF:-0.25}" \
  --selected_count_coef "${SELECTED_COUNT_COEF:-0.04}" \
  --over_select_coef "${OVER_SELECT_COEF:-0.04}" \
  --good_stop_bonus "${GOOD_STOP_BONUS:-0.18}" \
  --good_stop_obj_threshold "${GOOD_STOP_OBJ_THRESHOLD:-0.45}" \
  --good_stop_res_threshold "${GOOD_STOP_RES_THRESHOLD:-0.25}" \
  --geometry_grid_size "${GEOMETRY_GRID_SIZE:-14}" \
  --policy_coef "${POLICY_COEF:-0.25}" \
  --entropy_coef "${ENTROPY_COEF:-0.0}" \
  --label_smoothing "${LABEL_SMOOTHING:-0.01}" \
  --confidence_penalty "${CONFIDENCE_PENALTY:-0.03}" \
  --confidence_penalty_threshold "${CONFIDENCE_PENALTY_THRESHOLD:-0.85}" \
  --early_confidence_penalty_until_slots "${EARLY_CONFIDENCE_PENALTY_UNTIL_SLOTS:-3}" \
  --seed "${SEED:-8}"

echo "PPO80 checkpoint: ${OUT_DIR}/selector_ppo_best.pt"
