#!/usr/bin/env bash
#SBATCH --job-name=grpo56
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs GRPO56/checkpoints

RUN_NAME="${RUN_NAME:-grpo56_u_m2_free3_count035_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-GRPO56/checkpoints/${RUN_NAME}}"
PYTHON="${PYTHON:-.venv/bin/python}"

REWARD_PROBE_CHECKPOINT="${REWARD_PROBE_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/SET56/checkpoints/set56_u_20260708_030719/set56_best.pt}"

"${PYTHON}" GRPO56/train_grpo56.py \
  --output_dir "${OUT_DIR}" \
  --data "${DATA:-/vol/biomedic3/kw1025/dinosaur/analysis/coco_top2_clean_scenes_anchor009_evidence005_10cls_450_150_150/classification_dataset}" \
  --sa_checkpoint "${SA_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt}" \
  --structured_checkpoint "${STRUCTURED56_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/analysis/structured_slot_bottleneck/structured_slot_bottleneck_10cls450_20260708_011844/structured_slot_bottleneck_best.pt}" \
  --structured_mode "${STRUCTURED_MODE:-u}" \
  --reward_probe_checkpoint "${REWARD_PROBE_CHECKPOINT}" \
  --epochs "${EPOCHS:-40}" \
  --warmup_epochs "${WARMUP_EPOCHS:-1}" \
  --bs "${BS:-16}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --lr "${LR:-5e-4}" \
  --wd "${WD:-1e-4}" \
  --max_steps "${MAX_STEPS:-8}" \
  --min_steps "${MIN_STEPS:-2}" \
  --early_exit_conf "${EARLY_EXIT_CONF:-0.9}" \
  --reward_source "${REWARD_SOURCE:-probe_subset_margin}" \
  --subset_contrast "${SUBSET_CONTRAST:-complement}" \
  --subset_contrast_weight "${SUBSET_CONTRAST_WEIGHT:-1.0}" \
  --step_margin_reward_weights "${STEP_MARGIN_REWARD_WEIGHTS:-0.6,0.3,0.15}" \
  --grpo_group_size "${GRPO_GROUP_SIZE:-4}" \
  --free_slots "${FREE_SLOTS:-3}" \
  --count_penalty "${COUNT_PENALTY:-0.35}" \
  --policy_coef "${POLICY_COEF:-1.0}" \
  --entropy_coef "${ENTROPY_COEF:-0.01}" \
  --seed "${SEED:-8}"

echo "GRPO56 checkpoint: ${OUT_DIR}/selector_grpo_best.pt"
