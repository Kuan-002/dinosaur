#!/usr/bin/env bash
#SBATCH --job-name=grpo80
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs GRPO80/checkpoints

RUN_NAME="${RUN_NAME:-grpo80_slothead_u_m3_4_lightcount_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-GRPO80/checkpoints/${RUN_NAME}}"
PYTHON="${PYTHON:-.venv/bin/python}"

STRUCTURED80_CHECKPOINT="${STRUCTURED80_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/SET80/checkpoints/structured80_obj16_geo16_res48_20260708_185728/structured_slot_bottleneck_best.pt}"

"${PYTHON}" GRPO80/train_grpo80.py \
  --output_dir "${OUT_DIR}" \
  --data "${DATA:-/vol/biomedic3/kw1025/dinosaur/analysis/coco_top2_clean_scenes_anchor009_evidence005_10cls_450_150_150/classification_dataset}" \
  --sa_checkpoint "${SA_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt}" \
  --structured_checkpoint "${STRUCTURED80_CHECKPOINT}" \
  --structured_mode "${STRUCTURED_MODE:-u}" \
  --epochs "${EPOCHS:-40}" \
  --warmup_epochs "${WARMUP_EPOCHS:-1}" \
  --warmup_steps "${WARMUP_STEPS:-3}" \
  --bs "${BS:-16}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --lr "${LR:-5e-4}" \
  --wd "${WD:-1e-4}" \
  --max_steps "${MAX_STEPS:-6}" \
  --min_steps "${MIN_STEPS:-2}" \
  --reward_source "${REWARD_SOURCE:-classifier_logprob}" \
  --grpo_group_size "${GRPO_GROUP_SIZE:-4}" \
  --free_slots "${FREE_SLOTS:-4}" \
  --min_free_slots "${MIN_FREE_SLOTS:-3}" \
  --max_free_slots "${MAX_FREE_SLOTS:-4}" \
  --count_penalty "${COUNT_PENALTY:-0.08}" \
  --policy_coef "${POLICY_COEF:-1.0}" \
  --entropy_coef "${ENTROPY_COEF:-0.01}" \
  --label_smoothing "${LABEL_SMOOTHING:-0.05}" \
  --confidence_penalty "${CONFIDENCE_PENALTY:-0.03}" \
  --confidence_penalty_threshold "${CONFIDENCE_PENALTY_THRESHOLD:-0.85}" \
  --early_confidence_penalty_until_slots "${EARLY_CONFIDENCE_PENALTY_UNTIL_SLOTS:-3}" \
  --seed "${SEED:-8}"

echo "GRPO80 checkpoint: ${OUT_DIR}/selector_grpo_best.pt"
