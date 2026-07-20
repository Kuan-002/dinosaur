#!/usr/bin/env bash
#SBATCH --job-name=grpo6_fac
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --exclude=semois

set -euo pipefail
cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs GRPO6-factorized/checkpoints
RUN_NAME="${RUN_NAME:-grpo6_factorized_joint_accsel_margin_cpair_seed8_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-GRPO6-factorized/checkpoints/${RUN_NAME}}"
"${PYTHON:-.venv/bin/python}" GRPO6-factorized/train_grpo6_factorized.py \
  --require_cuda \
  --data "${DATA:-dataset/coco_compositional_pair6_clean_300_100_100/classification_dataset}" \
  --output_dir "${OUT_DIR}" --epochs "${EPOCHS:-40}" \
  --bs "${BS:-16}" --num_workers "${NUM_WORKERS:-4}" --lr "${LR:-3e-4}" --policy_lr "${POLICY_LR:-3e-4}" \
  --max_steps "${MAX_STEPS:-6}" --min_steps "${MIN_STEPS:-3}" \
  --classification_coef "${CLASSIFICATION_COEF:-1.0}" \
  --component_mil_coef "${COMPONENT_MIL_COEF:-1.0}" --component_loss_mode "${COMPONENT_LOSS_MODE:-positive_unknown}" \
  --pair_ce_coef "${PAIR_CE_COEF:-1.0}" --class_margin_coef "${CLASS_MARGIN_COEF:-1.0}" \
  --object_a_coef "${OBJECT_A_COEF:-0.0}" --object_b_coef "${OBJECT_B_COEF:-0.0}" \
  --component_pair_coef "${COMPONENT_PAIR_COEF:-1.0}" \
  --rank_discount "${RANK_DISCOUNT:-0.85}" \
  --grpo_group_size "${GRPO_GROUP_SIZE:-4}" \
  --quick_limit_train "${QUICK_LIMIT_TRAIN:-0}" --quick_limit_val "${QUICK_LIMIT_VAL:-0}" --seed "${SEED:-8}"
echo "checkpoint: ${OUT_DIR}/selector_grpo_best.pt"
