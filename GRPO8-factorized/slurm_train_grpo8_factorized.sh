#!/usr/bin/env bash
#SBATCH --job-name=grpo8_fac
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --exclude=semois

set -euo pipefail
cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs GRPO8-factorized/checkpoints
RUN_NAME="${RUN_NAME:-grpo8_factorized_joint_margin_a1_e2_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-GRPO8-factorized/checkpoints/${RUN_NAME}}"
"${PYTHON:-.venv/bin/python}" GRPO8-factorized/train_grpo8_factorized.py \
  --require_cuda \
  --data "${DATA:-dataset/coco_rule_graph8_v2_area015_012_300_100_100/classification_dataset}" \
  --output_dir "${OUT_DIR}" --epochs "${EPOCHS:-40}" \
  --bs "${BS:-16}" --num_workers "${NUM_WORKERS:-4}" --lr "${LR:-3e-4}" --policy_lr "${POLICY_LR:-3e-4}" \
  --max_steps "${MAX_STEPS:-6}" --min_steps "${MIN_STEPS:-3}" --early_exit_conf "${EARLY_EXIT_CONF:-0.90}" \
  --classification_coef "${CLASSIFICATION_COEF:-1.0}" \
  --component_mil_coef "${COMPONENT_MIL_COEF:-1.0}" --component_loss_mode "${COMPONENT_LOSS_MODE:-positive_unknown}" \
  --pair_ce_coef "${PAIR_CE_COEF:-1.0}" --anchor_coef "${ANCHOR_COEF:-1.0}" \
  --evidence_coef "${EVIDENCE_COEF:-2.0}" --pair_coef "${PAIR_COEF:-1.0}" \
  --class_margin_coef "${CLASS_MARGIN_COEF:-1.0}" \
  --rank_discount "${RANK_DISCOUNT:-0.85}" --grpo_group_size "${GRPO_GROUP_SIZE:-4}" \
  --quick_limit_train "${QUICK_LIMIT_TRAIN:-0}" --quick_limit_val "${QUICK_LIMIT_VAL:-0}" --seed "${SEED:-17}"
echo "checkpoint: ${OUT_DIR}/selector_grpo_best.pt"
