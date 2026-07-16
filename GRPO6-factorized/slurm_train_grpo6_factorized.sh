#!/usr/bin/env bash
#SBATCH --job-name=grpo6_fac
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail
cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs GRPO6-factorized/checkpoints

RUN_NAME="${RUN_NAME:-grpo6_factorized_m3_p085_n090_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-GRPO6-factorized/checkpoints/${RUN_NAME}}"

"${PYTHON:-.venv/bin/python}" GRPO6-factorized/train_grpo6_factorized.py \
  --output_dir "${OUT_DIR}" \
  --epochs "${EPOCHS:-60}" --warmup_epochs "${WARMUP_EPOCHS:-15}" \
  --bs "${BS:-16}" --num_workers "${NUM_WORKERS:-4}" \
  --lr "${LR:-3e-4}" --policy_lr "${POLICY_LR:-3e-4}" \
  --max_steps "${MAX_STEPS:-6}" --min_steps 3 --early_exit_conf 0.85 \
  --component_mil_coef "${COMPONENT_MIL_COEF:-1.0}" \
  --component_mil_temperature "${COMPONENT_MIL_TEMPERATURE:-5.0}" \
  --component_pair_coef "${COMPONENT_PAIR_COEF:-1.0}" \
  --balanced_margin_coef "${BALANCED_MARGIN_COEF:-0.0}" \
  --necessity_coef "${NECESSITY_COEF:-0.0}" --novelty_coef "${NOVELTY_COEF:-0.0}" \
  --novelty_stop_threshold "${NOVELTY_STOP_THRESHOLD:-0.90}" \
  --grpo_group_size "${GRPO_GROUP_SIZE:-4}" \
  --quick_limit_train "${QUICK_LIMIT_TRAIN:-0}" --quick_limit_val "${QUICK_LIMIT_VAL:-0}" \
  --seed "${SEED:-8}"

echo "checkpoint: ${OUT_DIR}/selector_grpo_best.pt"
