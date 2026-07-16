#!/usr/bin/env bash
#SBATCH --job-name=ac80cls
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs

PYTHON="${PYTHON:-.venv/bin/python}"
RUN_DIR="${RUN_DIR:-AC80/checkpoints/ac80_lr3e4_e60_m3_5_ce12_pc020_vc025_20260711_203906}"
MIN_STEPS="${MIN_STEPS:-3}"
EARLY_EXIT_CONF="${EARLY_EXIT_CONF:-0.85}"
OUT_JSON="${OUT_JSON:-${RUN_DIR}/ac80_classification_eval_min${MIN_STEPS}_p${EARLY_EXIT_CONF}.json}"

"${PYTHON}" AC80/evaluate_ac80_classification.py \
  --run_dir "${RUN_DIR}" \
  --checkpoint "${CHECKPOINT:-selector_ac_best.pt}" \
  --splits "${SPLITS:-valid,test}" \
  --min_steps "${MIN_STEPS}" \
  --early_exit_conf "${EARLY_EXIT_CONF}" \
  --out_json "${OUT_JSON}" \
  --bs "${BS:-32}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --device "${DEVICE:-auto}"

echo "AC80 classification eval: ${OUT_JSON}"
