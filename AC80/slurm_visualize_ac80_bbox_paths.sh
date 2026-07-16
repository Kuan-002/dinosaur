#!/usr/bin/env bash
#SBATCH --job-name=ac80vis
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
SPLIT="${SPLIT:-test}"
MIN_STEPS="${MIN_STEPS:-3}"
EARLY_EXIT_CONF="${EARLY_EXIT_CONF:-0.85}"
PER_CLASS_CORRECT="${PER_CLASS_CORRECT:-15}"
PER_CLASS_WRONG="${PER_CLASS_WRONG:-5}"
OUT_DIR="${OUT_DIR:-${RUN_DIR}/visualizations/${SPLIT}_slot_paths_bbox_min${MIN_STEPS}_p${EARLY_EXIT_CONF}_c${PER_CLASS_CORRECT}_w${PER_CLASS_WRONG}}"

"${PYTHON}" AC80/visualize_ac80_selector_paths.py \
  --run_dir "${RUN_DIR}" \
  --checkpoint "${CHECKPOINT:-selector_ac_best.pt}" \
  --split "${SPLIT}" \
  --out_dir "${OUT_DIR}" \
  --per_class_correct "${PER_CLASS_CORRECT}" \
  --per_class_wrong "${PER_CLASS_WRONG}" \
  --min_steps_override "${MIN_STEPS}" \
  --early_exit_conf_override "${EARLY_EXIT_CONF}" \
  --hit_threshold "${HIT_THRESHOLD:-0.20}" \
  --threshold_rel "${THRESHOLD_REL:-0.5}" \
  --device "${DEVICE:-auto}" \
  --seed "${SEED:-42}"

echo "AC80 bbox slot-path visualization: ${OUT_DIR}/index.html"
