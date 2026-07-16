#!/usr/bin/env bash
#SBATCH --job-name=grpo6_fac_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail
cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs
if [[ -z "${RUN_DIR:-}" ]]; then echo "RUN_DIR is required" >&2; exit 2; fi
OUT_DIR="${OUT_DIR:-${RUN_DIR}/bbox_eval_forced_top8}"
"${PYTHON:-.venv/bin/python}" GRPO6-factorized/evaluate_grpo6_factorized_bbox.py \
  --run_dir "${RUN_DIR}" --out_dir "${OUT_DIR}" --top_ks "3,4,8" \
  --hit_thresholds "0.2,0.4" --bs "${BS:-32}" --num_workers "${NUM_WORKERS:-4}"
echo "summary: ${OUT_DIR}/summary.json"
