#!/usr/bin/env bash
#SBATCH --job-name=grpo8_fac_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00

set -euo pipefail
cd /vol/biomedic3/kw1025/dinosaur
if [[ -z "${RUN_DIR:-}" ]]; then echo "RUN_DIR is required" >&2; exit 2; fi
"${PYTHON:-.venv/bin/python}" GRPO8-factorized/evaluate_grpo8_factorized_bbox.py \
  --run_dir "${RUN_DIR}" --out_dir "${OUT_DIR:-${RUN_DIR}/bbox_eval_forced_top3_top4}" \
  --top_ks 3,4 --hit_thresholds 0.2,0.4 --bs "${BS:-32}" --num_workers "${NUM_WORKERS:-4}"

