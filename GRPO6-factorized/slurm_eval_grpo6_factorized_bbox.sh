#!/usr/bin/env bash
#SBATCH --job-name=g6fac_bbox
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus48
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --exclude=semois

set -euo pipefail
cd /vol/biomedic3/kw1025/dinosaur
if [[ -z "${RUN_DIR:-}" ]]; then
  echo "RUN_DIR is required" >&2
  exit 2
fi
"${PYTHON:-.venv/bin/python}" GRPO6-factorized/evaluate_grpo6_factorized_bbox.py \
  --run_dir "${RUN_DIR}" \
  --split "${SPLIT:-test}" \
  --top_ks "${TOP_KS:-2,3,4}" \
  --hit_thresholds "${HIT_THRESHOLDS:-0.2,0.4}" \
  --out_dir "${OUT_DIR:-${RUN_DIR}/${SPLIT:-test}_bbox_top2_3_4_thr02_04}" \
  --bs "${BS:-32}" --num_workers "${NUM_WORKERS:-0}"
