#!/usr/bin/env bash
#SBATCH --job-name=set8_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

set -euo pipefail
cd /vol/biomedic3/kw1025/dinosaur
RUN_DIR="${RUN_DIR:-GRPO8-factorized/checkpoints/settransformer8_raw_predgreedy_seed8_20260716}"
DATA="${DATA:-dataset/coco_rule_graph8_v2_area015_012_300_100_100/classification_dataset}"
PYTHON="${PYTHON:-.venv/bin/python}"
"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
for threshold in 0.2 0.4; do
  "${PYTHON}" scripts/evaluate_set_bbox.py \
    --variant full --checkpoint "${RUN_DIR}/set_full_best.pt" --data "${DATA}" --split test \
    --out_dir "${RUN_DIR}/bbox_eval_predicted_greedy_thr${threshold/./}" \
    --top_ks 3,4 --hit_threshold "${threshold}" --rank_score predicted_greedy_prob \
    --bs "${BS:-32}" --num_workers "${NUM_WORKERS:-4}"
done
