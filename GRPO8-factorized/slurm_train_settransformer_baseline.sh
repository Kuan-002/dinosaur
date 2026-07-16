#!/usr/bin/env bash
#SBATCH --job-name=set8_base
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail
cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs GRPO8-factorized/checkpoints

RUN_NAME="${RUN_NAME:-settransformer8_raw_seed8_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-GRPO8-factorized/checkpoints/${RUN_NAME}}"
DATA="${DATA:-dataset/coco_rule_graph8_v2_area015_012_300_100_100/classification_dataset}"
PYTHON="${PYTHON:-.venv/bin/python}"

"${PYTHON}" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable; refusing CPU fallback"'
"${PYTHON}" SET_full/train_set_full.py \
  --output_dir "${OUT_DIR}" --data "${DATA}" \
  --epochs "${EPOCHS:-80}" --bs "${BS:-32}" --num_workers "${NUM_WORKERS:-4}" \
  --lr "${LR:-3e-4}" --wd "${WD:-1e-4}" --seed "${SEED:-8}" \
  --quick_limit_train "${QUICK_LIMIT_TRAIN:-0}" --quick_limit_val "${QUICK_LIMIT_VAL:-0}"

"${PYTHON}" scripts/evaluate_set_classification.py \
  --variant full --checkpoint "${OUT_DIR}/set_full_best.pt" --data "${DATA}" --split test \
  --out_dir "${OUT_DIR}/test_classification" --bs "${EVAL_BS:-32}" --num_workers "${NUM_WORKERS:-4}"

for threshold in 0.2 0.4; do
  "${PYTHON}" scripts/evaluate_set_bbox.py \
    --variant full --checkpoint "${OUT_DIR}/set_full_best.pt" --data "${DATA}" --split test \
    --out_dir "${OUT_DIR}/bbox_eval_predicted_greedy_thr${threshold/./}" \
    --top_ks 3,4 --hit_threshold "${threshold}" --rank_score predicted_greedy_prob \
    --bs "${EVAL_BS:-32}" --num_workers "${NUM_WORKERS:-4}"
done

echo "checkpoint: ${OUT_DIR}/set_full_best.pt"
