#!/usr/bin/env bash
#SBATCH --job-name=concept_bottleneck
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus24
#SBATCH --time=08:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs

RUN_NAME="${RUN_NAME:-concept_bottleneck_$(date +%Y%m%d_%H%M%S)}"
PYTHON="${PYTHON:-.venv/bin/python}"

echo "host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

"${PYTHON}" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; aborting instead of running on CPU")
PY

"${PYTHON}" analysis/train_concept_bottleneck_classifier.py \
  --output_dir "analysis/concept_bottleneck_classifier/${RUN_NAME}" \
  --data "${DATA:-/vol/biomedic3/kw1025/dinosaur/analysis/coco_top2_clean_scenes_anchor009_evidence005_10cls_450_150_150/classification_dataset}" \
  --checkpoint "${SA_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt}" \
  --probe_checkpoint "${PROBE_CHECKPOINT:-/vol/biomedic3/kw1025/dinosaur/checkpoints/slot_concept_probe/slot_concept_probe_20260706_184027/slot_concept_probe_best.pt}" \
  --epochs "${EPOCHS:-100}" \
  --bs "${BS:-64}" \
  --head_bs "${HEAD_BS:-256}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --lr "${LR:-1e-3}" \
  --wd "${WD:-1e-4}" \
  --hidden_dim "${HIDDEN_DIM:-256}" \
  --dropout "${DROPOUT:-0.1}" \
  ${REFRESH_CACHE:+--refresh_cache}
