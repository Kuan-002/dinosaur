#!/usr/bin/env bash
#SBATCH --job-name=slothead80recon
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs SET80/checkpoints

RUN_NAME="${RUN_NAME:-slothead80_recon_only_obj16_geo16_res48_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-SET80/checkpoints/${RUN_NAME}}"

LAMBDA_OBJ=0.0 \
LAMBDA_GEO=0.0 \
LAMBDA_CAT=0.0 \
LAMBDA_ORTH=0.0 \
LAMBDA_REC="${LAMBDA_REC:-1.0}" \
OUT_DIR="${OUT_DIR}" \
RUN_NAME="${RUN_NAME}" \
bash SET80/slurm_train_slothead80.sh

echo "SET80 recon-only slothead checkpoint: ${OUT_DIR}/slothead_best.pt"
