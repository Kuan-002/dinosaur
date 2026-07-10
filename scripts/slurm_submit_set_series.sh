#!/usr/bin/env bash
#SBATCH --job-name=submit_set_series
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --time=00:20:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs SET_full/checkpoints SET56/checkpoints SET80/checkpoints SET112/checkpoints

RUN_TAG="${RUN_TAG:-600_200_200_v1}"
DATA="/vol/biomedic3/kw1025/dinosaur/dataset/coco_top2_clean10_area006_004_600_200_200/classification_dataset"
COCO_ROOT="/vol/biomedic3/kw1025/dinosaur/dataset/coco2017"
SA_CHECKPOINT="/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt"
PYTHON="${PYTHON:-.venv/bin/python}"

SLOTHEAD56_RUN="slothead56_obj16_geo16_res24_${RUN_TAG}"
SLOTHEAD80_RUN="slothead80_obj16_geo16_res48_${RUN_TAG}"
SLOTHEAD112_RUN="slothead112_obj16_geo32_res64_${RUN_TAG}"
SETFULL_RUN="set_full_raw_${RUN_TAG}"
SET56_RUN="set56_u_${RUN_TAG}"
SET80_RUN="set80_u_${RUN_TAG}"
SET112_RUN="set112_u_${RUN_TAG}"

SLOTHEAD56_DIR="/vol/biomedic3/kw1025/dinosaur/SET56/checkpoints/${SLOTHEAD56_RUN}"
SLOTHEAD80_DIR="/vol/biomedic3/kw1025/dinosaur/SET80/checkpoints/${SLOTHEAD80_RUN}"
SLOTHEAD112_DIR="/vol/biomedic3/kw1025/dinosaur/SET112/checkpoints/${SLOTHEAD112_RUN}"
SETFULL_DIR="/vol/biomedic3/kw1025/dinosaur/SET_full/checkpoints/${SETFULL_RUN}"
SET56_DIR="/vol/biomedic3/kw1025/dinosaur/SET56/checkpoints/${SET56_RUN}"
SET80_DIR="/vol/biomedic3/kw1025/dinosaur/SET80/checkpoints/${SET80_RUN}"
SET112_DIR="/vol/biomedic3/kw1025/dinosaur/SET112/checkpoints/${SET112_RUN}"

SLOTHEAD56_CKPT="${SLOTHEAD56_DIR}/slothead_best.pt"
SLOTHEAD80_CKPT="${SLOTHEAD80_DIR}/slothead_best.pt"
SLOTHEAD112_CKPT="${SLOTHEAD112_DIR}/slothead_best.pt"
SETFULL_CKPT="${SETFULL_DIR}/set_full_best.pt"
SET56_CKPT="${SET56_DIR}/set56_best.pt"
SET80_CKPT="${SET80_DIR}/set80_best.pt"
SET112_CKPT="${SET112_DIR}/set112_best.pt"

COMMON_EXPORT="ALL,PYTHON=${PYTHON},DATA=${DATA},COCO_ROOT=${COCO_ROOT},SA_CHECKPOINT=${SA_CHECKPOINT}"

echo "RUN_TAG=${RUN_TAG}"
echo "DATA=${DATA}"
echo "SA_CHECKPOINT=${SA_CHECKPOINT}"

slot56_job=$(sbatch --parsable --export="${COMMON_EXPORT},RUN_NAME=${SLOTHEAD56_RUN},OUT_DIR=${SLOTHEAD56_DIR}" SET56/slurm_train_slothead56.sh)
slot80_job=$(sbatch --parsable --export="${COMMON_EXPORT},RUN_NAME=${SLOTHEAD80_RUN},OUT_DIR=${SLOTHEAD80_DIR}" SET80/slurm_train_slothead80.sh)
slot112_job=$(sbatch --parsable --export="${COMMON_EXPORT},RUN_NAME=${SLOTHEAD112_RUN},OUT_DIR=${SLOTHEAD112_DIR}" SET112/slurm_train_slothead112.sh)
setfull_job=$(sbatch --parsable --export="${COMMON_EXPORT},RUN_NAME=${SETFULL_RUN},OUT_DIR=${SETFULL_DIR}" SET_full/slurm_train_set_full.sh)

set56_job=$(sbatch --parsable --dependency="afterok:${slot56_job}" --export="${COMMON_EXPORT},RUN_NAME=${SET56_RUN},OUT_DIR=${SET56_DIR},SLOTHEAD56_CHECKPOINT=${SLOTHEAD56_CKPT},SLOTHEAD_MODE=u" SET56/slurm_train_set56.sh)
set80_job=$(sbatch --parsable --dependency="afterok:${slot80_job}" --export="${COMMON_EXPORT},RUN_NAME=${SET80_RUN},OUT_DIR=${SET80_DIR},SLOTHEAD80_CHECKPOINT=${SLOTHEAD80_CKPT},SLOTHEAD_MODE=u" SET80/slurm_train_set80.sh)
set112_job=$(sbatch --parsable --dependency="afterok:${slot112_job}" --export="${COMMON_EXPORT},RUN_NAME=${SET112_RUN},OUT_DIR=${SET112_DIR},SLOTHEAD112_CHECKPOINT=${SLOTHEAD112_CKPT},SLOTHEAD_MODE=u" SET112/slurm_train_set112.sh)

evalfull_job=$(sbatch --parsable --dependency="afterok:${setfull_job}" --export="${COMMON_EXPORT},VARIANT=full,CHECKPOINT=${SETFULL_CKPT},OUT_DIR=${SETFULL_DIR}/test_bbox_at3_at4_thr04,HIT_THRESHOLD=0.4" scripts/slurm_eval_set_bbox.sh)
eval56_job=$(sbatch --parsable --dependency="afterok:${set56_job}" --export="${COMMON_EXPORT},VARIANT=56,CHECKPOINT=${SET56_CKPT},SLOTHEAD_CHECKPOINT=${SLOTHEAD56_CKPT},SLOTHEAD_MODE=u,OUT_DIR=${SET56_DIR}/test_bbox_at3_at4_thr04,HIT_THRESHOLD=0.4" scripts/slurm_eval_set_bbox.sh)
eval80_job=$(sbatch --parsable --dependency="afterok:${set80_job}" --export="${COMMON_EXPORT},VARIANT=80,CHECKPOINT=${SET80_CKPT},SLOTHEAD_CHECKPOINT=${SLOTHEAD80_CKPT},SLOTHEAD_MODE=u,OUT_DIR=${SET80_DIR}/test_bbox_at3_at4_thr04,HIT_THRESHOLD=0.4" scripts/slurm_eval_set_bbox.sh)
eval112_job=$(sbatch --parsable --dependency="afterok:${set112_job}" --export="${COMMON_EXPORT},VARIANT=112,CHECKPOINT=${SET112_CKPT},SLOTHEAD_CHECKPOINT=${SLOTHEAD112_CKPT},SLOTHEAD_MODE=u,OUT_DIR=${SET112_DIR}/test_bbox_at3_at4_thr04,HIT_THRESHOLD=0.4" scripts/slurm_eval_set_bbox.sh)

cat <<EOF
Submitted SET series.

slothead56 job: ${slot56_job} -> ${SLOTHEAD56_CKPT}
slothead80 job: ${slot80_job} -> ${SLOTHEAD80_CKPT}
slothead112 job: ${slot112_job} -> ${SLOTHEAD112_CKPT}
SET_full job:   ${setfull_job} -> ${SETFULL_CKPT}
SET56 job:      ${set56_job} -> ${SET56_CKPT}
SET80 job:      ${set80_job} -> ${SET80_CKPT}
SET112 job:     ${set112_job} -> ${SET112_CKPT}

@3/@4 threshold=0.4 eval jobs:
SET_full eval:  ${evalfull_job} -> ${SETFULL_DIR}/test_bbox_at3_at4_thr04/summary.json
SET56 eval:     ${eval56_job} -> ${SET56_DIR}/test_bbox_at3_at4_thr04/summary.json
SET80 eval:     ${eval80_job} -> ${SET80_DIR}/test_bbox_at3_at4_thr04/summary.json
SET112 eval:    ${eval112_job} -> ${SET112_DIR}/test_bbox_at3_at4_thr04/summary.json
EOF
