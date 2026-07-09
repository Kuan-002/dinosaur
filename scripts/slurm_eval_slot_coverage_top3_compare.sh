#!/usr/bin/env bash
#SBATCH --job-name=slot_cov_top3
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpus24
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00

set -euo pipefail

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs

RUN_NAME="${RUN_NAME:-slot_coverage_top3_compare_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-analysis/slot_coverage_compare/${RUN_NAME}}"
PYTHON="${PYTHON:-.venv/bin/python}"

SPLIT="${SPLIT:-test}"
TOP_K="${TOP_K:-3}"
THRESHOLD="${THRESHOLD:-0.5}"
THRESHOLD_REL="${THRESHOLD_REL:-0.5}"
BS="${BS:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-auto}"
CONTACT_SHEETS="${CONTACT_SHEETS:-0}"
MAX_ITEMS="${MAX_ITEMS:-0}"

BBOX_PROBE_CHECKPOINT="${BBOX_PROBE_CHECKPOINT:-checkpoints/bbox-settransformer/bbox_settransformer_20260706_162939/bbox_settransformer_best.pt}"
SETTRANSFORMER_PROBE_CHECKPOINT="${SETTRANSFORMER_PROBE_CHECKPOINT:-checkpoints/settransformer/settransformer_anchor_evidence_20260705_174127/settransformer_discriminative_best.pt}"

BBOX_OUT="${OUT_ROOT}/bbox_supervised"
SETTRANSFORMER_OUT="${OUT_ROOT}/set_transformer"
RANDOM_OUT="${OUT_ROOT}/random_selection"
SUMMARY_OUT="${OUT_ROOT}/summary"

mkdir -p "${BBOX_OUT}" "${SETTRANSFORMER_OUT}" "${RANDOM_OUT}" "${SUMMARY_OUT}"

echo "Writing outputs under ${OUT_ROOT}"
echo "split=${SPLIT} top_k=${TOP_K} threshold=${THRESHOLD} threshold_rel=${THRESHOLD_REL}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
"${PYTHON}" analysis/set_transformer_diagnostics/evaluate_bbox_settransformer.py \
  --probe_checkpoint "${BBOX_PROBE_CHECKPOINT}" \
  --out_dir "${BBOX_OUT}" \
  --split "${SPLIT}" \
  --top_k "${TOP_K}" \
  --hit_threshold "${THRESHOLD}" \
  --threshold_rel "${THRESHOLD_REL}" \
  --contact_sheets "${CONTACT_SHEETS}" \
  --max_items "${MAX_ITEMS}" \
  --bs "${BS}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
"${PYTHON}" analysis/set_transformer_diagnostics/evaluate_set_transformer_bbox.py \
  --probe_checkpoint "${SETTRANSFORMER_PROBE_CHECKPOINT}" \
  --out_dir "${SETTRANSFORMER_OUT}" \
  --split "${SPLIT}" \
  --top_k "${TOP_K}" \
  --threshold_rel "${THRESHOLD_REL}" \
  --contact_sheets "${CONTACT_SHEETS}" \
  --max_items "${MAX_ITEMS}" \
  --bs "${BS}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" \
"${PYTHON}" analysis/set_transformer_diagnostics/evaluate_random_slot_baseline.py \
  --out_dir "${RANDOM_OUT}" \
  --split "${SPLIT}" \
  --top_k "${TOP_K}" \
  --thresholds "${THRESHOLD}" \
  --threshold_rel "${THRESHOLD_REL}" \
  --max_items "${MAX_ITEMS}" \
  --bs "${BS}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}"

"${PYTHON}" analysis/set_transformer_diagnostics/summarize_slot_coverage_methods.py \
  --out_dir "${SUMMARY_OUT}" \
  --threshold "${THRESHOLD}" \
  --method "bbox_supervised:${BBOX_OUT}/bbox_settransformer_eval.csv" \
  --method "set_transformer:${SETTRANSFORMER_OUT}/bbox_eval.csv" \
  --method "random_selection:${RANDOM_OUT}/random_slot_baseline.csv"

echo "Final per-class comparison:"
echo "${SUMMARY_OUT}/slot_coverage_method_compare.csv"
sed -n '1,40p' "${SUMMARY_OUT}/slot_coverage_method_compare.csv"
