#!/bin/bash
#SBATCH --job-name=grpo_settr174127
#SBATCH --output=logs/%x.%N.%j.out
#SBATCH --error=logs/%x.%N.%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=gpus24
#SBATCH --time=08:00:00

set -euo pipefail
set -x

cd /vol/biomedic3/kw1025/dinosaur
mkdir -p logs checkpoints

PYTHON=${PYTHON:-/vol/biomedic3/kw1025/dinosaur/.venv/bin/python}
DATA=${DATA:-/vol/biomedic3/kw1025/dinosaur/analysis/coco_top2_clean_scenes_anchor009_evidence005_10cls_450_150_150/classification_dataset}
SA_CKPT=${SA_CKPT:-/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt}
PROBE_CKPT=${PROBE_CKPT:-/vol/biomedic3/kw1025/dinosaur/checkpoints/settransformer/settransformer_anchor_evidence_20260705_174127/settransformer_discriminative_best.pt}
RUN_NAME=${RUN_NAME:-grpo_selector_disc_settr174127_m2_free3_count035_stepdelta_$(date +%Y%m%d_%H%M%S)}
OUT_DIR=${OUT_DIR:-/vol/biomedic3/kw1025/dinosaur/checkpoints/${RUN_NAME}}

EPOCHS=${EPOCHS:-40}
WARMUP_EPOCHS=${WARMUP_EPOCHS:-0}
BS=${BS:-16}
LR=${LR:-5e-4}
WD=${WD:-1e-4}
HIDDEN_DIM=${HIDDEN_DIM:-256}
POLICY_DIM=${POLICY_DIM:-256}
DROPOUT=${DROPOUT:-0.1}
MAX_STEPS=${MAX_STEPS:-8}
MIN_STEPS=${MIN_STEPS:-2}
EARLY_EXIT_CONF=${EARLY_EXIT_CONF:-0.85}
SUBSET_CONTRAST=${SUBSET_CONTRAST:-none}
SUBSET_CONTRAST_WEIGHT=${SUBSET_CONTRAST_WEIGHT:-1.0}
STEP_MARGIN_REWARD_WEIGHTS=${STEP_MARGIN_REWARD_WEIGHTS:-0.6,0.3,0.15}
GRPO_GROUP_SIZE=${GRPO_GROUP_SIZE:-4}
FREE_SLOTS=${FREE_SLOTS:-3}
COUNT_PENALTY=${COUNT_PENALTY:-0.35}
PROBE_REWARD_CLIP=${PROBE_REWARD_CLIP:-10.0}
POS_DIM=${POS_DIM:-0}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-8}

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
echo "Repository: $(pwd)"
echo "Data: ${DATA}"
echo "Frozen SA checkpoint: ${SA_CKPT}"
echo "Frozen discriminative SetTransformer reward probe: ${PROBE_CKPT}"
echo "Output dir: ${OUT_DIR}"

${PYTHON} - <<PY
import torch
from pathlib import Path

sa_path = Path("${SA_CKPT}")
probe_path = Path("${PROBE_CKPT}")
for path in (sa_path, probe_path):
    if not path.exists():
        raise FileNotFoundError(path)

sa = torch.load(sa_path, map_location="cpu", weights_only=False)
probe = torch.load(probe_path, map_location="cpu", weights_only=False)
print("SA num_slots=", sa.get("args", {}).get("num_slots"), "slot_dim=", sa.get("args", {}).get("slot_dim"), "step=", sa.get("step"))
print("Probe epoch=", probe.get("epoch"), "valid_selection_score=", probe.get("valid_selection_score"))
print("Probe config=", probe.get("probe_config"))
print("Probe classes=", probe.get("classes"))
PY

${PYTHON} -m settransformer.train_grpo_selector_discriminative_probe \
  --data "${DATA}" \
  --checkpoint "${SA_CKPT}" \
  --output_dir "${OUT_DIR}" \
  --input_res 224 \
  --epochs "${EPOCHS}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --bs "${BS}" \
  --lr "${LR}" \
  --wd "${WD}" \
  --hidden_dim "${HIDDEN_DIM}" \
  --policy_dim "${POLICY_DIM}" \
  --dropout "${DROPOUT}" \
  --max_steps "${MAX_STEPS}" \
  --min_steps "${MIN_STEPS}" \
  --early_exit_conf "${EARLY_EXIT_CONF}" \
  --disable_confidence_early_exit \
  --reward_source probe_subset_margin \
  --reward_probe_checkpoint "${PROBE_CKPT}" \
  --subset_contrast "${SUBSET_CONTRAST}" \
  --subset_contrast_weight "${SUBSET_CONTRAST_WEIGHT}" \
  --step_margin_reward_weights "${STEP_MARGIN_REWARD_WEIGHTS}" \
  --grpo_group_size "${GRPO_GROUP_SIZE}" \
  --free_slots "${FREE_SLOTS}" \
  --count_penalty "${COUNT_PENALTY}" \
  --probe_reward_clip "${PROBE_REWARD_CLIP}" \
  --pos_dim "${POS_DIM}" \
  --num_workers "${NUM_WORKERS}" \
  --seed "${SEED}"

${PYTHON} plot_grpo_selector_training_curves.py \
  --run_dir "${OUT_DIR}" \
  --out "${OUT_DIR}/visualizations/training_curves.png"

VIZ_PER_CLASS_CORRECT=${VIZ_PER_CLASS_CORRECT:-15}
VIZ_PER_CLASS_WRONG=${VIZ_PER_CLASS_WRONG:-5}
VIZ_OUT_DIR=${VIZ_OUT_DIR:-${OUT_DIR}/visualizations/test_slot_paths_${VIZ_PER_CLASS_CORRECT}c_${VIZ_PER_CLASS_WRONG}w}

${PYTHON} visualize_grpo_selector_paths.py \
  --run_dir "${OUT_DIR}" \
  --out_dir "${VIZ_OUT_DIR}" \
  --split "${VIZ_SPLIT:-test}" \
  --per_class_correct "${VIZ_PER_CLASS_CORRECT}" \
  --per_class_wrong "${VIZ_PER_CLASS_WRONG}" \
  --device auto
