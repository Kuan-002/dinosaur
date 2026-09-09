# DINOSAUR Slot Selection Experiments

This repository contains the minimal code needed to train a DINOSAUR-style slot-attention backbone and run the COCO-6 and COCO-8 contrastive-pair experiments with either actor-critic (AC) or group relative policy optimization (GRPO). Shell launchers, datasets, checkpoints, logs, plots, and generated reports are intentionally excluded.

## 1. Install the environment

Python 3.10 or newer and a CUDA-capable PyTorch installation are recommended.

```bash
git clone <repository-url> dinosaur
cd dinosaur
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The model loads `dino_vitb16` through `torch.hub`. The first run therefore needs network access to download the upstream DINO code and weights. Later runs use `.cache/torch`, which is ignored by Git.

## 2. Prepare the data

No dataset or checkpoint is stored in this repository.

### Raw COCO 2017

Raw COCO is required for slot-attention training and bbox evaluation. Arrange it as follows:

```text
/data/coco2017/
├── annotations/
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/
│   └── *.jpg
└── val2017/
    └── *.jpg
```

### COCO-6 and COCO-8 experiment datasets

Extract the supplied archives outside the repository:

```bash
mkdir -p /data/dinosaur
unzip COCO-6.zip -d /data/dinosaur
unzip COCO-8.zip -d /data/dinosaur
```

Each experiment script expects `--data` to name the extracted `classification_dataset` directory:

```text
/data/dinosaur/COCO-6/classification_dataset/
├── summary.json
├── metadata.csv
├── class_to_idx.json
├── train/<class-name>/*.jpg
├── val/<class-name>/*.jpg
└── test/<class-name>/*.jpg
```

COCO-8 has the same layout. `summary.json` is required: for COCO-6, every item in `pairs` maps `class_name` to `object_a` and `object_b`; for COCO-8, it maps `class_name` to `anchor` and `evidence`. Directory names under every split must agree with the class names in that file.

## 3. Train slot attention

Run the common slot-attention backbone on raw COCO:

```bash
python train.py \
  --dataset coco \
  --data_dir /data/coco2017 \
  --exp_name sa_coco \
  --num_slots 8 \
  --epochs 1000 \
  --bs 64 \
  --lr 1e-3 \
  --num_workers 8 \
  --wandb_mode disabled
```

The checkpoints are written to `checkpoints/sa_coco/`. Use `checkpoint_best_mbo_i_slots.pt` as `--sa_checkpoint` below. Checkpoints are local artifacts and are ignored by Git.

Important SA arguments are:

- `--dataset`: `coco`, `coco_rules`, or `pascal`; this workflow uses `coco`.
- `--data_dir`: raw dataset root in the layout above.
- `--num_slots`, `--slot_dim`: slot count and slot representation size.
- `--epochs`, `--bs`, `--lr`: epoch count, batch size, and learning rate.
- `--monitor_metric`: validation metric used for model selection; the default is `mBO_i_slots`.
- `--wandb_mode`: `disabled`, `offline`, or `online`. Install `wandb` separately only when enabling it.

## 4. Run AC or GRPO

All commands use Python directly; no shell launcher is required. Set reusable paths in your shell if desired:

```bash
SA_CKPT=/absolute/path/to/checkpoint_best_mbo_i_slots.pt
COCO6_DATA=/data/dinosaur/COCO-6/classification_dataset
COCO8_DATA=/data/dinosaur/COCO-8/classification_dataset
```

### COCO-6

```bash
python GRPO6-contrastive-pair/train_ac6_contrastive_pair.py \
  --data "$COCO6_DATA" \
  --sa_checkpoint "$SA_CKPT" \
  --output_dir outputs/coco6_ac \
  --require_cuda

python GRPO6-contrastive-pair/train_grpo6_contrastive_pair.py \
  --data "$COCO6_DATA" \
  --sa_checkpoint "$SA_CKPT" \
  --output_dir outputs/coco6_grpo \
  --require_cuda
```

COCO-6 treats its two objects symmetrically. Its reward coefficients are `--class_margin_coef`, `--object_a_coef`, `--object_b_coef`, and `--component_pair_coef`.

### COCO-8

```bash
python GRPO8-contrastive-pair/train_ac8_contrastive_pair.py \
  --data "$COCO8_DATA" \
  --sa_checkpoint "$SA_CKPT" \
  --output_dir outputs/coco8_ac \
  --require_cuda

python GRPO8-contrastive-pair/train_grpo8_contrastive_pair.py \
  --data "$COCO8_DATA" \
  --sa_checkpoint "$SA_CKPT" \
  --output_dir outputs/coco8_grpo \
  --require_cuda
```

COCO-8 uses a directed anchor-to-evidence rule. Its reward coefficients are `--class_margin_coef`, `--anchor_coef`, `--evidence_coef`, and `--pair_coef`.

Parameters shared by AC and GRPO include `--epochs`, `--bs`, `--lr`, `--policy_lr`, `--max_steps`, `--min_steps`, `--early_exit_conf`, `--seed`, and the auxiliary classification/component-loss coefficients. AC additionally exposes `--value_coef`, `--entropy_coef`, `--disable_return_norm`, and `--disable_advantage_norm`; GRPO exposes `--grpo_group_size`.

Every training output directory contains `experiment_meta.json`, a checkpoint-selection report, and either `selector_ac_best.pt` or `selector_grpo_best.pt`. Use `python <entrypoint> --help` for the complete argument definition.

## 5. Final bbox evaluation

The evaluator accepts either AC or GRPO checkpoints. Select the matching checkpoint filename explicitly:

```bash
python GRPO6-contrastive-pair/evaluate_grpo6_contrastive_bbox.py \
  --run_dir outputs/coco6_grpo \
  --checkpoint selector_grpo_best.pt \
  --data "$COCO6_DATA" \
  --sa_checkpoint "$SA_CKPT" \
  --coco_root /data/coco2017 \
  --split test \
  --out_dir outputs/coco6_grpo/test_bbox \
  --top_ks 2,3,4 \
  --hit_thresholds 0.2,0.4

python GRPO8-contrastive-pair/evaluate_grpo8_contrastive_bbox.py \
  --run_dir outputs/coco8_ac \
  --checkpoint selector_ac_best.pt \
  --data "$COCO8_DATA" \
  --sa_checkpoint "$SA_CKPT" \
  --coco_root /data/coco2017 \
  --split test \
  --out_dir outputs/coco8_ac/test_bbox \
  --top_ks 2,3,4 \
  --hit_thresholds 0.2,0.4
```

The evaluator writes machine-readable CSV and JSON metrics beneath `--out_dir`. These results, like all model outputs and reports, are ignored by Git.

## Repository boundary

Only the SA implementation, the shared selector dependencies, the COCO-6/8 AC and GRPO entrypoints, and the bbox evaluator are versioned. The ignore rules exclude all `.sh` files and common dataset, checkpoint, archive, image, report, cache, and experiment-output paths. Before committing, verify the boundary with:

```bash
git status --short
git ls-files | sort
```
