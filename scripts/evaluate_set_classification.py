#!/usr/bin/env python3
"""Evaluate SET classifier checkpoints on a split with per-class accuracy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for name in ("SET56", "SET80", "SET112"):
    path = REPO_ROOT / name
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".cache" / "torch"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.evaluate_set_bbox import load_probe, load_slothead, project_slots
from settransformer.model import true_class_margin
from train_slot_classifier import build_dataset, build_transforms, load_backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=["full", "56", "80", "112"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="/vol/biomedic3/kw1025/dinosaur/dataset/coco_top2_clean10_area006_004_600_200_200/classification_dataset")
    parser.add_argument("--split", default="test", choices=["valid", "val", "test"])
    parser.add_argument("--sa_checkpoint", default="")
    parser.add_argument("--slothead_checkpoint", default="")
    parser.add_argument("--slothead_mode", default="")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--input_res", type=int, default=0)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def class_names_from_dataset(dataset) -> list[str]:
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return list(getattr(dataset, "classes", []))


@torch.no_grad()
def encode_raw_slots(backbone, images: torch.Tensor, device: torch.device):
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.forward_dino(images)
    features = backbone.mlp(features)
    slots, _, _ = backbone.slot_attention(features)
    return slots.detach()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    split = "valid" if args.split == "val" else args.split
    device = choose_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args") or {}
    sa_checkpoint = args.sa_checkpoint or ckpt_args.get("sa_checkpoint")
    input_res = args.input_res or int(ckpt_args.get("input_res", 224))
    slothead_mode = args.slothead_mode or ckpt_args.get("slothead_mode", "u")
    if not sa_checkpoint:
        raise ValueError("--sa_checkpoint is required when it is not present in checkpoint args")

    tfm = build_transforms(input_res)
    dataset = build_dataset(args.data, split, tfm["valid"])
    classes = class_names_from_dataset(dataset) or ckpt.get("classes") or []
    loader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    backbone = load_backbone(sa_checkpoint, device)
    backbone.eval()
    backbone.requires_grad_(False)
    projector = load_slothead(args, device)
    model, _ = load_probe(args.checkpoint, device)

    rows: list[dict[str, Any]] = []
    totals = defaultdict(int)
    corrects = defaultdict(int)
    loss_sum = 0.0
    prob_sum = 0.0
    margin_sum = 0.0
    seen = 0

    for images, labels in tqdm(loader, desc=f"set{args.variant}-cls-eval", mininterval=1.0):
        labels = labels.to(device, non_blocking=device.type == "cuda")
        raw_slots = encode_raw_slots(backbone, images, device)
        slots = project_slots(projector, args.variant, raw_slots, slothead_mode)
        mask = torch.ones(slots.size(0), slots.size(1), dtype=torch.bool, device=device)
        logits = model(slots, mask)
        probs = logits.softmax(dim=1)
        pred = logits.argmax(dim=1)
        margin = true_class_margin(logits, labels)
        batch = labels.numel()
        seen += batch
        loss_sum += float(F.cross_entropy(logits, labels, reduction="sum").detach().cpu())
        prob_sum += float(probs.gather(1, labels[:, None]).sum().detach().cpu())
        margin_sum += float(margin.sum().detach().cpu())
        for i in range(batch):
            true_id = int(labels[i].detach().cpu())
            pred_id = int(pred[i].detach().cpu())
            ok = int(true_id == pred_id)
            totals[true_id] += 1
            corrects[true_id] += ok
            rows.append(
                {
                    "index": len(rows),
                    "true": true_id,
                    "true_name": classes[true_id] if true_id < len(classes) else str(true_id),
                    "pred": pred_id,
                    "pred_name": classes[pred_id] if pred_id < len(classes) else str(pred_id),
                    "correct": ok,
                    "true_prob": float(probs[i, true_id].detach().cpu()),
                    "true_margin": float(margin[i].detach().cpu()),
                }
            )

    per_class = {}
    for class_id in range(len(classes)):
        count = totals[class_id]
        per_class[f"{class_id}:{classes[class_id]}"] = {
            "items": count,
            "correct": corrects[class_id],
            "accuracy": (corrects[class_id] / count) if count else 0.0,
        }

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).resolve().parent / f"{split}_classification"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "classification_eval.csv", rows)
    summary = {
        "variant": args.variant,
        "checkpoint": args.checkpoint,
        "data": args.data,
        "split": split,
        "items": seen,
        "accuracy": sum(corrects.values()) / max(seen, 1),
        "loss": loss_sum / max(seen, 1),
        "true_prob": prob_sum / max(seen, 1),
        "true_margin": margin_sum / max(seen, 1),
        "classes": classes,
        "per_class": per_class,
        "slothead_checkpoint": args.slothead_checkpoint,
        "slothead_mode": slothead_mode,
        "outputs": {"csv": str(out_dir / "classification_eval.csv"), "summary": str(out_dir / "summary.json")},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
