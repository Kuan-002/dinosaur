#!/usr/bin/env python3
"""Evaluate attention-only top-k union masks against anchor/evidence bboxes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "torch"))

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_set_bbox import (
    boxes_to_mask,
    load_coco_boxes,
    load_metadata,
    sample_relative_path,
    transform_boxes_to_input,
    unwrap_dataset,
)
from misc_utils import seed_all
from train_slot_classifier import build_dataset, build_transforms, load_backbone
from visualize_attention_object_regions import attention_quality, encode_attention, make_heatmaps, select_per_class


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, label = self.base[idx]
        return image, label, idx


def choose_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dataset_indices(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        return [int(dataset.indices[i]) for i in range(len(dataset))]
    return list(range(len(dataset)))


def mask_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.bool()
    target = target.bool()
    inter = float((pred & target).sum().item())
    pred_area = float(pred.sum().item())
    target_area = float(target.sum().item())
    union = float((pred | target).sum().item())
    return {
        "coverage": inter / max(target_area, 1.0),
        "purity": inter / max(pred_area, 1.0),
        "iou": inter / max(union, 1.0),
        "pred_area_frac": pred_area / float(pred.numel()),
        "target_area_frac": target_area / float(target.numel()),
    }


def summarize(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(REPO_ROOT / "dataset/coco_top2_clean10_area006_004_600_200_200/classification_dataset"))
    p.add_argument("--coco_root", default=str(REPO_ROOT / "dataset/coco2017"))
    p.add_argument("--sa_checkpoint", default=str(REPO_ROOT / "checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt"))
    p.add_argument("--split", default="test", choices=["train", "valid", "val", "test"])
    p.add_argument("--out_dir", default=str(REPO_ROOT / "analysis/attention_object_regions_10_per_class_bbox_eval"))
    p.add_argument("--input_res", type=int, default=224)
    p.add_argument("--per_class", type=int, default=10)
    p.add_argument("--max_items", type=int, default=0)
    p.add_argument("--topk", type=int, default=3)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--threshold_rel", type=float, default=0.5)
    p.add_argument("--evidence_min_area", type=float, default=0.02)
    p.add_argument("--evidence_max_area", type=float, default=0.35)
    p.add_argument("--evidence_area_softness", type=float, default=0.015)
    p.add_argument("--evidence_max_entropy", type=float, default=0.85)
    p.add_argument("--evidence_entropy_softness", type=float, default=0.03)
    p.add_argument("--evidence_min_compactness", type=float, default=0.25)
    p.add_argument("--evidence_compactness_softness", type=float, default=0.06)
    p.add_argument("--evidence_max_duplicate", type=float, default=0.75)
    p.add_argument("--evidence_duplicate_softness", type=float, default=0.08)
    p.add_argument("--evidence_peakiness_norm", type=float, default=8.0)
    p.add_argument("--evidence_min_peakiness", type=float, default=0.5)
    p.add_argument("--evidence_peakiness_softness", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed, False)
    split = "valid" if args.split == "val" else args.split
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tfm = build_transforms(args.input_res)["valid"]
    base = build_dataset(args.data, split, tfm)
    if args.per_class > 0:
        selected_indices, classes = select_per_class(base, args.per_class)
    else:
        classes = list(getattr(base, "classes", []))
        selected_indices = list(range(len(base)))
        if args.max_items and args.max_items < len(selected_indices):
            selected_indices = selected_indices[: args.max_items]
    dataset = IndexedDataset(Subset(base, selected_indices))
    base_dataset = unwrap_dataset(base)
    metadata = load_metadata(args.data)

    needed: set[tuple[str, int, str]] = set()
    for sample_idx in selected_indices:
        rel = sample_relative_path(base_dataset, sample_idx, split)
        row = metadata[rel]
        image_id = int(row["image_id"])
        source_split = row["source_split"]
        needed.add((source_split, image_id, row["anchor_object"]))
        needed.add((source_split, image_id, row["evidence_object"]))
    coco_boxes = load_coco_boxes(Path(args.coco_root), needed)

    loader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    device = choose_device(args.device)
    backbone = load_backbone(args.sa_checkpoint, device)
    rows: list[dict[str, Any]] = []
    for images, labels, subset_indices in tqdm(loader, desc="attention-union-bbox", mininterval=1.0):
        original_indices = [selected_indices[int(i)] for i in subset_indices.tolist()]
        attn = encode_attention(backbone, images, device)
        heatmaps = make_heatmaps(attn, args.input_res)
        quality = attention_quality(heatmaps, args)
        scores = quality["raw_evidence"]
        masks = quality["masks"]
        for bi, sample_idx in enumerate(original_indices):
            rel = sample_relative_path(base_dataset, sample_idx, split)
            meta = metadata[rel]
            image_id = int(meta["image_id"])
            source_split = meta["source_split"]
            width = int(meta["width"])
            height = int(meta["height"])
            anchor_name = meta["anchor_object"]
            evidence_name = meta["evidence_object"]
            anchor_boxes = transform_boxes_to_input(
                coco_boxes.get((source_split, image_id, anchor_name), []),
                width,
                height,
                args.input_res,
            )
            evidence_boxes = transform_boxes_to_input(
                coco_boxes.get((source_split, image_id, evidence_name), []),
                width,
                height,
                args.input_res,
            )
            anchor_mask = boxes_to_mask(anchor_boxes, args.input_res)
            evidence_mask = boxes_to_mask(evidence_boxes, args.input_res)
            object_union = anchor_mask | evidence_mask
            top_idx = scores[bi].argsort(descending=True)[: min(args.topk, scores.size(1))]
            pred_union = masks[bi, top_idx].any(dim=0)
            object_metrics = mask_metrics(pred_union, object_union)
            anchor_metrics = mask_metrics(pred_union, anchor_mask)
            evidence_metrics = mask_metrics(pred_union, evidence_mask)
            row = {
                "class": meta["class_name"],
                "sample_idx": sample_idx,
                "relative_path": rel,
                "image_id": image_id,
                "anchor_object": anchor_name,
                "evidence_object": evidence_name,
                "top_slots_1based": json.dumps([int(i) + 1 for i in top_idx.tolist()]),
                "top_scores": json.dumps([round(float(scores[bi, i]), 6) for i in top_idx.tolist()]),
                "object_union_coverage": object_metrics["coverage"],
                "object_union_purity": object_metrics["purity"],
                "object_union_iou": object_metrics["iou"],
                "anchor_coverage": anchor_metrics["coverage"],
                "evidence_coverage": evidence_metrics["coverage"],
                "pred_area_frac": object_metrics["pred_area_frac"],
                "target_area_frac": object_metrics["target_area_frac"],
            }
            rows.append(row)

    write_csv(out_dir / "attention_union_bbox_eval.csv", rows)
    metric_keys = [
        "object_union_coverage",
        "object_union_purity",
        "object_union_iou",
        "anchor_coverage",
        "evidence_coverage",
        "pred_area_frac",
        "target_area_frac",
    ]
    by_class = defaultdict(list)
    for row in rows:
        by_class[row["class"]].append(row)
    summary = {
        "split": split,
        "per_class": args.per_class,
        "total_images": len(rows),
        "topk": args.topk,
        "threshold_rel": args.threshold_rel,
        "bbox_usage": "offline diagnostic only",
        "overall": summarize(rows, metric_keys),
        "by_class": {name: summarize(class_rows, metric_keys) for name, class_rows in sorted(by_class.items())},
        "rows": str(out_dir / "attention_union_bbox_eval.csv"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
