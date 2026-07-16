#!/usr/bin/env python3
"""One-pass test evaluation for SET full/56/80/112 classification and bbox metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
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
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from scripts.evaluate_set_bbox import (
    boxes_to_mask,
    hit_stats,
    load_coco_boxes,
    load_metadata,
    load_probe,
    load_slothead,
    make_heatmaps,
    project_slots,
    rank_slots,
    sample_relative_path,
    transform_boxes_to_input,
)
from train_slot_classifier import build_dataset, build_transforms, load_backbone


DEFAULT_DATA = "/vol/biomedic3/kw1025/dinosaur/dataset/coco_top2_clean10_area006_004_600_200_200/classification_dataset"
DEFAULT_COCO = "/vol/biomedic3/kw1025/dinosaur/dataset/coco2017"
DEFAULT_SA = "/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt"
DEFAULT_MODELS = {
    "full": {
        "checkpoint": "SET_full/checkpoints/set_full_raw_20260709_192356/set_full_best.pt",
        "slothead_checkpoint": "",
    },
    "56": {
        "checkpoint": "SET56/checkpoints/set56_u_20260710_093023/set56_best.pt",
        "slothead_checkpoint": "SET56/checkpoints/slothead56_obj16_geo16_res24_20260709_192411/slothead_best.pt",
    },
    "80": {
        "checkpoint": "SET80/checkpoints/set80_u_20260710_093223/set80_best.pt",
        "slothead_checkpoint": "SET80/checkpoints/slothead80_obj16_geo16_res48_20260709_192454/slothead_best.pt",
    },
    "112": {
        "checkpoint": "SET112/checkpoints/set112_u_20260710_093234/set112_best.pt",
        "slothead_checkpoint": "SET112/checkpoints/slothead112_obj16_geo32_res64_20260709_192510/slothead_best.pt",
    },
}


def parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--coco_root", default=DEFAULT_COCO)
    parser.add_argument("--sa_checkpoint", default=DEFAULT_SA)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--out_dir", default="analysis/set_suite_test_eval")
    parser.add_argument("--input_res", type=int, default=224)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top_ks", type=parse_int_list, default=parse_int_list("3,4"))
    parser.add_argument("--hit_thresholds", type=parse_float_list, default=parse_float_list("0.2,0.4"))
    parser.add_argument("--threshold_rel", type=float, default=0.5)
    parser.add_argument("--rank_score", choices=["true_prob", "true_margin"], default="true_prob")
    return parser.parse_args()


def choose_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dataset_indices(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        return [int(dataset.indices[i]) for i in range(len(dataset))]
    return list(range(len(dataset)))


def unwrap_dataset(dataset: Dataset) -> Dataset:
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def class_names_from_dataset(dataset: Dataset) -> list[str]:
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return list(getattr(dataset, "classes", []))


@torch.no_grad()
def encode_raw_slots(backbone, images: torch.Tensor, device: torch.device):
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.forward_dino(images)
    features = backbone.mlp(features)
    slots, attn, _ = backbone.slot_attention(features)
    return slots.detach(), attn.detach()


def zero_cls(classes: list[str]) -> dict[str, Any]:
    return {
        "items": 0,
        "correct": 0,
        "per_class": {f"{idx}:{name}": {"items": 0, "correct": 0} for idx, name in enumerate(classes)},
    }


def add_cls(stats: dict[str, Any], labels: torch.Tensor, preds: torch.Tensor, classes: list[str]) -> None:
    for true_id, pred_id in zip(labels.detach().cpu().tolist(), preds.detach().cpu().tolist()):
        ok = int(true_id == pred_id)
        stats["items"] += 1
        stats["correct"] += ok
        key = f"{true_id}:{classes[true_id]}"
        stats["per_class"][key]["items"] += 1
        stats["per_class"][key]["correct"] += ok


def finish_cls(stats: dict[str, Any]) -> dict[str, Any]:
    out = dict(stats)
    out["accuracy"] = out["correct"] / max(out["items"], 1)
    out["per_class"] = {
        key: {**value, "accuracy": value["correct"] / max(value["items"], 1)}
        for key, value in stats["per_class"].items()
    }
    return out


def metric_keys(top_ks: list[int]) -> list[str]:
    keys: list[str] = []
    for top_k in top_ks:
        keys.extend([f"anchor@{top_k}", f"evidence@{top_k}", f"pair@{top_k}"])
    return keys


def zero_bbox(classes: list[str], top_ks: list[int], thresholds: list[float]) -> dict[str, Any]:
    keys = metric_keys(top_ks)
    stats: dict[str, Any] = {}
    for thr in thresholds:
        stats[str(thr)] = {
            "items": 0,
            "sums": {key: 0.0 for key in keys},
            "per_class": {
                f"{idx}:{name}": {"items": 0, "sums": {key: 0.0 for key in keys}}
                for idx, name in enumerate(classes)
            },
        }
    return stats


def add_bbox(stats: dict[str, Any], threshold: float, class_key: str, row_metrics: dict[str, float]) -> None:
    bucket = stats[str(threshold)]
    bucket["items"] += 1
    class_bucket = bucket["per_class"][class_key]
    class_bucket["items"] += 1
    for key, value in row_metrics.items():
        bucket["sums"][key] += float(value)
        class_bucket["sums"][key] += float(value)


def finish_bbox(stats: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for threshold, bucket in stats.items():
        items = max(bucket["items"], 1)
        out[threshold] = {
            "items": bucket["items"],
            "metrics": {key: value / items for key, value in bucket["sums"].items()},
            "per_class": {},
        }
        for class_key, class_bucket in bucket["per_class"].items():
            class_items = max(class_bucket["items"], 1)
            out[threshold]["per_class"][class_key] = {
                "items": class_bucket["items"],
                "metrics": {key: value / class_items for key, value in class_bucket["sums"].items()},
            }
    return out


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    tfm = build_transforms(args.input_res)
    dataset = build_dataset(args.data, args.split, tfm["valid"])
    classes = class_names_from_dataset(dataset)
    base_dataset = unwrap_dataset(dataset)
    base_indices = dataset_indices(dataset)
    metadata = load_metadata(args.data)

    needed = set()
    for sample_idx in base_indices:
        rel = sample_relative_path(base_dataset, sample_idx, args.split)
        row = metadata[rel]
        needed.add((row["source_split"], int(row["image_id"]), row["anchor_object"]))
        needed.add((row["source_split"], int(row["image_id"]), row["evidence_object"]))
    coco_boxes = load_coco_boxes(Path(args.coco_root), needed)

    loader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    backbone = load_backbone(args.sa_checkpoint, device)
    backbone.eval()
    backbone.requires_grad_(False)

    variants: dict[str, dict[str, Any]] = {}
    for variant, spec in DEFAULT_MODELS.items():
        ns = SimpleNamespace(
            variant=variant,
            checkpoint=spec["checkpoint"],
            slothead_checkpoint=spec["slothead_checkpoint"],
            slothead_mode="u",
        )
        probe, ckpt = load_probe(spec["checkpoint"], device)
        variants[variant] = {
            "checkpoint": spec["checkpoint"],
            "slothead_checkpoint": spec["slothead_checkpoint"],
            "slothead_mode": (ckpt.get("args") or {}).get("slothead_mode", "u"),
            "projector": load_slothead(ns, device),
            "probe": probe,
            "classification": zero_cls(classes),
            "bbox": zero_bbox(classes, args.top_ks, args.hit_thresholds),
        }

    cursor = 0
    for images, labels in tqdm(loader, desc="set-suite-eval", mininterval=1.0):
        labels = labels.to(device, non_blocking=device.type == "cuda")
        raw_slots, attn = encode_raw_slots(backbone, images, device)
        batch = labels.numel()
        batch_meta = []
        for b in range(batch):
            sample_idx = base_indices[cursor + b]
            rel = sample_relative_path(base_dataset, sample_idx, args.split)
            meta_row = metadata[rel]
            image_id = int(meta_row["image_id"])
            source_split = meta_row["source_split"]
            width = int(meta_row["width"])
            height = int(meta_row["height"])
            anchor_boxes = transform_boxes_to_input(
                coco_boxes.get((source_split, image_id, meta_row["anchor_object"]), []),
                width,
                height,
                args.input_res,
            )
            evidence_boxes = transform_boxes_to_input(
                coco_boxes.get((source_split, image_id, meta_row["evidence_object"]), []),
                width,
                height,
                args.input_res,
            )
            batch_meta.append(
                {
                    "class_key": f"{int(labels[b].detach().cpu())}:{meta_row['class_name']}",
                    "anchor_mask": boxes_to_mask(anchor_boxes, args.input_res),
                    "evidence_mask": boxes_to_mask(evidence_boxes, args.input_res),
                    "heatmaps": make_heatmaps(attn[b].detach().cpu(), args.input_res),
                }
            )

        for variant, state in variants.items():
            slots = project_slots(state["projector"], variant, raw_slots, state["slothead_mode"])
            all_mask = torch.ones(slots.size(0), slots.size(1), dtype=torch.bool, device=device)
            logits = state["probe"](slots, all_mask)
            add_cls(state["classification"], labels, logits.argmax(dim=1), classes)
            ranked = rank_slots(state["probe"], slots, labels, args.rank_score)
            for b in range(batch):
                meta = batch_meta[b]
                for threshold in args.hit_thresholds:
                    row_metrics: dict[str, float] = {}
                    for top_k in args.top_ks:
                        top_slots = ranked[b][:top_k]
                        anchor_hit, _ = hit_stats(meta["heatmaps"], top_slots, meta["anchor_mask"], threshold, args.threshold_rel)
                        evidence_hit, _ = hit_stats(meta["heatmaps"], top_slots, meta["evidence_mask"], threshold, args.threshold_rel)
                        row_metrics[f"anchor@{top_k}"] = float(anchor_hit)
                        row_metrics[f"evidence@{top_k}"] = float(evidence_hit)
                        row_metrics[f"pair@{top_k}"] = float(anchor_hit and evidence_hit)
                    add_bbox(state["bbox"], threshold, meta["class_key"], row_metrics)
        cursor += batch

    summary = {
        "data": args.data,
        "split": args.split,
        "items": len(dataset),
        "sa_checkpoint": args.sa_checkpoint,
        "input_res": args.input_res,
        "top_ks": args.top_ks,
        "hit_thresholds": args.hit_thresholds,
        "threshold_rel": args.threshold_rel,
        "rank_score": args.rank_score,
        "classes": classes,
        "variants": {
            variant: {
                "checkpoint": state["checkpoint"],
                "slothead_checkpoint": state["slothead_checkpoint"],
                "slothead_mode": state["slothead_mode"],
                "classification": finish_cls(state["classification"]),
                "bbox": finish_bbox(state["bbox"]),
            }
            for variant, state in variants.items()
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
