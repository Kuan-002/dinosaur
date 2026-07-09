#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Optional

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".cache" / "torch"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from misc_utils import seed_all
from train_slot_classifier import build_dataset, build_transforms, load_backbone, subset_dataset

from model import BBoxSetTransformer, BBoxSetTransformerConfig


DEFAULT_DATA = (
    "/vol/biomedic3/kw1025/dinosaur/analysis/"
    "coco_top2_clean_scenes_anchor009_evidence005_10cls_450_150_150/classification_dataset"
)
DEFAULT_SA = (
    "/vol/biomedic3/kw1025/dinosaur/checkpoints/"
    "sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt"
)
DEFAULT_COCO = "/vol/biomedic3/kw1025/dinosaur/dataset/coco2017"

COCO_CATEGORY_BY_ID = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus",
    7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant",
    13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat",
    18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear",
    24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag",
    32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard",
    37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove",
    41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
    46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl",
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake",
    62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 67: "dining table",
    70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
    76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster",
    81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase",
    87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}


def unwrap_dataset(dataset: Dataset) -> Dataset:
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def dataset_indices(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        return [dataset.indices[i] for i in range(len(dataset))]
    return list(range(len(dataset)))


def fixed_sample_relative_path(base_dataset: Dataset, sample_index: int, split: str) -> str:
    samples = getattr(base_dataset, "samples", None)
    if samples is None:
        raise ValueError("Dataset does not expose samples")
    sample = str(samples[sample_index][0])
    split_name = "val" if split == "valid" else split
    marker = f"/{split_name}/"
    if marker in sample:
        return split_name + "/" + sample.split(marker, 1)[1]
    parts = sample.split("/")
    return "/".join(parts[-3:])


def load_metadata(data: str) -> dict[str, dict]:
    root = Path(data)
    if (root / "metadata.csv").exists():
        metadata_path = root / "metadata.csv"
    elif (root / "classification_dataset" / "metadata.csv").exists():
        metadata_path = root / "classification_dataset" / "metadata.csv"
    else:
        raise FileNotFoundError(f"metadata.csv not found under {root}")
    rows = {}
    with metadata_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[row["relative_path"]] = row
    return rows


def needed_metadata_keys(metadata: dict[str, dict], dataset: Dataset, split: str) -> dict[str, dict[int, set[str]]]:
    base_dataset = unwrap_dataset(dataset)
    out: dict[str, dict[int, set[str]]] = {"train": defaultdict(set), "val": defaultdict(set)}
    for sample_idx in dataset_indices(dataset):
        rel = fixed_sample_relative_path(base_dataset, sample_idx, split)
        row = metadata.get(rel)
        if row is None:
            continue
        image_id = int(row["image_id"])
        out[row["source_split"]][image_id].add(row["anchor_object"])
        out[row["source_split"]][image_id].add(row["evidence_object"])
    return out


def load_coco_boxes(coco_root: Path, needed: dict[str, dict[int, set[str]]]) -> dict[tuple[str, int, str], list[list[float]]]:
    out: dict[tuple[str, int, str], list[list[float]]] = defaultdict(list)
    object_pattern = re.compile(r"\{[^{}]*\}")
    for source_split in ["train", "val"]:
        if not needed.get(source_split):
            continue
        path = coco_root / "annotations" / f"instances_{source_split}2017.json"
        wanted_image_ids = set(needed[source_split])
        carry = ""
        in_annotations = False
        with path.open("r", encoding="utf-8") as f:
            while True:
                chunk = f.read(16 * 1024 * 1024)
                if not chunk:
                    break
                text = carry + chunk
                if not in_annotations:
                    pos = text.find('"annotations"')
                    if pos < 0:
                        carry = text[-64:]
                        continue
                    array_pos = text.find("[", pos)
                    if array_pos < 0:
                        carry = text[pos:]
                        continue
                    text = text[array_pos + 1 :]
                    in_annotations = True
                matches = list(object_pattern.finditer(text))
                last_end = 0
                for match in matches:
                    last_end = match.end()
                    raw = match.group(0)
                    if '"image_id"' not in raw or '"bbox"' not in raw:
                        continue
                    ann = json.loads(raw)
                    if ann.get("iscrowd", 0):
                        continue
                    image_id = int(ann["image_id"])
                    if image_id not in wanted_image_ids:
                        continue
                    class_name = COCO_CATEGORY_BY_ID.get(int(ann["category_id"]))
                    if class_name is None or class_name not in needed[source_split][image_id]:
                        continue
                    x, y, w, h = [float(v) for v in ann["bbox"]]
                    if w <= 0 or h <= 0:
                        continue
                    out[(source_split, image_id, class_name)].append([x, y, x + w, y + h])
                carry = text[last_end:]
                if '"categories"' in carry:
                    carry = carry.split('"categories"', 1)[0]
    return out


def transform_boxes_to_input(boxes: list[list[float]], width: int, height: int, input_res: int) -> torch.Tensor:
    if not boxes:
        return torch.empty(0, 4)
    if width < height:
        scale = input_res / float(width)
        resized_w = input_res
        resized_h = int(round(height * scale))
    else:
        scale = input_res / float(height)
        resized_h = input_res
        resized_w = int(round(width * scale))
    crop_x = max(0.0, (resized_w - input_res) / 2.0)
    crop_y = max(0.0, (resized_h - input_res) / 2.0)
    mapped = []
    for x1, y1, x2, y2 in boxes:
        nx1 = min(max(x1 * scale - crop_x, 0.0), float(input_res))
        nx2 = min(max(x2 * scale - crop_x, 0.0), float(input_res))
        ny1 = min(max(y1 * scale - crop_y, 0.0), float(input_res))
        ny2 = min(max(y2 * scale - crop_y, 0.0), float(input_res))
        if nx2 > nx1 and ny2 > ny1:
            mapped.append([nx1, ny1, nx2, ny2])
    return torch.tensor(mapped, dtype=torch.float32)


def boxes_to_mask(boxes: torch.Tensor, size: int) -> torch.Tensor:
    mask = torch.zeros(size, size, dtype=torch.bool)
    for box in boxes:
        x1, y1, x2, y2 = box.tolist()
        ix1 = max(0, min(size, int(math.floor(x1))))
        iy1 = max(0, min(size, int(math.floor(y1))))
        ix2 = max(0, min(size, int(math.ceil(x2))))
        iy2 = max(0, min(size, int(math.ceil(y2))))
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1:iy2, ix1:ix2] = True
    return mask


def make_heatmaps(attn: torch.Tensor, size: int) -> torch.Tensor:
    k, n = attn.shape
    side = int(math.sqrt(n))
    if side * side != n:
        raise ValueError(f"attention token count is not square: {n}")
    maps = attn.reshape(k, 1, side, side)
    maps = F.interpolate(maps, size=(size, size), mode="bilinear", align_corners=False)[:, 0]
    denom = maps.flatten(1).sum(dim=1).clamp_min(1e-8).view(k, 1, 1)
    return maps / denom


def slot_mass_targets(heatmaps: torch.Tensor, anchor_mask: torch.Tensor, evidence_mask: torch.Tensor) -> torch.Tensor:
    anchor = heatmaps[:, anchor_mask].sum(dim=1) if bool(anchor_mask.any()) else heatmaps.new_zeros(heatmaps.size(0))
    evidence = heatmaps[:, evidence_mask].sum(dim=1) if bool(evidence_mask.any()) else heatmaps.new_zeros(heatmaps.size(0))
    return torch.stack([anchor.clamp(0, 1), evidence.clamp(0, 1)], dim=1)


@torch.no_grad()
def encode_batch(backbone, images: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.forward_dino(images)
    features = backbone.mlp(features)
    slots, attn, _ = backbone.slot_attention(features)
    return slots.detach(), attn.detach()


class BBoxTargetBuilder:
    def __init__(self, data: str, coco_root: str, dataset: Dataset, split: str, input_res: int):
        self.metadata = load_metadata(data)
        self.base_dataset = unwrap_dataset(dataset)
        self.base_indices = dataset_indices(dataset)
        self.split = split
        self.input_res = input_res
        needed = needed_metadata_keys(self.metadata, dataset, split)
        self.coco_boxes = load_coco_boxes(Path(coco_root), needed)

    def batch_targets(self, attn: torch.Tensor, cursor: int, batch_size: int, device: torch.device) -> torch.Tensor:
        targets = []
        for b in range(batch_size):
            sample_idx = self.base_indices[cursor + b]
            rel = fixed_sample_relative_path(self.base_dataset, sample_idx, self.split)
            meta = self.metadata.get(rel)
            if meta is None:
                raise KeyError(f"No metadata row for relative_path={rel}")
            image_id = int(meta["image_id"])
            source_split = meta["source_split"]
            width = int(meta["width"])
            height = int(meta["height"])
            anchor_boxes = transform_boxes_to_input(
                self.coco_boxes.get((source_split, image_id, meta["anchor_object"]), []),
                width,
                height,
                self.input_res,
            )
            evidence_boxes = transform_boxes_to_input(
                self.coco_boxes.get((source_split, image_id, meta["evidence_object"]), []),
                width,
                height,
                self.input_res,
            )
            anchor_mask = boxes_to_mask(anchor_boxes, self.input_res).to(device)
            evidence_mask = boxes_to_mask(evidence_boxes, self.input_res).to(device)
            heatmaps = make_heatmaps(attn[b], self.input_res)
            targets.append(slot_mass_targets(heatmaps, anchor_mask, evidence_mask))
        return torch.stack(targets, dim=0).to(device)


def subset_targets(role_targets: torch.Tensor, subset_mask: torch.Tensor, hit_threshold: float) -> torch.Tensor:
    selected = subset_mask.to(role_targets.dtype)
    anchor_mass = (role_targets[:, :, 0] * selected).amax(dim=1)
    evidence_mass = (role_targets[:, :, 1] * selected).amax(dim=1)
    pair_hit = ((anchor_mass >= hit_threshold) & (evidence_mass >= hit_threshold)).to(role_targets.dtype)
    return torch.stack([anchor_mass, evidence_mass, pair_hit], dim=1)


def random_size_mask(batch: int, num_slots: int, device: torch.device, min_size: int, max_size: int) -> torch.Tensor:
    lo = max(0, min(min_size, num_slots))
    hi = max(lo, min(max_size, num_slots))
    sizes = torch.randint(lo, hi + 1, (batch,), device=device)
    order = torch.rand(batch, num_slots, device=device).argsort(dim=1)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(num_slots, device=device).expand(batch, -1))
    return ranks < sizes[:, None]


def make_subset_masks(role_targets: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    b, k, _ = role_targets.shape
    masks = []
    masks.append(torch.ones(b, k, dtype=torch.bool, device=role_targets.device))
    masks.append(random_size_mask(b, k, role_targets.device, args.min_subset_slots, args.max_subset_slots))
    anchor_idx = role_targets[:, :, 0].argmax(dim=1)
    evidence_idx = role_targets[:, :, 1].argmax(dim=1)
    oracle = torch.zeros(b, k, dtype=torch.bool, device=role_targets.device)
    oracle[torch.arange(b, device=role_targets.device), anchor_idx] = True
    oracle[torch.arange(b, device=role_targets.device), evidence_idx] = True
    masks.append(oracle)
    return torch.cat(masks, dim=0)


def compute_losses(model: BBoxSetTransformer, slots: torch.Tensor, role_targets: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    encoded = model.encode_slots(slots)
    role_logits = model.role_head(encoded)
    role_loss = F.binary_cross_entropy_with_logits(role_logits, role_targets)

    subset_masks = make_subset_masks(role_targets, args)
    slots_rep_targets = role_targets.repeat(3, 1, 1)
    subset_target = subset_targets(slots_rep_targets, subset_masks, args.hit_threshold)
    encoded_rep = encoded.repeat(3, 1, 1)
    subset_logits = model.subset_logits_from_encoded(encoded_rep, subset_masks)
    subset_loss = F.binary_cross_entropy_with_logits(subset_logits, subset_target)
    loss = args.role_loss_weight * role_loss + args.subset_loss_weight * subset_loss
    with torch.no_grad():
        probs = role_logits.sigmoid()
        anchor_pred = probs[:, :, 0].argmax(dim=1)
        evidence_pred = probs[:, :, 1].argmax(dim=1)
        anchor_true = role_targets[:, :, 0].argmax(dim=1)
        evidence_true = role_targets[:, :, 1].argmax(dim=1)
        pair_logits = subset_logits[:, 2]
        pair_target = subset_target[:, 2]
        pair_pred = pair_logits.sigmoid() >= 0.5
        stats = {
            "loss": float(loss.detach().cpu()),
            "role_loss": float(role_loss.detach().cpu()),
            "subset_loss": float(subset_loss.detach().cpu()),
            "anchor_top1": float((anchor_pred == anchor_true).float().mean().cpu()),
            "evidence_top1": float((evidence_pred == evidence_true).float().mean().cpu()),
            "pair_acc": float((pair_pred == pair_target.bool()).float().mean().cpu()),
            "target_anchor_mass": float(role_targets[:, :, 0].amax(dim=1).mean().cpu()),
            "target_evidence_mass": float(role_targets[:, :, 1].amax(dim=1).mean().cpu()),
        }
    return loss, stats


def run_epoch(model, backbone, loader, target_builder, device, optimizer, args, train: bool) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = {}
    seen = 0
    cursor = 0
    desc = "train" if train else "valid"
    iterator = tqdm(loader, desc=desc, mininterval=1.0) if sys.stdout.isatty() else loader
    for images, _labels in iterator:
        slots, attn = encode_batch(backbone, images, device)
        batch = slots.size(0)
        role_targets = target_builder.batch_targets(attn, cursor, batch, device)
        cursor += batch
        with torch.set_grad_enabled(train):
            loss, stats = compute_losses(model, slots, role_targets, args)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        seen += batch
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + value * batch
    return {key: value / max(seen, 1) for key, value in totals.items()}


def class_names_from_dataset(dataset: Dataset) -> list[str]:
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return list(getattr(dataset, "classes", []))


def write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a bbox-supervised SetTransformer anchor/evidence slot judge.")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--coco_root", default=DEFAULT_COCO)
    parser.add_argument("--checkpoint", default=DEFAULT_SA)
    parser.add_argument("--output_dir", default="checkpoints/bbox_settransformer")
    parser.add_argument("--input_res", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--bottleneck_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_sab_layers", type=int, default=2)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min_subset_slots", type=int, default=1)
    parser.add_argument("--max_subset_slots", type=int, default=4)
    parser.add_argument("--hit_threshold", type=float, default=0.20)
    parser.add_argument("--role_loss_weight", type=float, default=1.0)
    parser.add_argument("--subset_loss_weight", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--quick_limit_train", type=int, default=0)
    parser.add_argument("--quick_limit_val", type=int, default=0)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_min_epochs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed, False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tfm = build_transforms(args.input_res)
    train_set = subset_dataset(build_dataset(args.data, "train", tfm["train"]), args.quick_limit_train, args.seed)
    valid_set = subset_dataset(build_dataset(args.data, "valid", tfm["valid"]), args.quick_limit_val, args.seed)
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    valid_loader = DataLoader(valid_set, batch_size=args.bs, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)

    backbone = load_backbone(args.checkpoint, device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    train_targets = BBoxTargetBuilder(args.data, args.coco_root, train_set, "train", args.input_res)
    valid_targets = BBoxTargetBuilder(args.data, args.coco_root, valid_set, "valid", args.input_res)
    classes = class_names_from_dataset(train_set)
    cfg = BBoxSetTransformerConfig(
        num_slots=backbone.num_slots,
        slot_dim=backbone.slot_dim,
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        num_heads=args.num_heads,
        num_sab_layers=args.num_sab_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    model = BBoxSetTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bbox_settransformer_meta.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "config": asdict(cfg),
                "classes": classes,
                "supervision": "COCO bbox-derived anchor/evidence slot mass targets",
                "bbox_used_for_training": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"device={device} train={len(train_set)} valid={len(valid_set)} slots={cfg.num_slots} slot_dim={cfg.slot_dim}")
    print(f"trainable_params={sum(p.numel() for p in model.parameters()):,}")
    history: list[dict] = []
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, backbone, train_loader, train_targets, device, optimizer, args, train=True)
        with torch.no_grad():
            valid_stats = run_epoch(model, backbone, valid_loader, valid_targets, device, optimizer, args, train=False)
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        row = {"epoch": epoch, "elapsed": elapsed, **{f"train_{k}": v for k, v in train_stats.items()}, **{f"valid_{k}": v for k, v in valid_stats.items()}}
        history.append(row)
        write_history(out_dir / "history_metrics.csv", history)
        score = valid_stats["anchor_top1"] + valid_stats["evidence_top1"] + valid_stats["pair_acc"]
        print(
            f"epoch={epoch} elapsed={elapsed} score={score:.4f} "
            f"anchor={valid_stats['anchor_top1']:.3f} evidence={valid_stats['evidence_top1']:.3f} "
            f"pair={valid_stats['pair_acc']:.3f} loss={valid_stats['loss']:.4f}"
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "args": vars(args),
                    "config": asdict(cfg),
                    "classes": classes,
                    "epoch": epoch,
                    "valid_score": score,
                    "valid_stats": valid_stats,
                    "model_state_dict": model.state_dict(),
                    "model_class": "bbox-settransformer.model.BBoxSetTransformer",
                },
                out_dir / "bbox_settransformer_best.pt",
            )
            print(f"saved {out_dir / 'bbox_settransformer_best.pt'}")
        else:
            stale += 1
        if args.early_stop_patience > 0 and epoch >= args.early_stop_min_epochs and stale >= args.early_stop_patience:
            print(f"early_stop best_epoch={best_epoch} best_score={best_score:.4f}")
            break
    (out_dir / "final_metrics.json").write_text(
        json.dumps({"best_epoch": best_epoch, "best_score": best_score, "history": history}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
