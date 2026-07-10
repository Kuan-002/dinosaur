#!/usr/bin/env python3
"""Evaluate SET slot rankings against anchor/evidence boxes at @3 and @4."""

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
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from settransformer.model import DiscriminativeSetTransformer, ProbeConfig
from train_slot_classifier import build_dataset, build_transforms, load_backbone


def parse_top_ks(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("--top_ks must contain positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=["full", "56", "80", "112"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="/vol/biomedic3/kw1025/dinosaur/dataset/coco_top2_clean10_area006_004_600_200_200/classification_dataset")
    parser.add_argument("--split", default="test", choices=["valid", "val", "test"])
    parser.add_argument("--coco_root", default="/vol/biomedic3/kw1025/dinosaur/dataset/coco2017")
    parser.add_argument("--sa_checkpoint", default="")
    parser.add_argument("--slothead_checkpoint", default="")
    parser.add_argument("--slothead_mode", default="")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--input_res", type=int, default=0)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--top_ks", type=parse_top_ks, default=parse_top_ks("3,4"))
    parser.add_argument("--hit_threshold", type=float, default=0.4)
    parser.add_argument("--threshold_rel", type=float, default=0.5)
    parser.add_argument("--rank_score", choices=["true_prob", "true_margin"], default="true_prob")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_metadata(data_root: str) -> dict[str, dict[str, str]]:
    rows = read_csv(Path(data_root) / "metadata.csv")
    return {row["relative_path"]: row for row in rows}


def dataset_indices(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        return [int(dataset.indices[i]) for i in range(len(dataset))]
    return list(range(len(dataset)))


def unwrap_dataset(dataset: Dataset) -> Dataset:
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def sample_relative_path(base_dataset: Dataset, sample_idx: int, split: str) -> str:
    sample_path = Path(base_dataset.samples[sample_idx][0])
    root = Path(getattr(base_dataset, "root"))
    rel = sample_path.relative_to(root).as_posix()
    if rel.startswith(("train/", "valid/", "test/")):
        if split == "valid" and rel.startswith("val/"):
            return "valid/" + rel.split("/", 1)[1]
        return rel
    return f"{split}/{rel}"


def load_coco_categories(coco_root: Path, split: str) -> dict[int, str]:
    ann_path = coco_root / "annotations" / f"instances_{split}2017.json"
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(cat["id"]): str(cat["name"]) for cat in data["categories"]}


def load_coco_boxes(coco_root: Path, needed: set[tuple[str, int, str]]) -> dict[tuple[str, int, str], list[list[float]]]:
    by_split: dict[str, set[int]] = defaultdict(set)
    wanted_names: dict[tuple[str, int], set[str]] = defaultdict(set)
    for split, image_id, name in needed:
        by_split[split].add(image_id)
        wanted_names[(split, image_id)].add(name)

    out: dict[tuple[str, int, str], list[list[float]]] = defaultdict(list)
    for split, image_ids in by_split.items():
        ann_path = coco_root / "annotations" / f"instances_{split}2017.json"
        with ann_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        categories = {int(cat["id"]): str(cat["name"]) for cat in data["categories"]}
        for ann in data["annotations"]:
            image_id = int(ann["image_id"])
            if image_id not in image_ids or ann.get("iscrowd", 0):
                continue
            name = categories.get(int(ann["category_id"]))
            if name not in wanted_names[(split, image_id)]:
                continue
            x, y, w, h = [float(v) for v in ann["bbox"]]
            if w <= 1 or h <= 1:
                continue
            out[(split, image_id, name)].append([x, y, x + w, y + h])
    return out


def transform_boxes_to_input(boxes: list[list[float]], width: int, height: int, size: int) -> torch.Tensor:
    if not boxes:
        return torch.empty(0, 4)
    sx = size / max(float(width), 1.0)
    sy = size / max(float(height), 1.0)
    out = torch.tensor(boxes, dtype=torch.float32)
    out[:, [0, 2]] *= sx
    out[:, [1, 3]] *= sy
    out.clamp_(0, size - 1)
    return out


def boxes_to_mask(boxes: torch.Tensor, size: int) -> torch.Tensor:
    mask = torch.zeros(size, size, dtype=torch.bool)
    for x1, y1, x2, y2 in boxes.tolist():
        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(size, int(x2) + 1), min(size, int(y2) + 1)
        if ix2 > ix1 and iy2 > iy1:
            mask[iy1:iy2, ix1:ix2] = True
    return mask


def make_heatmaps(attn: torch.Tensor, size: int) -> torch.Tensor:
    k, n = attn.shape
    side = int(n**0.5)
    if side * side != n:
        raise ValueError(f"attention token count is not square: {n}")
    maps = attn.reshape(k, side, side).float()
    maps = F.interpolate(maps[:, None], size=(size, size), mode="bilinear", align_corners=False)[:, 0]
    maps = maps.clamp_min(0)
    maps = maps / maps.flatten(1).sum(dim=1).clamp_min(1e-8)[:, None, None]
    return maps


def slot_mask_mass(heatmap: torch.Tensor, mask: torch.Tensor, threshold_rel: float) -> float:
    if not mask.any():
        return 0.0
    active = heatmap >= (float(heatmap.max()) * threshold_rel)
    selected_mass = float(heatmap[mask].sum().item())
    active_overlap = float((active & mask).sum().item()) / float(active.sum().item() or 1)
    return max(selected_mass, active_overlap)


def hit_stats(heatmaps: torch.Tensor, slots: list[int], mask: torch.Tensor, hit_threshold: float, threshold_rel: float) -> tuple[bool, float]:
    if not slots:
        return False, 0.0
    masses = [slot_mask_mass(heatmaps[s], mask, threshold_rel) for s in slots]
    best = max(masses) if masses else 0.0
    return best >= hit_threshold, best


@torch.no_grad()
def encode_raw_slots(backbone, images: torch.Tensor, device: torch.device):
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.forward_dino(images)
    features = backbone.mlp(features)
    slots, attn, _ = backbone.slot_attention(features)
    return slots.detach(), attn.detach()


def load_slothead(args: argparse.Namespace, device: torch.device):
    if args.variant == "full":
        return None
    if not args.slothead_checkpoint:
        raise ValueError(f"--slothead_checkpoint is required for SET{args.variant}")
    if args.variant == "56":
        from SET56.slothead56 import load_slothead56

        return load_slothead56(args.slothead_checkpoint, device)[0]
    if args.variant == "80":
        from SET80.slothead80 import load_slothead80

        return load_slothead80(args.slothead_checkpoint, device)[0]
    if args.variant == "112":
        from SET112.slothead112 import load_slothead112

        return load_slothead112(args.slothead_checkpoint, device)[0]
    raise ValueError(args.variant)


def project_slots(projector, variant: str, raw_slots: torch.Tensor, mode: str) -> torch.Tensor:
    if variant == "full":
        return raw_slots
    return projector(raw_slots, mode=mode or "u")


def load_probe(path: str, device: torch.device) -> tuple[DiscriminativeSetTransformer, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ProbeConfig(**ckpt["probe_config"])
    model = DiscriminativeSetTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def rank_slots(model: DiscriminativeSetTransformer, slots: torch.Tensor, labels: torch.Tensor, score_mode: str) -> list[list[int]]:
    b, k, d = slots.shape
    masks = torch.eye(k, dtype=torch.bool, device=slots.device).unsqueeze(0).expand(b, -1, -1)
    flat_slots = slots[:, None].expand(-1, k, -1, -1).reshape(b * k, k, d)
    flat_masks = masks.reshape(b * k, k)
    logits = model(flat_slots, flat_masks).reshape(b, k, -1)
    if score_mode == "true_prob":
        scores = logits.softmax(dim=-1).gather(2, labels[:, None, None].expand(-1, k, 1)).squeeze(-1)
    else:
        true_logits = logits.gather(2, labels[:, None, None].expand(-1, k, 1)).squeeze(-1)
        other_logits = logits.masked_fill(F.one_hot(labels, logits.size(-1)).bool()[:, None], torch.finfo(logits.dtype).min).amax(dim=-1)
        scores = true_logits - other_logits
    return scores.argsort(dim=1, descending=True).detach().cpu().tolist()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_summary(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, float]:
    if not rows:
        return {key: 0.0 for key in keys}
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


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
    if args.max_items and args.max_items < len(dataset):
        dataset = Subset(dataset, list(range(args.max_items)))
    base_dataset = unwrap_dataset(dataset)
    base_indices = dataset_indices(dataset)

    metadata = load_metadata(args.data)
    needed = set()
    for sample_idx in base_indices:
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

    rows = []
    cursor = 0
    for images, labels in tqdm(loader, desc=f"set{args.variant}-bbox-eval", mininterval=1.0):
        labels = labels.to(device, non_blocking=device.type == "cuda")
        raw_slots, attn = encode_raw_slots(backbone, images, device)
        slots = project_slots(projector, args.variant, raw_slots, slothead_mode)
        ranked = rank_slots(model, slots, labels, args.rank_score)
        batch = labels.numel()
        for b in range(batch):
            sample_idx = base_indices[cursor + b]
            rel = sample_relative_path(base_dataset, sample_idx, split)
            meta_row = metadata[rel]
            image_id = int(meta_row["image_id"])
            source_split = meta_row["source_split"]
            width = int(meta_row["width"])
            height = int(meta_row["height"])
            anchor_name = meta_row["anchor_object"]
            evidence_name = meta_row["evidence_object"]
            anchor_boxes = transform_boxes_to_input(coco_boxes.get((source_split, image_id, anchor_name), []), width, height, input_res)
            evidence_boxes = transform_boxes_to_input(coco_boxes.get((source_split, image_id, evidence_name), []), width, height, input_res)
            anchor_mask = boxes_to_mask(anchor_boxes, input_res)
            evidence_mask = boxes_to_mask(evidence_boxes, input_res)
            union_mask = anchor_mask | evidence_mask
            heatmaps = make_heatmaps(attn[b].detach().cpu(), input_res)
            row = {
                "dataset_index": cursor + b,
                "relative_path": rel,
                "true": int(labels[b].detach().cpu().item()),
                "true_name": meta_row["class_name"],
                "image_id": image_id,
                "source_split": source_split,
                "anchor_object": anchor_name,
                "evidence_object": evidence_name,
                "ranked_slots_1based": json.dumps([slot_id + 1 for slot_id in ranked[b]]),
            }
            for top_k in args.top_ks:
                top_slots = ranked[b][:top_k]
                anchor_hit, anchor_mass = hit_stats(heatmaps, top_slots, anchor_mask, args.hit_threshold, args.threshold_rel)
                evidence_hit, evidence_mass = hit_stats(heatmaps, top_slots, evidence_mask, args.hit_threshold, args.threshold_rel)
                _union_hit, union_mass = hit_stats(heatmaps, top_slots, union_mask, args.hit_threshold, args.threshold_rel)
                row[f"top{top_k}_slots_1based"] = json.dumps([slot_id + 1 for slot_id in top_slots])
                row[f"anchor@{top_k}"] = float(anchor_hit)
                row[f"evidence@{top_k}"] = float(evidence_hit)
                row[f"pair@{top_k}"] = float(anchor_hit and evidence_hit)
                row[f"top{top_k}_best_anchor_mass"] = anchor_mass
                row[f"top{top_k}_best_evidence_mass"] = evidence_mass
                row[f"top{top_k}_best_union_mass"] = union_mass
            rows.append(row)
        cursor += batch

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.checkpoint).resolve().parent / f"{split}_bbox_at3_at4_thr{args.hit_threshold:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "set_bbox_eval.csv", rows)
    metric_keys = []
    for top_k in args.top_ks:
        metric_keys.extend(
            [
                f"anchor@{top_k}",
                f"evidence@{top_k}",
                f"pair@{top_k}",
                f"top{top_k}_best_anchor_mass",
                f"top{top_k}_best_evidence_mass",
                f"top{top_k}_best_union_mass",
            ]
        )
    rows_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_class[f"{row['true']}:{row['true_name']}"].append(row)
    summary = {
        "variant": args.variant,
        "checkpoint": args.checkpoint,
        "data": args.data,
        "split": split,
        "items": len(rows),
        "top_ks": args.top_ks,
        "hit_threshold": args.hit_threshold,
        "threshold_rel": args.threshold_rel,
        "rank_score": args.rank_score,
        "bbox_is_eval_only": True,
        "slothead_checkpoint": args.slothead_checkpoint,
        "slothead_mode": slothead_mode,
        "metrics": metric_summary(rows, metric_keys),
        "per_class": {
            key: {"items": len(class_rows), "metrics": metric_summary(class_rows, metric_keys)}
            for key, class_rows in sorted(rows_by_class.items(), key=lambda item: int(item[0].split(":", 1)[0]))
        },
        "outputs": {"csv": str(out_dir / "set_bbox_eval.csv"), "summary": str(out_dir / "summary.json")},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
