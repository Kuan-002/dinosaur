#!/usr/bin/env python3
"""Split frozen DINOSAUR slot embeddings into objectness, geometry, and residual subspaces.

This is an embedding-reshaping experiment, not a slot scoring model.  A single
projection P maps each frozen slot embedding z to [u_obj, u_geo, u_res].  Small
auxiliary heads shape the subspaces during training; the saved artifact is the
projection and diagnostics about what information is present or leaking in each
part.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFile
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm

from misc_utils import seed_all
from train_slot_classifier import build_transforms, load_backbone

ImageFile.LOAD_TRUNCATED_IMAGES = True


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


@dataclass
class BottleneckConfig:
    slot_dim: int
    obj_dim: int
    geo_dim: int
    res_dim: int
    hidden_dim: int
    dropout: float
    num_categories: int


class SlotheadProjector(nn.Module):
    def __init__(self, cfg: BottleneckConfig):
        super().__init__()
        self.cfg = cfg
        total_dim = cfg.obj_dim + cfg.geo_dim + cfg.res_dim
        self.projector = nn.Sequential(
            nn.LayerNorm(cfg.slot_dim),
            nn.Linear(cfg.slot_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, total_dim),
        )
        self.obj_head = nn.Linear(cfg.obj_dim, 1)
        self.geo_head = nn.Sequential(nn.LayerNorm(cfg.geo_dim), nn.Linear(cfg.geo_dim, 5))
        self.cat_head = nn.Sequential(
            nn.LayerNorm(cfg.res_dim),
            nn.Linear(cfg.res_dim, cfg.num_categories),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.slot_dim),
        )

    def split(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.projector(slots)
        o = self.cfg.obj_dim
        g = self.cfg.geo_dim
        return u, u[..., :o], u[..., o:o + g], u[..., o + g:]

    def forward(self, slots: torch.Tensor) -> dict[str, torch.Tensor]:
        u, u_obj, u_geo, u_res = self.split(slots)
        return {
            "u": u,
            "u_obj": u_obj,
            "u_geo": u_geo,
            "u_res": u_res,
            "obj_logit": self.obj_head(u_obj).squeeze(-1),
            "geo": self.geo_head(u_geo).sigmoid(),
            "cat_logit": self.cat_head(u_res),
            "recon": self.decoder(u),
        }


class CocoInstanceDataset(Dataset):
    def __init__(self, coco_root: str, split: str, transform, input_res: int, max_images: int = 0):
        self.root = Path(coco_root)
        self.split = "val" if split in {"valid", "val"} else "train"
        self.transform = transform
        self.input_res = input_res
        ann_path = self.root / "annotations" / f"instances_{self.split}2017.json"
        with ann_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self.images = sorted(data["images"], key=lambda row: int(row["id"]))
        if max_images > 0:
            self.images = self.images[:max_images]
        wanted = {int(row["id"]) for row in self.images}
        self.anns_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in wanted}
        for ann in data["annotations"]:
            image_id = int(ann["image_id"])
            if image_id not in wanted or ann.get("iscrowd", 0):
                continue
            if int(ann["category_id"]) not in COCO_CATEGORY_BY_ID:
                continue
            x, y, w, h = [float(v) for v in ann["bbox"]]
            if w <= 1 or h <= 1 or float(ann.get("area", 0.0)) <= 4:
                continue
            self.anns_by_image[image_id].append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        info = self.images[idx]
        image_id = int(info["id"])
        path = self.root / f"{self.split}2017" / info["file_name"]
        image = Image.open(path).convert("RGB")
        return self.transform(image), {
            "image_id": image_id,
            "width": int(info["width"]),
            "height": int(info["height"]),
            "annotations": self.anns_by_image.get(image_id, []),
        }


class ClassificationMetadataCocoInstanceDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        coco_root: str,
        split: str,
        transform,
        max_images: int = 0,
    ):
        self.root = Path(dataset_root)
        self.coco_root = Path(coco_root)
        self.split = split
        self.transform = transform
        metadata_path = self.root / "metadata.csv"
        with metadata_path.open("r", encoding="utf-8", newline="") as f:
            rows = [row for row in csv.DictReader(f) if row["split"] == split]
        self.rows = rows[:max_images] if max_images > 0 else rows

        needed_by_split: dict[str, set[int]] = {}
        for row in self.rows:
            needed_by_split.setdefault(row["source_split"], set()).add(int(row["image_id"]))

        self.anns_by_source_image: dict[tuple[str, int], list[dict[str, Any]]] = {}
        for source_split, image_ids in needed_by_split.items():
            ann_path = self.coco_root / "annotations" / f"instances_{source_split}2017.json"
            with ann_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            for image_id in image_ids:
                self.anns_by_source_image[(source_split, image_id)] = []
            for ann in data["annotations"]:
                image_id = int(ann["image_id"])
                if image_id not in image_ids or ann.get("iscrowd", 0):
                    continue
                if int(ann["category_id"]) not in COCO_CATEGORY_BY_ID:
                    continue
                x, y, w, h = [float(v) for v in ann["bbox"]]
                if w <= 1 or h <= 1 or float(ann.get("area", 0.0)) <= 4:
                    continue
                self.anns_by_source_image[(source_split, image_id)].append(ann)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image_id = int(row["image_id"])
        source_split = row["source_split"]
        path = self.root / row["relative_path"]
        image = Image.open(path).convert("RGB")
        return self.transform(image), {
            "image_id": image_id,
            "width": int(row["width"]),
            "height": int(row["height"]),
            "source_split": source_split,
            "annotations": self.anns_by_source_image.get((source_split, image_id), []),
        }


def collate_coco(batch):
    images, metas = zip(*batch)
    return torch.stack(list(images), dim=0), list(metas)


def choose_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def resize_crop_params(width: int, height: int, input_res: int) -> tuple[float, float, float]:
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
    return scale, crop_x, crop_y


def map_point(x: float, y: float, scale: float, crop_x: float, crop_y: float, size: int) -> tuple[float, float]:
    return (
        min(max(x * scale - crop_x, 0.0), float(size - 1)),
        min(max(y * scale - crop_y, 0.0), float(size - 1)),
    )


def shrink_box(box: list[float], factor: float) -> list[float]:
    x, y, w, h = [float(v) for v in box]
    cx, cy = x + 0.5 * w, y + 0.5 * h
    nw, nh = w * factor, h * factor
    return [cx - 0.5 * nw, cy - 0.5 * nh, nw, nh]


def annotation_mask(ann: dict[str, Any], width: int, height: int, size: int, bbox_shrink: float) -> tuple[torch.Tensor, str]:
    scale, crop_x, crop_y = resize_crop_params(width, height, size)
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    source = "polygon"
    seg = ann.get("segmentation")
    if isinstance(seg, list) and seg:
        drew = False
        for poly in seg:
            if len(poly) < 6:
                continue
            pts = [map_point(float(poly[i]), float(poly[i + 1]), scale, crop_x, crop_y, size) for i in range(0, len(poly), 2)]
            draw.polygon(pts, fill=1)
            drew = True
        if drew:
            out = torch.from_numpy_bool(mask)
            if bool(out.any()):
                return out, source
    source = "shrunken_bbox"
    x, y, w, h = shrink_box(ann["bbox"], bbox_shrink)
    x1, y1 = map_point(x, y, scale, crop_x, crop_y, size)
    x2, y2 = map_point(x + w, y + h, scale, crop_x, crop_y, size)
    ix1, ix2 = sorted([int(math.floor(x1)), int(math.ceil(x2))])
    iy1, iy2 = sorted([int(math.floor(y1)), int(math.ceil(y2))])
    if ix2 > ix1 and iy2 > iy1:
        draw.rectangle([ix1, iy1, ix2, iy2], fill=1)
    return torch.from_numpy_bool(mask), source


def pil_mask_to_bool(mask: Image.Image) -> torch.Tensor:
    return torch.as_tensor(bytearray(mask.tobytes()), dtype=torch.uint8).view(mask.height, mask.width).bool()


# torch.from_numpy is avoided here because old cluster images sometimes expose
# PIL/numpy ABI mismatches.  Monkey-patch a local name for readability above.
torch.from_numpy_bool = pil_mask_to_bool  # type: ignore[attr-defined]


def make_heatmaps(attn: torch.Tensor, size: int) -> torch.Tensor:
    k, n = attn.shape
    side = int(math.sqrt(n))
    maps = attn.reshape(k, 1, side, side)
    maps = F.interpolate(maps, size=(size, size), mode="bilinear", align_corners=False)[:, 0]
    denom = maps.flatten(1).sum(dim=1).clamp_min(1e-8).view(k, 1, 1)
    return (maps / denom).detach().cpu()


def binary_slot_mask(heatmap: torch.Tensor, threshold_rel: float) -> torch.Tensor:
    peak = float(heatmap.max().item())
    if peak <= 0:
        return torch.zeros_like(heatmap, dtype=torch.bool)
    return heatmap >= peak * threshold_rel


def mask_geometry(mask: torch.Tensor) -> torch.Tensor:
    ys, xs = mask.nonzero(as_tuple=True)
    if xs.numel() == 0:
        return torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0], dtype=torch.float32)
    h, w = mask.shape
    x1, x2 = xs.float().min(), xs.float().max()
    y1, y2 = ys.float().min(), ys.float().max()
    bw = (x2 - x1 + 1.0) / float(w)
    bh = (y2 - y1 + 1.0) / float(h)
    cx = (x1 + x2 + 1.0) / (2.0 * w)
    cy = (y1 + y2 + 1.0) / (2.0 * h)
    area = mask.float().mean()
    return torch.tensor([cx, cy, bw, bh, float(area)], dtype=torch.float32)


def prepare_targets(
    attn: torch.Tensor,
    metas: list[dict[str, Any]],
    input_res: int,
    threshold_rel: float,
    pos_coverage: float,
    pos_purity: float,
    ignore_coverage: float,
    ignore_purity: float,
    bbox_shrink: float,
    cat_to_idx: dict[int, int],
) -> dict[str, torch.Tensor | dict[str, float]]:
    batch = len(metas)
    num_slots = attn.size(1)
    obj = torch.zeros(batch, num_slots, dtype=torch.float32)
    obj_weight = torch.ones(batch, num_slots, dtype=torch.float32)
    geo = torch.zeros(batch, num_slots, 5, dtype=torch.float32)
    geo_weight = torch.zeros(batch, num_slots, dtype=torch.float32)
    cat = torch.full((batch, num_slots), -100, dtype=torch.long)
    source_counts = {"polygon": 0.0, "shrunken_bbox": 0.0, "empty_images": 0.0}
    for b, meta in enumerate(metas):
        heatmaps = make_heatmaps(attn[b].detach().cpu(), input_res)
        slot_masks = [binary_slot_mask(heatmaps[k], threshold_rel) for k in range(num_slots)]
        object_masks: list[tuple[torch.Tensor, int]] = []
        for ann in meta["annotations"]:
            mask, source = annotation_mask(ann, int(meta["width"]), int(meta["height"]), input_res, bbox_shrink)
            if not bool(mask.any()):
                continue
            object_masks.append((mask, int(ann["category_id"])))
            source_counts[source] += 1.0
        if not object_masks:
            source_counts["empty_images"] += 1.0
        for k, slot_mask in enumerate(slot_masks):
            geo[b, k] = mask_geometry(slot_mask)
            best = None
            slot_area = float(slot_mask.sum().item())
            for obj_mask, category_id in object_masks:
                inter = float((slot_mask & obj_mask).sum().item())
                coverage = inter / max(float(obj_mask.sum().item()), 1.0)
                purity = inter / max(slot_area, 1.0)
                score = math.sqrt(max(coverage, 0.0) * max(purity, 0.0))
                if best is None or score > best[0]:
                    best = (score, coverage, purity, obj_mask, category_id)
            if best is None:
                continue
            _score, coverage, purity, obj_mask, category_id = best
            if coverage >= pos_coverage and purity >= pos_purity:
                obj[b, k] = 1.0
                geo[b, k] = mask_geometry(obj_mask)
                geo_weight[b, k] = 1.0
                cat[b, k] = cat_to_idx[category_id]
            elif coverage >= ignore_coverage or purity >= ignore_purity:
                obj_weight[b, k] = 0.0
    return {
        "objectness": obj,
        "objectness_weight": obj_weight,
        "geometry": geo,
        "geometry_weight": geo_weight,
        "category": cat,
        "source_counts": source_counts,
    }


@torch.no_grad()
def encode_batch(backbone, images: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.forward_dino(images)
    features = backbone.mlp(features)
    slots, attn, _ = backbone.slot_attention(features)
    return slots.detach(), attn.detach()


def covariance_penalty(parts: list[torch.Tensor]) -> torch.Tensor:
    losses = []
    flat = [p.reshape(-1, p.size(-1)) for p in parts if p.numel() > 0 and p.size(-1) > 0]
    for i in range(len(flat)):
        xi = flat[i] - flat[i].mean(dim=0, keepdim=True)
        xi = xi / xi.std(dim=0, keepdim=True).clamp_min(1e-4)
        for j in range(i + 1, len(flat)):
            xj = flat[j] - flat[j].mean(dim=0, keepdim=True)
            xj = xj / xj.std(dim=0, keepdim=True).clamp_min(1e-4)
            cov = xi.T @ xj / max(xi.size(0) - 1, 1)
            losses.append(cov.square().mean())
    return torch.stack(losses).sum() if losses else torch.tensor(0.0, device=parts[0].device)


def weighted_bce(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, pos_weight: float) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=torch.tensor(pos_weight, device=logits.device),
        reduction="none",
    )
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


def train_epoch(model, backbone, loader, optimizer, args, device, cat_to_idx):
    model.train()
    totals: dict[str, float] = {}
    counts: dict[str, float] = {}
    count = 0
    source_counts = {"polygon": 0.0, "shrunken_bbox": 0.0, "empty_images": 0.0}
    for images, metas in tqdm(loader, desc="train", mininterval=1.0):
        slots, attn = encode_batch(backbone, images, device)
        targets = prepare_targets(
            attn, metas, args.input_res, args.threshold_rel, args.pos_coverage,
            args.pos_purity, args.ignore_coverage, args.ignore_purity, args.bbox_shrink,
            cat_to_idx,
        )
        for key, value in targets["source_counts"].items():
            source_counts[key] += float(value)
        slots = slots.to(device)
        out = model(slots)
        obj = targets["objectness"].to(device)
        obj_w = targets["objectness_weight"].to(device)
        geo = targets["geometry"].to(device)
        geo_w = targets["geometry_weight"].to(device)
        cat = targets["category"].to(device)
        loss_obj = weighted_bce(out["obj_logit"], obj, obj_w, args.obj_pos_weight)
        loss_geo = ((out["geo"] - geo).square().mean(dim=-1) * geo_w).sum() / geo_w.sum().clamp_min(1.0)
        pos = cat.ne(-100)
        loss_cat = F.cross_entropy(out["cat_logit"][pos], cat[pos]) if bool(pos.any()) else out["cat_logit"].sum() * 0.0
        loss_rec = F.mse_loss(out["recon"], slots)
        loss_orth = covariance_penalty([out["u_obj"], out["u_geo"], out["u_res"]])
        loss = (
            args.lambda_obj * loss_obj
            + args.lambda_geo * loss_geo
            + args.lambda_cat * loss_cat
            + args.lambda_rec * loss_rec
            + args.lambda_orth * loss_orth
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        n = images.size(0)
        count += n
        for name, value in [
            ("loss", loss), ("loss_obj", loss_obj), ("loss_geo", loss_geo),
            ("loss_cat", loss_cat), ("loss_rec", loss_rec), ("loss_orth", loss_orth),
        ]:
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * n
            counts[name] = counts.get(name, 0.0) + n
        for name, value in [("positive_slots", obj.sum()), ("ignored_slots", obj_w.eq(0).sum())]:
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            counts[name] = counts.get(name, 0.0) + 1.0
    stats = {key: value / max(counts.get(key, count), 1.0) for key, value in totals.items()}
    stats.update({f"mask_source_{key}": value for key, value in source_counts.items()})
    return stats


def roc_auc(scores: list[float], labels: list[int]) -> float | None:
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for idx in order[i:j]:
            ranks[idx] = rank
        i = j
    pos_rank_sum = sum(ranks[i] for i, label in enumerate(labels) if label)
    return (pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def r2_by_dim(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    mse = (pred - target).square().mean(dim=0)
    var = target.var(dim=0, unbiased=False).clamp_min(1e-8)
    return {name: float(1.0 - mse[i] / var[i]) for i, name in enumerate(["cx", "cy", "w", "h", "area"])}


@torch.no_grad()
def collect_subspace_dataset(model, backbone, loader, args, device, cat_to_idx, max_slots: int) -> dict[str, torch.Tensor]:
    model.eval()
    parts = {"u_obj": [], "u_geo": [], "u_res": []}
    obj_labels = []
    obj_weights = []
    geo_targets = []
    geo_weights = []
    cat_targets = []
    total_slots = 0
    for images, metas in tqdm(loader, desc="collect-diagnostics", mininterval=1.0):
        slots, attn = encode_batch(backbone, images, device)
        targets = prepare_targets(
            attn, metas, args.input_res, args.threshold_rel, args.pos_coverage,
            args.pos_purity, args.ignore_coverage, args.ignore_purity, args.bbox_shrink,
            cat_to_idx,
        )
        out = model(slots.to(device))
        flat_n = slots.size(0) * slots.size(1)
        for name in parts:
            parts[name].append(out[name].detach().cpu().reshape(flat_n, -1))
        obj_labels.append(targets["objectness"].reshape(-1))
        obj_weights.append(targets["objectness_weight"].reshape(-1))
        geo_targets.append(targets["geometry"].reshape(flat_n, 5))
        geo_weights.append(targets["geometry_weight"].reshape(-1))
        cat_targets.append(targets["category"].reshape(-1))
        total_slots += flat_n
        if max_slots > 0 and total_slots >= max_slots:
            break
    data = {name: torch.cat(chunks, dim=0) for name, chunks in parts.items()}
    data["objectness"] = torch.cat(obj_labels, dim=0)
    data["objectness_weight"] = torch.cat(obj_weights, dim=0)
    data["geometry"] = torch.cat(geo_targets, dim=0)
    data["geometry_weight"] = torch.cat(geo_weights, dim=0)
    data["category"] = torch.cat(cat_targets, dim=0)
    if max_slots > 0:
        keep = slice(0, max_slots)
        data = {key: value[keep] for key, value in data.items()}
    return data


def fit_linear_objectness(x_train, y_train, w_train, x_val, y_val, w_val, epochs: int, device: torch.device) -> dict[str, Any]:
    keep = w_train.bool()
    x_train, y_train = x_train[keep], y_train[keep]
    keep_val = w_val.bool()
    x_val, y_val = x_val[keep_val], y_val[keep_val]
    model = nn.Sequential(nn.LayerNorm(x_train.size(-1)), nn.Linear(x_train.size(-1), 1)).to(device)
    pos = y_train.sum().clamp_min(1.0)
    neg = y_train.numel() - pos
    pos_weight = (neg / pos).clamp(1.0, 50.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        order = torch.randperm(x_train.size(0))
        for start in range(0, order.numel(), 4096):
            idx = order[start:start + 4096]
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            logits = model(xb).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        scores = []
        for start in range(0, x_val.size(0), 8192):
            scores.append(model(x_val[start:start + 8192].to(device)).squeeze(-1).sigmoid().cpu())
    return {"auc": roc_auc(torch.cat(scores).tolist(), y_val.int().tolist()), "positive_rate": float(y_val.mean())}


def fit_linear_geometry(x_train, geo_train, w_train, x_val, geo_val, w_val, epochs: int, device: torch.device) -> dict[str, Any]:
    keep = w_train.bool()
    keep_val = w_val.bool()
    x_train, geo_train = x_train[keep], geo_train[keep]
    x_val, geo_val = x_val[keep_val], geo_val[keep_val]
    if x_train.numel() == 0 or x_val.numel() == 0:
        return {"r2": None, "positive_slots": 0}
    model = nn.Sequential(nn.LayerNorm(x_train.size(-1)), nn.Linear(x_train.size(-1), 5), nn.Sigmoid()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        order = torch.randperm(x_train.size(0))
        for start in range(0, order.numel(), 4096):
            idx = order[start:start + 4096]
            pred = model(x_train[idx].to(device))
            loss = F.mse_loss(pred, geo_train[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = []
        for start in range(0, x_val.size(0), 8192):
            pred.append(model(x_val[start:start + 8192].to(device)).cpu())
    return {"r2": r2_by_dim(torch.cat(pred, dim=0), geo_val), "positive_slots": int(x_val.size(0))}


def fit_linear_category(x_train, cat_train, x_val, cat_val, num_classes: int, epochs: int, device: torch.device) -> dict[str, Any]:
    keep = cat_train.ne(-100)
    keep_val = cat_val.ne(-100)
    x_train, cat_train = x_train[keep], cat_train[keep]
    x_val, cat_val = x_val[keep_val], cat_val[keep_val]
    if x_train.numel() == 0 or x_val.numel() == 0:
        return {"acc": None, "positive_slots": 0}
    model = nn.Sequential(nn.LayerNorm(x_train.size(-1)), nn.Linear(x_train.size(-1), num_classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        order = torch.randperm(x_train.size(0))
        for start in range(0, order.numel(), 4096):
            idx = order[start:start + 4096]
            logits = model(x_train[idx].to(device))
            loss = F.cross_entropy(logits, cat_train[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, x_val.size(0), 8192):
            pred = model(x_val[start:start + 8192].to(device)).argmax(dim=-1).cpu()
            y = cat_val[start:start + 8192]
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return {"acc": correct / max(total, 1), "positive_slots": total}


def run_leakage_diagnostics(model, backbone, train_loader, val_loader, args, device, cat_to_idx) -> dict[str, Any]:
    train = collect_subspace_dataset(model, backbone, train_loader, args, device, cat_to_idx, args.diagnostic_max_train_slots)
    val = collect_subspace_dataset(model, backbone, val_loader, args, device, cat_to_idx, args.diagnostic_max_val_slots)
    out: dict[str, Any] = {
        "train_slots": int(train["objectness"].numel()),
        "val_slots": int(val["objectness"].numel()),
        "note": "fresh linear probes fitted after freezing P; high cross-subspace values indicate leakage",
    }
    for part in ["u_obj", "u_geo", "u_res"]:
        out[part] = {
            "objectness": fit_linear_objectness(
                train[part], train["objectness"], train["objectness_weight"],
                val[part], val["objectness"], val["objectness_weight"],
                args.diagnostic_epochs, device,
            ),
            "geometry": fit_linear_geometry(
                train[part], train["geometry"], train["geometry_weight"],
                val[part], val["geometry"], val["geometry_weight"],
                args.diagnostic_epochs, device,
            ),
            "category": fit_linear_category(
                train[part], train["category"], val[part], val["category"],
                len(cat_to_idx), args.diagnostic_epochs, device,
            ),
        }
    return out


@torch.no_grad()
def evaluate(model, backbone, loader, args, device, cat_to_idx) -> dict[str, Any]:
    model.eval()
    obj_scores: list[float] = []
    obj_labels: list[int] = []
    geo_pred: list[torch.Tensor] = []
    geo_true: list[torch.Tensor] = []
    cat_correct = 0
    cat_total = 0
    recon_losses = []
    source_counts = {"polygon": 0.0, "shrunken_bbox": 0.0, "empty_images": 0.0}
    for images, metas in tqdm(loader, desc="eval", mininterval=1.0):
        slots, attn = encode_batch(backbone, images, device)
        targets = prepare_targets(
            attn, metas, args.input_res, args.threshold_rel, args.pos_coverage,
            args.pos_purity, args.ignore_coverage, args.ignore_purity, args.bbox_shrink,
            cat_to_idx,
        )
        for key, value in targets["source_counts"].items():
            source_counts[key] += float(value)
        slots = slots.to(device)
        out = model(slots)
        obj = targets["objectness"]
        obj_w = targets["objectness_weight"].bool()
        scores = out["obj_logit"].sigmoid().detach().cpu()
        obj_scores.extend(scores[obj_w].flatten().tolist())
        obj_labels.extend(obj[obj_w].int().flatten().tolist())
        geo_w = targets["geometry_weight"].bool()
        if bool(geo_w.any()):
            geo_pred.append(out["geo"].detach().cpu()[geo_w])
            geo_true.append(targets["geometry"][geo_w])
        cat = targets["category"].to(device)
        pos = cat.ne(-100)
        if bool(pos.any()):
            pred = out["cat_logit"][pos].argmax(dim=-1)
            cat_correct += int((pred == cat[pos]).sum().item())
            cat_total += int(pos.sum().item())
        recon_losses.append(float(F.mse_loss(out["recon"], slots).detach().cpu()))
    metrics: dict[str, Any] = {
        "objectness_auc_from_u_obj": roc_auc(obj_scores, obj_labels),
        "objectness_positive_rate": sum(obj_labels) / max(len(obj_labels), 1),
        "category_acc_from_u_res": cat_correct / max(cat_total, 1),
        "category_positive_slots": cat_total,
        "reconstruction_mse": sum(recon_losses) / max(len(recon_losses), 1),
        "mask_sources": source_counts,
    }
    if geo_pred:
        pred = torch.cat(geo_pred, dim=0)
        true = torch.cat(geo_true, dim=0)
        mse = (pred - true).square().mean(dim=0)
        var = true.var(dim=0).clamp_min(1e-8)
        metrics["geometry_mse_from_u_geo"] = {
            name: float(mse[i]) for i, name in enumerate(["cx", "cy", "w", "h", "area"])
        }
        metrics["geometry_r2_from_u_geo"] = {
            name: float(1.0 - mse[i] / var[i]) for i, name in enumerate(["cx", "cy", "w", "h", "area"])
        }
    return metrics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coco_root", default="/vol/biomedic3/kw1025/dinosaur/dataset/coco2017")
    p.add_argument("--classification_dataset", default="", help="Optional classification_dataset root with metadata.csv; uses its train/val splits instead of full COCO.")
    p.add_argument("--sa_checkpoint", default="/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--input_res", type=int, default=224)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--obj_dim", type=int, default=16)
    p.add_argument("--geo_dim", type=int, default=16)
    p.add_argument("--res_dim", type=int, default=24)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lambda_obj", type=float, default=1.0)
    p.add_argument("--lambda_geo", type=float, default=2.0)
    p.add_argument("--lambda_cat", type=float, default=0.5)
    p.add_argument("--lambda_rec", type=float, default=0.1)
    p.add_argument("--lambda_orth", type=float, default=0.02)
    p.add_argument("--obj_pos_weight", type=float, default=4.0)
    p.add_argument("--threshold_rel", type=float, default=0.5)
    p.add_argument("--pos_coverage", type=float, default=0.25)
    p.add_argument("--pos_purity", type=float, default=0.20)
    p.add_argument("--ignore_coverage", type=float, default=0.12)
    p.add_argument("--ignore_purity", type=float, default=0.10)
    p.add_argument("--bbox_shrink", type=float, default=0.70)
    p.add_argument(
        "--category_mode",
        choices=["object", "coco"],
        default="object",
        help="Use one class named object for bbox-derived category targets by default; coco restores COCO category labels.",
    )
    p.add_argument("--quick_limit_train", type=int, default=0)
    p.add_argument("--quick_limit_val", type=int, default=0)
    p.add_argument("--diagnostic_max_train_slots", type=int, default=300000)
    p.add_argument("--diagnostic_max_val_slots", type=int, default=0)
    p.add_argument("--diagnostic_epochs", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=8)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed, False)
    device = choose_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tfm = build_transforms(args.input_res)
    if args.classification_dataset:
        train_set = ClassificationMetadataCocoInstanceDataset(
            args.classification_dataset, args.coco_root, "train", tfm["valid"], args.quick_limit_train
        )
        val_set = ClassificationMetadataCocoInstanceDataset(
            args.classification_dataset, args.coco_root, "val", tfm["valid"], args.quick_limit_val
        )
    else:
        train_set = CocoInstanceDataset(args.coco_root, "train", tfm["valid"], args.input_res, args.quick_limit_train)
        val_set = CocoInstanceDataset(args.coco_root, "val", tfm["valid"], args.input_res, args.quick_limit_val)
    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=True, drop_last=False, num_workers=args.num_workers, collate_fn=collate_coco, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_set, batch_size=args.bs, shuffle=False, drop_last=False, num_workers=args.num_workers, collate_fn=collate_coco, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
    if args.category_mode == "object":
        cat_ids = sorted(COCO_CATEGORY_BY_ID)
        cat_to_idx = {cat_id: 0 for cat_id in cat_ids}
        saved_category_ids = [0]
        category_index = {"0": "object"}
        category_names = ["object"]
        num_categories = 1
    else:
        cat_ids = sorted(COCO_CATEGORY_BY_ID)
        cat_to_idx = {cat_id: i for i, cat_id in enumerate(cat_ids)}
        saved_category_ids = cat_ids
        category_index = {str(cat_to_idx[k]): COCO_CATEGORY_BY_ID[k] for k in cat_ids}
        category_names = [COCO_CATEGORY_BY_ID[k] for k in cat_ids]
        num_categories = len(cat_ids)
    backbone = load_backbone(args.sa_checkpoint, device)
    backbone.eval()
    backbone.requires_grad_(False)
    slot_dim = int(getattr(backbone, "slot_dim"))
    cfg = BottleneckConfig(slot_dim, args.obj_dim, args.geo_dim, args.res_dim, args.hidden_dim, args.dropout, num_categories)
    model = SlotheadProjector(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    meta = {
        "setting": "single projection splitting frozen DINOSAUR slot embeddings into u_obj/u_geo/u_res",
        "dataset": "classification_metadata" if args.classification_dataset else "full_coco_instances",
        "train_images": len(train_set),
        "val_images": len(val_set),
        "bbox_compensation": "COCO polygon masks are used when possible; otherwise bbox is center-shrunk before overlap matching.",
        "args": vars(args),
        "config": asdict(cfg),
        "category_index": category_index,
    }
    (out_dir / "experiment_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    history: list[dict[str, Any]] = []
    best_auc = -1.0
    best_epoch = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(model, backbone, train_loader, optimizer, args, device, cat_to_idx)
        val_metrics = evaluate(model, backbone, val_loader, args, device, cat_to_idx)
        val_auc = val_metrics.get("objectness_auc_from_u_obj") or 0.0
        row = {
            "epoch": epoch,
            "elapsed": time.strftime("%H:%M:%S", time.gmtime(time.time() - start)),
            **train_stats,
            "val_objectness_auc_from_u_obj": val_auc,
            "val_category_acc_from_u_res": val_metrics["category_acc_from_u_res"],
            "val_reconstruction_mse": val_metrics["reconstruction_mse"],
        }
        history.append(row)
        write_csv(out_dir / "history_metrics.csv", history)
        print(json.dumps(row, sort_keys=True))
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "config": asdict(cfg),
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "category_ids": saved_category_ids,
                    "category_names": category_names,
                },
                out_dir / "slothead_best.pt",
            )
    ckpt = torch.load(out_dir / "slothead_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    final = {
        "best_epoch": best_epoch,
        "best_val_objectness_auc_from_u_obj": best_auc,
        "validation": evaluate(model, backbone, val_loader, args, device, cat_to_idx),
        "leakage_diagnostics": run_leakage_diagnostics(model, backbone, train_loader, val_loader, args, device, cat_to_idx),
        "outputs": {
            "checkpoint": str(out_dir / "slothead_best.pt"),
            "meta": str(out_dir / "experiment_meta.json"),
            "history": str(out_dir / "history_metrics.csv"),
        },
    }
    (out_dir / "final_metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
