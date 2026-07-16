#!/usr/bin/env python3
"""Visualize object-like regions reconstructed from DINOSAUR attention masks."""

from __future__ import annotations

import argparse
import csv
import html
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCH_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "torch"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from misc_utils import seed_all
from train_slot_classifier import build_dataset, build_transforms, load_backbone


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base
        self.classes = getattr(base, "classes", None)
        self.samples = getattr(base, "samples", None)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, label = self.base[idx]
        return image, label, idx


def choose_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def select_per_class(dataset: Dataset, per_class: int) -> tuple[list[int], list[str]]:
    classes = list(getattr(dataset, "classes", []))
    samples = getattr(dataset, "samples", None)
    if not classes or samples is None:
        raise ValueError("Dataset must expose ImageFolder-style classes and samples.")
    counts = {idx: 0 for idx in range(len(classes))}
    selected: list[int] = []
    for idx, (_path, label) in enumerate(samples):
        label = int(label)
        if counts[label] >= per_class:
            continue
        selected.append(idx)
        counts[label] += 1
        if all(count >= per_class for count in counts.values()):
            break
    missing = {classes[label]: per_class - count for label, count in counts.items() if count < per_class}
    if missing:
        raise ValueError(f"Not enough samples for requested per_class={per_class}: {missing}")
    return selected, classes


@torch.no_grad()
def encode_attention(backbone, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.forward_dino(images)
    features = backbone.mlp(features)
    _slots, attn, _ = backbone.slot_attention(features)
    return attn.detach().cpu()


def make_heatmaps(attn: torch.Tensor, size: int) -> torch.Tensor:
    b, k, n = attn.shape
    side = int(math.sqrt(n))
    if side * side != n:
        raise ValueError(f"Attention token count is not square: {n}")
    maps = attn.reshape(b * k, 1, side, side).float().clamp_min(0)
    maps = F.interpolate(maps, size=(size, size), mode="bilinear", align_corners=False)[:, 0]
    maps = maps / maps.flatten(1).sum(dim=1).clamp_min(1e-8).view(b * k, 1, 1)
    return maps.reshape(b, k, size, size)


def binary_slot_masks(heatmaps: torch.Tensor, threshold_rel: float) -> torch.Tensor:
    peaks = heatmaps.flatten(2).amax(dim=2).view(heatmaps.size(0), heatmaps.size(1), 1, 1)
    return heatmaps >= peaks.clamp_min(1e-12) * float(threshold_rel)


def mask_geometry(masks: torch.Tensor) -> torch.Tensor:
    b, k, h, w = masks.shape
    out = masks.new_zeros((b, k, 5), dtype=torch.float32)
    for bi in range(b):
        for ki in range(k):
            ys, xs = masks[bi, ki].nonzero(as_tuple=True)
            if xs.numel() == 0:
                out[bi, ki] = torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0])
                continue
            x1, x2 = xs.float().min(), xs.float().max()
            y1, y2 = ys.float().min(), ys.float().max()
            bw = (x2 - x1 + 1.0) / float(w)
            bh = (y2 - y1 + 1.0) / float(h)
            cx = (x1 + x2 + 1.0) / (2.0 * w)
            cy = (y1 + y2 + 1.0) / (2.0 * h)
            area = masks[bi, ki].float().mean()
            out[bi, ki] = torch.stack([cx, cy, bw, bh, area])
    return out


def normalized_entropy(heatmaps: torch.Tensor) -> torch.Tensor:
    flat = heatmaps.flatten(2).clamp_min(1e-12)
    flat = flat / flat.sum(dim=2, keepdim=True).clamp_min(1e-12)
    return -(flat * flat.log()).sum(dim=2) / math.log(float(flat.size(2)))


def duplicate_overlap(masks: torch.Tensor) -> torch.Tensor:
    b, k, _h, _w = masks.shape
    flat = masks.flatten(2)
    inter = (flat[:, :, None] & flat[:, None, :]).sum(dim=-1).float()
    area = flat.sum(dim=-1).float().clamp_min(1.0)
    overlap = inter / area[:, :, None]
    eye = torch.eye(k, dtype=torch.bool).unsqueeze(0)
    return overlap.masked_fill(eye, 0.0).amax(dim=2)


def soft_range_score(x: torch.Tensor, lo: float, hi: float, softness: float) -> torch.Tensor:
    left = torch.sigmoid((x - lo) / max(softness, 1e-6))
    right = torch.sigmoid((hi - x) / max(softness, 1e-6))
    return left * right


def attention_quality(heatmaps: torch.Tensor, args: argparse.Namespace) -> dict[str, torch.Tensor]:
    masks = binary_slot_masks(heatmaps, args.threshold_rel)
    geo = mask_geometry(masks)
    area = geo[..., 4]
    enclosing_area = (geo[..., 2] * geo[..., 3]).clamp_min(1e-6)
    compactness = (area / enclosing_area).clamp(0.0, 1.0)
    entropy = normalized_entropy(heatmaps)
    dup = duplicate_overlap(masks)
    peak = heatmaps.flatten(2).amax(dim=2)
    mean = heatmaps.flatten(2).mean(dim=2).clamp_min(1e-8)
    peakiness = ((peak / mean) / args.evidence_peakiness_norm).clamp(0.0, 1.0)
    area_score = soft_range_score(area, args.evidence_min_area, args.evidence_max_area, args.evidence_area_softness)
    entropy_score = torch.sigmoid((args.evidence_max_entropy - entropy) / args.evidence_entropy_softness)
    compact_score = torch.sigmoid((compactness - args.evidence_min_compactness) / args.evidence_compactness_softness)
    non_redundancy = torch.sigmoid((args.evidence_max_duplicate - dup) / args.evidence_duplicate_softness)
    foreground_score = torch.sigmoid((peakiness - args.evidence_min_peakiness) / args.evidence_peakiness_softness)
    coverage_score = (area_score * non_redundancy).clamp(0.0, 1.0)
    raw_evidence = (
        foreground_score
        * area_score
        * entropy_score
        * compact_score
        * coverage_score
    ).clamp(0.0, 1.0)
    return {
        "masks": masks,
        "raw_evidence": raw_evidence,
        "area": area,
        "compactness": compactness,
        "entropy": entropy,
        "duplicate_overlap": dup,
        "peakiness": peakiness,
    }


def denorm(image: torch.Tensor) -> torch.Tensor:
    return (image.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0.0, 1.0)


def draw_one(
    image: torch.Tensor,
    heatmaps: torch.Tensor,
    masks: torch.Tensor,
    scores: torch.Tensor,
    top_idx: torch.Tensor,
    class_name: str,
    sample_idx: int,
    path: Path,
) -> None:
    rgb = denorm(image).permute(1, 2, 0)
    top_masks = masks[top_idx]
    union = top_masks.any(dim=0).float()
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.2), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title(f"{class_name}\nidx={sample_idx}", fontsize=8)
    axes[0, 1].imshow(rgb)
    axes[0, 1].imshow(union, cmap="spring", alpha=0.42)
    axes[0, 1].set_title("top-3 union", fontsize=8)
    for col, slot_id in enumerate(top_idx.tolist(), start=2):
        axes[0, col].imshow(rgb)
        axes[0, col].imshow(heatmaps[slot_id], cmap="magma", alpha=0.52)
        axes[0, col].contour(masks[slot_id].float(), levels=[0.5], colors=["cyan"], linewidths=1.0)
        axes[0, col].set_title(f"slot {slot_id + 1}\nscore={scores[slot_id]:.3f}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_index(out_dir: Path, rows: list[dict[str, Any]], classes: list[str]) -> None:
    csv_path = out_dir / "index.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Attention Object Regions</title>",
        "<style>body{font-family:sans-serif;margin:20px} img{max-width:100%;border:1px solid #ddd}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:18px}"
        "h2{margin-top:32px}.card{break-inside:avoid}.meta{font-size:12px;color:#444}</style>",
        "</head><body>",
        "<h1>Attention Mask Object-Like Regions</h1>",
        "<p>Top-3 slots are ranked by attention-only evidence quality. No bbox is used.</p>",
    ]
    for class_name in classes:
        class_rows = [row for row in rows if row["class"] == class_name]
        parts.append(f"<h2>{html.escape(class_name)}</h2><div class='grid'>")
        for row in class_rows:
            rel = html.escape(row["image"])
            meta = html.escape(f"sample={row['sample_idx']} top_slots={row['top_slots']} scores={row['top_scores']}")
            parts.append(f"<div class='card'><img src='{rel}'><div class='meta'>{meta}</div></div>")
        parts.append("</div>")
    parts.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default=str(REPO_ROOT / "dataset/coco_top2_clean10_area006_004_600_200_200/classification_dataset"))
    p.add_argument("--sa_checkpoint", default=str(REPO_ROOT / "checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt"))
    p.add_argument("--split", default="test", choices=["train", "valid", "val", "test"])
    p.add_argument("--out_dir", default=str(REPO_ROOT / "analysis/attention_object_regions_10_per_class"))
    p.add_argument("--input_res", type=int, default=224)
    p.add_argument("--per_class", type=int, default=10)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--topk", type=int, default=3)
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
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    tfm = build_transforms(args.input_res)["valid"]
    base = build_dataset(args.data, split, tfm)
    selected_indices, classes = select_per_class(base, args.per_class)
    dataset = IndexedDataset(Subset(base, selected_indices))
    dataset.classes = classes
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
    for images, labels, subset_indices in tqdm(loader, desc="visualize-attention", mininterval=1.0):
        original_indices = [selected_indices[int(i)] for i in subset_indices.tolist()]
        attn = encode_attention(backbone, images, device)
        heatmaps = make_heatmaps(attn, args.input_res)
        quality = attention_quality(heatmaps, args)
        scores = quality["raw_evidence"]
        masks = quality["masks"]
        for bi in range(images.size(0)):
            label = int(labels[bi])
            class_name = classes[label]
            topk = min(args.topk, scores.size(1))
            top_idx = scores[bi].argsort(descending=True)[:topk]
            stem = f"{class_name}_{original_indices[bi]:06d}"
            rel_path = Path("images") / f"{stem}.png"
            draw_one(
                images[bi],
                heatmaps[bi],
                masks[bi],
                scores[bi],
                top_idx,
                class_name,
                original_indices[bi],
                out_dir / rel_path,
            )
            rows.append(
                {
                    "class": class_name,
                    "sample_idx": original_indices[bi],
                    "image": str(rel_path),
                    "top_slots": ",".join(str(int(i) + 1) for i in top_idx.tolist()),
                    "top_scores": ",".join(f"{float(scores[bi, i]):.4f}" for i in top_idx.tolist()),
                    "top_areas": ",".join(f"{float(quality['area'][bi, i]):.4f}" for i in top_idx.tolist()),
                    "top_peakiness": ",".join(f"{float(quality['peakiness'][bi, i]):.4f}" for i in top_idx.tolist()),
                }
            )
    write_index(out_dir, rows, classes)
    summary = {
        "split": split,
        "per_class": args.per_class,
        "total_images": len(rows),
        "topk": args.topk,
        "threshold_rel": args.threshold_rel,
        "uses_bbox": False,
        "source": "raw DINOSAUR slot attention masks",
        "html": str(out_dir / "index.html"),
        "csv": str(out_dir / "index.csv"),
    }
    (out_dir / "summary.json").write_text(__import__("json").dumps(summary, indent=2), encoding="utf-8")
    print(__import__("json").dumps(summary, indent=2))


if __name__ == "__main__":
    main()
