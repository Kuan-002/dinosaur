#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".cache" / "torch"))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from misc_utils import seed_all
from settransformer.model import DiscriminativeSetTransformer, ProbeConfig, true_class_margin
from settransformer.train import DEFAULT_DATA, DEFAULT_SA, all_single_masks, encode_slots, fixed_size_mask
from train_slot_classifier import build_dataset, build_transforms, load_backbone, subset_dataset


def load_probe(path: str, device: torch.device) -> DiscriminativeSetTransformer:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_raw = ckpt["probe_config"]
    allowed = {field.name for field in fields(ProbeConfig)}
    cfg = ProbeConfig(**{key: value for key, value in cfg_raw.items() if key in allowed})
    model = DiscriminativeSetTransformer(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


@torch.no_grad()
def diagnose(args: argparse.Namespace) -> None:
    seed_all(args.seed, False)
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    tfm = build_transforms(args.input_res)
    dataset = subset_dataset(build_dataset(args.data, args.split, tfm["valid"]), args.max_items, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    backbone = load_backbone(args.sa_checkpoint, device)
    probe = load_probe(args.probe_checkpoint, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_rows: list[dict] = []
    subset_totals: dict[int, list[float]] = {size: [] for size in args.subset_sizes}
    marginal_spreads: list[float] = []
    consistency_errors: list[float] = []

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="diagnose", mininterval=1.0)):
        labels = labels.to(device, non_blocking=device.type == "cuda")
        slots = encode_slots(backbone, images, device)
        b, k, d = slots.shape

        single_masks = all_single_masks(b, k, device)
        flat_slots = slots[:, None].expand(-1, k, -1, -1).reshape(b * k, k, d)
        flat_masks = single_masks.reshape(b * k, k)
        flat_labels = labels[:, None].expand(-1, k).reshape(b * k)
        single_margins = true_class_margin(probe(flat_slots, flat_masks), flat_labels).reshape(b, k)
        order = single_margins.argsort(dim=1, descending=True)

        for size in args.subset_sizes:
            mask = fixed_size_mask(b, k, size, device)
            subset_margin = true_class_margin(probe(slots, mask), labels)
            subset_totals[size].extend(float(v) for v in subset_margin.cpu())

        base_mask = torch.zeros(b, k, dtype=torch.bool, device=device)
        empty_margin = true_class_margin(probe(slots, base_mask), labels)
        first_gains = single_margins - empty_margin[:, None]
        marginal_spreads.extend(float(v) for v in (first_gains.max(dim=1).values - first_gains.min(dim=1).values).cpu())

        two_mask = fixed_size_mask(b, k, min(2, k), device)
        logits_a = probe(slots, two_mask)
        perm = torch.rand(b, k, device=device).argsort(dim=1)
        logits_b = probe(slots.gather(1, perm.unsqueeze(-1).expand(-1, -1, d)), two_mask.gather(1, perm))
        consistency_errors.extend(
            float(v)
            for v in (true_class_margin(logits_a, labels) - true_class_margin(logits_b, labels)).abs().cpu()
        )

        for i in range(b):
            row = {
                "global_index": batch_idx * args.bs + i,
                "label": int(labels[i].item()),
                "single_margin_range": float((single_margins[i].max() - single_margins[i].min()).cpu()),
                "first_gain_range": float((first_gains[i].max() - first_gains[i].min()).cpu()),
            }
            for rank in range(min(args.top_k, k)):
                slot_idx = int(order[i, rank].item())
                row[f"top{rank + 1}_slot"] = slot_idx
                row[f"top{rank + 1}_single_margin"] = float(single_margins[i, slot_idx].cpu())
                row[f"top{rank + 1}_first_gain"] = float(first_gains[i, slot_idx].cpu())
            image_rows.append(row)

    with (out_dir / "image_slot_ranking.csv").open("w", newline="") as f:
        keys = sorted({key for row in image_rows for key in row})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(image_rows)

    subset_curve = {
        str(size): {
            "mean_margin": sum(values) / max(len(values), 1),
            "count": len(values),
        }
        for size, values in subset_totals.items()
    }
    summary = {
        "num_images": len(image_rows),
        "subset_curve": subset_curve,
        "mean_first_step_gain_range": sum(marginal_spreads) / max(len(marginal_spreads), 1),
        "mean_permutation_abs_error": sum(consistency_errors) / max(len(consistency_errors), 1),
        "outputs": {
            "image_slot_ranking": str(out_dir / "image_slot_ranking.csv"),
            "summary": str(out_dir / "summary.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_ints(raw: str) -> list[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose discriminative Set Transformer slot ranking behavior.")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--split", default="valid", choices=["train", "valid", "val", "test"])
    parser.add_argument("--sa_checkpoint", default=DEFAULT_SA)
    parser.add_argument("--probe_checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--input_res", type=int, default=224)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--subset_sizes", type=parse_ints, default=parse_ints("1,2,3,4,8"))
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    diagnose(parse_args())


if __name__ == "__main__":
    main()
