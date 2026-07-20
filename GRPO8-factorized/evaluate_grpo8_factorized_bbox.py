#!/usr/bin/env python3
"""Evaluate forced top-k GRPO8 rankings with standard bbox hit metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from scripts.evaluate_set_bbox import (
    boxes_to_mask,
    hit_stats,
    load_coco_boxes,
    load_metadata,
    make_heatmaps,
    metric_summary,
    sample_relative_path,
    transform_boxes_to_input,
    unwrap_dataset,
)
from selector_grpo import GRPOSelectorConfig, SlotSelectorGRPO
from train_slot_classifier import build_dataset, build_transforms, load_backbone


def comma_ints(value: str) -> list[int]:
    return sorted({int(item) for item in value.split(",") if item.strip()})


def comma_floats(value: str) -> list[float]:
    return sorted({float(item) for item in value.split(",") if item.strip()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="selector_grpo_best.pt")
    parser.add_argument("--data", default="")
    parser.add_argument("--sa_checkpoint", default="")
    parser.add_argument("--coco_root", default=str(ROOT / "dataset/coco2017"))
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--top_ks", type=comma_ints, default=comma_ints("3,4"))
    parser.add_argument("--hit_thresholds", type=comma_floats, default=comma_floats("0.2,0.4"))
    parser.add_argument("--threshold_rel", type=float, default=0.5)
    parser.add_argument("--input_res", type=int, default=224)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def dataset_indices(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        return [int(dataset.indices[index]) for index in range(len(dataset))]
    return list(range(len(dataset)))


@torch.no_grad()
def encode(backbone, images: torch.Tensor, device: torch.device):
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.mlp(backbone.forward_dino(images))
    slots, attention, _ = backbone.slot_attention(features)
    return slots.detach(), attention.detach()


@torch.no_grad()
def forced_ranking(model: SlotSelectorGRPO, slots: torch.Tensor, max_k: int):
    embeds = model.embed_slots(slots, None)
    batch, num_slots, _ = embeds.shape
    hidden = model.initial_state(embeds)
    selected = torch.zeros(batch, num_slots, dtype=torch.bool, device=slots.device)
    active = torch.ones(batch, dtype=torch.bool, device=slots.device)
    actions, logits_by_k = [], {}
    for step in range(max_k):
        action = model.slot_policy_logits(hidden, embeds, selected, step).argmax(dim=1)
        hidden, selected = model.update_with_action(hidden, selected, embeds, action, active)
        actions.append(action)
        logits_by_k[step + 1] = model.classify(hidden)
    return torch.stack(actions, dim=1), logits_by_k


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda:0" if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    run_dir = Path(args.run_dir)
    checkpoint_path = run_dir / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args") or {}
    data = args.data or checkpoint_args.get("data")
    sa_checkpoint = args.sa_checkpoint or checkpoint_args.get("sa_checkpoint")
    if not data or not sa_checkpoint:
        raise ValueError("data and sa_checkpoint are required")

    dataset = build_dataset(data, args.split, build_transforms(args.input_res)["valid"])
    base_dataset = unwrap_dataset(dataset)
    base_indices = dataset_indices(dataset)
    metadata = load_metadata(data)
    needed = set()
    for sample_idx in base_indices:
        row = metadata[sample_relative_path(base_dataset, sample_idx, args.split)]
        needed.add((row["source_split"], int(row["image_id"]), row["anchor_object"]))
        needed.add((row["source_split"], int(row["image_id"]), row["evidence_object"]))
    coco_boxes = load_coco_boxes(Path(args.coco_root), needed)

    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    backbone = load_backbone(sa_checkpoint, device).eval()
    backbone.requires_grad_(False)
    model = SlotSelectorGRPO(GRPOSelectorConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []
    cursor = 0
    for images, labels in tqdm(loader, desc="grpo8-factorized-bbox", mininterval=1.0):
        labels = labels.to(device)
        slots, attention = encode(backbone, images, device)
        ranking, logits_by_k = forced_ranking(model, slots, max(args.top_ks))
        ranking = ranking.cpu()
        predictions = {k: logits_by_k[k].argmax(1).cpu() for k in args.top_ks}
        for batch_idx in range(labels.numel()):
            sample_idx = base_indices[cursor + batch_idx]
            relative_path = sample_relative_path(base_dataset, sample_idx, args.split)
            meta = metadata[relative_path]
            image_id = int(meta["image_id"])
            source_split = meta["source_split"]
            width, height = int(meta["width"]), int(meta["height"])
            anchor_boxes = transform_boxes_to_input(
                coco_boxes.get((source_split, image_id, meta["anchor_object"]), []),
                width, height, args.input_res,
            )
            evidence_boxes = transform_boxes_to_input(
                coco_boxes.get((source_split, image_id, meta["evidence_object"]), []),
                width, height, args.input_res,
            )
            anchor_mask = boxes_to_mask(anchor_boxes, args.input_res)
            evidence_mask = boxes_to_mask(evidence_boxes, args.input_res)
            heatmaps = make_heatmaps(attention[batch_idx].cpu(), args.input_res)
            true_id = int(labels[batch_idx].item())
            row = {
                "dataset_index": cursor + batch_idx,
                "relative_path": relative_path,
                "true": true_id,
                "true_name": meta["class_name"],
                "anchor_object": meta["anchor_object"],
                "evidence_object": meta["evidence_object"],
                "ranked_slots_1based": json.dumps((ranking[batch_idx] + 1).tolist()),
            }
            for top_k in args.top_ks:
                chosen = ranking[batch_idx, :top_k].tolist()
                row[f"accuracy@{top_k}"] = float(int(predictions[top_k][batch_idx]) == true_id)
                for threshold in args.hit_thresholds:
                    anchor_hit, _ = hit_stats(heatmaps, chosen, anchor_mask, threshold, args.threshold_rel)
                    evidence_hit, _ = hit_stats(heatmaps, chosen, evidence_mask, threshold, args.threshold_rel)
                    suffix = f"@{top_k}/thr{threshold:g}"
                    row[f"anchor{suffix}"] = float(anchor_hit)
                    row[f"evidence{suffix}"] = float(evidence_hit)
                    row[f"pair{suffix}"] = float(anchor_hit and evidence_hit)
            rows.append(row)
        cursor += labels.numel()

    metric_keys = [f"accuracy@{k}" for k in args.top_ks]
    for threshold in args.hit_thresholds:
        for top_k in args.top_ks:
            metric_keys += [f"{name}@{top_k}/thr{threshold:g}" for name in ("anchor", "evidence", "pair")]
    rows_by_class = defaultdict(list)
    for row in rows:
        rows_by_class[f"{row['true']}:{row['true_name']}"].append(row)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "bbox_eval_forced_top3_top4"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_valid_pair_score": checkpoint.get("valid_pair_score"),
        "split": args.split,
        "items": len(rows),
        "top_ks": args.top_ks,
        "hit_thresholds": args.hit_thresholds,
        "threshold_rel": args.threshold_rel,
        "metrics": metric_summary(rows, metric_keys),
        "per_class": {
            key: {"items": len(value), "metrics": metric_summary(value, metric_keys)}
            for key, value in sorted(rows_by_class.items(), key=lambda item: int(item[0].split(":", 1)[0]))
        },
    }
    write_csv(out_dir / "rows.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
