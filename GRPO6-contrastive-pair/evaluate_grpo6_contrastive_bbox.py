#!/usr/bin/env python3
"""Evaluate forced top-k GRPO6 rankings with standard bbox hit metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "GRPO6-contrastive-pair"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

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
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--top_ks", type=comma_ints, default=comma_ints("2,3,4"))
    parser.add_argument("--hit_thresholds", type=comma_floats, default=comma_floats("0.2,0.4"))
    parser.add_argument("--threshold_rel", type=float, default=0.5)
    parser.add_argument("--input_res", type=int, default=224)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def choose_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    if max_k > num_slots:
        raise ValueError(f"requested top-{max_k}, but model only has {num_slots} slots")
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


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    labels = labels.float()
    pos = labels.sum()
    neg = labels.numel() - pos
    if pos <= 0 or neg <= 0:
        return None
    sorted_scores, order = scores.sort()
    sorted_ranks = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    _unique_scores, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    rank_sums = torch.split(sorted_ranks, counts.tolist())
    average_ranks = torch.cat(
        [
            ranks.new_full((count,), float(ranks.mean()))
            for ranks, count in zip(rank_sums, counts.tolist())
        ]
    )
    ranks = torch.empty_like(scores, dtype=torch.float64)
    ranks[order] = average_ranks
    pos_rank_sum = ranks[labels.bool()].sum()
    auc = (pos_rank_sum - pos.double() * (pos.double() + 1.0) / 2.0) / (
        pos.double() * neg.double()
    )
    return float(auc)


def multiclass_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = logits.softmax(dim=-1).cpu()
    labels = labels.cpu()
    num_classes = probs.size(1)
    pred = probs.argmax(dim=-1)
    tp = torch.zeros(num_classes, dtype=torch.float64)
    fp = torch.zeros(num_classes, dtype=torch.float64)
    fn = torch.zeros(num_classes, dtype=torch.float64)
    for cls in range(num_classes):
        pred_cls = pred == cls
        true_cls = labels == cls
        tp[cls] = (pred_cls & true_cls).sum()
        fp[cls] = (pred_cls & ~true_cls).sum()
        fn[cls] = (~pred_cls & true_cls).sum()
    precision = tp / (tp + fp).clamp_min(1e-8)
    recall = tp / (tp + fn).clamp_min(1e-8)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1e-8)
    one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()
    aucs = [binary_auc(probs[:, cls], one_hot[:, cls]) for cls in range(num_classes)]
    valid_aucs = [auc for auc in aucs if auc is not None]
    micro_auc = binary_auc(probs.flatten(), one_hot.flatten())
    macro_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else 0.0
    accuracy = pred.eq(labels).float().mean().item()
    return {
        "acc": float(accuracy),
        "accuracy": float(accuracy),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
        "auc": macro_auc,
        "macro_auc": macro_auc,
        "micro_auc": micro_auc if micro_auc is not None else 0.0,
    }


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    run_dir = Path(args.run_dir)
    checkpoint_path = run_dir / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args") or {}
    data = args.data or checkpoint_args.get("data")
    sa_checkpoint = args.sa_checkpoint or checkpoint_args.get("sa_checkpoint")
    if not data or not sa_checkpoint:
        raise ValueError("data and sa_checkpoint are required")

    dataset = build_dataset(data, args.split, build_transforms(args.input_res)["valid"])
    if args.max_items and args.max_items < len(dataset):
        dataset = Subset(dataset, list(range(args.max_items)))
    base_dataset = unwrap_dataset(dataset)
    base_indices = dataset_indices(dataset)
    metadata = load_metadata(data)
    rules = checkpoint["rules"]
    needed = set()
    for sample_idx in base_indices:
        row = metadata[sample_relative_path(base_dataset, sample_idx, args.split)]
        class_name = row["class_name"]
        needed.add((row["source_split"], int(row["image_id"]), rules[class_name]["object_a"]))
        needed.add((row["source_split"], int(row["image_id"]), rules[class_name]["object_b"]))
    coco_boxes = load_coco_boxes(Path(args.coco_root), needed)

    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
    backbone = load_backbone(sa_checkpoint, device).eval()
    backbone.requires_grad_(False)
    model = SlotSelectorGRPO(GRPOSelectorConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    rows = []
    logits_by_top_k = {top_k: [] for top_k in args.top_ks}
    labels_all = []
    cursor = 0
    for images, labels in tqdm(loader, desc="grpo6-contrastive-bbox", mininterval=1.0):
        labels = labels.to(device)
        slots, attention = encode(backbone, images, device)
        ranking, logits_by_k = forced_ranking(model, slots, max(args.top_ks))
        ranking = ranking.cpu()
        predictions = {k: logits_by_k[k].argmax(1).cpu() for k in args.top_ks}
        for top_k in args.top_ks:
            logits_by_top_k[top_k].append(logits_by_k[top_k].detach().cpu())
        labels_all.append(labels.detach().cpu())
        for batch_idx in range(labels.numel()):
            sample_idx = base_indices[cursor + batch_idx]
            relative_path = sample_relative_path(base_dataset, sample_idx, args.split)
            meta = metadata[relative_path]
            image_id = int(meta["image_id"])
            source_split = meta["source_split"]
            width, height = int(meta["width"]), int(meta["height"])
            true_id = int(labels[batch_idx].item())
            class_name = meta["class_name"]
            object_a_name = rules[class_name]["object_a"]
            object_b_name = rules[class_name]["object_b"]
            object_a_boxes = transform_boxes_to_input(
                coco_boxes.get((source_split, image_id, object_a_name), []),
                width,
                height,
                args.input_res,
            )
            object_b_boxes = transform_boxes_to_input(
                coco_boxes.get((source_split, image_id, object_b_name), []),
                width,
                height,
                args.input_res,
            )
            object_a_mask = boxes_to_mask(object_a_boxes, args.input_res)
            object_b_mask = boxes_to_mask(object_b_boxes, args.input_res)
            heatmaps = make_heatmaps(attention[batch_idx].cpu(), args.input_res)
            row = {
                "dataset_index": cursor + batch_idx,
                "relative_path": relative_path,
                "true": true_id,
                "true_name": class_name,
                "image_id": image_id,
                "source_split": source_split,
                "object_a": object_a_name,
                "object_b": object_b_name,
                "ranked_slots_1based": json.dumps((ranking[batch_idx] + 1).tolist()),
            }
            for top_k in args.top_ks:
                chosen = ranking[batch_idx, :top_k].tolist()
                row[f"avg_selected@{top_k}"] = float(top_k)
                row[f"accuracy@{top_k}"] = float(int(predictions[top_k][batch_idx]) == true_id)
                row[f"top{top_k}_slots_1based"] = json.dumps([slot + 1 for slot in chosen])
                for threshold in args.hit_thresholds:
                    suffix = f"@{top_k}/thr{threshold:g}"
                    object_a_hit, object_a_mass = hit_stats(heatmaps, chosen, object_a_mask, threshold, args.threshold_rel)
                    object_b_hit, object_b_mass = hit_stats(heatmaps, chosen, object_b_mask, threshold, args.threshold_rel)
                    row[f"object_a{suffix}"] = float(object_a_hit)
                    row[f"object_b{suffix}"] = float(object_b_hit)
                    row[f"pair{suffix}"] = float(object_a_hit and object_b_hit)
                    row[f"object_a_mass{suffix}"] = object_a_mass
                    row[f"object_b_mass{suffix}"] = object_b_mass
            rows.append(row)
        cursor += labels.numel()

    metric_keys = [key for k in args.top_ks for key in (f"accuracy@{k}", f"avg_selected@{k}")]
    for threshold in args.hit_thresholds:
        for top_k in args.top_ks:
            metric_keys += [f"{name}@{top_k}/thr{threshold:g}" for name in ("object_a", "object_b", "pair")]
    rows_by_class = defaultdict(list)
    for row in rows:
        rows_by_class[f"{row['true']}:{row['true_name']}"].append(row)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "bbox_eval_forced_top2_top3_top4"
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = metric_summary(rows, metric_keys)
    labels_cat = torch.cat(labels_all, dim=0) if labels_all else torch.empty(0, dtype=torch.long)
    class_metrics = {
        f"top_{top_k}": multiclass_metrics(torch.cat(logits_by_top_k[top_k], dim=0), labels_cat)
        for top_k in args.top_ks
        if logits_by_top_k[top_k]
    }
    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_valid_pair_margin": checkpoint.get("valid_pair_margin"),
        "split": args.split,
        "items": len(rows),
        "top_ks": args.top_ks,
        "hit_thresholds": args.hit_thresholds,
        "threshold_rel": args.threshold_rel,
        "bbox_is_evaluation_only": True,
        "classification": {
            f"acc@{top_k}": flat[f"accuracy@{top_k}"]
            for top_k in args.top_ks
        } | class_metrics,
        "thresholds": {
            f"{threshold:g}": {
                f"{metric}@{top_k}": flat[f"{metric}@{top_k}/thr{threshold:g}"]
                for top_k in args.top_ks
                for metric in ("object_a", "object_b", "pair")
            }
            for threshold in args.hit_thresholds
        },
        "overall_flat": flat,
        "per_class": {
            key: {"items": len(value), "metrics": metric_summary(value, metric_keys)}
            for key, value in sorted(rows_by_class.items(), key=lambda item: int(item[0].split(":", 1)[0]))
        },
    }
    write_csv(out_dir / "bbox_eval.csv", rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
