#!/usr/bin/env python3
"""Evaluate forced @3/@4 GRPO6 slot rankings against COCO object boxes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".cache" / "torch"))

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from selector_grpo import GRPOSelectorConfig, SlotSelectorGRPO
from train_slot_classifier import build_dataset, build_transforms, load_backbone
from scripts.evaluate_set_bbox import (
    boxes_to_mask,
    hit_stats,
    load_coco_boxes,
    load_metadata,
    make_heatmaps,
    metric_summary,
    transform_boxes_to_input,
    unwrap_dataset,
)


DEFAULT_RUN = REPO_ROOT / "GRPO6-factorized/checkpoints/grpo6_factorized_m3_p085_n090_seed8_20260715"
DEFAULT_DATA = REPO_ROOT / "dataset/coco_compositional_pair6_clean_300_100_100/classification_dataset"
DEFAULT_COCO = REPO_ROOT / "dataset/coco2017"


def parse_float_list(raw: str) -> list[float]:
    values = sorted({float(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("thresholds must be comma-separated values in [0, 1]")
    return values


def parse_int_list(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("top-k values must be positive integers")
    return values


def choose_device(raw: str) -> torch.device:
    if raw != "auto":
        return torch.device(raw)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dataset_indices(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        return [int(dataset.indices[index]) for index in range(len(dataset))]
    return list(range(len(dataset)))


def relative_path(base_dataset: Dataset, sample_idx: int, split: str) -> str:
    sample_path = Path(base_dataset.samples[sample_idx][0])
    root = Path(getattr(base_dataset, "root"))
    rel = sample_path.relative_to(root).as_posix()
    return rel if rel.startswith(f"{split}/") else f"{split}/{rel}"


def load_selector(checkpoint_path: Path, device: torch.device) -> tuple[SlotSelectorGRPO, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = GRPOSelectorConfig(**checkpoint["config"])
    model = SlotSelectorGRPO(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def encode_raw_slots(backbone, images: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.forward_dino(images)
    features = backbone.mlp(features)
    slots, attn, _ = backbone.slot_attention(features)
    return slots.detach(), attn.detach()


@torch.no_grad()
def force_rank_slots(
    model: SlotSelectorGRPO,
    slots: torch.Tensor,
    max_k: int,
) -> tuple[list[list[int]], dict[int, torch.Tensor]]:
    """Greedily select exactly max_k slots, ignoring confidence early exit."""
    slot_embeds = model.embed_slots(slots, None)
    batch_size, num_slots, _ = slot_embeds.shape
    if max_k > num_slots:
        raise ValueError(f"requested top-{max_k}, but model only has {num_slots} slots")
    h = model.initial_state(slot_embeds)
    selected = torch.zeros(batch_size, num_slots, dtype=torch.bool, device=slots.device)
    actions: list[torch.Tensor] = []
    logits_by_k: dict[int, torch.Tensor] = {}
    active = torch.ones(batch_size, dtype=torch.bool, device=slots.device)
    for step in range(max_k):
        policy_logits = model.slot_policy_logits(h, slot_embeds, selected, step)
        action = policy_logits.argmax(dim=-1)
        h, selected = model.update_with_action(h, selected, slot_embeds, action, active)
        actions.append(action)
        logits_by_k[step + 1] = model.classify(h).detach()
    action_tensor = torch.stack(actions, dim=1).detach().cpu()
    return [[int(slot) for slot in row] for row in action_tensor.tolist()], logits_by_k


def needed_box_keys(metadata: dict[str, dict[str, str]], dataset: Dataset, split: str) -> set[tuple[str, int, str]]:
    base = unwrap_dataset(dataset)
    keys: set[tuple[str, int, str]] = set()
    for sample_idx in dataset_indices(dataset):
        row = metadata[relative_path(base, sample_idx, split)]
        source = row["source_split"]
        image_id = int(row["image_id"])
        keys.add((source, image_id, row["anchor_object"]))
        keys.add((source, image_id, row["evidence_object"]))
    return keys


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def threshold_key(value: float) -> str:
    return f"{value:g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN))
    parser.add_argument("--checkpoint", default="selector_grpo_best.pt")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--coco_root", default=str(DEFAULT_COCO))
    parser.add_argument("--sa_checkpoint", default="")
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--input_res", type=int, default=224)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--top_ks", type=parse_int_list, default=parse_int_list("3,4"))
    parser.add_argument("--hit_thresholds", type=parse_float_list, default=parse_float_list("0.2,0.4"))
    parser.add_argument("--threshold_rel", type=float, default=0.5)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    run_dir = Path(args.run_dir)
    model, checkpoint = load_selector(run_dir / args.checkpoint, device)
    checkpoint_args = checkpoint.get("args") or {}
    sa_checkpoint = args.sa_checkpoint or checkpoint_args.get("sa_checkpoint")
    if not sa_checkpoint:
        raise ValueError("SA checkpoint is missing")
    max_k = max(args.top_ks)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "bbox_eval_forced_top4"
    out_dir.mkdir(parents=True, exist_ok=True)

    transforms = build_transforms(args.input_res)
    dataset = build_dataset(args.data, args.split, transforms["valid"])
    if args.max_items and args.max_items < len(dataset):
        dataset = Subset(dataset, list(range(args.max_items)))
    base_dataset = unwrap_dataset(dataset)
    base_indices = dataset_indices(dataset)
    metadata = load_metadata(args.data)
    coco_boxes = load_coco_boxes(Path(args.coco_root), needed_box_keys(metadata, dataset, args.split))
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

    rows: list[dict] = []
    cursor = 0
    for images, labels in tqdm(loader, desc="grpo6-factorized-bbox", mininterval=1.0):
        labels_device = labels.to(device, non_blocking=device.type == "cuda")
        slots, attn = encode_raw_slots(backbone, images, device)
        ranked_slots, logits_by_k = force_rank_slots(model, slots, max_k)
        predictions = {top_k: logits_by_k[top_k].argmax(dim=1).cpu() for top_k in args.top_ks}
        for batch_index in range(labels.numel()):
            sample_idx = base_indices[cursor + batch_index]
            rel = relative_path(base_dataset, sample_idx, args.split)
            meta = metadata[rel]
            source = meta["source_split"]
            image_id = int(meta["image_id"])
            width, height = int(meta["width"]), int(meta["height"])
            anchor_name, evidence_name = meta["anchor_object"], meta["evidence_object"]
            anchor_boxes = transform_boxes_to_input(
                coco_boxes.get((source, image_id, anchor_name), []), width, height, args.input_res
            )
            evidence_boxes = transform_boxes_to_input(
                coco_boxes.get((source, image_id, evidence_name), []), width, height, args.input_res
            )
            anchor_mask = boxes_to_mask(anchor_boxes, args.input_res)
            evidence_mask = boxes_to_mask(evidence_boxes, args.input_res)
            heatmaps = make_heatmaps(attn[batch_index].cpu(), args.input_res)
            row = {
                "dataset_index": cursor + batch_index,
                "relative_path": rel,
                "true": int(labels[batch_index]),
                "true_name": meta["class_name"],
                "image_id": image_id,
                "source_split": source,
                "object_a": meta["object_a"],
                "object_b": meta["object_b"],
                "anchor_object": anchor_name,
                "evidence_object": evidence_name,
                "ranked_slots_1based": json.dumps([slot + 1 for slot in ranked_slots[batch_index]]),
            }
            for top_k in args.top_ks:
                selected = ranked_slots[batch_index][:top_k]
                row[f"selector_correct@{top_k}"] = float(predictions[top_k][batch_index] == labels[batch_index])
                for threshold in args.hit_thresholds:
                    key = threshold_key(threshold)
                    anchor_hit, anchor_mass = hit_stats(
                        heatmaps, selected, anchor_mask, threshold, args.threshold_rel
                    )
                    evidence_hit, evidence_mass = hit_stats(
                        heatmaps, selected, evidence_mask, threshold, args.threshold_rel
                    )
                    row[f"anchor@{top_k}_thr{key}"] = float(anchor_hit)
                    row[f"evidence@{top_k}_thr{key}"] = float(evidence_hit)
                    row[f"pair@{top_k}_thr{key}"] = float(anchor_hit and evidence_hit)
                    row[f"anchor_mass@{top_k}"] = anchor_mass
                    row[f"evidence_mass@{top_k}"] = evidence_mass
            rows.append(row)
        cursor += labels.numel()

    write_csv(out_dir / "bbox_eval.csv", rows)
    rows_by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_class[f"{row['true']}:{row['true_name']}"].append(row)

    metric_keys = [f"selector_correct@{top_k}" for top_k in args.top_ks]
    for threshold in args.hit_thresholds:
        key = threshold_key(threshold)
        for top_k in args.top_ks:
            metric_keys.extend(
                [f"anchor@{top_k}_thr{key}", f"evidence@{top_k}_thr{key}", f"pair@{top_k}_thr{key}"]
            )
    overall_flat = metric_summary(rows, metric_keys)
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_valid_acc": checkpoint.get("valid_acc"),
        "data": args.data,
        "split": args.split,
        "items": len(rows),
        "top_ks": args.top_ks,
        "hit_thresholds": args.hit_thresholds,
        "threshold_rel": args.threshold_rel,
        "forced_ranking": True,
        "confidence_early_exit_used": False,
        "bbox_is_evaluation_only": True,
        "classification": {f"acc@{top_k}": overall_flat[f"selector_correct@{top_k}"] for top_k in args.top_ks},
        "thresholds": {
            threshold_key(threshold): {
                f"anchor@{top_k}": overall_flat[f"anchor@{top_k}_thr{threshold_key(threshold)}"]
                for top_k in args.top_ks
            }
            for threshold in args.hit_thresholds
        },
        "overall_flat": overall_flat,
        "per_class": {
            class_key: {"items": len(class_rows), "metrics": metric_summary(class_rows, metric_keys)}
            for class_key, class_rows in sorted(rows_by_class.items(), key=lambda item: int(item[0].split(":", 1)[0]))
        },
        "csv": str(out_dir / "bbox_eval.csv"),
    }
    # Add evidence/pair values next to anchor values in the compact threshold view.
    for threshold in args.hit_thresholds:
        key = threshold_key(threshold)
        for top_k in args.top_ks:
            summary["thresholds"][key][f"evidence@{top_k}"] = overall_flat[f"evidence@{top_k}_thr{key}"]
            summary["thresholds"][key][f"pair@{top_k}"] = overall_flat[f"pair@{top_k}_thr{key}"]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
