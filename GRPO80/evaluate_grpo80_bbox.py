#!/usr/bin/env python3
"""BBox evaluation for GRPO80 selector-chosen slots at @3 and @4."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "SET80") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "SET80"))

os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from SET80.structured80 import load_structured80, project_slots80
from train_slot_classifier import build_dataset, build_transforms, load_backbone
from visualize_grpo_selector_paths import choose_device, load_selector, trace_controls_from_meta

from analysis.set_transformer_diagnostics.evaluate_set_transformer_bbox import (
    boxes_to_mask,
    denorm_image,
    draw_boxes,
    fixed_sample_relative_path,
    load_coco_boxes,
    load_metadata,
    make_heatmaps,
    metric_summary,
    needed_metadata_keys,
    slot_bbox_metrics,
    transform_boxes_to_input,
    unwrap_dataset,
)


def parse_top_ks(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError("--top_ks must contain positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="selector_grpo_best.pt")
    parser.add_argument("--data", default="")
    parser.add_argument("--split", default="test", choices=["valid", "val", "test"])
    parser.add_argument("--coco_root", default="/vol/biomedic3/kw1025/dinosaur/dataset/coco2017")
    parser.add_argument("--sa_checkpoint", default="")
    parser.add_argument("--structured_checkpoint", default="")
    parser.add_argument("--structured_mode", default="")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--input_res", type=int, default=0)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--top_ks", type=parse_top_ks, default=parse_top_ks("3,4"))
    parser.add_argument("--hit_threshold", type=float, default=0.20)
    parser.add_argument("--threshold_rel", type=float, default=0.5)
    parser.add_argument("--contact_sheets", type=int, default=80)
    parser.add_argument("--contact_sheets_per_class", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def dataset_indices(dataset: Dataset) -> list[int]:
    if isinstance(dataset, Subset):
        return [dataset.indices[i] for i in range(len(dataset))]
    return list(range(len(dataset)))


def meta_arg(meta: dict, name: str, default=None):
    return (meta.get("args") or {}).get(name, default)


@torch.no_grad()
def encode_slots80(backbone, projector, images: torch.Tensor, device: torch.device, mode: str):
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.forward_dino(images)
    features = backbone.mlp(features)
    raw_slots, attn, _ = backbone.slot_attention(features)
    slots80 = project_slots80(projector, raw_slots.detach(), mode=mode)
    return slots80, attn.detach()


@torch.no_grad()
def selector_selected_slots(model, slots: torch.Tensor, controls: dict, confidence_early_exit: bool):
    out = model.forward_greedy(
        slots,
        None,
        min_steps=int(controls["min_steps"]),
        early_exit_conf=float(controls["early_exit_conf"]),
        confidence_early_exit=confidence_early_exit,
    )
    selected_by_item: list[list[int]] = []
    for actions in out.actions.detach().cpu().tolist():
        selected_by_item.append([int(action) for action in actions if 0 <= int(action) < model.cfg.num_slots])
    return out, selected_by_item


def save_selector_contact_sheet(
    path: Path,
    image: torch.Tensor,
    heatmaps: torch.Tensor,
    selected_slots: list[int],
    anchor_boxes: torch.Tensor,
    evidence_boxes: torch.Tensor,
    title: str,
) -> None:
    cols = min(5, 1 + len(selected_slots))
    rows = math.ceil((1 + len(selected_slots)) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.0 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    image_rgb = denorm_image(image).permute(1, 2, 0)
    axes.flat[0].imshow(image_rgb)
    draw_boxes(axes.flat[0], anchor_boxes, "lime", "anchor")
    draw_boxes(axes.flat[0], evidence_boxes, "cyan", "evidence")
    axes.flat[0].set_title(title, fontsize=8)
    for panel_idx, slot_id in enumerate(selected_slots, start=1):
        ax = axes.flat[panel_idx]
        ax.imshow(image_rgb)
        ax.imshow(heatmaps[slot_id], cmap="magma", alpha=0.45)
        draw_boxes(ax, anchor_boxes, "lime", "anchor")
        draw_boxes(ax, evidence_boxes, "cyan", "evidence")
        ax.set_title(f"rank={panel_idx} slot={slot_id + 1}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def hit_stats(heatmaps: torch.Tensor, slots: list[int], mask: torch.Tensor, hit_threshold: float, threshold_rel: float) -> tuple[bool, float]:
    if not slots:
        return False, 0.0
    masses = [slot_bbox_metrics(heatmaps[s], mask, threshold_rel)["mass"] for s in slots]
    best = max(masses) if masses else 0.0
    return best >= hit_threshold, best


def main() -> None:
    args = parse_args()
    split = "valid" if args.split == "val" else args.split
    device = choose_device(args.device)
    run_dir = Path(args.run_dir)
    meta, model, _checkpoint = load_selector(run_dir, args.checkpoint, device)
    if meta.get("slot_embedding_source") != "structured80_slothead":
        raise ValueError(f"Expected structured80_slothead selector, got {meta.get('slot_embedding_source')!r}")

    data = args.data or meta_arg(meta, "data")
    sa_checkpoint = args.sa_checkpoint or meta_arg(meta, "sa_checkpoint") or meta_arg(meta, "checkpoint")
    structured_checkpoint = args.structured_checkpoint or meta_arg(meta, "structured_checkpoint")
    structured_mode = args.structured_mode or meta_arg(meta, "structured_mode", "u")
    input_res = args.input_res or int(meta_arg(meta, "input_res", 224))
    if not data or not sa_checkpoint or not structured_checkpoint:
        raise ValueError("Missing data, sa_checkpoint, or structured_checkpoint; pass them explicitly.")

    top_k_max = max(args.top_ks)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "visualizations" / f"{split}_grpo80_bbox_at3_at4"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "contact_sheets").mkdir(exist_ok=True)

    tfm = build_transforms(input_res)
    dataset = build_dataset(data, split, tfm["valid"])
    if args.max_items and args.max_items < len(dataset):
        dataset = Subset(dataset, list(range(args.max_items)))
    base_dataset = unwrap_dataset(dataset)
    base_indices = dataset_indices(dataset)
    metadata = load_metadata(data)
    needed = needed_metadata_keys(metadata, dataset, split)
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
    projector, _structured_ckpt = load_structured80(structured_checkpoint, device)
    controls = trace_controls_from_meta(meta)
    confidence_early_exit = not bool(meta_arg(meta, "disable_confidence_early_exit", False))

    rows = []
    contact_rows = []
    contact_counts_by_class: dict[int, int] = defaultdict(int)
    cursor = 0
    for images, labels in tqdm(loader, desc="grpo80-bbox-eval", mininterval=1.0):
        labels = labels.to(device, non_blocking=device.type == "cuda")
        slots80, attn = encode_slots80(backbone, projector, images, device, structured_mode)
        out, selected_by_item = selector_selected_slots(model, slots80, controls, confidence_early_exit)
        pred = out.logits.argmax(dim=1).detach().cpu()
        batch = labels.numel()
        for b in range(batch):
            sample_idx = base_indices[cursor + b]
            rel = fixed_sample_relative_path(base_dataset, sample_idx, split)
            meta_row = metadata.get(rel)
            if meta_row is None:
                raise KeyError(f"No metadata row for relative_path={rel}")
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
            selected_slots = selected_by_item[b]
            if not selected_slots:
                selected_slots = [0]

            row = {
                "dataset_index": cursor + b,
                "relative_path": rel,
                "true": int(labels[b].item()),
                "pred": int(pred[b].item()),
                "selector_correct": float(pred[b].item() == labels[b].detach().cpu().item()),
                "true_name": meta_row["class_name"],
                "image_id": image_id,
                "source_split": source_split,
                "anchor_object": anchor_name,
                "evidence_object": evidence_name,
                "selected_slots_1based": json.dumps([s + 1 for s in selected_slots]),
                "selected_count": len(selected_by_item[b]),
            }
            for top_k in args.top_ks:
                top_slots = selected_slots[:top_k]
                anchor_hit, anchor_mass = hit_stats(heatmaps, top_slots, anchor_mask, args.hit_threshold, args.threshold_rel)
                evidence_hit, evidence_mass = hit_stats(heatmaps, top_slots, evidence_mask, args.hit_threshold, args.threshold_rel)
                _union_hit, union_mass = hit_stats(heatmaps, top_slots, union_mask, args.hit_threshold, args.threshold_rel)
                row[f"top{top_k}_slots_1based"] = json.dumps([s + 1 for s in top_slots])
                row[f"anchor@{top_k}"] = float(anchor_hit)
                row[f"evidence@{top_k}"] = float(evidence_hit)
                row[f"pair@{top_k}"] = float(anchor_hit and evidence_hit)
                row[f"top{top_k}_best_anchor_mass"] = anchor_mass
                row[f"top{top_k}_best_evidence_mass"] = evidence_mass
                row[f"top{top_k}_best_union_mass"] = union_mass
            rows.append(row)

            class_id = int(labels[b].item())
            if args.contact_sheets_per_class > 0:
                should_save_contact = contact_counts_by_class[class_id] < args.contact_sheets_per_class
            else:
                should_save_contact = len(contact_rows) < args.contact_sheets
            if should_save_contact:
                fname = f"{split}_idx{cursor + b:05d}_true{row['true']}_pred{row['pred']}_grpo80_bbox.png"
                save_selector_contact_sheet(
                    out_dir / "contact_sheets" / fname,
                    images[b].detach().cpu(),
                    heatmaps,
                    selected_slots[:top_k_max],
                    anchor_boxes,
                    evidence_boxes,
                    f"{meta_row['class_name']} anchor={anchor_name} evidence={evidence_name}",
                )
                contact_rows.append({**row, "file": f"contact_sheets/{fname}"})
                contact_counts_by_class[class_id] += 1
        cursor += batch

    write_csv(out_dir / "grpo80_bbox_eval.csv", rows)
    write_csv(out_dir / "contact_sheet_index.csv", contact_rows)
    metric_keys = ["selector_correct"]
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
    rows_by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_class[f"{row['true']}:{row['true_name']}"].append(row)
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "sa_checkpoint": sa_checkpoint,
        "structured_checkpoint": structured_checkpoint,
        "structured_mode": structured_mode,
        "data": data,
        "split": split,
        "items": len(rows),
        "top_ks": args.top_ks,
        "hit_threshold": args.hit_threshold,
        "threshold_rel": args.threshold_rel,
        "bbox_is_test_only": True,
        "selection_source": "GRPO80 greedy selected slots",
        "metrics": metric_summary(rows, metric_keys),
        "per_class": {
            class_key: {"items": len(class_rows), "metrics": metric_summary(class_rows, metric_keys)}
            for class_key, class_rows in sorted(rows_by_class.items(), key=lambda item: int(item[0].split(":", 1)[0]))
        },
        "outputs": {
            "csv": str(out_dir / "grpo80_bbox_eval.csv"),
            "contact_sheet_index": str(out_dir / "contact_sheet_index.csv"),
            "contact_sheets": str(out_dir / "contact_sheets"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    html = ["<html><body><h1>GRPO80 BBox Evaluation @3/@4</h1>"]
    html.append("<pre>" + json.dumps(summary, indent=2) + "</pre>")
    for rec in contact_rows:
        html.append(
            f"<h3>idx={rec['dataset_index']} true={rec['true_name']} pred={rec['pred']} "
            f"anchor={rec['anchor_object']} evidence={rec['evidence_object']} "
            f"@3 A/E/P={rec.get('anchor@3', 0):.0f}/{rec.get('evidence@3', 0):.0f}/{rec.get('pair@3', 0):.0f} "
            f"@4 A/E/P={rec.get('anchor@4', 0):.0f}/{rec.get('evidence@4', 0):.0f}/{rec.get('pair@4', 0):.0f}</h3>"
        )
        html.append(f"<img src='{rec['file']}' style='max-width:100%; border:1px solid #ccc;'>")
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
