#!/usr/bin/env python3
"""Visualize AC80 selector paths over 80-dim slothead features."""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter
import torch

from misc_utils import seed_all
from SET80.slothead80 import load_slothead80, project_slots80
from train_slot_classifier import build_dataset, build_transforms, load_backbone
from visualize_grpo_selector_paths import (
    apply_trace_overrides,
    annotate_slot_overlay,
    choose_device,
    class_names_from_dataset,
    denorm_image,
    format_prob_tick,
    greedy_trace,
    load_selector,
    make_heatmaps,
    make_slot_overlay,
    trace_controls_from_meta,
)
from scripts.evaluate_set_bbox import (
    boxes_to_mask,
    hit_stats,
    load_coco_boxes,
    load_metadata,
    make_heatmaps as make_bbox_heatmaps,
    transform_boxes_to_input,
    unwrap_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="selector_ac_best.pt")
    parser.add_argument("--data", default="")
    parser.add_argument("--sa_checkpoint", default="")
    parser.add_argument("--slothead_checkpoint", default="")
    parser.add_argument("--slothead_mode", default="")
    parser.add_argument("--coco_root", default="/vol/biomedic3/kw1025/dinosaur/dataset/coco2017")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--split", default="test", choices=["train", "val", "valid", "test", "confounding_test"])
    parser.add_argument("--per_class_correct", type=int, default=2)
    parser.add_argument("--per_class_wrong", type=int, default=2)
    parser.add_argument("--class_ids", default="")
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--input_res", type=int, default=0)
    parser.add_argument("--min_steps_override", type=int, default=-1)
    parser.add_argument("--early_exit_conf_override", type=float, default=-1.0)
    parser.add_argument("--hit_threshold", type=float, default=0.20)
    parser.add_argument("--threshold_rel", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def batch_slots_attn80(backbone, projector, images: torch.Tensor, device: torch.device, mode: str):
    backbone.eval()
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.forward_dino(images)
    features = backbone.mlp(features)
    raw_slots, attn, _ = backbone.slot_attention(features)
    slots80 = project_slots80(projector, raw_slots.detach(), mode=mode)
    return slots80, attn.detach()


def meta_arg(meta: dict, name: str, default=None):
    return (meta.get("args") or {}).get(name, default)


def fixed_sample_relative_path(base_dataset, sample_idx: int, split: str) -> str:
    sample_path = Path(base_dataset.samples[sample_idx][0])
    root = Path(getattr(base_dataset, "root"))
    rel = sample_path.relative_to(root).as_posix()
    if rel.startswith(("train/", "valid/", "test/")):
        return rel
    return f"{split}/{rel}"


def needed_metadata_keys(metadata: dict[str, dict[str, str]], dataset, split: str) -> set[tuple[str, int, str]]:
    base_dataset = unwrap_dataset(dataset)
    keys = set()
    for sample_idx in range(len(dataset)):
        rel = fixed_sample_relative_path(base_dataset, sample_idx, split)
        row = metadata[rel]
        image_id = int(row["image_id"])
        source_split = row["source_split"]
        keys.add((source_split, image_id, row["anchor_object"]))
        keys.add((source_split, image_id, row["evidence_object"]))
    return keys


def bbox_metrics_for_slots(
    attn: torch.Tensor,
    selected_slots: list[int],
    anchor_boxes: torch.Tensor,
    evidence_boxes: torch.Tensor,
    input_res: int,
    hit_threshold: float,
    threshold_rel: float,
) -> dict[str, float | str]:
    heatmaps = make_bbox_heatmaps(attn.detach().cpu(), input_res)
    anchor_mask = boxes_to_mask(anchor_boxes, input_res)
    evidence_mask = boxes_to_mask(evidence_boxes, input_res)
    union_mask = anchor_mask | evidence_mask
    out: dict[str, float | str] = {}
    for top_k in (3, 4):
        top_slots = selected_slots[:top_k]
        anchor_hit, anchor_mass = hit_stats(heatmaps, top_slots, anchor_mask, hit_threshold, threshold_rel)
        evidence_hit, evidence_mass = hit_stats(heatmaps, top_slots, evidence_mask, hit_threshold, threshold_rel)
        _union_hit, union_mass = hit_stats(heatmaps, top_slots, union_mask, hit_threshold, threshold_rel)
        out[f"top{top_k}_slots_1based"] = json.dumps([slot_id + 1 for slot_id in top_slots])
        out[f"anchor@{top_k}"] = float(anchor_hit)
        out[f"evidence@{top_k}"] = float(evidence_hit)
        out[f"pair@{top_k}"] = float(anchor_hit and evidence_hit)
        out[f"top{top_k}_best_anchor_mass"] = anchor_mass
        out[f"top{top_k}_best_evidence_mass"] = evidence_mass
        out[f"top{top_k}_best_union_mass"] = union_mass
    return out


def draw_boxes(ax, boxes: torch.Tensor, color: str, label: str) -> None:
    for idx, (x1, y1, x2, y2) in enumerate(boxes.tolist()):
        rect = Rectangle(
            (x1, y1),
            max(x2 - x1, 1.0),
            max(y2 - y1, 1.0),
            fill=False,
            edgecolor=color,
            linewidth=1.5,
        )
        ax.add_patch(rect)
        if idx == 0:
            ax.text(
                x1,
                y1,
                label,
                color=color,
                fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 1},
            )


def plot_trace_with_boxes(
    image: torch.Tensor,
    attn: torch.Tensor,
    label: int,
    trace: dict,
    class_names: list[str],
    anchor_boxes: torch.Tensor,
    evidence_boxes: torch.Tensor,
    anchor_name: str,
    evidence_name: str,
    out_path: Path,
) -> None:
    image_rgb = denorm_image(image).permute(1, 2, 0)
    image_chw = denorm_image(image)
    heatmaps = make_heatmaps(attn, image.shape[-1])
    selected_steps = [step for step in trace["steps"] if not step["is_stop"]]
    n_slot_cols = max(1, len(selected_steps))
    ncols = 2 + n_slot_cols + 1
    fig = plt.figure(figsize=(2.25 * ncols, 3.05), constrained_layout=True)
    gs = fig.add_gridspec(1, ncols, width_ratios=[1.05, 1.35] + [1.0] * n_slot_cols + [1.35])

    true_name = class_names[label] if label < len(class_names) else str(label)
    pred_name = class_names[trace["pred"]] if trace["pred"] < len(class_names) else str(trace["pred"])
    correct_text = "CORRECT" if trace["pred"] == label else "WRONG"

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image_rgb)
    draw_boxes(ax, anchor_boxes, "lime", f"anchor:{anchor_name}")
    draw_boxes(ax, evidence_boxes, "cyan", f"evidence:{evidence_name}")
    ax.set_title(f"input + bbox\ntrue={label} {true_name}", fontsize=8)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    slot_overlay, slot_labels = make_slot_overlay(image_chw, heatmaps)
    ax.imshow(slot_overlay)
    draw_boxes(ax, anchor_boxes, "lime", "anchor")
    draw_boxes(ax, evidence_boxes, "cyan", "evidence")
    annotate_slot_overlay(ax, slot_labels, heatmaps)
    ax.set_title("SA all slots + bbox", fontsize=8)
    ax.axis("off")

    for i, step in enumerate(selected_steps):
        slot_id = step["action"]
        slot_map = heatmaps[slot_id]
        masked = image_chw * slot_map.unsqueeze(0) + (1.0 - slot_map.unsqueeze(0))
        ax = fig.add_subplot(gs[0, 2 + i])
        ax.imshow(masked.permute(1, 2, 0).clamp(0.0, 1.0))
        ax.imshow(slot_map, cmap="magma", alpha=0.18, vmin=0, vmax=1)
        draw_boxes(ax, anchor_boxes, "lime", "anchor")
        draw_boxes(ax, evidence_boxes, "cyan", "evidence")
        suffix = " early" if step.get("early_exit") else ""
        ax.set_title(
            f"t={step['step']} slot={slot_id + 1}{suffix}\n"
            f"pi={step['policy_prob']:.2f} pred={step['post_pred']} p={step['post_conf']:.2f}",
            fontsize=7,
        )
        ax.axis("off")

    ax = fig.add_subplot(gs[0, -1])
    curve = [point for point in trace["confidence_curve"] if point["step"] >= 1]
    xs = [point["step"] for point in curve]
    pred_conf = [point["pred_conf"] for point in curve]
    if xs:
        ax.plot(xs, pred_conf, marker="o", label="posthoc p(pred)", color="#d62728", linewidth=1.4)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
    threshold = float(trace.get("trace_controls", {}).get("early_exit_conf", 0.8))
    ax.axhline(threshold, color="#888888", linestyle="--", linewidth=0.9)
    plotted_probs = [value for value in pred_conf if value > 0]
    ymin = max(1e-4, min(plotted_probs) * 0.8) if plotted_probs else 1e-4
    ax.set_ylim(ymin, 1.02)
    ax.yaxis.set_major_formatter(FuncFormatter(format_prob_tick))
    ax.set_xlabel("selection step", fontsize=7)
    ax.set_ylabel("prob", fontsize=7)
    ax.set_title(
        f"{correct_text}\npred={trace['pred']} {pred_name}\n"
        f"conf={trace['conf']:.2f} dM={trace.get('margin_delta', 0.0):.2f}",
        fontsize=8,
    )
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc="lower right")

    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_all(args.seed, False)
    run_dir = Path(args.run_dir)
    device = choose_device(args.device)
    meta, model, _checkpoint = load_selector(run_dir, args.checkpoint, device)
    if meta.get("slot_embedding_source") != "slothead80":
        raise ValueError(f"Expected slothead80 selector, got {meta.get('slot_embedding_source')!r}")

    trace_controls = apply_trace_overrides(trace_controls_from_meta(meta), args)
    data = args.data or meta_arg(meta, "data")
    sa_checkpoint = args.sa_checkpoint or meta_arg(meta, "sa_checkpoint") or meta_arg(meta, "checkpoint")
    slothead_checkpoint = args.slothead_checkpoint or meta_arg(meta, "slothead_checkpoint")
    slothead_mode = args.slothead_mode or meta_arg(meta, "slothead_mode", "u")
    input_res = args.input_res or int(meta_arg(meta, "input_res", 224))
    if not data or not sa_checkpoint or not slothead_checkpoint:
        raise ValueError("Missing data, sa_checkpoint, or slothead_checkpoint; pass them explicitly.")

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "visualizations" / f"{args.split}_slot_paths"
    out_dir.mkdir(parents=True, exist_ok=True)

    split = "valid" if args.split == "val" else args.split
    tfm = build_transforms(input_res)
    transform = tfm["train"] if split == "train" else tfm["valid"]
    dataset = build_dataset(data, split, transform)
    base_dataset = unwrap_dataset(dataset)
    metadata = load_metadata(data)
    needed = needed_metadata_keys(metadata, dataset, split)
    coco_boxes = load_coco_boxes(Path(args.coco_root), needed)
    class_names = class_names_from_dataset(dataset) or list(meta.get("classes") or [])
    num_classes = len(class_names) if class_names else model.cfg.num_classes
    if args.class_ids.strip():
        target_classes = {int(part) for part in args.class_ids.split(",") if part.strip()}
    else:
        target_classes = set(range(num_classes))

    backbone = load_backbone(sa_checkpoint, device)
    backbone.eval()
    backbone.requires_grad_(False)
    projector, _slothead_ckpt = load_slothead80(slothead_checkpoint, device)

    per_class = {label: {"correct": 0, "wrong": 0} for label in sorted(target_classes)}
    records = []
    seen = 0
    for idx in range(len(dataset)):
        if args.max_items and len(records) >= args.max_items:
            break
        image, label_tensor = dataset[idx]
        label = int(label_tensor)
        if label not in per_class:
            continue
        rel = fixed_sample_relative_path(base_dataset, idx, split)
        meta_row = metadata[rel]
        image_id = int(meta_row["image_id"])
        source_split = meta_row["source_split"]
        width = int(meta_row["width"])
        height = int(meta_row["height"])
        anchor_name = meta_row["anchor_object"]
        evidence_name = meta_row["evidence_object"]
        anchor_boxes = transform_boxes_to_input(
            coco_boxes.get((source_split, image_id, anchor_name), []),
            width,
            height,
            input_res,
        )
        evidence_boxes = transform_boxes_to_input(
            coco_boxes.get((source_split, image_id, evidence_name), []),
            width,
            height,
            input_res,
        )

        slots, attn = batch_slots_attn80(backbone, projector, image.unsqueeze(0), device, slothead_mode)
        trace = greedy_trace(model, slots, None, label, trace_controls)
        correct = trace["pred"] == label
        bucket = "correct" if correct else "wrong"
        limit = args.per_class_correct if correct else args.per_class_wrong
        seen += 1
        if per_class[label][bucket] >= limit:
            if all(
                counts["correct"] >= args.per_class_correct and counts["wrong"] >= args.per_class_wrong
                for counts in per_class.values()
            ):
                break
            continue

        fname = f"{split}_idx{idx:05d}_true{label}_pred{trace['pred']}_{bucket}.png"
        selected_slots = trace["selected_slots"]
        bbox_metrics = bbox_metrics_for_slots(
            attn[0],
            selected_slots,
            anchor_boxes,
            evidence_boxes,
            input_res,
            args.hit_threshold,
            args.threshold_rel,
        )
        plot_trace_with_boxes(
            image,
            attn[0].cpu(),
            label,
            trace,
            class_names,
            anchor_boxes,
            evidence_boxes,
            anchor_name,
            evidence_name,
            out_dir / fname,
        )
        record = {
            "split": split,
            "dataset_index": idx,
            "relative_path": rel,
            "true": label,
            "true_name": class_names[label] if label < len(class_names) else str(label),
            "pred": trace["pred"],
            "pred_name": class_names[trace["pred"]] if trace["pred"] < len(class_names) else str(trace["pred"]),
            "correct": correct,
            "image_id": image_id,
            "source_split": source_split,
            "anchor_object": anchor_name,
            "evidence_object": evidence_name,
            "conf": trace["conf"],
            "true_prob": trace["true_prob"],
            "selected_count": trace["selected_count"],
            "selected_slots": selected_slots,
            "selected_slots_1based": [slot_id + 1 for slot_id in selected_slots],
            "steps": trace["steps"],
            "confidence_curve": trace["confidence_curve"],
            "file": fname,
            **bbox_metrics,
        }
        records.append(record)
        per_class[label][bucket] += 1

        if all(
            counts["correct"] >= args.per_class_correct and counts["wrong"] >= args.per_class_wrong
            for counts in per_class.values()
        ):
            break

    (out_dir / "index.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    html = ["<html><body><h1>DINOSAUR AC80 selector slot paths</h1>"]
    html.append(
        f"<p>run_dir={run_dir}<br>split={split}<br>records={len(records)} seen={seen}"
        f"<br>slothead_mode={slothead_mode}<br>min_steps={trace_controls['min_steps']} "
        f"threshold={trace_controls['early_exit_conf']}<br>"
        f"bbox hit_threshold={args.hit_threshold} threshold_rel={args.threshold_rel}</p>"
    )
    for rec in records:
        html.append(
            f"<h3>{rec['file']} | true={rec['true']} {rec['true_name']} | "
            f"pred={rec['pred']} {rec['pred_name']} | conf={rec['conf']:.3f} | "
            f"slots(1-based)={rec['selected_slots_1based']}<br>"
            f"anchor={rec['anchor_object']} evidence={rec['evidence_object']} | "
            f"@3 A/E/P={rec.get('anchor@3', 0):.0f}/{rec.get('evidence@3', 0):.0f}/{rec.get('pair@3', 0):.0f} "
            f"@4 A/E/P={rec.get('anchor@4', 0):.0f}/{rec.get('evidence@4', 0):.0f}/{rec.get('pair@4', 0):.0f}</h3>"
        )
        html.append(f"<img src='{rec['file']}' style='max-width:100%; border:1px solid #ccc;'>")
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "records": len(records), "seen": seen}, sort_keys=True))


if __name__ == "__main__":
    main()
