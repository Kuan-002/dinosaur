#!/usr/bin/env python3
"""Visualize GRPO56 selector paths over structured 56-dim slot embeddings."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "SET56") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "SET56"))

os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".cache" / "torch"))

import torch

from misc_utils import seed_all
from SET56.structured56 import load_structured56, project_slots56
from train_slot_classifier import build_dataset, build_transforms, load_backbone
from visualize_grpo_selector_paths import (
    apply_trace_overrides,
    choose_device,
    class_names_from_dataset,
    greedy_trace,
    load_selector,
    plot_trace,
    trace_controls_from_meta,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="selector_grpo_best.pt")
    parser.add_argument("--data", default="")
    parser.add_argument("--sa_checkpoint", default="")
    parser.add_argument("--structured_checkpoint", default="")
    parser.add_argument("--structured_mode", default="")
    parser.add_argument("--out_dir", default="")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "valid", "test", "confounding_test"],
    )
    parser.add_argument("--per_class_correct", type=int, default=2)
    parser.add_argument("--per_class_wrong", type=int, default=2)
    parser.add_argument(
        "--class_ids",
        default="",
        help="Comma-separated class ids to visualize. Defaults to all classes.",
    )
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--input_res", type=int, default=0)
    parser.add_argument("--min_steps_override", type=int, default=-1)
    parser.add_argument("--early_exit_conf_override", type=float, default=-1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def batch_slots_attn56(backbone, projector, images: torch.Tensor, device: torch.device, mode: str):
    backbone.eval()
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.forward_dino(images)
    features = backbone.mlp(features)
    raw_slots, attn, _ = backbone.slot_attention(features)
    slots56 = project_slots56(projector, raw_slots.detach(), mode=mode)
    return slots56, attn.detach()


def meta_arg(meta: dict, name: str, default=None):
    return (meta.get("args") or {}).get(name, default)


def main() -> None:
    args = parse_args()
    seed_all(args.seed, False)
    run_dir = Path(args.run_dir)
    device = choose_device(args.device)
    meta, model, _checkpoint = load_selector(run_dir, args.checkpoint, device)
    if meta.get("slot_embedding_source") != "structured56":
        raise ValueError(f"Expected structured56 selector, got {meta.get('slot_embedding_source')!r}")

    trace_controls = apply_trace_overrides(trace_controls_from_meta(meta), args)
    data = args.data or meta_arg(meta, "data")
    sa_checkpoint = args.sa_checkpoint or meta_arg(meta, "sa_checkpoint") or meta_arg(meta, "checkpoint")
    structured_checkpoint = args.structured_checkpoint or meta_arg(meta, "structured_checkpoint")
    structured_mode = args.structured_mode or meta_arg(meta, "structured_mode", "u")
    input_res = args.input_res or int(meta_arg(meta, "input_res", 224))
    if not data or not sa_checkpoint or not structured_checkpoint:
        raise ValueError("Missing data, sa_checkpoint, or structured_checkpoint; pass them explicitly.")

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "visualizations" / f"{args.split}_slot_paths"
    out_dir.mkdir(parents=True, exist_ok=True)

    split = "valid" if args.split == "val" else args.split
    tfm = build_transforms(input_res)
    transform = tfm["train"] if split == "train" else tfm["valid"]
    dataset = build_dataset(data, split, transform)
    class_names = class_names_from_dataset(dataset) or list(meta.get("classes") or [])
    num_classes = len(class_names) if class_names else model.cfg.num_classes
    if args.class_ids.strip():
        target_classes = {int(part) for part in args.class_ids.split(",") if part.strip()}
    else:
        target_classes = set(range(num_classes))

    backbone = load_backbone(sa_checkpoint, device)
    backbone.eval()
    backbone.requires_grad_(False)
    projector, _structured_ckpt = load_structured56(structured_checkpoint, device)

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
        slots, attn = batch_slots_attn56(backbone, projector, image.unsqueeze(0), device, structured_mode)
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
        plot_trace(image, attn[0].cpu(), label, trace, class_names, out_dir / fname)
        record = {
            "split": split,
            "dataset_index": idx,
            "true": label,
            "true_name": class_names[label] if label < len(class_names) else str(label),
            "pred": trace["pred"],
            "pred_name": class_names[trace["pred"]] if trace["pred"] < len(class_names) else str(trace["pred"]),
            "correct": correct,
            "conf": trace["conf"],
            "true_prob": trace["true_prob"],
            "unselect_pred": trace.get("unselect_pred"),
            "unselect_conf": trace.get("unselect_conf"),
            "unselect_true_prob": trace.get("unselect_true_prob"),
            "select_margin": trace.get("select_margin"),
            "unselect_margin": trace.get("unselect_margin"),
            "margin_delta": trace.get("margin_delta"),
            "selected_count": trace["selected_count"],
            "selected_slots": trace["selected_slots"],
            "selected_slots_1based": [slot_id + 1 for slot_id in trace["selected_slots"]],
            "steps": trace["steps"],
            "steps_1based": [
                {**step, "action_1based": step["action"] + 1 if not step["is_stop"] else None}
                for step in trace["steps"]
            ],
            "confidence_curve": trace["confidence_curve"],
            "file": fname,
        }
        records.append(record)
        per_class[label][bucket] += 1

        if all(
            counts["correct"] >= args.per_class_correct and counts["wrong"] >= args.per_class_wrong
            for counts in per_class.values()
        ):
            break

    (out_dir / "index.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    html = ["<html><body><h1>DINOSAUR GRPO56 selector slot paths</h1>"]
    html.append(
        f"<p>run_dir={run_dir}<br>split={split}<br>records={len(records)} seen={seen}"
        f"<br>structured_mode={structured_mode}<br>min_steps={trace_controls['min_steps']} "
        f"threshold={trace_controls['early_exit_conf']}</p>"
    )
    for rec in records:
        html.append(
            f"<h3>{rec['file']} | true={rec['true']} {rec['true_name']} | "
            f"pred={rec['pred']} {rec['pred_name']} | conf={rec['conf']:.3f} | "
            f"slots(1-based)={rec['selected_slots_1based']}</h3>"
        )
        html.append(f"<img src='{rec['file']}' style='max-width:100%; border:1px solid #ccc;'>")
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "records": len(records), "seen": seen}, sort_keys=True))


if __name__ == "__main__":
    main()
