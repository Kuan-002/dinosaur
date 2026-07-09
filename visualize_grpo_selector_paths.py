#!/usr/bin/env python3
"""Visualize DINOSAUR GRPO slot-selection sequences."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
import math
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import torch
import torch.nn.functional as F

from selector_grpo import GRPOSelectorConfig, SlotSelectorGRPO, true_class_margin
from train_grpo_selector import (
    attention_to_xy,
    augment_slots_with_slothead,
    load_slothead_probe,
)
from train_slot_classifier import build_dataset, build_transforms, load_backbone
from misc_utils import seed_all


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
SLOT_COLORS = torch.tensor(
    [
        [0.90, 0.10, 0.10],
        [0.10, 0.45, 0.95],
        [0.10, 0.70, 0.25],
        [0.95, 0.65, 0.10],
        [0.60, 0.25, 0.90],
        [0.00, 0.75, 0.75],
        [0.95, 0.25, 0.65],
        [0.55, 0.35, 0.10],
        [0.45, 0.45, 0.45],
        [0.75, 0.85, 0.10],
        [0.05, 0.05, 0.05],
        [0.80, 0.40, 0.00],
    ],
    dtype=torch.float32,
)


def format_prob_tick(value: float, _pos) -> str:
    if value <= 0:
        return ""
    if value < 0.001:
        return f"{value:.1e}"
    if value < 0.1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.2f}".rstrip("0").rstrip(".")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="selector_grpo_best.pt")
    parser.add_argument("--data", default="")
    parser.add_argument("--sa_checkpoint", default="")
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


def choose_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def denorm_image(image: torch.Tensor) -> torch.Tensor:
    return (image.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0.0, 1.0)


def class_names_from_dataset(dataset) -> list[str]:
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    classes = getattr(dataset, "classes", None)
    if classes is None:
        return []
    return list(classes)


def load_selector(run_dir: Path, checkpoint_name: str, device: torch.device) -> tuple[dict, SlotSelectorGRPO, dict]:
    meta_path = run_dir / "selector_grpo_meta.json"
    if not meta_path.exists():
        meta_path = run_dir / "selector_ac_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg_raw = dict(meta.get("grpo_config", meta.get("ac_config", {})))
    allowed = {field.name for field in fields(GRPOSelectorConfig)}
    cfg = GRPOSelectorConfig(**{key: value for key, value in cfg_raw.items() if key in allowed})
    model = SlotSelectorGRPO(cfg).to(device)
    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.exists() and checkpoint_name == "selector_grpo_best.pt":
        checkpoint_path = run_dir / "selector_ac_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return meta, model, checkpoint


def trace_controls_from_meta(meta: dict) -> dict:
    cfg = meta.get("grpo_config", meta.get("ac_config", {}))
    args = meta.get("args", {})
    has_strict_stop = "min_steps" in cfg or "min_steps" in args
    confidence_exit_enabled = not bool(args.get("disable_confidence_early_exit", False))
    return {
        "min_steps": int(cfg.get("min_steps", args.get("min_steps", 0)) or 0),
        "early_exit_conf": float(
            cfg.get("early_exit_conf", args.get("early_exit_conf", 0.8))
        ),
        "strict_conf_stop": bool(has_strict_stop and confidence_exit_enabled),
    }


def load_slothead_from_meta(meta: dict, device: torch.device):
    slothead_meta = meta.get("slothead")
    if not slothead_meta:
        return None
    checkpoint = slothead_meta.get("checkpoint")
    if not checkpoint:
        return None
    probe, _ckpt = load_slothead_probe(str(checkpoint), device)
    return probe


def augment_slots_from_meta(slots: torch.Tensor, slothead_probe, meta: dict) -> torch.Tensor:
    if slothead_probe is None:
        return slots
    slothead_meta = meta.get("slothead") or {}
    return augment_slots_with_slothead(
        slots,
        slothead_probe,
        float(slothead_meta.get("slot_embedding_scale", 1.0)),
        float(slothead_meta.get("slothead_feature_scale", 1.0)),
        float(slothead_meta.get("slothead_score_scale", 1.0)),
        str(slothead_meta.get("slothead_feature_mode", "scores")),
    )


def apply_trace_overrides(controls: dict, args: argparse.Namespace) -> dict:
    controls = dict(controls)
    if args.min_steps_override >= 0:
        controls["min_steps"] = int(args.min_steps_override)
    if args.early_exit_conf_override >= 0:
        controls["early_exit_conf"] = float(args.early_exit_conf_override)
        controls["strict_conf_stop"] = True
    return controls


@torch.no_grad()
def load_race_from_selector_checkpoint(
    meta: dict,
    checkpoint: dict,
    model: SlotSelectorGRPO,
    device: torch.device,
) -> Optional[RACEEncoder]:
    if meta.get("slot_embedding_source") != "race":
        return None
    if "race_state_dict" not in checkpoint:
        raise ValueError("Selector checkpoint was trained with RACE but has no race_state_dict")
    from race_encoder import RACEEncoder

    args = meta.get("args", {})
    race = RACEEncoder(
        model.cfg.slot_dim,
        backbone_name=str(meta.get("race_backbone", args.get("race_backbone", "convnext_tiny"))),
        pretrained=False,
        mask_threshold=float(meta.get("race_mask_threshold", args.get("mask_threshold", 0.1))),
        patch_size=int(meta.get("race_patch_size", args.get("patch_size", 16))),
        region_mode=str(meta.get("race_region_mode", args.get("region_mode", "masked_image"))),
        crop_output_size=int(meta.get("race_crop_output_size", args.get("crop_output_size", 96))),
        crop_padding=int(meta.get("race_crop_padding", args.get("crop_padding", 4))),
        crop_threshold_rel=float(meta.get("race_crop_threshold_rel", args.get("crop_threshold_rel", 0.2))),
        input_normalized=True,
        encoder_batch_size=int(meta.get("race_encoder_batch_size", args.get("race_encoder_batch_size", 64))),
    )
    race.load_state_dict(checkpoint["race_state_dict"], strict=True)
    race.to(device)
    race.eval()
    return race


@torch.no_grad()
def batch_slots_and_attn(
    backbone,
    race: Optional[RACEEncoder],
    images: torch.Tensor,
    device: torch.device,
    pos_dim: int,
):
    backbone.eval()
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.forward_dino(images)
    features = backbone.mlp(features)
    slots, attn, _ = backbone.slot_attention(features)
    if race is not None:
        slots = race(images, attn)
    slot_pos = attention_to_xy(attn) if pos_dim > 0 else None
    return slots, attn, slot_pos


@torch.no_grad()
def greedy_trace(
    model: SlotSelectorGRPO,
    slots: torch.Tensor,
    slot_pos: Optional[torch.Tensor],
    label: int,
    trace_controls: dict,
):
    cfg = model.cfg
    slot_embeds = model.embed_slots(slots, slot_pos)
    b, k, _ = slot_embeds.shape
    if b != 1:
        raise ValueError("greedy_trace expects batch size 1")

    h = model.initial_state(slot_embeds)
    selected_mask = torch.zeros(b, k, dtype=torch.bool, device=slots.device)
    active = torch.ones(b, dtype=torch.bool, device=slots.device)
    final_logits = model.classify(
        h,
    )
    steps = []

    init_prob = final_logits.softmax(dim=-1)
    init_unselected_mask = ~selected_mask
    init_unselected_logits = model.mask_order_logits(slot_embeds, init_unselected_mask)
    init_unselected_prob = init_unselected_logits.softmax(dim=-1)
    init_selected_margin = true_class_margin(final_logits, torch.tensor([label], device=slots.device))
    init_unselected_margin = true_class_margin(init_unselected_logits, torch.tensor([label], device=slots.device))
    conf_curve = [
        {
            "step": 0,
            "pred": int(init_prob.argmax(dim=-1).item()),
            "pred_conf": float(init_prob.max(dim=-1).values.item()),
            "true_prob": float(init_prob[0, label].item()),
            "select_true_prob": float(init_prob[0, label].item()),
            "unselect_true_prob": float(init_unselected_prob[0, label].item()),
            "select_margin": float(init_selected_margin.item()),
            "unselect_margin": float(init_unselected_margin.item()),
            "selected_count": 0,
        }
    ]
    label_tensor = torch.tensor([label], device=slots.device)

    min_steps = int(trace_controls.get("min_steps", 0))
    early_exit_conf = float(trace_controls.get("early_exit_conf", cfg.early_exit_conf))

    for step in range(min(cfg.max_steps, k)):
        current_conf = final_logits.softmax(dim=-1).max(dim=-1).values
        selected_count = selected_mask.sum(dim=1)
        if trace_controls.get("strict_conf_stop", False):
            can_stop = (selected_count >= min_steps) & (current_conf >= early_exit_conf)
        else:
            can_stop = selected_count >= min_steps
        if model.cfg.decoupled_stop_policy:
            stop_prob = torch.sigmoid(model.stop_head(h).squeeze(-1))
            slot_logits = model.slot_policy_logits(h, slot_embeds, selected_mask, step=step)
            slot_prob = slot_logits.softmax(dim=-1)
            slot_action = slot_logits.argmax(dim=-1)
            is_stop = can_stop & (stop_prob >= 0.5)
            action = torch.where(is_stop, torch.full_like(slot_action, model.stop_idx), slot_action)
            action_prob_value = float(stop_prob.item()) if bool(is_stop.item()) else float(slot_prob[0, slot_action.item()].item())
        else:
            action_logits = model.policy_logits(h, slot_embeds, selected_mask, step=step)
            if not bool(can_stop.item()):
                action_logits[:, model.stop_idx] = torch.finfo(action_logits.dtype).min
            action_prob = action_logits.softmax(dim=-1)
            action = action_logits.argmax(dim=-1)
            is_stop = action == model.stop_idx
            action_prob_value = float(action_prob[0, action.item()].item())
        select = active & ~is_stop

        step_record = {
            "step": step + 1,
            "action": int(action.item()),
            "is_stop": bool(is_stop.item()),
            "policy_prob": action_prob_value,
            "selected_before": int(selected_mask.sum(dim=1).item()),
        }
        steps.append(step_record)

        h, selected_mask = model.update_with_action(
            h,
            selected_mask,
            slot_embeds,
            action,
            select,
        )
        step_logits = model.classify(
            h,
        )
        step_prob = step_logits.softmax(dim=-1)
        unselected_mask = ~selected_mask
        unselected_logits = model.mask_order_logits(slot_embeds, unselected_mask)
        unselected_prob = unselected_logits.softmax(dim=-1)
        selected_margin = true_class_margin(step_logits, label_tensor)
        unselected_margin = true_class_margin(unselected_logits, label_tensor)
        step_record["post_pred"] = int(step_prob.argmax(dim=-1).item())
        step_record["post_conf"] = float(step_prob.max(dim=-1).values.item())
        step_record["post_true_prob"] = float(step_prob[0, label].item())
        step_record["unselect_pred"] = int(unselected_prob.argmax(dim=-1).item())
        step_record["unselect_conf"] = float(unselected_prob.max(dim=-1).values.item())
        step_record["unselect_true_prob"] = float(unselected_prob[0, label].item())
        step_record["select_margin"] = float(selected_margin.item())
        step_record["unselect_margin"] = float(unselected_margin.item())
        step_record["margin_delta"] = float((selected_margin - unselected_margin).item())
        step_record["selected_after"] = int(selected_mask.sum(dim=1).item())
        final_logits = torch.where(active.unsqueeze(-1), step_logits, final_logits)
        active = active & ~is_stop
        conf = step_prob.max(dim=-1).values
        enough_steps = selected_mask.sum(dim=1) >= min_steps
        early_exit = active & enough_steps & (conf >= early_exit_conf) if trace_controls.get("strict_conf_stop", False) else torch.zeros_like(active)
        if bool(early_exit.item()):
            step_record["early_exit"] = True
        active = active & ~early_exit
        conf_curve.append(
            {
                "step": step + 1,
                "pred": step_record["post_pred"],
                "pred_conf": step_record["post_conf"],
                "true_prob": step_record["post_true_prob"],
                "select_true_prob": step_record["post_true_prob"],
                "unselect_true_prob": step_record["unselect_true_prob"],
                "select_margin": step_record["select_margin"],
                "unselect_margin": step_record["unselect_margin"],
                "selected_count": step_record["selected_after"],
            }
        )
        if not active.any():
            break

    final_prob = final_logits.softmax(dim=-1)
    final_unselected_mask = ~selected_mask
    final_unselected_logits = model.mask_order_logits(slot_embeds, final_unselected_mask)
    final_unselected_prob = final_unselected_logits.softmax(dim=-1)
    final_selected_margin = true_class_margin(final_logits, label_tensor)
    final_unselected_margin = true_class_margin(final_unselected_logits, label_tensor)
    return {
        "steps": steps,
        "confidence_curve": conf_curve,
        "pred": int(final_prob.argmax(dim=-1).item()),
        "conf": float(final_prob.max(dim=-1).values.item()),
        "true_prob": float(final_prob[0, label].item()),
        "unselect_pred": int(final_unselected_prob.argmax(dim=-1).item()),
        "unselect_conf": float(final_unselected_prob.max(dim=-1).values.item()),
        "unselect_true_prob": float(final_unselected_prob[0, label].item()),
        "select_margin": float(final_selected_margin.item()),
        "unselect_margin": float(final_unselected_margin.item()),
        "margin_delta": float((final_selected_margin - final_unselected_margin).item()),
        "selected_count": int(selected_mask.sum(dim=1).item()),
        "selected_slots": [s["action"] for s in steps if not s["is_stop"]],
        "trace_controls": {
            "min_steps": min_steps,
            "early_exit_conf": early_exit_conf,
        },
    }


def make_heatmaps(attn: torch.Tensor, size: int) -> torch.Tensor:
    k, n = attn.shape
    side = int(n**0.5)
    if side * side != n:
        raise ValueError(f"Attention map size is not square: {n}")
    maps = attn.reshape(k, 1, side, side)
    maps = F.interpolate(maps, size=(size, size), mode="bilinear", align_corners=False)
    maps = maps[:, 0]
    maps = maps / maps.flatten(1).amax(dim=1).clamp_min(1e-8).view(k, 1, 1)
    return maps.detach().cpu()


def make_slot_grid(image_chw: torch.Tensor, heatmaps: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    k, h, w = heatmaps.shape
    cols = min(4, k)
    rows = math.ceil(k / cols)
    cells = []
    for idx in range(rows * cols):
        if idx < k:
            slot_map = heatmaps[idx]
            cell = image_chw * slot_map.unsqueeze(0) + (1.0 - slot_map.unsqueeze(0))
        else:
            cell = torch.ones_like(image_chw)
        cells.append(cell)
    row_tensors = []
    for row in range(rows):
        row_tensors.append(torch.cat(cells[row * cols : (row + 1) * cols], dim=2))
    return torch.cat(row_tensors, dim=1).permute(1, 2, 0).clamp(0.0, 1.0), rows, cols


def make_slot_overlay(image_chw: torch.Tensor, heatmaps: torch.Tensor, alpha: float = 0.42) -> tuple[torch.Tensor, torch.Tensor]:
    labels = heatmaps.argmax(dim=0)
    colors = SLOT_COLORS.to(dtype=image_chw.dtype)
    if heatmaps.shape[0] > colors.shape[0]:
        repeat = math.ceil(heatmaps.shape[0] / colors.shape[0])
        colors = colors.repeat(repeat, 1)
    color_map = colors[labels].permute(2, 0, 1)
    confidence = heatmaps.amax(dim=0).clamp(0.0, 1.0)
    blend_alpha = alpha * confidence.unsqueeze(0)
    overlay = image_chw * (1.0 - blend_alpha) + color_map * blend_alpha
    return overlay.permute(1, 2, 0).clamp(0.0, 1.0), labels


def annotate_slot_overlay(ax, labels: torch.Tensor, heatmaps: torch.Tensor) -> None:
    k = heatmaps.shape[0]
    h, w = labels.shape
    for slot_id in range(k):
        mask = labels == slot_id
        if not bool(mask.any()):
            continue
        ys, xs = torch.where(mask)
        weights = heatmaps[slot_id, ys, xs].clamp_min(1e-8)
        x = float((xs.float() * weights).sum() / weights.sum())
        y = float((ys.float() * weights).sum() / weights.sum())
        ax.text(
            x,
            y,
            str(slot_id + 1),
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 1.5},
        )


def plot_trace(
    image: torch.Tensor,
    attn: torch.Tensor,
    label: int,
    trace: dict,
    class_names: list[str],
    out_path: Path,
) -> None:
    image_rgb = denorm_image(image).permute(1, 2, 0)
    image_chw = denorm_image(image)
    heatmaps = make_heatmaps(attn, image.shape[-1])
    selected_steps = [step for step in trace["steps"] if not step["is_stop"]]
    n_slot_cols = max(1, len(selected_steps))
    ncols = 2 + n_slot_cols + 1
    fig = plt.figure(figsize=(2.2 * ncols, 2.8), constrained_layout=True)
    gs = fig.add_gridspec(1, ncols, width_ratios=[1.0, 1.35] + [1.0] * n_slot_cols + [1.35])

    true_name = class_names[label] if label < len(class_names) else str(label)
    pred_name = class_names[trace["pred"]] if trace["pred"] < len(class_names) else str(trace["pred"])
    correct_text = "CORRECT" if trace["pred"] == label else "WRONG"

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(image_rgb)
    ax.set_title(f"input\ntrue={label} {true_name}", fontsize=8)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    slot_overlay, slot_labels = make_slot_overlay(image_chw, heatmaps)
    ax.imshow(slot_overlay)
    annotate_slot_overlay(ax, slot_labels, heatmaps)
    ax.set_title("SA all slots on input", fontsize=8)
    ax.axis("off")

    for i, step in enumerate(selected_steps):
        slot_id = step["action"]
        slot_map = heatmaps[slot_id]
        masked = image_chw * slot_map.unsqueeze(0) + (1.0 - slot_map.unsqueeze(0))
        ax = fig.add_subplot(gs[0, 2 + i])
        ax.imshow(masked.permute(1, 2, 0).clamp(0.0, 1.0))
        ax.imshow(slot_map, cmap="magma", alpha=0.18, vmin=0, vmax=1)
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
    meta, model, checkpoint = load_selector(run_dir, args.checkpoint, device)
    trace_controls = apply_trace_overrides(trace_controls_from_meta(meta), args)

    data = args.data or meta["args"]["data"]
    sa_checkpoint = args.sa_checkpoint or meta["args"]["checkpoint"]
    input_res = args.input_res or int(meta["args"].get("input_res", 224))
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
    for param in backbone.parameters():
        param.requires_grad = False
    race = load_race_from_selector_checkpoint(meta, checkpoint, model, device)
    slothead_probe = load_slothead_from_meta(meta, device)

    per_class = {label: {"correct": 0, "wrong": 0} for label in sorted(target_classes)}
    records = []
    seen = 0
    for idx in range(len(dataset)):
        if args.max_items and len(records) >= args.max_items:
            break
        image, label_tensor = dataset[idx]
        label = int(label_tensor)
        if label not in target_classes or label not in per_class:
            continue
        image_batch = image.unsqueeze(0).to(device)
        slots, attn, slot_pos = batch_slots_and_attn(backbone, race, image_batch, device, model.cfg.pos_dim)
        slots = augment_slots_from_meta(slots, slothead_probe, meta)
        trace = greedy_trace(model, slots, slot_pos, label, trace_controls)
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
        plot_trace(image, attn[0].detach().cpu(), label, trace, class_names, out_dir / fname)
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
    html = ["<html><body><h1>DINOSAUR GRPO selector slot paths</h1>"]
    html.append(
        f"<p>run_dir={run_dir}<br>split={split}<br>records={len(records)} seen={seen}"
        f"<br>min_steps={trace_controls['min_steps']} threshold={trace_controls['early_exit_conf']}</p>"
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
