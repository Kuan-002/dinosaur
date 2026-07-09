#!/usr/bin/env python3
"""Counterfactual diagnostics for DINOSAUR GRPO slot selectors."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import fields
import json
import random
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from selector_grpo import GRPOSelectorConfig, SlotSelectorGRPO
from train_grpo_selector import attention_to_xy
from train_slot_classifier import build_dataset, build_transforms, load_backbone
from misc_utils import seed_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="selector_grpo_best.pt")
    parser.add_argument("--data", default="")
    parser.add_argument("--sa_checkpoint", default="")
    parser.add_argument("--split", default="test", choices=["train", "val", "valid", "test", "confounding_test"])
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--input_res", type=int, default=0)
    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_items", type=int, default=0)
    parser.add_argument("--random_trials", type=int, default=1)
    parser.add_argument(
        "--slot_transform",
        default="none",
        choices=["none", "mean_subtract"],
        help="Optional test-time slot transform before selector/classifier evaluation.",
    )
    parser.add_argument(
        "--fixed_budgets",
        default="2,3,4",
        help="Comma-separated policy-forced slot budgets to evaluate, or empty to disable.",
    )
    parser.add_argument("--min_steps_override", type=int, default=-1)
    parser.add_argument("--early_exit_conf_override", type=float, default=-1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def choose_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def class_names_from_dataset(dataset) -> list[str]:
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    classes = getattr(dataset, "classes", None)
    return list(classes) if classes is not None else []


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
        "early_exit_conf": float(cfg.get("early_exit_conf", args.get("early_exit_conf", 0.8))),
        "strict_conf_stop": bool(has_strict_stop and confidence_exit_enabled),
    }


def apply_trace_overrides(controls: dict, args: argparse.Namespace) -> dict:
    controls = dict(controls)
    if args.min_steps_override >= 0:
        controls["min_steps"] = int(args.min_steps_override)
    if args.early_exit_conf_override >= 0:
        controls["early_exit_conf"] = float(args.early_exit_conf_override)
        controls["strict_conf_stop"] = True
    return controls


def parse_fixed_budgets(raw: str, num_slots: int) -> list[int]:
    budgets: list[int] = []
    if not raw.strip():
        return budgets
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        budget = int(item)
        if budget < 0:
            raise ValueError(f"fixed budget must be non-negative, got {budget}")
        budgets.append(min(budget, num_slots))
    return sorted(set(budgets))


@torch.no_grad()
def batch_slots(
    backbone,
    race: Optional[Any],
    images: torch.Tensor,
    device: torch.device,
    pos_dim: int,
):
    backbone.eval()
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.forward_dino(images)
    slot_features = backbone.mlp(features)
    slots, attn, _ = backbone.slot_attention(slot_features)
    if race is not None:
        race.eval()
        slots = race(images, attn)
    slot_pos = attention_to_xy(attn) if pos_dim > 0 else None
    return slots, slot_pos


def load_race_from_selector_checkpoint(
    meta: dict,
    checkpoint: dict,
    backbone,
    model: SlotSelectorGRPO,
    device: torch.device,
) -> Optional[Any]:
    if meta.get("slot_embedding_source") != "race":
        return None
    if "race_state_dict" not in checkpoint:
        raise ValueError("Selector checkpoint was trained with RACE but has no race_state_dict")
    from race_encoder import RACEEncoder

    args = meta.get("args", {})
    race = RACEEncoder(
        model.cfg.slot_dim,
        backbone_name=str(args.get("race_backbone", meta.get("race_backbone", "convnext_tiny"))),
        pretrained=False,
        mask_threshold=float(args.get("mask_threshold", meta.get("race_mask_threshold", 0.1))),
        patch_size=int(args.get("patch_size", meta.get("race_patch_size", 16))),
        region_mode=str(args.get("region_mode", meta.get("race_region_mode", "masked_image"))),
        crop_output_size=int(args.get("crop_output_size", meta.get("race_crop_output_size", 96))),
        crop_padding=int(args.get("crop_padding", meta.get("race_crop_padding", 4))),
        crop_threshold_rel=float(args.get("crop_threshold_rel", meta.get("race_crop_threshold_rel", 0.2))),
        input_normalized=True,
        encoder_batch_size=int(args.get("race_encoder_batch_size", meta.get("race_encoder_batch_size", 64))),
    )
    race.load_state_dict(checkpoint["race_state_dict"], strict=True)
    race.to(device)
    race.eval()
    return race


@torch.no_grad()
def classify_sequence(
    model: SlotSelectorGRPO,
    slot_embeds: torch.Tensor,
    actions: list[int],
) -> torch.Tensor:
    b, k, _ = slot_embeds.shape
    if b != 1:
        raise ValueError("classify_sequence expects batch size 1")
    h = model.initial_state(slot_embeds)
    selected = torch.zeros(b, k, dtype=torch.bool, device=slot_embeds.device)
    active = torch.ones(b, dtype=torch.bool, device=slot_embeds.device)
    for action_id in actions:
        if action_id < 0 or action_id >= k:
            continue
        if bool(selected[0, action_id].item()):
            continue
        action = torch.tensor([action_id], dtype=torch.long, device=slot_embeds.device)
        h, selected = model.update_with_action(h, selected, slot_embeds, action, active)
    return model.classify(h)


@torch.no_grad()
def forced_policy_order(
    model: SlotSelectorGRPO,
    slot_embeds: torch.Tensor,
    budget: int,
) -> list[int]:
    b, k, _ = slot_embeds.shape
    if b != 1:
        raise ValueError("forced_policy_order expects batch size 1")
    h = model.initial_state(slot_embeds)
    selected_mask = torch.zeros(b, k, dtype=torch.bool, device=slot_embeds.device)
    active = torch.ones(b, dtype=torch.bool, device=slot_embeds.device)
    selected: list[int] = []
    for step in range(min(budget, model.cfg.max_steps, k)):
        if model.cfg.decoupled_stop_policy:
            action_logits = model.slot_policy_logits(h, slot_embeds, selected_mask, step=step)
        else:
            action_logits = model.policy_logits(h, slot_embeds, selected_mask, step=step)
            action_logits[:, model.stop_idx] = torch.finfo(action_logits.dtype).min
        action = action_logits.argmax(dim=-1)
        action_id = int(action.item())
        selected.append(action_id)
        h, selected_mask = model.update_with_action(h, selected_mask, slot_embeds, action, active)
    return selected


@torch.no_grad()
def greedy_trace(
    model: SlotSelectorGRPO,
    slot_embeds: torch.Tensor,
    label: int,
    controls: dict,
) -> dict:
    b, k, _ = slot_embeds.shape
    if b != 1:
        raise ValueError("greedy_trace expects batch size 1")
    min_steps = int(controls["min_steps"])
    early_exit_conf = float(controls["early_exit_conf"])

    h = model.initial_state(slot_embeds)
    selected_mask = torch.zeros(b, k, dtype=torch.bool, device=slot_embeds.device)
    active = torch.ones(b, dtype=torch.bool, device=slot_embeds.device)
    final_logits = model.classify(
        h,
    )
    init_prob = final_logits.softmax(dim=-1)
    curve = [
        {
            "step": 0,
            "pred": int(init_prob.argmax(dim=-1).item()),
            "pred_conf": float(init_prob.max(dim=-1).values.item()),
            "true_prob": float(init_prob[0, label].item()),
            "selected_count": 0,
        }
    ]
    steps = []
    stop_reason = "horizon_stop"

    for step in range(min(model.cfg.max_steps, k)):
        current_conf = final_logits.softmax(dim=-1).max(dim=-1).values
        selected_count = selected_mask.sum(dim=1)
        if controls.get("strict_conf_stop", False):
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
        record = {
            "step": step + 1,
            "action": int(action.item()),
            "is_stop": bool(is_stop.item()),
            "policy_prob": action_prob_value,
            "selected_before": int(selected_mask.sum(dim=1).item()),
        }
        steps.append(record)

        h, selected_mask = model.update_with_action(h, selected_mask, slot_embeds, action, select)
        logits = model.classify(
            h,
        )
        prob = logits.softmax(dim=-1)
        record["post_pred"] = int(prob.argmax(dim=-1).item())
        record["post_conf"] = float(prob.max(dim=-1).values.item())
        record["post_true_prob"] = float(prob[0, label].item())
        record["selected_after"] = int(selected_mask.sum(dim=1).item())
        final_logits = torch.where(active.unsqueeze(-1), logits, final_logits)

        if bool(is_stop.item()):
            stop_reason = "policy_stop"
            active = active & ~is_stop
        else:
            conf = prob.max(dim=-1).values
            enough = selected_mask.sum(dim=1) >= min_steps
            early_exit = active & enough & (conf >= early_exit_conf) if controls.get("strict_conf_stop", False) else torch.zeros_like(active)
            if bool(early_exit.item()):
                stop_reason = "confidence_stop"
                record["early_exit"] = True
            active = active & ~early_exit

        curve.append(
            {
                "step": step + 1,
                "pred": record["post_pred"],
                "pred_conf": record["post_conf"],
                "true_prob": record["post_true_prob"],
                "selected_count": record["selected_after"],
            }
        )
        if not active.any():
            break

    final_prob = final_logits.softmax(dim=-1)
    selected_slots = [step["action"] for step in steps if not step["is_stop"]]
    return {
        "steps": steps,
        "confidence_curve": curve,
        "stop_reason": stop_reason,
        "pred": int(final_prob.argmax(dim=-1).item()),
        "conf": float(final_prob.max(dim=-1).values.item()),
        "true_prob": float(final_prob[0, label].item()),
        "selected_count": len(selected_slots),
        "selected_slots": selected_slots,
    }


def pred_record(logits: torch.Tensor, label: int) -> dict:
    prob = logits.softmax(dim=-1)
    pred = int(prob.argmax(dim=-1).item())
    return {
        "pred": pred,
        "correct": pred == label,
        "conf": float(prob.max(dim=-1).values.item()),
        "true_prob": float(prob[0, label].item()),
        "loss": float(F.cross_entropy(logits, torch.tensor([label], device=logits.device)).item()),
    }


def transform_slots(slots: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return slots
    if mode == "mean_subtract":
        return slots - slots.mean(dim=1, keepdim=True)
    raise ValueError(f"Unknown slot_transform: {mode}")


def init_metric_dict() -> dict:
    return {
        "total": 0,
        "correct": 0,
        "loss_sum": 0.0,
        "conf_sum": 0.0,
        "true_prob_sum": 0.0,
    }


def update_metric(metric: dict, rec: dict) -> None:
    metric["total"] += 1
    metric["correct"] += float(rec["correct"])
    metric["loss_sum"] += rec["loss"]
    metric["conf_sum"] += rec["conf"]
    metric["true_prob_sum"] += rec["true_prob"]


def finalize_metric(metric: dict) -> dict:
    total = max(metric["total"], 1)
    return {
        "total": metric["total"],
        "accuracy": metric["correct"] / total,
        "loss": metric["loss_sum"] / total,
        "avg_conf": metric["conf_sum"] / total,
        "avg_true_prob": metric["true_prob_sum"] / total,
    }


def main() -> None:
    args = parse_args()
    seed_all(args.seed, False)
    device = choose_device(args.device)
    run_dir = Path(args.run_dir)
    meta, model, selector_checkpoint = load_selector(run_dir, args.checkpoint, device)
    controls = apply_trace_overrides(trace_controls_from_meta(meta), args)

    data = args.data or meta["args"]["data"]
    sa_checkpoint = args.sa_checkpoint or meta["args"]["checkpoint"]
    input_res = args.input_res or int(meta["args"].get("input_res", 224))
    split = "valid" if args.split == "val" else args.split
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "diagnostics" / f"{split}_counterfactual"
    out_dir.mkdir(parents=True, exist_ok=True)

    tfm = build_transforms(input_res)
    transform = tfm["train"] if split == "train" else tfm["valid"]
    dataset = build_dataset(data, split, transform)
    class_names = class_names_from_dataset(dataset) or list(meta.get("classes") or [])
    num_classes = len(class_names) if class_names else model.cfg.num_classes
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    backbone = load_backbone(sa_checkpoint, device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    race = load_race_from_selector_checkpoint(meta, selector_checkpoint, backbone, model, device)
    if race is not None:
        print("Loaded fine-tuned RACE weights from selector checkpoint.", flush=True)

    fixed_budgets = parse_fixed_budgets(args.fixed_budgets, model.cfg.num_slots)
    fixed_variants = [f"policy_budget_{budget}" for budget in fixed_budgets]
    variants = ["greedy", "selected_order", "removed_selected", "full_order", "random_same_count", *fixed_variants]
    metrics = {name: init_metric_dict() for name in variants}
    by_class = {str(cls): {name: init_metric_dict() for name in variants} for cls in range(num_classes)}
    stop_reasons: dict[str, int] = defaultdict(int)
    step_bins: dict[str, int] = defaultdict(int)
    first_slot_high_conf = 0
    records_path = out_dir / "counterfactual_records.jsonl"
    rng = random.Random(args.seed)
    total_seen = 0

    with records_path.open("w", encoding="utf-8") as records_file:
        iterator = tqdm(loader, desc=f"{split} counterfactual", mininterval=1.0)
        for idx, (images, labels) in enumerate(iterator):
            if args.max_items and total_seen >= args.max_items:
                break
            label = int(labels.item())
            slots, slot_pos = batch_slots(backbone, race, images, device, model.cfg.pos_dim)
            slots = transform_slots(slots, args.slot_transform)
            slot_embeds = model.embed_slots(slots, slot_pos)
            trace = greedy_trace(model, slot_embeds, label, controls)
            selected = list(trace["selected_slots"])
            removed = [slot_id for slot_id in range(model.cfg.num_slots) if slot_id not in set(selected)]
            full = list(range(model.cfg.num_slots))

            variant_logits = {
                "greedy": classify_sequence(model, slot_embeds, selected),
                "selected_order": classify_sequence(model, slot_embeds, selected),
                "removed_selected": classify_sequence(model, slot_embeds, removed),
                "full_order": classify_sequence(model, slot_embeds, full),
            }
            fixed_budget_records = {}
            for budget, name in zip(fixed_budgets, fixed_variants):
                budget_slots = forced_policy_order(model, slot_embeds, budget)
                variant_logits[name] = classify_sequence(model, slot_embeds, budget_slots)
                fixed_budget_records[name] = {"slots": budget_slots}
            random_records = []
            random_correct = 0
            random_loss = 0.0
            random_conf = 0.0
            random_true_prob = 0.0
            for trial in range(max(args.random_trials, 1)):
                sample_count = min(len(selected), model.cfg.num_slots)
                random_slots = rng.sample(full, sample_count) if sample_count > 0 else []
                random_rec = pred_record(classify_sequence(model, slot_embeds, random_slots), label)
                random_rec["slots"] = random_slots
                random_records.append(random_rec)
                random_correct += int(random_rec["correct"])
                random_loss += random_rec["loss"]
                random_conf += random_rec["conf"]
                random_true_prob += random_rec["true_prob"]

            variant_records = {name: pred_record(logits, label) for name, logits in variant_logits.items()}
            for name, extra in fixed_budget_records.items():
                variant_records[name].update(extra)
            variant_records["random_same_count"] = {
                "pred": None,
                "correct": random_correct / max(args.random_trials, 1),
                "conf": random_conf / max(args.random_trials, 1),
                "true_prob": random_true_prob / max(args.random_trials, 1),
                "loss": random_loss / max(args.random_trials, 1),
                "trials": random_records,
            }

            for name in variants:
                update_metric(metrics[name], variant_records[name])
                update_metric(by_class[str(label)][name], variant_records[name])
            stop_reasons[trace["stop_reason"]] += 1
            step_bins[str(trace["selected_count"])] += 1
            if len(trace["confidence_curve"]) > 1 and trace["confidence_curve"][1]["pred_conf"] >= controls["early_exit_conf"]:
                first_slot_high_conf += 1

            record = {
                "dataset_index": idx,
                "true": label,
                "true_name": class_names[label] if label < len(class_names) else str(label),
                "selected_slots": selected,
                "removed_slots": removed,
                "stop_reason": trace["stop_reason"],
                "selected_count": trace["selected_count"],
                "steps": trace["steps"],
                "confidence_curve": trace["confidence_curve"],
                "variants": variant_records,
            }
            records_file.write(json.dumps(record) + "\n")
            total_seen += 1

    final_metrics = {
        "run_dir": str(run_dir),
        "split": split,
        "checkpoint": args.checkpoint,
        "total": total_seen,
        "slot_transform": args.slot_transform,
        "trace_controls": controls,
        "fixed_budgets": fixed_budgets,
        "metrics": {name: finalize_metric(metric) for name, metric in metrics.items()},
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "selected_count_hist": {key: step_bins[key] for key in sorted(step_bins, key=lambda x: int(x))},
        "first_slot_high_conf_rate": first_slot_high_conf / max(total_seen, 1),
        "class_names": class_names,
    }
    final_by_class = {
        class_id: {name: finalize_metric(metric) for name, metric in variant_metrics.items()}
        for class_id, variant_metrics in by_class.items()
    }
    (out_dir / "counterfactual_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    (out_dir / "counterfactual_by_class.json").write_text(json.dumps(final_by_class, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "total": total_seen, "metrics": final_metrics["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
