#!/usr/bin/env python3
"""Joint contrastive-pair GRPO training for the 8-class COCO rule graph."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

import torch
import torch.nn.functional as F

import contrastive_core as fac
import train_grpo_selector as grpo
from misc_utils import seed_all
from selector_grpo import GRPOSelectorConfig, SlotSelectorGRPO
from train_slot_classifier import build_dataset, build_transforms, load_backbone, subset_dataset


DATA = ROOT / "dataset/coco_rule_graph8_v2_area015_012_300_100_100/classification_dataset"
SA = ROOT / "checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--sa_checkpoint", default=str(SA))
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--min_steps", type=int, default=3)
    parser.add_argument("--early_exit_conf", type=float, default=0.85)
    parser.add_argument("--classification_coef", type=float, default=1.0)
    parser.add_argument("--component_mil_coef", type=float, default=1.0)
    parser.add_argument("--component_mil_temperature", type=float, default=5.0)
    parser.add_argument("--component_loss_mode", choices=("positive_unknown", "positive_weak_negative", "binary"), default="positive_weak_negative")
    parser.add_argument("--component_negative_coef", type=float, default=0.05)
    parser.add_argument("--pair_ce_coef", type=float, default=1.0)
    parser.add_argument("--anchor_coef", type=float, default=0.25)
    parser.add_argument("--evidence_coef", type=float, default=2.0)
    parser.add_argument("--pair_coef", type=float, default=1.0)
    parser.add_argument("--class_margin_coef", type=float, default=1.0)
    parser.add_argument("--rank_discount", type=float, default=0.85)
    parser.add_argument("--grpo_group_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--quick_limit_train", type=int, default=0)
    parser.add_argument("--quick_limit_val", type=int, default=0)
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--checkpoint_acc_tolerance", type=float, default=0.02)
    parser.add_argument("--checkpoint_report_tolerances", default="0.02,0.03")
    return parser.parse_args()


def auxiliary_loss(model, slots, labels, args, anchor_ids, evidence_ids, num_objects):
    """Train component support heads and classifier on random partial sets."""
    slot_embeds = model.embed_slots(slots, None)
    batch, num_slots, _ = slot_embeds.shape
    hidden = model.initial_state(slot_embeds)
    selected_mask = torch.zeros(batch, num_slots, dtype=torch.bool, device=slots.device)
    max_random_steps = min(args.max_steps, num_slots)
    min_random_steps = min(args.min_steps, max_random_steps)
    selected_counts = torch.randint(min_random_steps, max_random_steps + 1, (batch,), device=slots.device)
    for step in range(max_random_steps):
        action = torch.rand(batch, num_slots, device=slots.device).masked_fill(selected_mask, -1).argmax(1)
        active_mask = step < selected_counts
        hidden, selected_mask = model.update_with_action(hidden, selected_mask, slot_embeds, action, active_mask)

    class_logits = model.classify(hidden)
    slot_component_logits = model.classify_components(slot_embeds)
    targets = fac.component_targets(labels, anchor_ids, evidence_ids, num_objects)
    image_component_logits = fac.mil_logits(slot_component_logits, args.component_mil_temperature)
    if args.component_loss_mode == "positive_unknown":
        component_loss = -(F.logsigmoid(image_component_logits) * targets).sum() / targets.sum().clamp_min(1.0)
    elif args.component_loss_mode == "positive_weak_negative":
        positive_loss = -(F.logsigmoid(image_component_logits) * targets).sum() / targets.sum().clamp_min(1.0)
        negative_targets = 1.0 - targets
        negative_loss = -(F.logsigmoid(-image_component_logits) * negative_targets).sum() / negative_targets.sum().clamp_min(1.0)
        component_loss = positive_loss + args.component_negative_coef * negative_loss
    else:
        component_loss = F.binary_cross_entropy_with_logits(image_component_logits, targets)
    rule_pair_logits = fac.rule_pair_logits(
        slot_component_logits.sigmoid(), torch.ones_like(selected_mask), anchor_ids, evidence_ids
    )
    class_loss = F.cross_entropy(class_logits, labels)
    pair_ce_loss = F.cross_entropy(rule_pair_logits, labels)
    total = (
        args.classification_coef * class_loss
        + args.component_mil_coef * component_loss
        + args.pair_ce_coef * pair_ce_loss
    )
    return total, class_logits, class_loss.detach(), component_loss.detach(), pair_ce_loss.detach()


def main() -> None:
    args = parse_args()
    seed_all(args.seed, False)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing silent CPU fallback")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transforms = build_transforms(224)
    train_set = subset_dataset(build_dataset(args.data, "train", transforms["train"]), args.quick_limit_train, args.seed)
    valid_set = subset_dataset(build_dataset(args.data, "valid", transforms["valid"]), args.quick_limit_val, args.seed)
    base = train_set
    while hasattr(base, "dataset"):
        base = base.dataset
    classes = list(base.classes)
    objects, anchor_ids, evidence_ids, rules = fac.load_rules(args.data, classes)
    anchor_ids = anchor_ids.to(device)
    evidence_ids = evidence_ids.to(device)
    train_loader = grpo.make_loader(train_set, args, device, shuffle=True)
    valid_loader = grpo.make_loader(valid_set, args, device, shuffle=False)
    backbone = load_backbone(args.sa_checkpoint, device).eval()
    backbone.requires_grad_(False)

    config = GRPOSelectorConfig(
        num_slots=backbone.num_slots,
        slot_dim=backbone.slot_dim,
        num_classes=len(classes),
        hidden_dim=256,
        policy_dim=256,
        max_steps=args.max_steps,
        min_steps=args.min_steps,
        early_exit_conf=args.early_exit_conf,
        policy_context_attention=True,
        first_step_num_heads=4,
        num_components=len(objects),
    )
    model = SlotSelectorGRPO(config).to(device)
    auxiliary_optimizer = torch.optim.AdamW(fac.ce_parameters(model), lr=args.lr, weight_decay=1e-4)
    policy_optimizer = torch.optim.AdamW(fac.policy_parameters(model), lr=args.policy_lr, weight_decay=1e-4)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    meta = {
        "args": vars(args), "config": asdict(config), "classes": classes,
        "objects": objects, "rules": rules, "uses_bbox": False,
        "training_schedule": "joint auxiliary + GRPO from epoch 1; no warmup",
        "selection_objective": "class-margin GRPO plus evidence-upweighted anchor/evidence increments and object-sharing-aware contrastive rule-pair margin increment",
        "checkpoint_selection": "highest valid_pair_margin@3 among epochs with valid_acc@3 within checkpoint_acc_tolerance of the best valid_acc@3",
    }
    (output / "experiment_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    checkpoint_candidates = []
    best_acc_seen = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        seen = correct = 0
        loss_sums = {"aux": 0.0, "class": 0.0, "component": 0.0, "pair_ce": 0.0, "grpo": 0.0}
        for images, labels in train_loader:
            labels = labels.to(device)
            slots = fac.encode(backbone, images, device)

            auxiliary_optimizer.zero_grad(set_to_none=True)
            aux_loss, class_logits, class_loss, component_loss, pair_ce_loss = auxiliary_loss(
                model, slots, labels, args, anchor_ids, evidence_ids, len(objects)
            )
            aux_loss.backward()
            torch.nn.utils.clip_grad_norm_(fac.ce_parameters(model), 1.0)
            auxiliary_optimizer.step()

            group = args.grpo_group_size
            repeated_slots = slots.repeat_interleave(group, 0)
            repeated_labels = labels.repeat_interleave(group, 0)
            rollout = fac.rollout(model, repeated_slots, repeated_labels, args, True, anchor_ids, evidence_ids)
            active_mask = rollout["mask"]
            advantage = (
                args.anchor_coef * fac.group_advantage(rollout["anchor"], active_mask, labels.numel(), group)
                + args.evidence_coef * fac.group_advantage(rollout["evidence"], active_mask, labels.numel(), group)
                + args.pair_coef * fac.group_advantage(rollout["pair"], active_mask, labels.numel(), group)
                + args.class_margin_coef * fac.group_advantage(rollout["class_margin"], active_mask, labels.numel(), group)
            )
            grpo_loss = -(rollout["log_probs"] * advantage * active_mask).sum() / active_mask.sum().clamp_min(1.0)
            policy_optimizer.zero_grad(set_to_none=True)
            grpo_loss.backward()
            torch.nn.utils.clip_grad_norm_(fac.policy_parameters(model), 1.0)
            policy_optimizer.step()

            batch = labels.numel()
            seen += batch
            correct += class_logits.argmax(1).eq(labels).sum().item()
            for name, value in (
                ("aux", aux_loss), ("class", class_loss), ("component", component_loss),
                ("pair_ce", pair_ce_loss), ("grpo", grpo_loss.detach()),
            ):
                loss_sums[name] += float(value.detach()) * batch

        model.eval()
        sums = {k: {name: 0.0 for name in ("pair_margin", "correct")} for k in (3, 4)}
        valid_logits = {k: [] for k in (3, 4)}
        valid_labels = []
        valid_total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                labels = labels.to(device)
                slots = fac.encode(backbone, images, device)
                metrics = fac.forced_metrics(model, slots, labels, (3, 4), anchor_ids, evidence_ids)
                valid_total += labels.numel()
                valid_labels.append(labels.detach().cpu())
                for k in (3, 4):
                    for name in sums[k]:
                        sums[k][name] += metrics[k][name].float().sum().item()
                    valid_logits[k].append(metrics[k]["logits"].detach().cpu())
        labels_cat = torch.cat(valid_labels, dim=0)
        cls_metrics = {
            k: fac.multiclass_metrics(torch.cat(valid_logits[k], dim=0), labels_cat)
            for k in (3, 4)
        }
        valid_pair_margin = 0.5 * (sums[3]["pair_margin"] + sums[4]["pair_margin"]) / valid_total
        valid_pair_margin_at3 = sums[3]["pair_margin"] / valid_total
        valid_pair_margin_at4 = sums[4]["pair_margin"] / valid_total
        valid_acc_at3 = sums[3]["correct"] / valid_total
        valid_acc_at4 = sums[4]["correct"] / valid_total
        valid_mean_acc = 0.5 * (valid_acc_at3 + valid_acc_at4)
        losses = " ".join(f"{name}={value/max(seen,1):.4f}" for name, value in loss_sums.items())
        print(
            f"epoch={epoch} phase=joint train_acc={100*correct/max(seen,1):.2f} {losses} "
            f"class_margin_coef={args.class_margin_coef:g} "
            f"anchor_coef={args.anchor_coef:g} evidence_coef={args.evidence_coef:g} pair_coef={args.pair_coef:g} "
            f"valid_pair_margin@3={sums[3]['pair_margin']/valid_total:.4f} "
            f"valid_pair_margin@4={sums[4]['pair_margin']/valid_total:.4f} "
            f"valid_acc@3={100*valid_acc_at3:.2f} "
            f"valid_acc@4={100*valid_acc_at4:.2f} "
            f"valid_precision@3={cls_metrics[3]['precision']:.4f} "
            f"valid_recall@3={cls_metrics[3]['recall']:.4f} "
            f"valid_f1@3={cls_metrics[3]['f1']:.4f} "
            f"valid_auc@3={cls_metrics[3]['auc']:.4f} "
            f"valid_precision@4={cls_metrics[4]['precision']:.4f} "
            f"valid_recall@4={cls_metrics[4]['recall']:.4f} "
            f"valid_f1@4={cls_metrics[4]['f1']:.4f} "
            f"valid_auc@4={cls_metrics[4]['auc']:.4f} "
            f"valid_mean_acc={100*valid_mean_acc:.2f} "
            f"valid_selection_acc={100*valid_acc_at3:.2f}",
            flush=True,
        )
        best_acc_seen = max(best_acc_seen, valid_acc_at3)
        checkpoint_candidates.append({
            "epoch": epoch, "valid_acc": valid_acc_at3, "valid_mean_acc": valid_mean_acc,
            "valid_acc_at3": valid_acc_at3, "valid_acc_at4": valid_acc_at4,
            "valid_pair_margin": valid_pair_margin,
            "valid_pair_margin_at3": valid_pair_margin_at3,
            "valid_pair_margin_at4": valid_pair_margin_at4,
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "config": asdict(config), "args": vars(args), "classes": classes,
            "objects": objects, "rules": rules, "valid_metrics": cls_metrics,
        })
    def select_checkpoint(tolerance: float) -> tuple[dict, float]:
        acc_floor = best_acc_seen - tolerance
        eligible = [c for c in checkpoint_candidates if c["valid_acc_at3"] >= acc_floor]
        return max(
            eligible,
            key=lambda c: (c["valid_pair_margin_at3"], c["valid_acc_at3"], c["valid_pair_margin_at4"]),
        ), acc_floor

    best_checkpoint, acc_floor = select_checkpoint(args.checkpoint_acc_tolerance)
    best_checkpoint["best_valid_acc_at3"] = best_acc_seen
    best_checkpoint["checkpoint_acc_floor"] = acc_floor
    best_checkpoint["checkpoint_acc_tolerance"] = args.checkpoint_acc_tolerance
    torch.save(best_checkpoint, output / "selector_grpo_best.pt")
    report_tolerances = sorted({
        float(raw)
        for raw in args.checkpoint_report_tolerances.split(",")
        if raw.strip()
    } | {float(args.checkpoint_acc_tolerance)})
    report = {
        "best_valid_acc_at3": best_acc_seen,
        "default_tolerance": args.checkpoint_acc_tolerance,
        "default_checkpoint_epoch": best_checkpoint["epoch"],
        "candidates": [
            {
                key: c[key]
                for key in ("epoch", "valid_acc_at3", "valid_acc_at4", "valid_pair_margin_at3", "valid_pair_margin_at4")
            }
            for c in checkpoint_candidates
        ],
        "selections": {},
    }
    for tolerance in report_tolerances:
        selected, floor = select_checkpoint(tolerance)
        tag = str(tolerance).replace(".", "p")
        selected_path = output / f"selector_grpo_best_tol{tag}.pt"
        selected_for_save = dict(selected)
        selected_for_save["best_valid_acc_at3"] = best_acc_seen
        selected_for_save["checkpoint_acc_floor"] = floor
        selected_for_save["checkpoint_acc_tolerance"] = tolerance
        torch.save(selected_for_save, selected_path)
        report["selections"][f"{tolerance:g}"] = {
            "path": str(selected_path),
            "epoch": selected["epoch"],
            "valid_acc_at3": selected["valid_acc_at3"],
            "valid_acc_at4": selected["valid_acc_at4"],
            "valid_pair_margin_at3": selected["valid_pair_margin_at3"],
            "valid_pair_margin_at4": selected["valid_pair_margin_at4"],
            "checkpoint_acc_floor": floor,
        }
    (output / "checkpoint_selection_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"checkpoint: {output / 'selector_grpo_best.pt'}", flush=True)


if __name__ == "__main__":
    main()
