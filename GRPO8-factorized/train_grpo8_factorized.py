#!/usr/bin/env python3
"""Joint factorized-MIL + GRPO training for the 8-class COCO rule graph."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "GRPO10-factorized"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

import torch
import torch.nn.functional as F

import train_grpo10_factorized as fac
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
    parser.add_argument("--component_loss_mode", choices=("positive_unknown", "binary"), default="positive_unknown")
    parser.add_argument("--pair_ce_coef", type=float, default=1.0)
    parser.add_argument("--anchor_coef", type=float, default=0.25)
    parser.add_argument("--evidence_coef", type=float, default=2.0)
    parser.add_argument("--pair_coef", type=float, default=1.0)
    parser.add_argument("--rule_margin_coef", type=float, default=0.5)
    parser.add_argument("--completion_threshold", type=float, default=0.25)
    parser.add_argument("--completion_bonus", type=float, default=0.5)
    parser.add_argument("--rank_discount", type=float, default=0.85)
    parser.add_argument("--grpo_group_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--quick_limit_train", type=int, default=0)
    parser.add_argument("--quick_limit_val", type=int, default=0)
    parser.add_argument("--require_cuda", action="store_true")
    return parser.parse_args()


def auxiliary_loss(model, slots, labels, args, anchor_ids, evidence_ids, num_objects):
    """Train A/E/P and the classifier on random partial sets every epoch."""
    embeds = model.embed_slots(slots, None)
    batch, num_slots, _ = embeds.shape
    hidden = model.initial_state(embeds)
    selected = torch.zeros(batch, num_slots, dtype=torch.bool, device=slots.device)
    high = min(args.max_steps, num_slots)
    low = min(args.min_steps, high)
    counts = torch.randint(low, high + 1, (batch,), device=slots.device)
    for step in range(high):
        action = torch.rand(batch, num_slots, device=slots.device).masked_fill(selected, -1).argmax(1)
        active = step < counts
        hidden, selected = model.update_with_action(hidden, selected, embeds, action, active)

    class_logits = model.classify(hidden)
    component_logits = model.classify_components(embeds)
    targets = fac.component_targets(labels, anchor_ids, evidence_ids, num_objects)
    image_component_logits = fac.mil_logits(component_logits, args.component_mil_temperature)
    if args.component_loss_mode == "positive_unknown":
        component_loss = -(F.logsigmoid(image_component_logits) * targets).sum() / targets.sum().clamp_min(1.0)
    else:
        component_loss = F.binary_cross_entropy_with_logits(image_component_logits, targets)
    pair_logits = fac.all_pair_logits(
        component_logits.sigmoid(), torch.ones_like(selected), anchor_ids, evidence_ids
    )
    class_loss = F.cross_entropy(class_logits, labels)
    pair_loss = F.cross_entropy(pair_logits, labels)
    total = (
        args.classification_coef * class_loss
        + args.component_mil_coef * component_loss
        + args.pair_ce_coef * pair_loss
    )
    return total, class_logits, class_loss.detach(), component_loss.detach(), pair_loss.detach()


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
    anchor_ids, evidence_ids = anchor_ids.to(device), evidence_ids.to(device)
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
        "selection_objective": "evidence-upweighted distinct-slot factorized pair@k",
    }
    (output / "experiment_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    best_score = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        seen = correct = 0
        loss_sums = {"aux": 0.0, "class": 0.0, "component": 0.0, "pair_ce": 0.0, "grpo": 0.0}
        for images, labels in train_loader:
            labels = labels.to(device)
            slots = fac.encode(backbone, images, device)

            auxiliary_optimizer.zero_grad(set_to_none=True)
            aux_loss, class_logits, class_loss, component_loss, pair_loss = auxiliary_loss(
                model, slots, labels, args, anchor_ids, evidence_ids, len(objects)
            )
            aux_loss.backward()
            torch.nn.utils.clip_grad_norm_(fac.ce_parameters(model), 1.0)
            auxiliary_optimizer.step()

            group = args.grpo_group_size
            repeated_slots = slots.repeat_interleave(group, 0)
            repeated_labels = labels.repeat_interleave(group, 0)
            result = fac.rollout(model, repeated_slots, repeated_labels, args, True, anchor_ids, evidence_ids)
            mask = result["mask"]
            advantage = (
                args.anchor_coef * fac.group_advantage(result["anchor"], mask, labels.numel(), group)
                + args.evidence_coef * fac.group_advantage(result["evidence"], mask, labels.numel(), group)
                + args.pair_coef * fac.group_advantage(result["pair"], mask, labels.numel(), group)
                + args.rule_margin_coef * fac.group_advantage(result["rule"], mask, labels.numel(), group)
            )
            grpo_loss = -(result["log_probs"] * advantage * mask).sum() / mask.sum().clamp_min(1.0)
            policy_optimizer.zero_grad(set_to_none=True)
            grpo_loss.backward()
            torch.nn.utils.clip_grad_norm_(fac.policy_parameters(model), 1.0)
            policy_optimizer.step()

            batch = labels.numel()
            seen += batch
            correct += class_logits.argmax(1).eq(labels).sum().item()
            for name, value in (
                ("aux", aux_loss), ("class", class_loss), ("component", component_loss),
                ("pair_ce", pair_loss), ("grpo", grpo_loss.detach()),
            ):
                loss_sums[name] += float(value.detach()) * batch

        model.eval()
        sums = {k: {name: 0.0 for name in ("pair", "pair_hit", "correct")} for k in (3, 4)}
        valid_total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                labels = labels.to(device)
                slots = fac.encode(backbone, images, device)
                metrics = fac.forced_metrics(model, slots, labels, (3, 4), anchor_ids, evidence_ids)
                valid_total += labels.numel()
                for k in (3, 4):
                    for name in sums[k]:
                        sums[k][name] += metrics[k][name].float().sum().item()
        score = 0.5 * (sums[3]["pair"] + sums[4]["pair"]) / valid_total
        losses = " ".join(f"{name}={value/max(seen,1):.4f}" for name, value in loss_sums.items())
        print(
            f"epoch={epoch} phase=joint train_acc={100*correct/max(seen,1):.2f} {losses} "
            f"valid_pair_score@3={sums[3]['pair']/valid_total:.4f} "
            f"valid_pair_score@4={sums[4]['pair']/valid_total:.4f} "
            f"valid_pair_hit@3={100*sums[3]['pair_hit']/valid_total:.2f} "
            f"valid_pair_hit@4={100*sums[4]['pair_hit']/valid_total:.2f} "
            f"valid_acc@3={100*sums[3]['correct']/valid_total:.2f} "
            f"valid_acc@4={100*sums[4]['correct']/valid_total:.2f}",
            flush=True,
        )
        if score > best_score:
            best_score = score
            torch.save({
                "epoch": epoch, "valid_pair_score": score, "model_state_dict": model.state_dict(),
                "config": asdict(config), "args": vars(args), "classes": classes,
                "objects": objects, "rules": rules,
            }, output / "selector_grpo_best.pt")
    print(f"checkpoint: {output / 'selector_grpo_best.pt'}", flush=True)


if __name__ == "__main__":
    main()
