#!/usr/bin/env python3
"""Joint undirected factorized-MIL + GRPO training for the 6-class pair dataset."""

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

import factorized_core as fac
import train_grpo_selector as grpo
from misc_utils import seed_all
from selector_grpo import GRPOSelectorConfig, SlotSelectorGRPO
from train_slot_classifier import build_dataset, build_transforms, load_backbone, subset_dataset


DATA = ROOT / "dataset/coco_compositional_pair6_clean_300_100_100/classification_dataset"
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
    parser.add_argument("--class_margin_coef", type=float, default=1.0)
    parser.add_argument("--object_a_coef", type=float, default=0.0)
    parser.add_argument("--object_b_coef", type=float, default=0.0)
    parser.add_argument("--component_pair_coef", type=float, default=1.0)
    parser.add_argument("--rank_discount", type=float, default=0.85)
    parser.add_argument("--grpo_group_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--quick_limit_train", type=int, default=0)
    parser.add_argument("--quick_limit_val", type=int, default=0)
    parser.add_argument("--require_cuda", action="store_true")
    return parser.parse_args()


def auxiliary_loss(model, slots, labels, args, object_a_ids, object_b_ids, num_objects):
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
    targets = fac.component_targets(labels, object_a_ids, object_b_ids, num_objects)
    image_component_logits = fac.mil_logits(slot_component_logits, args.component_mil_temperature)
    if args.component_loss_mode == "positive_unknown":
        component_loss = -(F.logsigmoid(image_component_logits) * targets).sum() / targets.sum().clamp_min(1.0)
    else:
        component_loss = F.binary_cross_entropy_with_logits(image_component_logits, targets)
    rule_pair_logits = fac.rule_pair_logits(
        slot_component_logits.sigmoid(), torch.ones_like(selected_mask), object_a_ids, object_b_ids
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
    objects, object_a_ids, object_b_ids, rules = fac.load_rules(args.data, classes)
    object_a_ids = object_a_ids.to(device)
    object_b_ids = object_b_ids.to(device)
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
        "training_schedule": "joint auxiliary + GRPO from epoch 1; no warmup/freeze",
        "selection_objective": "class-margin GRPO plus optional object-a/object-b dense increments and undirected component-pair increment",
        "checkpoint_selection": "best valid_acc@3; valid_mean_acc and pair metrics are logged only",
        "class_margin_reward": "true-class logit-margin increment",
        "object_a_reward": "single-component support increment for object_a",
        "object_b_reward": "single-component support increment for object_b",
        "component_pair_reward": "raw undirected pair-support increment without completion bonus",
    }
    (output / "experiment_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    best_acc = -1.0
    best_mean_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        seen = correct = 0
        loss_sums = {"aux": 0.0, "class": 0.0, "component": 0.0, "pair_ce": 0.0, "grpo": 0.0}
        for images, labels in train_loader:
            labels = labels.to(device)
            slots = fac.encode(backbone, images, device)

            auxiliary_optimizer.zero_grad(set_to_none=True)
            aux_loss, class_logits, class_loss, component_loss, pair_ce_loss = auxiliary_loss(
                model, slots, labels, args, object_a_ids, object_b_ids, len(objects)
            )
            aux_loss.backward()
            torch.nn.utils.clip_grad_norm_(fac.ce_parameters(model), 1.0)
            auxiliary_optimizer.step()

            group = args.grpo_group_size
            repeated_slots = slots.repeat_interleave(group, 0)
            repeated_labels = labels.repeat_interleave(group, 0)
            rollout = fac.rollout(model, repeated_slots, repeated_labels, args, True, object_a_ids, object_b_ids)
            active_mask = rollout["mask"]
            advantage = (
                args.class_margin_coef * fac.group_advantage(rollout["class_margin"], active_mask, labels.numel(), group)
                + args.object_a_coef * fac.group_advantage(rollout["object_a"], active_mask, labels.numel(), group)
                + args.object_b_coef * fac.group_advantage(rollout["object_b"], active_mask, labels.numel(), group)
                + args.component_pair_coef * fac.group_advantage(rollout["component_pair"], active_mask, labels.numel(), group)
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
        sums = {k: {name: 0.0 for name in ("pair", "pair_hit", "correct")} for k in (3, 4)}
        valid_total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                labels = labels.to(device)
                slots = fac.encode(backbone, images, device)
                metrics = fac.forced_metrics(model, slots, labels, (3, 4), object_a_ids, object_b_ids)
                valid_total += labels.numel()
                for k in (3, 4):
                    for name in sums[k]:
                        sums[k][name] += metrics[k][name].float().sum().item()
        valid_pair_score = 0.5 * (sums[3]["pair"] + sums[4]["pair"]) / valid_total
        valid_acc_at3 = sums[3]["correct"] / valid_total
        valid_acc_at4 = sums[4]["correct"] / valid_total
        valid_mean_acc = 0.5 * (valid_acc_at3 + valid_acc_at4)
        losses = " ".join(f"{name}={value/max(seen,1):.4f}" for name, value in loss_sums.items())
        print(
            f"epoch={epoch} phase=joint train_acc={100*correct/max(seen,1):.2f} {losses} "
            f"class_margin_coef={args.class_margin_coef:.3g} "
            f"object_a_coef={args.object_a_coef:.3g} "
            f"object_b_coef={args.object_b_coef:.3g} "
            f"component_pair_coef={args.component_pair_coef:.3g} "
            f"valid_pair_score@3={sums[3]['pair']/valid_total:.4f} "
            f"valid_pair_score@4={sums[4]['pair']/valid_total:.4f} "
            f"valid_pair_hit@3={100*sums[3]['pair_hit']/valid_total:.2f} "
            f"valid_pair_hit@4={100*sums[4]['pair_hit']/valid_total:.2f} "
            f"valid_acc@3={100*valid_acc_at3:.2f} "
            f"valid_acc@4={100*valid_acc_at4:.2f} "
            f"valid_mean_acc={100*valid_mean_acc:.2f} "
            f"valid_selection_acc={100*valid_acc_at3:.2f}",
            flush=True,
        )
        if valid_acc_at3 > best_acc or (valid_acc_at3 == best_acc and valid_mean_acc > best_mean_acc):
            best_acc = valid_acc_at3
            best_mean_acc = valid_mean_acc
            torch.save({
                "epoch": epoch, "valid_acc": best_acc, "valid_mean_acc": best_mean_acc,
                "valid_pair_score": valid_pair_score, "model_state_dict": model.state_dict(),
                "config": asdict(config), "args": vars(args), "classes": classes,
                "objects": objects, "rules": rules,
            }, output / "selector_grpo_best.pt")
    print(f"checkpoint: {output / 'selector_grpo_best.pt'}", flush=True)


if __name__ == "__main__":
    main()
