#!/usr/bin/env python3
"""Actor-critic counterpart for the 6-class contrastive pair experiment."""

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
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import contrastive_core as fac
import train_grpo_selector as grpo
from misc_utils import seed_all
from selector_grpo import GRPOSelectorConfig, SlotSelectorGRPO
from train_grpo6_contrastive_pair import DATA, SA, auxiliary_loss
from train_slot_classifier import build_dataset, build_transforms, load_backbone, subset_dataset


class SlotSelectorAC(SlotSelectorGRPO):
    def __init__(self, cfg: GRPOSelectorConfig):
        super().__init__(cfg)
        self.value_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def value(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.value_head(hidden).squeeze(-1)


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
    parser.add_argument("--class_margin_coef", type=float, default=1.0)
    parser.add_argument("--object_a_coef", type=float, default=0.0)
    parser.add_argument("--object_b_coef", type=float, default=0.0)
    parser.add_argument("--component_pair_coef", type=float, default=1.0)
    parser.add_argument("--rank_discount", type=float, default=0.85)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.005)
    parser.add_argument("--checkpoint_acc_tolerance", type=float, default=0.02)
    parser.add_argument("--checkpoint_report_tolerances", default="0.01,0.015,0.02,0.03")
    parser.add_argument("--disable_return_norm", action="store_true")
    parser.add_argument("--disable_advantage_norm", action="store_true")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--quick_limit_train", type=int, default=0)
    parser.add_argument("--quick_limit_val", type=int, default=0)
    parser.add_argument("--require_cuda", action="store_true")
    return parser.parse_args()


def rollout_ac(model: SlotSelectorAC, slots: torch.Tensor, labels: torch.Tensor, args, object_a_ids, object_b_ids) -> dict[str, torch.Tensor]:
    slot_embeds = model.embed_slots(slots, None)
    component_prob = model.classify_components(slot_embeds).sigmoid().detach()
    batch, num_slots, _ = slot_embeds.shape
    hidden = model.initial_state(slot_embeds)
    selected = torch.zeros(batch, num_slots, dtype=torch.bool, device=slots.device)
    active = torch.ones(batch, dtype=torch.bool, device=slots.device)
    previous_a = slots.new_zeros(batch)
    previous_b = slots.new_zeros(batch)
    previous_pair_margin = fac.rule_pair_margin(
        component_prob, selected, labels, object_a_ids, object_b_ids
    )
    previous_margin = fac.true_logit_margin(model.classify(hidden), labels).detach()
    log_probs, entropies, values, masks, rewards = [], [], [], [], []
    class_logits = model.classify(hidden)
    for step in range(min(int(args.max_steps), num_slots)):
        values.append(model.value(hidden))
        policy_logits = model.slot_policy_logits(hidden.detach(), slot_embeds.detach(), selected, step)
        distribution = Categorical(logits=policy_logits)
        action = distribution.sample()
        log_probs.append(distribution.log_prob(action))
        entropies.append(distribution.entropy())
        masks.append(active.to(slots.dtype))
        hidden, selected = model.update_with_action(hidden, selected, slot_embeds, action, active)
        a, b, pair = fac.selected_support(component_prob, selected, labels, object_a_ids, object_b_ids)
        discount = float(args.rank_discount) ** step
        class_logits = model.classify(hidden)
        margin = fac.true_logit_margin(class_logits, labels).detach()
        pair_margin = fac.rule_pair_margin(
            component_prob, selected, labels, object_a_ids, object_b_ids
        )
        reward = (
            args.class_margin_coef * (margin - previous_margin)
            + args.object_a_coef * (a - previous_a)
            + args.object_b_coef * (b - previous_b)
            + args.component_pair_coef * (pair_margin - previous_pair_margin)
        ) * discount
        rewards.append(reward)
        previous_a, previous_b = a, b
        previous_pair_margin, previous_margin = pair_margin, margin
    stack = lambda values: torch.stack(values, dim=1)
    return {
        "logits": class_logits,
        "selected": selected,
        "log_probs": stack(log_probs),
        "entropies": stack(entropies),
        "values": stack(values),
        "mask": stack(masks),
        "reward": stack(rewards),
    }


def masked_standardize(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    mean = (values * mask).sum() / denom
    var = (((values - mean) * mask) ** 2).sum() / denom
    return (values - mean) / (var.sqrt() + eps)


def reward_to_go(rewards: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_rewards = rewards * mask
    return torch.flip(torch.cumsum(torch.flip(masked_rewards, dims=(1,)), dim=1), dims=(1,))


def main() -> None:
    args = parse_args()
    seed_all(args.seed, False)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
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
    model = SlotSelectorAC(config).to(device)
    auxiliary_optimizer = torch.optim.AdamW(fac.ce_parameters(model), lr=args.lr, weight_decay=1e-4)
    ac_params = [*fac.policy_parameters(model), *model.value_head.parameters()]
    ac_optimizer = torch.optim.AdamW(ac_params, lr=args.policy_lr, weight_decay=1e-4)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    meta = {
        "args": vars(args),
        "config": asdict(config),
        "classes": classes,
        "objects": objects,
        "rules": rules,
        "algorithm": "actor_critic",
        "standard": "same contrastive pair reward/forced top-k eval as GRPO6; group advantage replaced by learned value baseline",
        "ac_stabilization": "reward-to-go targets with masked return and advantage normalization by default",
        "checkpoint_selection": "highest valid_pair_margin@3 among epochs with valid_acc@3 within checkpoint_acc_tolerance of the best valid_acc@3",
    }
    (output / "experiment_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    checkpoint_candidates = []
    best_acc_seen = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        seen = correct = 0
        loss_sums = {"aux": 0.0, "class": 0.0, "component": 0.0, "pair_ce": 0.0, "policy": 0.0, "value": 0.0}
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
            rollout = rollout_ac(model, slots, labels, args, object_a_ids, object_b_ids)
            mask = rollout["mask"]
            denom = mask.sum().clamp_min(1.0)
            returns = reward_to_go(rollout["reward"], mask)
            value_targets = returns if args.disable_return_norm else masked_standardize(returns, mask)
            advantage = value_targets.detach() - rollout["values"].detach()
            if not args.disable_advantage_norm:
                advantage = masked_standardize(advantage, mask)
            policy_loss = -(rollout["log_probs"] * advantage * mask).sum() / denom
            value_loss = (((rollout["values"] - value_targets.detach()) ** 2) * mask).sum() / denom
            entropy = (rollout["entropies"] * mask).sum() / denom
            ac_loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
            ac_optimizer.zero_grad(set_to_none=True)
            ac_loss.backward()
            torch.nn.utils.clip_grad_norm_(ac_params, 1.0)
            ac_optimizer.step()
            batch = labels.numel()
            seen += batch
            correct += class_logits.argmax(1).eq(labels).sum().item()
            for name, value in (("aux", aux_loss), ("class", class_loss), ("component", component_loss), ("pair_ce", pair_ce_loss), ("policy", policy_loss), ("value", value_loss)):
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
                metrics = fac.forced_metrics(model, slots, labels, (3, 4), object_a_ids, object_b_ids)
                valid_total += labels.numel()
                valid_labels.append(labels.detach().cpu())
                for k in (3, 4):
                    for name in sums[k]:
                        sums[k][name] += metrics[k][name].float().sum().item()
                    valid_logits[k].append(metrics[k]["logits"].detach().cpu())
        labels_cat = torch.cat(valid_labels, dim=0)
        cls_metrics = {k: fac.multiclass_metrics(torch.cat(valid_logits[k], dim=0), labels_cat) for k in (3, 4)}
        valid_pair_margin_at3 = sums[3]["pair_margin"] / valid_total
        valid_pair_margin_at4 = sums[4]["pair_margin"] / valid_total
        valid_acc_at3 = sums[3]["correct"] / valid_total
        valid_acc_at4 = sums[4]["correct"] / valid_total
        losses = " ".join(f"{name}={value/max(seen,1):.4f}" for name, value in loss_sums.items())
        print(
            f"epoch={epoch} train_acc={100*correct/max(seen,1):.2f} {losses} "
            f"valid_pair_margin@3={valid_pair_margin_at3:.4f} "
            f"valid_pair_margin@4={valid_pair_margin_at4:.4f} "
            f"valid_acc@3={100*valid_acc_at3:.2f} valid_acc@4={100*valid_acc_at4:.2f} "
            f"valid_precision@3={cls_metrics[3]['precision']:.4f} valid_recall@3={cls_metrics[3]['recall']:.4f} "
            f"valid_f1@3={cls_metrics[3]['f1']:.4f} valid_auc@3={cls_metrics[3]['auc']:.4f} "
            f"valid_precision@4={cls_metrics[4]['precision']:.4f} valid_recall@4={cls_metrics[4]['recall']:.4f} "
            f"valid_f1@4={cls_metrics[4]['f1']:.4f} valid_auc@4={cls_metrics[4]['auc']:.4f}",
            flush=True,
        )
        best_acc_seen = max(best_acc_seen, valid_acc_at3)
        checkpoint_candidates.append({
            "epoch": epoch,
            "valid_acc": valid_acc_at3,
            "valid_pair_margin": valid_pair_margin_at3,
            "valid_pair_margin_at3": valid_pair_margin_at3,
            "valid_pair_margin_at4": valid_pair_margin_at4,
            "valid_acc_at3": valid_acc_at3,
            "valid_acc_at4": valid_acc_at4,
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "config": asdict(config),
            "args": vars(args),
            "classes": classes,
            "objects": objects,
            "rules": rules,
            "valid_metrics": cls_metrics,
        })
    def select_checkpoint(tolerance: float) -> tuple[dict, float]:
        acc_floor = best_acc_seen - tolerance
        eligible = [
            candidate for candidate in checkpoint_candidates
            if candidate["valid_acc_at3"] >= acc_floor
        ]
        return max(
            eligible,
            key=lambda candidate: (
                candidate["valid_pair_margin_at3"],
                candidate["valid_acc_at3"],
                candidate["valid_pair_margin_at4"],
            ),
        ), acc_floor

    best_checkpoint, acc_floor = select_checkpoint(args.checkpoint_acc_tolerance)
    best_checkpoint["best_valid_acc_at3"] = best_acc_seen
    best_checkpoint["checkpoint_acc_floor"] = acc_floor
    best_checkpoint["checkpoint_acc_tolerance"] = args.checkpoint_acc_tolerance
    torch.save(best_checkpoint, output / "selector_ac_best.pt")
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
                key: candidate[key]
                for key in (
                    "epoch",
                    "valid_acc_at3",
                    "valid_acc_at4",
                    "valid_pair_margin_at3",
                    "valid_pair_margin_at4",
                )
            }
            for candidate in checkpoint_candidates
        ],
        "selections": {},
    }
    for tolerance in report_tolerances:
        selected, floor = select_checkpoint(tolerance)
        tag = str(tolerance).replace(".", "p")
        selected_path = output / f"selector_ac_best_tol{tag}.pt"
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
    (output / "checkpoint_selection_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"checkpoint: {output / 'selector_ac_best.pt'}", flush=True)


if __name__ == "__main__":
    main()
