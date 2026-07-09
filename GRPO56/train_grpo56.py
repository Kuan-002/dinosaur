#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "SET56") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "SET56"))

os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".cache" / "torch"))

import torch
import torch.nn as nn
import torch.nn.functional as F

import train_grpo_selector as grpo
from misc_utils import seed_all
from selector_grpo import GRPOSelectorConfig, SlotSelectorGRPO
from settransformer.model import DiscriminativeSetTransformer, ProbeConfig
from structured56 import load_structured56, project_slots56
from train_slot_classifier import build_dataset, build_transforms, load_backbone, subset_dataset


DEFAULT_DATA = "/vol/biomedic3/kw1025/dinosaur/analysis/coco_top2_clean_scenes_anchor009_evidence005_10cls_450_150_150/classification_dataset"
DEFAULT_SA = "/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt"
DEFAULT_STRUCTURED56 = "/vol/biomedic3/kw1025/dinosaur/analysis/structured_slot_bottleneck/structured_slot_bottleneck_10cls450_20260708_011844/structured_slot_bottleneck_best.pt"


def class_names_from_dataset(dataset) -> list[str]:
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return list(getattr(dataset, "classes", []))


def load_reward_probe(path: str, device: torch.device) -> DiscriminativeSetTransformer:
    ckpt: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    cfg = ProbeConfig(**ckpt["probe_config"])
    probe = DiscriminativeSetTransformer(cfg).to(device)
    probe.load_state_dict(ckpt["model_state_dict"], strict=True)
    probe.eval()
    for param in probe.parameters():
        param.requires_grad = False
    return probe


@torch.no_grad()
def encode_slots56(backbone, projector, images: torch.Tensor, device: torch.device, mode: str) -> torch.Tensor:
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.forward_dino(images)
    features = backbone.mlp(features)
    slots, _, _ = backbone.slot_attention(features)
    return project_slots56(projector, slots.detach(), mode=mode)


def run_epoch(
    model: SlotSelectorGRPO,
    backbone,
    projector,
    loader,
    device: torch.device,
    ce_optimizer,
    policy_optimizer,
    args: argparse.Namespace,
    train: bool,
    reward_probe: DiscriminativeSetTransformer,
) -> dict[str, float]:
    model.train(train)
    total = 0
    loss_sum = 0.0
    correct = 0
    count_sum = 0.0
    reward_sum = 0.0
    prob_sum = 0.0
    for images, labels in loader:
        labels = labels.to(device, non_blocking=device.type == "cuda")
        slots = encode_slots56(backbone, projector, images, device, args.structured_mode)
        slot_pos = None
        metric_labels = labels
        if train and args.warmup_epochs_remaining > 0:
            out = grpo.forced_warmup_rollout(model, slots, slot_pos, args)
            loss = F.cross_entropy(out["logits"], labels)
        elif train:
            out, cls_loss, policy_loss, entropy_loss = grpo.grpo_loss(
                model,
                slots,
                slot_pos,
                labels,
                args,
                reward_probe=reward_probe,
            )
            loss = cls_loss + args.policy_coef * policy_loss + args.entropy_coef * entropy_loss
            metric_labels = labels.repeat_interleave(int(args.grpo_group_size))
        else:
            with torch.no_grad():
                out = grpo.rollout(model, slots, slot_pos, labels, args, sample=False, reward_probe=reward_probe)
                loss = F.cross_entropy(out["logits"], labels)
        if train:
            ce_optimizer.zero_grad(set_to_none=True)
            policy_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            ce_optimizer.step()
            policy_optimizer.step()

        with torch.no_grad():
            probs = out["logits"].softmax(dim=1).gather(1, metric_labels[:, None]).squeeze(1)
        batch = metric_labels.numel()
        total += batch
        loss_sum += float(loss.detach().cpu()) * batch
        correct += int((out["logits"].argmax(dim=1) == metric_labels).sum().item())
        count_sum += float(out["selected_count"].sum().detach().cpu())
        reward_sum += float(out.get("terminal_reward", torch.zeros(batch, device=device)).sum().detach().cpu())
        prob_sum += float(probs.sum().detach().cpu())
    return {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "avg_selected": count_sum / max(total, 1),
        "avg_reward": reward_sum / max(total, 1),
        "true_prob": prob_sum / max(total, 1),
    }


def write_history(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RNN-GRPO selector over 56-dim structured slots.")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--sa_checkpoint", default=DEFAULT_SA)
    parser.add_argument("--structured_checkpoint", default=DEFAULT_STRUCTURED56)
    parser.add_argument("--structured_mode", choices=["u", "obj", "geo", "res", "obj_geo", "obj_res", "geo_res"], default="u")
    parser.add_argument("--reward_probe_checkpoint", required=True)
    parser.add_argument("--output_dir", default="GRPO56/checkpoints/grpo56")
    parser.add_argument("--input_res", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--policy_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--min_steps", type=int, default=2)
    parser.add_argument("--early_exit_conf", type=float, default=0.9)
    parser.add_argument("--disable_confidence_early_exit", action="store_true", default=True)
    parser.add_argument("--enable_confidence_early_exit", dest="disable_confidence_early_exit", action="store_false")
    parser.add_argument("--confidence_early_exit_min_steps", type=int, default=2)
    parser.add_argument("--decoupled_stop_policy", action="store_true")
    parser.add_argument("--reward_source", choices=["probe_subset_margin", "probe_margin", "classifier"], default="probe_subset_margin")
    parser.add_argument("--probe_selected_weight", type=float, default=1.0)
    parser.add_argument("--probe_necessity_weight", type=float, default=1.0)
    parser.add_argument("--probe_reward_clip", type=float, default=10.0)
    parser.add_argument("--subset_contrast", choices=["none", "complement"], default="complement")
    parser.add_argument("--subset_contrast_weight", type=float, default=1.0)
    parser.add_argument("--step_margin_reward_weights", default="0.6,0.3,0.15")
    parser.add_argument("--grpo_group_size", type=int, default=4)
    parser.add_argument("--grpo_adv_eps", type=float, default=1e-4)
    parser.add_argument("--free_slots", type=int, default=3)
    parser.add_argument("--count_penalty", type=float, default=0.35)
    parser.add_argument("--cls_reward_negative", type=float, default=0.0)
    parser.add_argument("--policy_coef", type=float, default=1.0)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--quick_limit_train", type=int, default=0)
    parser.add_argument("--quick_limit_val", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.step_margin_reward_weights = grpo.parse_float_list(args.step_margin_reward_weights)
    args.pos_dim = 0
    args.first_step_cross_attention = False
    args.first_step_num_heads = 4
    args.warmup_policy = "random"
    if args.grpo_group_size <= 1:
        raise ValueError("--grpo_group_size must be greater than 1 for GRPO")

    seed_all(args.seed, False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tfm = build_transforms(args.input_res)
    train_set = subset_dataset(build_dataset(args.data, "train", tfm["train"]), args.quick_limit_train, args.seed)
    valid_set = subset_dataset(build_dataset(args.data, "valid", tfm["valid"]), args.quick_limit_val, args.seed)
    classes = class_names_from_dataset(train_set)
    if not classes:
        raise RuntimeError("Could not infer class names from dataset.")
    train_loader = grpo.make_loader(train_set, args, device, shuffle=True)
    valid_loader = grpo.make_loader(valid_set, args, device, shuffle=False)

    backbone = load_backbone(args.sa_checkpoint, device)
    backbone.eval()
    backbone.requires_grad_(False)
    projector, structured_ckpt = load_structured56(args.structured_checkpoint, device)
    reward_probe = load_reward_probe(args.reward_probe_checkpoint, device)
    if reward_probe.cfg.num_slots != backbone.num_slots:
        raise ValueError(f"Reward probe slots={reward_probe.cfg.num_slots}, backbone slots={backbone.num_slots}")
    if reward_probe.cfg.slot_dim != projector.cfg.out_dim and args.structured_mode == "u":
        raise ValueError(f"Reward probe slot_dim={reward_probe.cfg.slot_dim}, structured56 dim={projector.cfg.out_dim}")
    slot_dim = int(reward_probe.cfg.slot_dim)
    if args.max_steps <= 0:
        args.max_steps = backbone.num_slots

    cfg = GRPOSelectorConfig(
        num_slots=backbone.num_slots,
        slot_dim=slot_dim,
        num_classes=len(classes),
        hidden_dim=args.hidden_dim,
        policy_dim=args.policy_dim,
        max_steps=args.max_steps,
        dropout=args.dropout,
        pos_dim=0,
        min_steps=args.min_steps,
        early_exit_conf=args.early_exit_conf,
        decoupled_stop_policy=args.decoupled_stop_policy,
        first_step_cross_attention=False,
        first_step_num_heads=4,
    )
    model = SlotSelectorGRPO(cfg).to(device)
    ce_params = grpo.ce_parameters(model)
    policy_params = grpo.policy_parameters(model)
    ce_ids = {id(param) for param in ce_params}
    policy_ids = {id(param) for param in policy_params}
    if ce_ids & policy_ids:
        raise ValueError("CE and policy parameter groups overlap")
    missing = {id(param) for param in model.parameters()} - ce_ids - policy_ids
    if missing:
        raise ValueError(f"Some parameters are not assigned to optimizers: {len(missing)}")
    ce_optimizer = torch.optim.AdamW(ce_params, lr=args.lr, weight_decay=args.wd)
    policy_optimizer = torch.optim.AdamW(policy_params, lr=args.lr, weight_decay=args.wd)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "args": vars(args),
        "grpo_config": asdict(cfg),
        "classes": classes,
        "slot_embedding_source": "structured56",
        "structured56_config": structured_ckpt["config"],
        "reward_probe_config": asdict(reward_probe.cfg),
    }
    (out_dir / "selector_grpo_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"device={device} train={len(train_set)} valid={len(valid_set)} classes={len(classes)} slot_dim={slot_dim}")
    print(f"reward_probe={args.reward_probe_checkpoint}")

    best_acc = -1.0
    history: list[dict] = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        args.warmup_epochs_remaining = max(args.warmup_epochs - epoch + 1, 0)
        train_metrics = run_epoch(model, backbone, projector, train_loader, device, ce_optimizer, policy_optimizer, args, True, reward_probe)
        args.warmup_epochs_remaining = 0
        with torch.no_grad():
            valid_metrics = run_epoch(model, backbone, projector, valid_loader, device, ce_optimizer, policy_optimizer, args, False, reward_probe)
        row = {
            "epoch": epoch,
            "elapsed": time.strftime("%H:%M:%S", time.gmtime(time.time() - start)),
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
        }
        history.append(row)
        write_history(out_dir / "history_metrics.csv", history)
        print(
            f"epoch={epoch} elapsed={row['elapsed']} "
            f"train_acc={100*row['train_acc']:.2f} train_slots={row['train_avg_selected']:.2f} "
            f"valid_acc={100*row['valid_acc']:.2f} valid_slots={row['valid_avg_selected']:.2f} "
            f"valid_prob={row['valid_true_prob']:.3f} reward={row['valid_avg_reward']:.3f}"
        )
        if valid_metrics["acc"] > best_acc:
            best_acc = valid_metrics["acc"]
            torch.save(
                {
                    "epoch": epoch,
                    "valid_acc": best_acc,
                    "model_state_dict": model.state_dict(),
                    "ce_optimizer_state_dict": ce_optimizer.state_dict(),
                    "policy_optimizer_state_dict": policy_optimizer.state_dict(),
                },
                out_dir / "selector_grpo_best.pt",
            )
            print(f"saved best selector: {out_dir / 'selector_grpo_best.pt'}")
    (out_dir / "final_metrics.json").write_text(json.dumps({"best_valid_acc": best_acc, "history": history}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
