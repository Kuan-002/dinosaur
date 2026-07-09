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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "SET80") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "SET80"))

os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".cache" / "torch"))

import torch
import torch.nn as nn
import torch.nn.functional as F

import train_grpo_selector as grpo
from misc_utils import seed_all
from selector_grpo import GRPOSelectorConfig, SlotSelectorGRPO
from structured80 import load_structured80, project_slots80
from train_slot_classifier import build_dataset, build_transforms, load_backbone, subset_dataset


DEFAULT_DATA = "/vol/biomedic3/kw1025/dinosaur/analysis/coco_top2_clean_scenes_anchor009_evidence005_10cls_450_150_150/classification_dataset"
DEFAULT_SA = "/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt"
DEFAULT_STRUCTURED80 = "/vol/biomedic3/kw1025/dinosaur/SET80/checkpoints/structured80_obj16_geo16_res48_20260708_185728/structured_slot_bottleneck_best.pt"


def class_names_from_dataset(dataset) -> list[str]:
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return list(getattr(dataset, "classes", []))


@torch.no_grad()
def encode_slots80(backbone, projector, images: torch.Tensor, device: torch.device, mode: str) -> torch.Tensor:
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.forward_dino(images)
    features = backbone.mlp(features)
    slots, _, _ = backbone.slot_attention(features)
    return project_slots80(projector, slots.detach(), mode=mode)


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
        slots = encode_slots80(backbone, projector, images, device, args.structured_mode)
        slot_pos = None
        metric_labels = labels
        if train and args.warmup_epochs_remaining > 0:
            out = grpo.forced_warmup_rollout(model, slots, slot_pos, args)
            loss = F.cross_entropy(out["logits"], labels, label_smoothing=args.label_smoothing)
        elif train:
            out, cls_loss, policy_loss, entropy_loss = grpo.grpo_loss(
                model,
                slots,
                slot_pos,
                labels,
                args,
                reward_probe=None,
            )
            loss = cls_loss + args.policy_coef * policy_loss + args.entropy_coef * entropy_loss
            metric_labels = labels.repeat_interleave(int(args.grpo_group_size))
        else:
            with torch.no_grad():
                out = grpo.rollout(model, slots, slot_pos, labels, args, sample=False, reward_probe=None)
                loss = F.cross_entropy(out["logits"], labels, label_smoothing=args.label_smoothing)
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
    parser = argparse.ArgumentParser(description="Train RNN-GRPO selector over 80-dim structured slothead features.")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--sa_checkpoint", default=DEFAULT_SA)
    parser.add_argument("--structured_checkpoint", default=DEFAULT_STRUCTURED80)
    parser.add_argument("--structured_mode", choices=["u", "obj", "geo", "res", "obj_geo", "obj_res", "geo_res"], default="u")
    parser.add_argument("--output_dir", default="GRPO80/checkpoints/grpo80")
    parser.add_argument("--input_res", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--policy_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--min_steps", type=int, default=2)
    parser.add_argument("--early_exit_conf", type=float, default=0.9)
    parser.add_argument("--disable_confidence_early_exit", action="store_true", default=True)
    parser.add_argument("--enable_confidence_early_exit", dest="disable_confidence_early_exit", action="store_false")
    parser.add_argument("--confidence_early_exit_min_steps", type=int, default=3)
    parser.add_argument("--decoupled_stop_policy", action="store_true", default=True)
    parser.add_argument("--coupled_stop_policy", dest="decoupled_stop_policy", action="store_false")
    parser.add_argument("--policy_context_attention", action="store_true", default=True)
    parser.add_argument("--no_policy_context_attention", dest="policy_context_attention", action="store_false")
    parser.add_argument("--first_step_num_heads", type=int, default=4)
    parser.add_argument("--reward_source", choices=["classifier", "classifier_logprob", "classifier_margin"], default="classifier_logprob")
    parser.add_argument("--grpo_group_size", type=int, default=4)
    parser.add_argument("--grpo_adv_eps", type=float, default=1e-4)
    parser.add_argument("--free_slots", type=int, default=4)
    parser.add_argument("--min_free_slots", type=int, default=3)
    parser.add_argument("--max_free_slots", type=int, default=4)
    parser.add_argument("--count_penalty", type=float, default=0.08)
    parser.add_argument("--cls_reward_negative", type=float, default=0.0)
    parser.add_argument("--policy_coef", type=float, default=1.0)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--confidence_penalty", type=float, default=0.03)
    parser.add_argument("--confidence_penalty_threshold", type=float, default=0.85)
    parser.add_argument("--early_confidence_penalty_until_slots", type=int, default=3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--quick_limit_train", type=int, default=0)
    parser.add_argument("--quick_limit_val", type=int, default=0)
    args = parser.parse_args()
    args.pos_dim = 0
    args.first_step_cross_attention = False
    args.warmup_policy = "random"
    args.step_margin_reward_weights = ()
    args.subset_contrast = "none"
    args.probe_selected_weight = 0.0
    args.probe_necessity_weight = 0.0
    args.probe_reward_clip = 0.0
    if args.grpo_group_size <= 1:
        raise ValueError("--grpo_group_size must be greater than 1 for GRPO")
    return args


def main() -> None:
    args = parse_args()
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
    projector, structured_ckpt = load_structured80(args.structured_checkpoint, device)
    if args.max_steps <= 0:
        args.max_steps = backbone.num_slots
    slot_dim = int(projector.cfg.out_dim if args.structured_mode == "u" else encode_slots80(backbone, projector, next(iter(valid_loader))[0], device, args.structured_mode).size(-1))

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
        first_step_num_heads=args.first_step_num_heads,
        policy_context_attention=args.policy_context_attention,
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
        "slot_embedding_source": "structured80_slothead",
        "structured80_config": structured_ckpt["config"],
        "design_note": "No SetTransformer reward probe or bbox supervision is used. Policy can attend over all slothead features; classifier state is built only from selected slots.",
    }
    (out_dir / "selector_grpo_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"device={device} train={len(train_set)} valid={len(valid_set)} classes={len(classes)} slot_dim={slot_dim}")
    print(f"structured80={args.structured_checkpoint} mode={args.structured_mode}")

    best_acc = -1.0
    history: list[dict] = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        args.warmup_epochs_remaining = max(args.warmup_epochs - epoch + 1, 0)
        train_metrics = run_epoch(model, backbone, projector, train_loader, device, ce_optimizer, policy_optimizer, args, True)
        args.warmup_epochs_remaining = 0
        with torch.no_grad():
            valid_metrics = run_epoch(model, backbone, projector, valid_loader, device, ce_optimizer, policy_optimizer, args, False)
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
