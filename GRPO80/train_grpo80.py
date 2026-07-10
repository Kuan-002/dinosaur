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
from torch.distributions import Bernoulli, Categorical

import train_grpo_selector as grpo
from misc_utils import seed_all
from selector_grpo import GRPOSelectorConfig, SlotSelectorGRPO
from slothead80 import load_slothead80, project_slots80
from train_slot_classifier import build_dataset, build_transforms, load_backbone, subset_dataset


DEFAULT_DATA = "/vol/biomedic3/kw1025/dinosaur/dataset/coco_top2_clean10_area006_004_600_200_200/classification_dataset"
DEFAULT_SA = "/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt"
DEFAULT_SLOTHEAD80 = "/vol/biomedic3/kw1025/dinosaur/SET80/checkpoints/slothead80_obj16_geo16_res48_20260709_192454/slothead_best.pt"


def class_names_from_dataset(dataset) -> list[str]:
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return list(getattr(dataset, "classes", []))


@torch.no_grad()
def encode_slots80(backbone, projector, images: torch.Tensor, device: torch.device, mode: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.forward_dino(images)
    features = backbone.mlp(features)
    slots, _, _ = backbone.slot_attention(features)
    raw_slots = slots.detach()
    reward_features = projector.reward_features(raw_slots)
    return project_slots80(projector, raw_slots, mode=mode), reward_features


def geometry_box_masks(geometry: torch.Tensor, grid_size: int) -> torch.Tensor:
    b, k, _ = geometry.shape
    if grid_size <= 0:
        raise ValueError("--geometry_grid_size must be positive")
    centers = (torch.arange(grid_size, device=geometry.device, dtype=geometry.dtype) + 0.5) / float(grid_size)
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    x = xx.reshape(1, 1, grid_size, grid_size)
    y = yy.reshape(1, 1, grid_size, grid_size)
    cx = geometry[..., 0].clamp(0.0, 1.0).view(b, k, 1, 1)
    cy = geometry[..., 1].clamp(0.0, 1.0).view(b, k, 1, 1)
    w = geometry[..., 2].clamp_min(0.0).clamp_max(1.0).view(b, k, 1, 1)
    h = geometry[..., 3].clamp_min(0.0).clamp_max(1.0).view(b, k, 1, 1)
    return (x >= cx - 0.5 * w) & (x <= cx + 0.5 * w) & (y >= cy - 0.5 * h) & (y <= cy + 0.5 * h)


def objectness_mass_reward(objectness: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    selected_f = selected.to(objectness.dtype)
    return (objectness * selected_f).sum(dim=1) / objectness.sum(dim=1).clamp_min(1e-6)


def geometry_novelty_reward(
    geometry: torch.Tensor,
    objectness: torch.Tensor,
    actions: torch.Tensor,
    stop_idx: int,
    grid_size: int,
) -> torch.Tensor:
    b, k, _ = geometry.shape
    box_masks = geometry_box_masks(geometry, grid_size)
    box_area = box_masks.to(objectness.dtype).mean(dim=(2, 3))
    denom = (objectness * box_area).sum(dim=1).clamp_min(1e-6)
    union = torch.zeros(b, grid_size, grid_size, dtype=torch.bool, device=geometry.device)
    reward = objectness.new_zeros(b)
    batch_idx = torch.arange(b, device=geometry.device)
    for step in range(actions.size(1)):
        action = actions[:, step]
        valid = action != stop_idx
        slot_idx = action.clamp(min=0, max=k - 1)
        slot_mask = box_masks[batch_idx, slot_idx]
        incremental = (slot_mask & ~union).to(objectness.dtype).mean(dim=(1, 2))
        slot_obj = objectness[batch_idx, slot_idx]
        reward = reward + valid.to(objectness.dtype) * slot_obj * incremental
        union = union | (slot_mask & valid.view(b, 1, 1))
    return (reward / denom).clamp(0.0, 1.0)


def residual_novelty_reward(
    u_res: torch.Tensor,
    objectness: torch.Tensor,
    actions: torch.Tensor,
    stop_idx: int,
) -> torch.Tensor:
    b, k, _ = u_res.shape
    res_norm = u_res.norm(dim=-1)
    res_unit = F.normalize(u_res, dim=-1, eps=1e-6)
    denom = (objectness * res_norm).sum(dim=1).clamp_min(1e-6)
    selected = torch.zeros(b, k, dtype=torch.bool, device=u_res.device)
    reward = objectness.new_zeros(b)
    batch_idx = torch.arange(b, device=u_res.device)
    for step in range(actions.size(1)):
        action = actions[:, step]
        valid = action != stop_idx
        slot_idx = action.clamp(min=0, max=k - 1)
        unit = res_unit[batch_idx, slot_idx]
        cosine = (res_unit * unit[:, None, :]).sum(dim=-1)
        masked_cosine = cosine.masked_fill(~selected, -1.0)
        max_prev = masked_cosine.max(dim=1).values
        has_prev = selected.any(dim=1)
        novelty = torch.where(has_prev, (1.0 - max_prev).clamp(0.0, 1.0), torch.ones_like(max_prev))
        value = objectness[batch_idx, slot_idx] * res_norm[batch_idx, slot_idx]
        reward = reward + valid.to(objectness.dtype) * value * novelty
        selected[batch_idx, slot_idx] = selected[batch_idx, slot_idx] | valid
    return (reward / denom).clamp(0.0, 1.0)


def slothead_terminal_reward(
    selected: torch.Tensor,
    reward_features: dict[str, torch.Tensor],
    args: argparse.Namespace,
    stopped: torch.Tensor,
    actions: torch.Tensor,
    stop_idx: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    objectness = reward_features["objectness"].to(dtype=reward_features["u"].dtype)
    geometry = reward_features["geometry"].to(dtype=reward_features["u"].dtype)
    u_res = reward_features["u_res"].to(dtype=reward_features["u"].dtype)
    selected_f = selected.to(objectness.dtype)
    count = selected_f.sum(dim=1)
    obj_reward = objectness_mass_reward(objectness, selected)
    geo_reward = geometry_novelty_reward(geometry, objectness, actions, stop_idx, int(args.geometry_grid_size))
    res_reward = residual_novelty_reward(u_res, objectness, actions, stop_idx)
    lo = float(args.min_free_slots if args.min_free_slots > 0 else args.free_slots)
    hi = float(args.max_free_slots if args.max_free_slots > 0 else args.free_slots)
    count_ok = (count >= lo) & (count <= hi)
    stop_quality_ok = (obj_reward >= float(args.good_stop_obj_threshold)) & (res_reward >= float(args.good_stop_res_threshold))
    good_stop = (stopped & count_ok & stop_quality_ok).to(objectness.dtype)
    over_selected = (count - hi).clamp_min(0.0)
    reward = (
        float(args.objectness_coef) * obj_reward
        + float(args.geometry_coef) * geo_reward
        + float(args.residual_coef) * res_reward
        - float(args.selected_count_coef) * count
        - float(args.over_select_coef) * over_selected
        + float(args.good_stop_bonus) * good_stop
    )
    parts = {
        "objectness_mass": obj_reward.detach(),
        "geometry_novelty": geo_reward.detach(),
        "residual_novelty": res_reward.detach(),
        "good_stop": good_stop.detach(),
        "selected_count": count.detach(),
        "over_selected": over_selected.detach(),
    }
    return reward.detach(), parts


def rollout_slothead_reward(
    model: SlotSelectorGRPO,
    slots: torch.Tensor,
    reward_features: dict[str, torch.Tensor],
    args: argparse.Namespace,
    sample: bool,
) -> dict[str, torch.Tensor]:
    slot_embeds = model.embed_slots(slots, None)
    b, k, _ = slot_embeds.shape
    h = model.initial_state(slot_embeds)
    selected = torch.zeros(b, k, dtype=torch.bool, device=slots.device)
    active = torch.ones(b, dtype=torch.bool, device=slots.device)
    stopped = torch.zeros(b, dtype=torch.bool, device=slots.device)
    log_probs = []
    entropies = []
    actions = []
    masks = []
    min_steps = int(args.min_steps)

    for step in range(min(args.max_steps, k)):
        can_stop = (selected.sum(dim=1) >= min_steps) | ~active
        if model.cfg.decoupled_stop_policy:
            stop_logits = model.stop_head(h.detach()).squeeze(-1)
            stop_dist = Bernoulli(logits=stop_logits)
            raw_stop = stop_dist.sample().bool() if sample else (torch.sigmoid(stop_logits) >= 0.5)
            is_stop = active & can_stop & raw_stop
            slot_logits = model.slot_policy_logits(h.detach(), slot_embeds.detach(), selected, step)
            slot_dist = Categorical(logits=slot_logits)
            slot_action = slot_dist.sample() if sample else slot_logits.argmax(dim=-1)
            action = torch.where(is_stop, torch.full_like(slot_action, model.stop_idx), slot_action)
            action = torch.where(active, action, torch.full_like(action, model.stop_idx))
            active_f = active.to(slots.dtype)
            step_log_prob = stop_dist.log_prob(is_stop.to(dtype=stop_logits.dtype)) * active_f
            step_log_prob = step_log_prob + slot_dist.log_prob(slot_action) * (active & ~is_stop).to(slots.dtype)
            step_entropy = (stop_dist.entropy() + slot_dist.entropy() * (~is_stop).to(slots.dtype)) * active_f
        else:
            logits = model.policy_logits(h.detach(), slot_embeds.detach(), selected, step)
            logits[:, model.stop_idx] = torch.where(can_stop, logits[:, model.stop_idx], torch.full_like(logits[:, model.stop_idx], torch.finfo(logits.dtype).min))
            dist = Categorical(logits=logits)
            action = dist.sample() if sample else logits.argmax(dim=-1)
            action = torch.where(active, action, torch.full_like(action, model.stop_idx))
            is_stop = action == model.stop_idx
            active_f = active.to(slots.dtype)
            step_log_prob = dist.log_prob(action) * active_f
            step_entropy = dist.entropy() * active_f

        is_stop = action == model.stop_idx
        do_select = active & ~is_stop
        log_probs.append(step_log_prob)
        entropies.append(step_entropy)
        actions.append(action)
        masks.append(active_f)
        h, selected = model.update_with_action(h, selected, slot_embeds, action, do_select)
        stopped = stopped | (active & is_stop)
        active = active & ~is_stop
        if not active.any():
            break

    if actions:
        action_t = torch.stack(actions, dim=1)
        log_prob_t = torch.stack(log_probs, dim=1)
        entropy_t = torch.stack(entropies, dim=1)
        mask_t = torch.stack(masks, dim=1)
    else:
        action_t = torch.empty(b, 0, dtype=torch.long, device=slots.device)
        log_prob_t = slots.new_zeros((b, 0))
        entropy_t = slots.new_zeros((b, 0))
        mask_t = slots.new_zeros((b, 0))
    logits = model.classify(h)
    terminal_reward, reward_parts = slothead_terminal_reward(selected, reward_features, args, stopped, action_t, model.stop_idx)
    return {
        "logits": logits,
        "selected_mask": selected,
        "actions": action_t,
        "mask": mask_t,
        "log_probs": log_prob_t,
        "entropies": entropy_t,
        "terminal_reward": terminal_reward,
        "trajectory_reward": terminal_reward,
        "stopped": stopped.detach(),
        "selected_count": reward_parts["selected_count"],
        **reward_parts,
    }


def grpo_loss_slothead_reward(
    model: SlotSelectorGRPO,
    slots: torch.Tensor,
    reward_features: dict[str, torch.Tensor],
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = labels.numel()
    group_size = int(args.grpo_group_size)
    slots_rep = slots.repeat_interleave(group_size, dim=0)
    labels_rep = labels.repeat_interleave(group_size, dim=0)
    reward_features_rep = {key: value.repeat_interleave(group_size, dim=0) for key, value in reward_features.items()}
    out = rollout_slothead_reward(model, slots_rep, reward_features_rep, args, sample=True)
    advantages = grpo.grpo_group_advantages(
        out["trajectory_reward"],
        batch_size=batch_size,
        group_size=group_size,
        eps=float(args.grpo_adv_eps),
    )
    seq_log_prob = (out["log_probs"] * out["mask"]).sum(dim=1)
    active_steps = out["mask"].sum().clamp_min(1.0)
    policy_loss = -(seq_log_prob * advantages).mean()
    entropy_loss = -((out["entropies"] * out["mask"]).sum() / active_steps)
    cls_loss = F.cross_entropy(out["logits"], labels_rep, label_smoothing=float(args.label_smoothing))
    return out, cls_loss, policy_loss, entropy_loss


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
    part_sums = {
        "objectness_mass": 0.0,
        "geometry_novelty": 0.0,
        "residual_novelty": 0.0,
        "good_stop": 0.0,
        "over_selected": 0.0,
        "stopped": 0.0,
    }
    for images, labels in loader:
        labels = labels.to(device, non_blocking=device.type == "cuda")
        slots, reward_features = encode_slots80(backbone, projector, images, device, args.slothead_mode)
        metric_labels = labels
        if train and args.warmup_epochs_remaining > 0:
            out = grpo.forced_warmup_rollout(model, slots, None, args)
            loss = float(args.classification_coef) * F.cross_entropy(out["logits"], labels, label_smoothing=args.label_smoothing)
        elif train:
            out, cls_loss, policy_loss, entropy_loss = grpo_loss_slothead_reward(model, slots, reward_features, labels, args)
            greedy_ce_loss = slots.new_zeros(())
            if float(args.greedy_ce_coef) > 0.0:
                greedy_out = rollout_slothead_reward(model, slots, reward_features, args, sample=False)
                greedy_ce_loss = F.cross_entropy(greedy_out["logits"], labels, label_smoothing=args.label_smoothing)
            loss = (
                float(args.classification_coef) * cls_loss
                + float(args.greedy_ce_coef) * greedy_ce_loss
                + args.policy_coef * policy_loss
                + args.entropy_coef * entropy_loss
            )
            metric_labels = labels.repeat_interleave(int(args.grpo_group_size))
        else:
            with torch.no_grad():
                out = rollout_slothead_reward(model, slots, reward_features, args, sample=False)
                loss = float(args.classification_coef) * F.cross_entropy(out["logits"], labels, label_smoothing=args.label_smoothing)
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
        for key in part_sums:
            if key in out:
                part_sums[key] += float(out[key].to(torch.float32).sum().detach().cpu())
    metrics = {
        "loss": loss_sum / max(total, 1),
        "acc": correct / max(total, 1),
        "avg_selected": count_sum / max(total, 1),
        "avg_reward": reward_sum / max(total, 1),
        "true_prob": prob_sum / max(total, 1),
    }
    metrics.update({f"avg_{key}": value / max(total, 1) for key, value in part_sums.items()})
    return metrics


def write_history(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RNN-GRPO selector over 80-dim slothead features.")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--sa_checkpoint", default=DEFAULT_SA)
    parser.add_argument("--slothead_checkpoint", default=DEFAULT_SLOTHEAD80)
    parser.add_argument("--slothead_mode", choices=["u", "obj", "geo", "res", "obj_geo", "obj_res", "geo_res"], default="u")
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
    parser.add_argument("--grpo_group_size", type=int, default=4)
    parser.add_argument("--grpo_adv_eps", type=float, default=1e-4)
    parser.add_argument("--free_slots", type=int, default=4)
    parser.add_argument("--min_free_slots", type=int, default=3)
    parser.add_argument("--max_free_slots", type=int, default=4)
    parser.add_argument("--classification_coef", type=float, default=1.0)
    parser.add_argument("--greedy_ce_coef", type=float, default=0.0)
    parser.add_argument("--objectness_coef", type=float, default=0.4)
    parser.add_argument("--geometry_coef", type=float, default=0.3)
    parser.add_argument("--residual_coef", type=float, default=0.5)
    parser.add_argument("--selected_count_coef", type=float, default=0.08)
    parser.add_argument("--over_select_coef", type=float, default=0.0)
    parser.add_argument("--good_stop_bonus", type=float, default=0.2)
    parser.add_argument("--good_stop_obj_threshold", type=float, default=0.6)
    parser.add_argument("--good_stop_res_threshold", type=float, default=0.35)
    parser.add_argument("--geometry_grid_size", type=int, default=14)
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
    if args.grpo_group_size <= 1:
        raise ValueError("--grpo_group_size must be greater than 1 for GRPO")
    if args.geometry_grid_size <= 0:
        raise ValueError("--geometry_grid_size must be positive")
    return args


def main() -> None:
    args = parse_args()
    seed_all(args.seed, False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tfm = build_transforms(args.input_res)
    train_set = subset_dataset(build_dataset(args.data, "train", tfm["train"]), args.quick_limit_train, args.seed)
    valid_set = subset_dataset(build_dataset(args.data, "valid", tfm["valid"]), args.quick_limit_val, args.seed)
    if not args.slothead_checkpoint:
        raise ValueError("--slothead_checkpoint is required for GRPO80; pass a fresh object-mode slothead checkpoint for the current dataset.")
    classes = class_names_from_dataset(train_set)
    if not classes:
        raise RuntimeError("Could not infer class names from dataset.")
    train_loader = grpo.make_loader(train_set, args, device, shuffle=True)
    valid_loader = grpo.make_loader(valid_set, args, device, shuffle=False)

    backbone = load_backbone(args.sa_checkpoint, device)
    backbone.eval()
    backbone.requires_grad_(False)
    projector, slothead_ckpt = load_slothead80(args.slothead_checkpoint, device)
    if args.max_steps <= 0:
        args.max_steps = backbone.num_slots
    slot_dim = int(projector.cfg.out_dim if args.slothead_mode == "u" else encode_slots80(backbone, projector, next(iter(valid_loader))[0], device, args.slothead_mode)[0].size(-1))

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
        "slot_embedding_source": "slothead80",
        "slothead80_config": slothead_ckpt["config"],
        "reward": {
            "formula": "A*objectness_mass + B*geometry_novelty + C*residual_novelty - D*selected_count + E*good_stop_bonus",
            "uses_labels": False,
            "label_usage": "labels are used only by the CE classification loss",
            "classification_coef": args.classification_coef,
            "greedy_ce_coef": args.greedy_ce_coef,
            "objectness_coef": args.objectness_coef,
            "geometry_coef": args.geometry_coef,
            "residual_coef": args.residual_coef,
            "selected_count_coef": args.selected_count_coef,
            "over_select_coef": args.over_select_coef,
            "good_stop_bonus": args.good_stop_bonus,
            "good_stop_obj_threshold": args.good_stop_obj_threshold,
            "good_stop_res_threshold": args.good_stop_res_threshold,
            "geometry_grid_size": args.geometry_grid_size,
        },
        "design_note": "Slothead80 supplies u=[u_obj,u_geo,u_res]. Reward is label-free: objectness keeps selected slots object-like, geometry rewards objectness-weighted incremental non-overlap coverage, residual novelty rewards non-duplicate u_res evidence. Scene labels are used only in CE loss.",
    }
    (out_dir / "selector_grpo_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"device={device} train={len(train_set)} valid={len(valid_set)} classes={len(classes)} slot_dim={slot_dim}")
    print(f"slothead80={args.slothead_checkpoint} mode={args.slothead_mode}")

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
