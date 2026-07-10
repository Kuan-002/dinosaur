#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

os.environ.setdefault(
    "TORCH_HOME",
    str(Path(__file__).resolve().parent / ".cache" / "torch"),
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli, Categorical
from torch.utils.data import DataLoader
from tqdm import tqdm

from misc_utils import seed_all
from selector_grpo import GRPOSelectorConfig, SlotSelectorGRPO, true_class_margin
from settransformer.model import DiscriminativeSetTransformer as SetTransformerSlotProbe
from settransformer.model import ProbeConfig
from train_slot_classifier import build_dataset, build_transforms, load_backbone, subset_dataset


def attention_to_xy(attn: torch.Tensor) -> torch.Tensor:
    b, k, n = attn.shape
    side = int(math.sqrt(n))
    if side * side != n:
        raise ValueError(f"attention token count is not square: {n}")
    weights = attn.clamp_min(0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, side, device=attn.device, dtype=attn.dtype),
        torch.linspace(-1.0, 1.0, side, device=attn.device, dtype=attn.dtype),
        indexing="ij",
    )
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
    return torch.einsum("bkn,nd->bkd", weights, grid)


@torch.no_grad()
def encode_batch(backbone, images: torch.Tensor, device: torch.device, pos_dim: int):
    backbone.eval()
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.forward_dino(images)
    features = backbone.mlp(features)
    slots, attn, _ = backbone.slot_attention(features)
    return slots.detach(), attn.detach(), attention_to_xy(attn).detach() if pos_dim > 0 else None


def make_loader(dataset, args, device, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=shuffle,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def class_names_from_dataset(dataset) -> Optional[list[str]]:
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    classes = getattr(dataset, "classes", None)
    return list(classes) if classes is not None else None


def load_reward_probe(path: str, device: torch.device) -> SetTransformerSlotProbe:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ProbeConfig(**ckpt["probe_config"])
    probe = SetTransformerSlotProbe(cfg).to(device)
    probe.load_state_dict(ckpt["model_state_dict"])
    probe.eval()
    for param in probe.parameters():
        param.requires_grad = False
    return probe


class SlotSemanticProbeForGRPO(nn.Module):
    def __init__(self, slot_dim: int, hidden_dim: int, dropout: float, num_heads: int):
        super().__init__()
        if hidden_dim <= 0:
            self.net = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, num_heads),
            )
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_heads),
            )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        return self.net(slots)

    def features(self, slots: torch.Tensor) -> torch.Tensor:
        if len(self.net) == 2:
            return self.net[0](slots)
        x = self.net[0](slots)
        x = self.net[1](x)
        x = self.net[2](x)
        return x


def load_slothead_probe(path: str, device: torch.device) -> tuple[SlotSemanticProbeForGRPO, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["probe_config"]
    probe = SlotSemanticProbeForGRPO(
        slot_dim=int(cfg["slot_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg["dropout"]),
        num_heads=int(cfg.get("num_heads", 3)),
    ).to(device)
    probe.load_state_dict(ckpt["model_state_dict"])
    probe.eval()
    for param in probe.parameters():
        param.requires_grad = False
    return probe, ckpt


@torch.no_grad()
def augment_slots_with_slothead(
    slots: torch.Tensor,
    slothead_probe: Optional[SlotSemanticProbeForGRPO],
    slot_embedding_scale: float,
    slothead_feature_scale: float,
    slothead_score_scale: float,
    slothead_feature_mode: str,
) -> torch.Tensor:
    if slothead_probe is None:
        return slots
    parts = [slots * float(slot_embedding_scale)]
    if slothead_feature_mode in {"hidden", "hidden_scores"}:
        hidden = slothead_probe.features(slots).to(dtype=slots.dtype)
        parts.append(hidden * float(slothead_feature_scale))
    if slothead_feature_mode in {"scores", "hidden_scores"}:
        scores = slothead_probe(slots).sigmoid().to(dtype=slots.dtype)
        parts.append(scores * float(slothead_score_scale))
    return torch.cat(parts, dim=-1)


@torch.no_grad()
def probe_margin_reward(
    probe: SetTransformerSlotProbe,
    slots: torch.Tensor,
    selected: torch.Tensor,
    labels: torch.Tensor,
    args,
) -> torch.Tensor:
    all_mask = torch.ones_like(selected, dtype=torch.bool)
    selected_mask = selected.bool()
    removed_mask = ~selected_mask

    all_margin = true_class_margin(probe(slots, all_mask), labels)
    selected_margin = true_class_margin(probe(slots, selected_mask), labels)
    removed_margin = true_class_margin(probe(slots, removed_mask), labels)
    necessity = all_margin - removed_margin

    reward = (
        float(args.probe_selected_weight) * selected_margin
        + float(args.probe_necessity_weight) * necessity
    )
    if args.probe_reward_clip > 0:
        reward = reward.clamp(-float(args.probe_reward_clip), float(args.probe_reward_clip))
    return reward.detach()


@torch.no_grad()
def probe_subset_margin_reward(
    probe: SetTransformerSlotProbe,
    slots: torch.Tensor,
    selected: torch.Tensor,
    labels: torch.Tensor,
    args,
) -> torch.Tensor:
    """Reward a selected subset against other subsets from the same image.

    This intentionally does not evaluate an all-slot classifier state. The
    selected subset and optional complement are scored as separate masked
    subsets, so the reward cannot rely on a pooled all-slots representation.
    """

    selected_mask = selected.bool()
    selected_margin = true_class_margin(probe(slots, selected_mask), labels)
    reward = selected_margin
    if args.subset_contrast == "complement":
        complement_mask = ~selected_mask
        complement_margin = true_class_margin(probe(slots, complement_mask), labels)
        reward = reward - float(args.subset_contrast_weight) * complement_margin
    if args.probe_reward_clip > 0:
        reward = reward.clamp(-float(args.probe_reward_clip), float(args.probe_reward_clip))
    return reward.detach()


@torch.no_grad()
def probe_full_pseudo_subset_margin_reward(
    probe: SetTransformerSlotProbe,
    slots: torch.Tensor,
    selected: torch.Tensor,
    args,
) -> torch.Tensor:
    """Label-free rationale reward using the full-set prediction as teacher.

    The full set defines a pseudo target. A good rationale is sufficient for
    that prediction, while its complement is not equally sufficient.
    """

    all_mask = torch.ones_like(selected, dtype=torch.bool)
    selected_mask = selected.bool()
    complement_mask = ~selected_mask
    full_logits = probe(slots, all_mask)
    pseudo_labels = full_logits.argmax(dim=1)
    full_margin = true_class_margin(full_logits, pseudo_labels)
    selected_margin = true_class_margin(probe(slots, selected_mask), pseudo_labels)
    complement_margin = true_class_margin(probe(slots, complement_mask), pseudo_labels)

    count = selected_mask.sum(dim=1).to(slots.dtype)
    k_min = float(args.pseudo_k_min)
    k_max = float(args.pseudo_k_max)
    sufficiency = selected_margin >= (full_margin - float(args.pseudo_sufficiency_eps))
    in_range = (count >= k_min) & (count <= k_max)

    count_cost = (
        float(args.pseudo_count_alpha) * (count - k_min).clamp_min(0.0)
        + float(args.pseudo_under_min_gamma) * (k_min - count).clamp_min(0.0)
        + float(args.pseudo_over_max_delta) * (count - k_max).clamp_min(0.0).pow(2)
    )
    sufficiency_gap = (full_margin - float(args.pseudo_sufficiency_eps) - selected_margin).clamp_min(0.0)
    stop_quality = torch.zeros_like(selected_margin)
    stop_quality = torch.where(
        in_range & sufficiency,
        stop_quality + float(args.pseudo_stop_bonus),
        stop_quality,
    )
    stop_quality = torch.where(
        (count < k_min) & ~sufficiency,
        stop_quality - float(args.pseudo_stop_bonus),
        stop_quality,
    )
    stop_quality = torch.where(
        count > k_max,
        stop_quality - 0.5 * float(args.pseudo_stop_bonus),
        stop_quality,
    )

    reward = (
        selected_margin
        - float(args.pseudo_beta) * complement_margin
        - float(args.pseudo_sufficiency_gap_weight) * sufficiency_gap
        - count_cost
        + stop_quality
    )
    if args.probe_reward_clip > 0:
        reward = reward.clamp(-float(args.probe_reward_clip), float(args.probe_reward_clip))
    return reward.detach()


def parse_float_list(raw: str) -> tuple[float, ...]:
    raw = raw.strip()
    if not raw:
        return ()
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if any(value < 0 for value in values):
        raise ValueError("step margin reward weights must be non-negative")
    return values


def rollout(
    model: SlotSelectorGRPO,
    slots: torch.Tensor,
    slot_pos: Optional[torch.Tensor],
    labels: torch.Tensor,
    args,
    sample: bool,
    reward_probe: Optional[SetTransformerSlotProbe] = None,
):
    slot_embeds = model.embed_slots(slots, slot_pos)
    b, k, _ = slot_embeds.shape
    h = model.initial_state(slot_embeds)
    selected = torch.zeros(b, k, dtype=torch.bool, device=slots.device)
    active = torch.ones(b, dtype=torch.bool, device=slots.device)

    log_probs = []
    entropies = []
    actions = []
    masks = []

    min_steps = int(args.min_steps)
    confidence_exit_min_steps = int(getattr(args, "confidence_early_exit_min_steps", 0) or min_steps)
    use_confidence_exit = not bool(args.disable_confidence_early_exit)
    early_exit_conf = float(args.early_exit_conf)
    step_margin_weights = getattr(args, "step_margin_reward_weights", ())
    use_step_margin_reward = (
        bool(step_margin_weights)
        and args.reward_source == "probe_subset_margin"
        and reward_probe is not None
    )
    step_margin_reward = torch.zeros(b, device=slots.device, dtype=slots.dtype)
    prev_step_margin = None
    if use_step_margin_reward:
        with torch.no_grad():
            prev_step_margin = true_class_margin(reward_probe(slots, selected.bool()), labels).detach()
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
            stop_log_prob = stop_dist.log_prob(is_stop.to(dtype=stop_logits.dtype)) * active_f
            slot_log_prob = slot_dist.log_prob(slot_action) * (active & ~is_stop).to(slots.dtype)
            step_log_prob = stop_log_prob + slot_log_prob
            step_entropy = (stop_dist.entropy() + slot_dist.entropy() * (~is_stop).to(slots.dtype)) * active_f
        else:
            logits = model.policy_logits(h.detach(), slot_embeds.detach(), selected, step)
            logits[:, model.stop_idx] = torch.where(
                can_stop,
                logits[:, model.stop_idx],
                torch.full_like(logits[:, model.stop_idx], torch.finfo(logits.dtype).min),
            )
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
        if use_step_margin_reward and step < len(step_margin_weights):
            if prev_step_margin is None:
                raise RuntimeError("prev_step_margin is not initialized")
            with torch.no_grad():
                current_margin = true_class_margin(reward_probe(slots, selected.bool()), labels).detach()
            delta_margin = (current_margin - prev_step_margin).to(slots.dtype)
            weight = float(step_margin_weights[step])
            step_reward = weight * delta_margin * do_select.to(slots.dtype)
            step_margin_reward = step_margin_reward + step_reward
            prev_step_margin = current_margin
        step_logits = model.classify(h)
        if use_confidence_exit:
            conf_stop = (
                active
                & ~is_stop
                & (selected.sum(dim=1) >= confidence_exit_min_steps)
                & (step_logits.softmax(dim=-1).amax(dim=1) >= early_exit_conf)
            )
        else:
            conf_stop = torch.zeros_like(active)
        active = active & ~is_stop & ~conf_stop
        if not active.any():
            break

    logits = model.classify(h)
    if args.reward_source == "classifier":
        pred = logits.argmax(dim=-1)
        task_reward = (pred == labels).to(slots.dtype) if args.cls_reward_negative <= 0 else torch.where(
            pred == labels,
            torch.ones_like(labels, dtype=slots.dtype),
            torch.full_like(labels, -float(args.cls_reward_negative), dtype=slots.dtype),
        )
    elif args.reward_source == "classifier_logprob":
        task_reward = F.log_softmax(logits, dim=-1).gather(1, labels[:, None]).squeeze(1).to(slots.dtype)
    elif args.reward_source == "classifier_margin":
        task_reward = true_class_margin(logits, labels).to(slots.dtype)
    elif args.reward_source == "probe_margin":
        if reward_probe is None:
            raise ValueError("--reward_source probe_margin requires --reward_probe_checkpoint")
        task_reward = probe_margin_reward(reward_probe, slots, selected, labels, args).to(slots.dtype)
    elif args.reward_source == "probe_subset_margin":
        if reward_probe is None:
            raise ValueError("--reward_source probe_subset_margin requires --reward_probe_checkpoint")
        task_reward = probe_subset_margin_reward(reward_probe, slots, selected, labels, args).to(slots.dtype)
    elif args.reward_source == "probe_full_pseudo_subset_margin":
        if reward_probe is None:
            raise ValueError("--reward_source probe_full_pseudo_subset_margin requires --reward_probe_checkpoint")
        task_reward = probe_full_pseudo_subset_margin_reward(reward_probe, slots, selected, args).to(slots.dtype)
    else:
        raise ValueError(f"Unknown reward_source: {args.reward_source}")
    count = selected.sum(dim=1).to(slots.dtype)
    min_free_slots = float(getattr(args, "min_free_slots", 0) or 0)
    max_free_slots = float(getattr(args, "max_free_slots", 0) or 0)
    if min_free_slots > 0 or max_free_slots > 0:
        lo = min_free_slots if min_free_slots > 0 else float(args.free_slots)
        hi = max_free_slots if max_free_slots > 0 else float(args.free_slots)
        count_penalty = -float(args.count_penalty) * ((lo - count).clamp_min(0) + (count - hi).clamp_min(0))
    else:
        count_penalty = -args.count_penalty * (count - args.free_slots).clamp_min(0)
    terminal_reward = task_reward + count_penalty
    trajectory_reward = terminal_reward + step_margin_reward
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
    return {
        "logits": logits,
        "selected_mask": selected,
        "actions": action_t,
        "mask": mask_t,
        "log_probs": log_prob_t,
        "entropies": entropy_t,
        "terminal_reward": terminal_reward.detach(),
        "trajectory_reward": trajectory_reward.detach(),
        "step_margin_reward": step_margin_reward.detach(),
        "selected_count": count.detach(),
    }


def repeat_for_grpo(
    slots: torch.Tensor,
    slot_pos: Optional[torch.Tensor],
    labels: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    if group_size <= 1:
        raise ValueError("--grpo_group_size must be greater than 1")
    slots_rep = slots.repeat_interleave(group_size, dim=0)
    labels_rep = labels.repeat_interleave(group_size, dim=0)
    if slot_pos is None:
        return slots_rep, None, labels_rep
    return slots_rep, slot_pos.repeat_interleave(group_size, dim=0), labels_rep


def grpo_group_advantages(rewards: torch.Tensor, batch_size: int, group_size: int, eps: float) -> torch.Tensor:
    reward_group = rewards.detach().view(batch_size, group_size)
    centered = reward_group - reward_group.mean(dim=1, keepdim=True)
    scale = reward_group.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return (centered / scale).reshape(batch_size * group_size)


def grpo_loss(
    model: SlotSelectorGRPO,
    slots: torch.Tensor,
    slot_pos: Optional[torch.Tensor],
    labels: torch.Tensor,
    args,
    reward_probe: Optional[SetTransformerSlotProbe],
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = labels.numel()
    group_size = int(args.grpo_group_size)
    slots_rep, slot_pos_rep, labels_rep = repeat_for_grpo(slots, slot_pos, labels, group_size)
    out = rollout(
        model,
        slots_rep,
        slot_pos_rep,
        labels_rep,
        args,
        sample=True,
        reward_probe=reward_probe,
    )
    advantages = grpo_group_advantages(
        out["trajectory_reward"],
        batch_size=batch_size,
        group_size=group_size,
        eps=float(args.grpo_adv_eps),
    )
    seq_log_prob = (out["log_probs"] * out["mask"]).sum(dim=1)
    active_steps = out["mask"].sum().clamp_min(1.0)
    policy_loss = -(seq_log_prob * advantages).mean()
    entropy_loss = -((out["entropies"] * out["mask"]).sum() / active_steps)
    cls_loss = F.cross_entropy(
        out["logits"],
        labels_rep,
        label_smoothing=float(getattr(args, "label_smoothing", 0.0) or 0.0),
    )
    confidence_penalty = float(getattr(args, "confidence_penalty", 0.0) or 0.0)
    if confidence_penalty > 0:
        max_prob = out["logits"].softmax(dim=-1).amax(dim=1)
        threshold = float(getattr(args, "confidence_penalty_threshold", 0.0) or 0.0)
        excess = (max_prob - threshold).clamp_min(0.0)
        until_slots = int(getattr(args, "early_confidence_penalty_until_slots", 0) or 0)
        if until_slots > 0:
            excess = excess * (out["selected_count"] < float(until_slots)).to(excess.dtype)
        cls_loss = cls_loss + confidence_penalty * excess.pow(2).mean()
    return out, cls_loss, policy_loss, entropy_loss


def forced_warmup_rollout(
    model: SlotSelectorGRPO,
    slots: torch.Tensor,
    slot_pos: Optional[torch.Tensor],
    args,
) -> dict:
    slot_embeds = model.embed_slots(slots, slot_pos)
    b, k, _ = slot_embeds.shape
    h = model.initial_state(slot_embeds)
    selected = torch.zeros(b, k, dtype=torch.bool, device=slots.device)
    steps = min(args.warmup_steps, args.max_steps, k)
    for _ in range(steps):
        score = torch.rand(b, k, device=slots.device, dtype=slots.dtype).masked_fill(
            selected,
            torch.finfo(slots.dtype).min,
        )
        action = score.argmax(dim=1)
        active = selected.sum(dim=1) < k
        h, selected = model.update_with_action(h, selected, slot_embeds, action, active)
    logits = model.classify(h)
    return {
        "logits": logits,
        "selected_mask": selected,
        "selected_count": selected.sum(dim=1).to(slots.dtype),
    }


def ce_parameters(model: SlotSelectorGRPO) -> list[nn.Parameter]:
    return [
        model.h0,
        *model.slot_embed.parameters(),
        *model.gru.parameters(),
        *model.classifier.parameters(),
    ]


def policy_parameters(model: SlotSelectorGRPO) -> list[nn.Parameter]:
    params = [
        *model.query.parameters(),
        *model.key.parameters(),
        model.stop_key,
        model.stop_bias,
        *model.stop_head.parameters(),
    ]
    if model.first_step_query is not None:
        params.append(model.first_step_query)
    if model.first_step_attention is not None:
        params.extend(model.first_step_attention.parameters())
    if model.first_step_norm is not None:
        params.extend(model.first_step_norm.parameters())
    return params


def run_epoch(
    model,
    backbone,
    loader,
    device,
    ce_optimizer,
    policy_optimizer,
    args,
    train: bool,
    reward_probe: Optional[SetTransformerSlotProbe] = None,
    slothead_probe: Optional[SlotSemanticProbeForGRPO] = None,
):
    model.train(train)
    total = 0
    loss_sum = 0.0
    correct = 0
    count_sum = 0.0
    desc = "train" if train else "valid"
    batches = tqdm(loader, desc=desc, mininterval=1.0) if sys.stdout.isatty() else loader
    for images, labels in batches:
        labels = labels.to(device, non_blocking=device.type == "cuda")
        slots, attn, slot_pos = encode_batch(backbone, images, device, model.cfg.pos_dim)
        slots = augment_slots_with_slothead(
            slots,
            slothead_probe,
            args.slot_embedding_scale,
            args.slothead_feature_scale,
            args.slothead_score_scale,
            args.slothead_feature_mode,
        )
        metric_labels = labels
        if train and args.warmup_epochs_remaining > 0:
            out = forced_warmup_rollout(model, slots, slot_pos, args)
            loss = F.cross_entropy(out["logits"], labels)
        elif train:
            with torch.set_grad_enabled(True):
                out, cls_loss, policy_loss, entropy_loss = grpo_loss(
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
                out = rollout(model, slots, slot_pos, labels, args, sample=train, reward_probe=reward_probe)
                loss = F.cross_entropy(out["logits"], labels)
        if train:
            ce_optimizer.zero_grad(set_to_none=True)
            policy_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            ce_optimizer.step()
            policy_optimizer.step()
        batch = metric_labels.numel()
        total += batch
        loss_sum += float(loss.detach()) * batch
        correct += int((out["logits"].argmax(dim=1) == metric_labels).sum())
        count_sum += float(out["selected_count"].sum())
    return {"loss": loss_sum / max(total, 1), "acc": correct / max(total, 1), "avg_selected": count_sum / max(total, 1)}


def write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Train GRPO selector hard slot selector.")
    parser.add_argument("--data", default="/vol/biomedic3/kw1025/dinosaur/dataset/coco_scene_guidelines_10_v2_classification.zip")
    parser.add_argument("--checkpoint", default="/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt")
    parser.add_argument("--output_dir", default="./checkpoints/selector_grpo")
    parser.add_argument("--input_res", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--warmup_steps", type=int, default=3)
    parser.add_argument("--warmup_policy", choices=["random"], default="random")
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--policy_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=0, help="Maximum selection steps. Use 0 to allow all slots.")
    parser.add_argument("--min_steps", type=int, default=1)
    parser.add_argument("--early_exit_conf", type=float, default=0.8)
    parser.add_argument(
        "--confidence_early_exit_min_steps",
        type=int,
        default=0,
        help="Minimum selected slots before confidence early exit may fire. Use 0 to match --min_steps.",
    )
    parser.add_argument(
        "--first_step_cross_attention",
        action="store_true",
        help="Use learned-query multi-head cross-attention over all slots only for the first selected slot.",
    )
    parser.add_argument("--first_step_num_heads", type=int, default=4)
    parser.add_argument(
        "--policy_context_attention",
        action="store_true",
        help="Let policy queries attend over all slot embeddings at every selection step.",
    )
    parser.add_argument("--decoupled_stop_policy", action="store_true")
    parser.add_argument("--disable_confidence_early_exit", action="store_true", default=True)
    parser.add_argument("--enable_confidence_early_exit", dest="disable_confidence_early_exit", action="store_false")
    parser.add_argument(
        "--reward_source",
        choices=[
            "classifier",
            "classifier_logprob",
            "classifier_margin",
            "probe_margin",
            "probe_subset_margin",
            "probe_full_pseudo_subset_margin",
        ],
        default="classifier",
    )
    parser.add_argument("--reward_probe_checkpoint", default="")
    parser.add_argument("--probe_selected_weight", type=float, default=1.0)
    parser.add_argument("--probe_necessity_weight", type=float, default=1.0)
    parser.add_argument("--probe_reward_clip", type=float, default=10.0)
    parser.add_argument(
        "--subset_contrast",
        choices=["none", "complement"],
        default="complement",
        help="Extra subset-only contrast for probe_subset_margin. Does not score all slots jointly.",
    )
    parser.add_argument("--subset_contrast_weight", type=float, default=1.0)
    parser.add_argument("--pseudo_beta", type=float, default=0.5)
    parser.add_argument("--pseudo_k_min", type=int, default=2)
    parser.add_argument("--pseudo_k_max", type=int, default=4)
    parser.add_argument("--pseudo_count_alpha", type=float, default=0.05)
    parser.add_argument("--pseudo_under_min_gamma", type=float, default=0.5)
    parser.add_argument("--pseudo_over_max_delta", type=float, default=0.15)
    parser.add_argument("--pseudo_sufficiency_eps", type=float, default=0.5)
    parser.add_argument("--pseudo_sufficiency_gap_weight", type=float, default=1.0)
    parser.add_argument("--pseudo_stop_bonus", type=float, default=1.0)
    parser.add_argument(
        "--step_margin_reward_weights",
        default="",
        help=(
            "Comma-separated non-negative weights for per-selection selected-subset "
            "margin gains, e.g. '1.0,0.5,0.3'. Only used with "
            "--reward_source probe_subset_margin."
        ),
    )
    parser.add_argument("--grpo_group_size", type=int, default=4)
    parser.add_argument("--grpo_adv_eps", type=float, default=1e-4)
    parser.add_argument("--free_slots", type=int, default=3)
    parser.add_argument("--min_free_slots", type=int, default=0)
    parser.add_argument("--max_free_slots", type=int, default=0)
    parser.add_argument("--count_penalty", type=float, default=0.1)
    parser.add_argument("--cls_reward_negative", type=float, default=0.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--confidence_penalty", type=float, default=0.0)
    parser.add_argument("--confidence_penalty_threshold", type=float, default=0.0)
    parser.add_argument("--early_confidence_penalty_until_slots", type=int, default=0)
    parser.add_argument("--policy_coef", type=float, default=1.0)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--pos_dim", type=int, choices=[0, 2], default=0)
    parser.add_argument(
        "--slothead_checkpoint",
        default="",
        help="Optional supervised semantic slot-head checkpoint. Its 3 sigmoid scores are concatenated to each slot.",
    )
    parser.add_argument(
        "--slot_embedding_scale",
        type=float,
        default=1.0,
        help="Scale applied to the original DINOSAUR slot embedding before slothead concatenation.",
    )
    parser.add_argument(
        "--slothead_score_scale",
        type=float,
        default=1.0,
        help="Scale applied to slothead anchor/evidence/background probabilities before concatenation.",
    )
    parser.add_argument(
        "--slothead_feature_scale",
        type=float,
        default=1.0,
        help="Scale applied to the slothead hidden representation before concatenation.",
    )
    parser.add_argument(
        "--slothead_feature_mode",
        choices=["scores", "hidden", "hidden_scores"],
        default="scores",
        help="Which slothead outputs to concatenate to the original slot embedding.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--quick_limit_train", type=int, default=0)
    parser.add_argument("--quick_limit_val", type=int, default=0)
    args = parser.parse_args()
    args.step_margin_reward_weights = parse_float_list(args.step_margin_reward_weights)

    if args.grpo_group_size <= 1:
        raise ValueError("--grpo_group_size must be greater than 1 for GRPO")
    if args.first_step_cross_attention:
        raise ValueError(
            "GRPO mode disallows --first_step_cross_attention because it pools all slots "
            "into the first policy state."
        )

    seed_all(args.seed, False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tfm = build_transforms(args.input_res)
    train_set = subset_dataset(build_dataset(args.data, "train", tfm["train"]), args.quick_limit_train, args.seed)
    valid_set = subset_dataset(build_dataset(args.data, "valid", tfm["valid"]), args.quick_limit_val, args.seed)
    classes = class_names_from_dataset(train_set)
    num_classes = len(classes) if classes is not None else 10

    train_loader = make_loader(train_set, args, device, shuffle=True)
    valid_loader = make_loader(valid_set, args, device, shuffle=False)
    backbone = load_backbone(args.checkpoint, device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    if args.max_steps <= 0:
        args.max_steps = backbone.num_slots
    reward_probe = None
    slothead_probe = None
    slothead_ckpt = None
    if args.reward_source in {"probe_margin", "probe_subset_margin", "probe_full_pseudo_subset_margin"}:
        if not args.reward_probe_checkpoint:
            raise ValueError(
                "--reward_probe_checkpoint is required when using a probe-based reward source"
            )
        reward_probe = load_reward_probe(args.reward_probe_checkpoint, device)
        if reward_probe.cfg.num_slots != backbone.num_slots or reward_probe.cfg.slot_dim != backbone.slot_dim:
            raise ValueError(
                "Reward probe shape does not match SA backbone: "
                f"probe slots={reward_probe.cfg.num_slots} dim={reward_probe.cfg.slot_dim}, "
                f"backbone slots={backbone.num_slots} dim={backbone.slot_dim}"
            )
    elif args.step_margin_reward_weights:
        raise ValueError("--step_margin_reward_weights requires a probe-based reward source")
    if args.slothead_checkpoint:
        if args.reward_source != "classifier":
            raise ValueError("--slothead_checkpoint is intended for classifier-only GRPO in this trainer")
        slothead_probe, slothead_ckpt = load_slothead_probe(args.slothead_checkpoint, device)
        probe_cfg = slothead_ckpt["probe_config"]
        if probe_cfg["slot_dim"] != backbone.slot_dim:
            raise ValueError(
                "Slothead probe shape does not match SA backbone: "
                f"slothead dim={probe_cfg['slot_dim']}, backbone dim={backbone.slot_dim}"
            )

    slothead_extra_dim = 0
    if slothead_probe is not None:
        probe_cfg = slothead_ckpt["probe_config"]
        if args.slothead_feature_mode in {"hidden", "hidden_scores"}:
            slothead_extra_dim += int(probe_cfg["hidden_dim"]) if int(probe_cfg["hidden_dim"]) > 0 else int(probe_cfg["slot_dim"])
        if args.slothead_feature_mode in {"scores", "hidden_scores"}:
            slothead_extra_dim += int(probe_cfg.get("num_heads", 3))
    selector_slot_dim = backbone.slot_dim + slothead_extra_dim
    cfg = GRPOSelectorConfig(
        num_slots=backbone.num_slots,
        slot_dim=selector_slot_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        policy_dim=args.policy_dim,
        max_steps=args.max_steps,
        dropout=args.dropout,
        pos_dim=args.pos_dim,
        min_steps=args.min_steps,
        early_exit_conf=args.early_exit_conf,
        decoupled_stop_policy=args.decoupled_stop_policy,
        first_step_cross_attention=args.first_step_cross_attention,
        first_step_num_heads=args.first_step_num_heads,
        policy_context_attention=args.policy_context_attention,
    )
    model = SlotSelectorGRPO(cfg).to(device)
    ce_params = ce_parameters(model)
    policy_params = policy_parameters(model)
    ce_param_ids = {id(param) for param in ce_params}
    policy_param_ids = {id(param) for param in policy_params}
    overlap = ce_param_ids & policy_param_ids
    missing = {id(param) for param in model.parameters()} - ce_param_ids - policy_param_ids
    if overlap:
        raise ValueError("CE and policy parameter groups overlap")
    if missing:
        raise ValueError(f"Some model parameters are not assigned to CE or policy groups: {len(missing)}")
    ce_optimizer = torch.optim.AdamW(ce_params, lr=args.lr, weight_decay=args.wd)
    policy_optimizer = torch.optim.AdamW(policy_params, lr=args.lr, weight_decay=args.wd)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "selector_grpo_meta.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "grpo_config": asdict(cfg),
                "classes": classes,
                "slot_embedding_source": "sa_plus_slothead" if slothead_probe is not None else "sa",
                "slothead": None
                if slothead_probe is None
                else {
                    "checkpoint": args.slothead_checkpoint,
                    "heads": slothead_ckpt.get("heads"),
                    "slot_embedding_scale": args.slot_embedding_scale,
                    "slothead_feature_mode": args.slothead_feature_mode,
                    "slothead_feature_scale": args.slothead_feature_scale,
                    "slothead_score_scale": args.slothead_score_scale,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Using device={device} train={len(train_set)} valid={len(valid_set)} classes={num_classes}")
    print(
        f"GRPO selector slots={cfg.num_slots} slot_dim={cfg.slot_dim} hidden={cfg.hidden_dim} "
        f"max_steps={cfg.max_steps} min_steps={cfg.min_steps} early_exit_conf={cfg.early_exit_conf} "
        f"confidence_early_exit_min_steps={args.confidence_early_exit_min_steps or args.min_steps} "
        f"decoupled_stop_policy={cfg.decoupled_stop_policy} "
        f"first_step_cross_attention={cfg.first_step_cross_attention} "
        f"disable_confidence_early_exit={args.disable_confidence_early_exit} "
        f"reward_source={args.reward_source}"
    )
    if slothead_probe is not None:
        print(
            "slothead-assisted input enabled: "
            f"embedding_scale={args.slot_embedding_scale} "
            f"slothead_feature_mode={args.slothead_feature_mode} "
            f"slothead_feature_scale={args.slothead_feature_scale} "
            f"slothead_score_scale={args.slothead_score_scale} "
            f"checkpoint={args.slothead_checkpoint}"
        )
    print(f"parameter groups: ce={sum(p.numel() for p in ce_params):,} policy={sum(p.numel() for p in policy_params):,}")
    best_acc = -1.0
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        args.warmup_epochs_remaining = max(args.warmup_epochs - epoch + 1, 0)
        train_metrics = run_epoch(
            model,
            backbone,
            train_loader,
            device,
            ce_optimizer,
            policy_optimizer,
            args,
            train=True,
            reward_probe=reward_probe,
            slothead_probe=slothead_probe,
        )
        args.warmup_epochs_remaining = 0
        with torch.no_grad():
            valid_metrics = run_epoch(
                model,
                backbone,
                valid_loader,
                device,
                ce_optimizer,
                policy_optimizer,
                args,
                train=False,
                reward_probe=reward_probe,
                slothead_probe=slothead_probe,
            )
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        row = {
            "epoch": epoch,
            "elapsed": elapsed,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "train_avg_selected": train_metrics["avg_selected"],
            "valid_loss": valid_metrics["loss"],
            "valid_acc": valid_metrics["acc"],
            "valid_avg_selected": valid_metrics["avg_selected"],
        }
        history.append(row)
        write_history(out_dir / "history_metrics.csv", history)
        print(
            f"epoch={epoch} elapsed={elapsed} "
            f"train_acc={100 * row['train_acc']:.2f} train_slots={row['train_avg_selected']:.2f} "
            f"valid_acc={100 * row['valid_acc']:.2f} valid_slots={row['valid_avg_selected']:.2f}"
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
