#!/usr/bin/env python3
"""Factorized-MIL RNN-GRPO training for the six-class compositional dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".cache" / "torch"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import train_grpo_selector as grpo
from misc_utils import seed_all
from selector_grpo import GRPOSelectorConfig, SlotSelectorGRPO, true_class_margin
from train_slot_classifier import build_dataset, build_transforms, load_backbone, subset_dataset


DEFAULT_DATA = str(REPO_ROOT / "dataset/coco_compositional_pair6_clean_300_100_100/classification_dataset")
DEFAULT_SA = str(REPO_ROOT / "checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt")


def class_names_from_dataset(dataset) -> list[str]:
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return list(getattr(dataset, "classes", []))


def class_neighbor_groups(classes: list[str]) -> list[list[list[int]]]:
    """Single-component-sharing competitors derived only from class names."""
    components = [set(name.removesuffix("_scene").split("__")) for name in classes]
    groups: list[list[list[int]]] = []
    for class_index, pair in enumerate(components):
        if len(pair) != 2:
            raise ValueError(f"expected a two-component class, got {classes[class_index]!r}")
        class_groups = [
            [other for other, other_pair in enumerate(components) if other != class_index and component in other_pair]
            for component in sorted(pair)
        ]
        if any(not group for group in class_groups):
            raise ValueError(f"class {classes[class_index]!r} has a component without a confusable neighbor")
        groups.append(class_groups)
    return groups


def class_component_structure(classes: list[str]) -> tuple[list[str], list[list[int]]]:
    pairs = [sorted(name.removesuffix("_scene").split("__")) for name in classes]
    components = sorted({component for pair in pairs for component in pair})
    component_to_index = {component: index for index, component in enumerate(components)}
    return components, [[component_to_index[component] for component in pair] for pair in pairs]


def balanced_confusable_margin(
    logits: torch.Tensor,
    labels: torch.Tensor,
    neighbor_groups: list[list[list[int]]],
) -> torch.Tensor:
    """Worst margin over the two single-component-sharing competitor groups."""
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    max_neighbors = max(len(group) for class_groups in neighbor_groups for group in class_groups)
    padded = [
        [group + [-1] * (max_neighbors - len(group)) for group in class_groups]
        for class_groups in neighbor_groups
    ]
    all_indices = torch.tensor(padded, dtype=torch.long, device=logits.device)
    indices = all_indices[labels]
    valid = indices >= 0
    candidates = logits[:, None, :].expand(-1, 2, -1).gather(2, indices.clamp_min(0))
    competitor_logits = candidates.masked_fill(~valid, torch.finfo(logits.dtype).min).amax(dim=2)
    return (true_logits[:, None] - competitor_logits).amin(dim=1)


@torch.no_grad()
def encode_slots_and_attention(backbone, images: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.forward_dino(images)
    features = backbone.mlp(features)
    slots, attention, _ = backbone.slot_attention(features)
    return slots.detach(), attention.detach()


def normalized_attention_and_quality(attention: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-slot token distributions and relative concentration quality."""
    probability = attention.float().clamp_min(0.0)
    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    num_tokens = probability.size(-1)
    entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=-1)
    quality = (1.0 - entropy / float(torch.log(torch.tensor(float(num_tokens))))).clamp(0.0, 1.0)
    # Relative scaling makes the stop threshold interpretable across images.
    quality = quality / quality.amax(dim=1, keepdim=True).clamp_min(1e-6)
    return probability.to(attention.dtype), quality.to(attention.dtype)


def candidate_novelty(
    attention_probability: torch.Tensor,
    attention_quality: torch.Tensor,
    coverage: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    gain = (attention_probability - coverage[:, None, :]).clamp_min(0.0).sum(dim=-1)
    novelty = attention_quality * gain
    return novelty.masked_fill(selected, -1.0)


def distinct_component_pair_score(
    component_probability: torch.Tensor,
    selected: torch.Tensor,
    labels: torch.Tensor,
    class_component_indices: list[list[int]],
) -> torch.Tensor:
    """Best two-distinct-slot soft assignment to the two true components."""
    target_indices = torch.tensor(class_component_indices, dtype=torch.long, device=labels.device)[labels]
    target_probability = component_probability.gather(
        2, target_indices[:, None, :].expand(-1, component_probability.size(1), -1)
    )
    first, second = target_probability[..., 0], target_probability[..., 1]
    forward = first[:, :, None] * second[:, None, :]
    backward = second[:, :, None] * first[:, None, :]
    valid = selected[:, :, None] & selected[:, None, :]
    eye = torch.eye(selected.size(1), dtype=torch.bool, device=selected.device)[None]
    score = torch.maximum(forward, backward).masked_fill(~valid | eye, -1.0)
    return score.flatten(1).amax(dim=1).clamp_min(0.0)


@torch.no_grad()
def leave_one_out_necessity(
    model: SlotSelectorGRPO,
    slot_embeds: torch.Tensor,
    actions: torch.Tensor,
    masks: torch.Tensor,
    final_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Margin loss caused by removing each selected action from the final path."""
    batch_size, num_slots, _ = slot_embeds.shape
    num_steps = actions.size(1)
    if num_steps == 0:
        return slot_embeds.new_zeros((batch_size, 0))
    final_margin = true_class_margin(final_logits, labels)
    rewards: list[torch.Tensor] = []
    for omitted_step in range(num_steps):
        h = model.initial_state(slot_embeds)
        selected = torch.zeros(batch_size, num_slots, dtype=torch.bool, device=slot_embeds.device)
        for replay_step in range(num_steps):
            replay_active = masks[:, replay_step].bool()
            if replay_step == omitted_step:
                replay_active = torch.zeros_like(replay_active)
            h, selected = model.update_with_action(
                h, selected, slot_embeds, actions[:, replay_step], replay_active
            )
        omitted_margin = true_class_margin(model.classify(h), labels)
        rewards.append((final_margin - omitted_margin) * masks[:, omitted_step])
    return torch.stack(rewards, dim=1).detach()


def rollout_novelty(
    model: SlotSelectorGRPO,
    slots: torch.Tensor,
    attention: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
    sample: bool,
) -> dict[str, torch.Tensor]:
    slot_embeds = model.embed_slots(slots, None)
    batch_size, num_slots, _ = slot_embeds.shape
    attention_probability, attention_quality = normalized_attention_and_quality(attention)
    component_probability = (
        model.classify_components(slot_embeds).sigmoid().detach()
        if model.component_head is not None else None
    )
    coverage = attention_probability.new_zeros((batch_size, attention_probability.size(-1)))
    h = model.initial_state(slot_embeds)
    selected = torch.zeros(batch_size, num_slots, dtype=torch.bool, device=slots.device)
    active = torch.ones(batch_size, dtype=torch.bool, device=slots.device)
    stopped = torch.zeros(batch_size, dtype=torch.bool, device=slots.device)
    logits = model.classify(h)
    previous_margin = true_class_margin(logits, labels).detach()
    previous_balanced_margin = balanced_confusable_margin(logits, labels, args.neighbor_groups).detach()
    previous_component_score = slots.new_zeros(batch_size)

    log_probs: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    margin_increments: list[torch.Tensor] = []
    balanced_margin_increments: list[torch.Tensor] = []
    component_pair_increments: list[torch.Tensor] = []
    novelty_rewards: list[torch.Tensor] = []
    max_remaining_values: list[torch.Tensor] = []

    batch_indices = torch.arange(batch_size, device=slots.device)
    for step in range(min(int(args.max_steps), num_slots)):
        active_f = active.to(slots.dtype)
        novelty_before = candidate_novelty(attention_probability, attention_quality, coverage, selected)
        policy_logits = model.slot_policy_logits(h.detach(), slot_embeds.detach(), selected, step)
        distribution = Categorical(logits=policy_logits)
        slot_action = distribution.sample() if sample else policy_logits.argmax(dim=-1)
        recorded_action = torch.where(active, slot_action, torch.full_like(slot_action, model.stop_idx))
        chosen_novelty = novelty_before[batch_indices, slot_action].clamp_min(0.0) * active_f
        if step + 1 < int(args.novelty_start_step):
            chosen_novelty = torch.zeros_like(chosen_novelty)
        log_probs.append(distribution.log_prob(slot_action) * active_f)
        masks.append(active_f)
        actions.append(recorded_action)
        novelty_rewards.append(chosen_novelty)

        h, selected = model.update_with_action(h, selected, slot_embeds, slot_action, active)
        if component_probability is not None:
            current_component_score = distinct_component_pair_score(
                component_probability, selected, labels, args.class_component_indices
            )
            component_pair_increments.append(
                (current_component_score - previous_component_score) * active_f
            )
            previous_component_score = torch.where(
                active, current_component_score, previous_component_score
            )
        else:
            component_pair_increments.append(slots.new_zeros(batch_size))
        chosen_attention = attention_probability[batch_indices, slot_action]
        coverage = torch.where(active[:, None], torch.maximum(coverage, chosen_attention), coverage)
        step_logits = model.classify(h)
        current_margin = true_class_margin(step_logits, labels).detach()
        current_balanced_margin = balanced_confusable_margin(
            step_logits, labels, args.neighbor_groups
        ).detach()
        margin_increments.append((current_margin - previous_margin) * active_f)
        balanced_margin_increments.append(
            (current_balanced_margin - previous_balanced_margin) * active_f
        )
        previous_margin = torch.where(active, current_margin, previous_margin)
        previous_balanced_margin = torch.where(
            active, current_balanced_margin, previous_balanced_margin
        )
        logits = torch.where(active[:, None], step_logits, logits)

        remaining_novelty = candidate_novelty(attention_probability, attention_quality, coverage, selected)
        max_remaining = remaining_novelty.clamp_min(0.0).amax(dim=1)
        max_remaining_values.append(max_remaining.detach())
        max_confidence = step_logits.softmax(dim=-1).amax(dim=1).detach()
        confidence_ready = max_confidence >= float(args.early_exit_conf)
        novelty_exhausted = max_remaining <= float(args.novelty_stop_threshold)
        stop_now = (
            active
            & (selected.sum(dim=1) >= int(args.min_steps))
            & confidence_ready
            & novelty_exhausted
        )
        stopped = stopped | stop_now
        active = active & ~stop_now
        if not active.any():
            break

    def stack_or_empty(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values, dim=1) if values else slots.new_zeros((batch_size, 0))

    action_tensor = torch.stack(actions, dim=1) if actions else torch.empty(
        batch_size, 0, dtype=torch.long, device=slots.device
    )
    mask_tensor = stack_or_empty(masks)
    necessity_rewards = leave_one_out_necessity(
        model, slot_embeds.detach(), action_tensor, mask_tensor, logits.detach(), labels
    )
    return {
        "logits": logits,
        "selected_mask": selected,
        "selected_count": selected.sum(dim=1).to(slots.dtype).detach(),
        "actions": action_tensor,
        "log_probs": stack_or_empty(log_probs),
        "mask": mask_tensor,
        "margin_increments": stack_or_empty(margin_increments).detach(),
        "balanced_margin_increments": stack_or_empty(balanced_margin_increments).detach(),
        "component_pair_increments": stack_or_empty(component_pair_increments).detach(),
        "necessity_rewards": necessity_rewards,
        "novelty_rewards": stack_or_empty(novelty_rewards).detach(),
        "max_remaining_novelty": stack_or_empty(max_remaining_values).detach(),
        "stopped": stopped.detach(),
        "final_true_conf": logits.softmax(dim=-1).gather(1, labels[:, None]).squeeze(1).detach(),
    }


def masked_group_advantages(
    rewards: torch.Tensor,
    masks: torch.Tensor,
    batch_size: int,
    group_size: int,
    eps: float,
) -> torch.Tensor:
    steps = rewards.size(1)
    reward_group = rewards.detach().view(batch_size, group_size, steps)
    mask_group = masks.detach().view(batch_size, group_size, steps)
    denom = mask_group.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (reward_group * mask_group).sum(dim=1, keepdim=True) / denom
    centered = (reward_group - mean) * mask_group
    scale = (centered.pow(2).sum(dim=1, keepdim=True) / denom).sqrt().clamp_min(float(eps))
    return (centered / scale).reshape(batch_size * group_size, steps)


def policy_loss(
    model: SlotSelectorGRPO,
    slots: torch.Tensor,
    attention: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    batch_size = labels.numel()
    group_size = int(args.grpo_group_size)
    slots_rep = slots.repeat_interleave(group_size, dim=0)
    attention_rep = attention.repeat_interleave(group_size, dim=0)
    labels_rep = labels.repeat_interleave(group_size, dim=0)
    out = rollout_novelty(model, slots_rep, attention_rep, labels_rep, args, sample=True)
    margin_advantage = masked_group_advantages(
        out["margin_increments"], out["mask"], batch_size, group_size, args.grpo_adv_eps
    )
    balanced_advantage = masked_group_advantages(
        out["balanced_margin_increments"], out["mask"], batch_size, group_size, args.grpo_adv_eps
    )
    component_advantage = masked_group_advantages(
        out["component_pair_increments"], out["mask"], batch_size, group_size, args.grpo_adv_eps
    )
    novelty_advantage = masked_group_advantages(
        out["novelty_rewards"], out["mask"], batch_size, group_size, args.grpo_adv_eps
    )
    necessity_advantage = masked_group_advantages(
        out["necessity_rewards"], out["mask"], batch_size, group_size, args.grpo_adv_eps
    )
    advantage = (
        margin_advantage
        + float(args.balanced_margin_coef) * balanced_advantage
        + float(args.component_pair_coef) * component_advantage
        + float(args.necessity_coef) * necessity_advantage
        + float(args.novelty_coef) * novelty_advantage
    )
    active_steps = out["mask"].sum().clamp_min(1.0)
    loss_policy = -((out["log_probs"] * advantage * out["mask"]).sum() / active_steps)
    loss_ce = F.cross_entropy(out["logits"], labels_rep, label_smoothing=float(args.label_smoothing))
    return out, loss_ce, loss_policy


def random_subset_rollout(
    model: SlotSelectorGRPO,
    slots: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    """Classifier warmup on random ordered subsets with sizes min_steps..max_steps."""
    slot_embeds = model.embed_slots(slots, None)
    batch_size, num_slots, _ = slot_embeds.shape
    h = model.initial_state(slot_embeds)
    selected = torch.zeros(batch_size, num_slots, dtype=torch.bool, device=slots.device)
    target_counts = torch.randint(
        int(args.min_steps), min(int(args.max_steps), num_slots) + 1, (batch_size,), device=slots.device
    )
    for step in range(min(int(args.max_steps), num_slots)):
        score = torch.rand(batch_size, num_slots, device=slots.device).masked_fill(selected, -1.0)
        action = score.argmax(dim=1)
        active = step < target_counts
        h, selected = model.update_with_action(h, selected, slot_embeds, action, active)
    logits = model.classify(h)
    return {
        "logits": logits,
        "selected_count": selected.sum(dim=1).to(slots.dtype).detach(),
        "stopped": torch.zeros(batch_size, dtype=torch.bool, device=slots.device),
        "margin_increments": slots.new_zeros((batch_size, 0)),
        "balanced_margin_increments": slots.new_zeros((batch_size, 0)),
        "component_pair_increments": slots.new_zeros((batch_size, 0)),
        "necessity_rewards": slots.new_zeros((batch_size, 0)),
        "novelty_rewards": slots.new_zeros((batch_size, 0)),
        "max_remaining_novelty": slots.new_zeros((batch_size, 0)),
        "mask": slots.new_zeros((batch_size, 0)),
        "final_true_conf": logits.softmax(dim=-1).gather(1, labels[:, None]).squeeze(1).detach(),
    }


def ce_parameters(model: SlotSelectorGRPO) -> list[nn.Parameter]:
    params = [model.h0, *model.slot_embed.parameters(), *model.gru.parameters(), *model.classifier.parameters()]
    if model.component_head is not None:
        params.extend(model.component_head.parameters())
    return params


def policy_parameters(model: SlotSelectorGRPO) -> list[nn.Parameter]:
    params: list[nn.Parameter] = [*model.query.parameters(), *model.key.parameters()]
    if model.first_step_query is not None:
        params.append(model.first_step_query)
    if model.first_step_attention is not None:
        params.extend(model.first_step_attention.parameters())
    if model.first_step_norm is not None:
        params.extend(model.first_step_norm.parameters())
    return params


def freeze_classifier(model: SlotSelectorGRPO) -> None:
    for parameter in ce_parameters(model):
        parameter.requires_grad_(False)
    model.slot_embed.eval()
    model.gru.eval()
    model.classifier.eval()
    if model.component_head is not None:
        model.component_head.eval()


def run_epoch(
    model: SlotSelectorGRPO,
    backbone,
    loader,
    device: torch.device,
    optimizer,
    args: argparse.Namespace,
    phase: str,
) -> dict[str, float]:
    train = phase in {"warmup", "policy"}
    model.train(train)
    if phase == "policy" and args.freeze_classifier_after_warmup:
        freeze_classifier(model)
    total = 0
    sums = {key: 0.0 for key in (
        "loss", "ce", "policy", "correct", "selected", "stopped", "confidence", "margin",
        "balanced_margin", "component_pair", "necessity", "novelty", "remaining"
    )}
    for images, labels in loader:
        labels = labels.to(device, non_blocking=device.type == "cuda")
        slots, attention = encode_slots_and_attention(backbone, images, device)
        metric_labels = labels
        if phase == "warmup":
            group_size = int(args.grpo_group_size)
            warmup_slots = slots.repeat_interleave(group_size, dim=0)
            warmup_labels = labels.repeat_interleave(group_size, dim=0)
            out = random_subset_rollout(model, warmup_slots, warmup_labels, args)
            loss_ce = F.cross_entropy(out["logits"], warmup_labels, label_smoothing=float(args.label_smoothing))
            if model.component_head is not None:
                all_slot_embeds = model.embed_slots(slots, None)
                slot_component_logits = model.classify_components(all_slot_embeds)
                temperature = float(args.component_mil_temperature)
                image_component_logits = torch.logsumexp(
                    slot_component_logits * temperature, dim=1
                ) / temperature - float(torch.log(torch.tensor(float(slot_component_logits.size(1))))) / temperature
                component_targets = slots.new_zeros((labels.numel(), len(args.components)))
                target_indices = torch.tensor(
                    args.class_component_indices, dtype=torch.long, device=device
                )[labels]
                component_targets.scatter_(1, target_indices, 1.0)
                component_loss = F.binary_cross_entropy_with_logits(
                    image_component_logits, component_targets
                )
                loss_ce = loss_ce + float(args.component_mil_coef) * component_loss
            loss_policy = loss_ce.new_zeros(())
            loss = loss_ce
            metric_labels = warmup_labels
        elif phase == "policy":
            out, loss_ce, loss_policy = policy_loss(model, slots, attention, labels, args)
            loss = loss_ce + loss_policy
            metric_labels = labels.repeat_interleave(int(args.grpo_group_size))
        else:
            with torch.no_grad():
                out = rollout_novelty(model, slots, attention, labels, args, sample=False)
                loss_ce = F.cross_entropy(out["logits"], labels, label_smoothing=float(args.label_smoothing))
                loss_policy = loss_ce.new_zeros(())
                loss = loss_ce
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), float(args.grad_clip))
            optimizer.step()

        batch = metric_labels.numel()
        total += batch
        sums["loss"] += float(loss.detach()) * batch
        sums["ce"] += float(loss_ce.detach()) * batch
        sums["policy"] += float(loss_policy.detach()) * batch
        sums["correct"] += float((out["logits"].argmax(dim=1) == metric_labels).sum())
        sums["selected"] += float(out["selected_count"].sum())
        sums["stopped"] += float(out["stopped"].float().sum())
        sums["confidence"] += float(out["final_true_conf"].sum())
        sums["margin"] += float(out["margin_increments"].sum())
        sums["balanced_margin"] += float(out["balanced_margin_increments"].sum())
        sums["component_pair"] += float(out["component_pair_increments"].sum())
        sums["necessity"] += float(out["necessity_rewards"].sum())
        sums["novelty"] += float(out["novelty_rewards"].sum())
        if out["max_remaining_novelty"].numel():
            last_index = out["mask"].sum(dim=1).long().clamp_min(1) - 1
            last_remaining = out["max_remaining_novelty"].gather(1, last_index[:, None]).squeeze(1)
            sums["remaining"] += float(last_remaining.sum())
    return {
        "loss": sums["loss"] / max(total, 1),
        "train_ce": sums["ce"] / max(total, 1),
        "policy_loss": sums["policy"] / max(total, 1),
        "acc": sums["correct"] / max(total, 1),
        "avg_selected": sums["selected"] / max(total, 1),
        "stopped_rate": sums["stopped"] / max(total, 1),
        "final_true_conf": sums["confidence"] / max(total, 1),
        "avg_margin_increment": sums["margin"] / max(total, 1),
        "avg_balanced_margin_increment": sums["balanced_margin"] / max(total, 1),
        "avg_component_pair_increment": sums["component_pair"] / max(total, 1),
        "avg_necessity_reward": sums["necessity"] / max(total, 1),
        "avg_novelty_reward": sums["novelty"] / max(total, 1),
        "final_max_remaining_novelty": sums["remaining"] / max(total, 1),
    }


def write_history(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--sa_checkpoint", default=DEFAULT_SA)
    parser.add_argument("--output_dir", default="GRPO6-factorized/checkpoints/grpo6_factorized")
    parser.add_argument("--input_res", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--warmup_epochs", type=int, default=15)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--policy_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--min_steps", type=int, default=3)
    parser.add_argument("--early_exit_conf", type=float, default=0.85)
    parser.add_argument("--novelty_stop_threshold", type=float, default=0.90)
    parser.add_argument("--novelty_start_step", type=int, default=2)
    parser.add_argument("--novelty_coef", type=float, default=0.5)
    parser.add_argument("--necessity_coef", type=float, default=0.0)
    parser.add_argument("--balanced_margin_coef", type=float, default=0.0)
    parser.add_argument("--component_pair_coef", type=float, default=0.0)
    parser.add_argument("--component_mil_coef", type=float, default=0.0)
    parser.add_argument("--component_mil_temperature", type=float, default=5.0)
    parser.add_argument("--grpo_group_size", type=int, default=4)
    parser.add_argument("--grpo_adv_eps", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--freeze_classifier_after_warmup", action="store_true", default=True)
    parser.add_argument("--no_freeze_classifier_after_warmup", dest="freeze_classifier_after_warmup", action="store_false")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--quick_limit_train", type=int, default=0)
    parser.add_argument("--quick_limit_val", type=int, default=0)
    args = parser.parse_args()
    if not 0 < args.warmup_epochs < args.epochs:
        raise ValueError("warmup_epochs must be in (0, epochs)")
    if args.min_steps < 1 or args.max_steps < args.min_steps:
        raise ValueError("invalid min/max steps")
    if args.grpo_group_size <= 1:
        raise ValueError("grpo_group_size must exceed one")
    return args


def main() -> None:
    args = parse_args()
    seed_all(int(args.seed), False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transforms = build_transforms(int(args.input_res))
    train_set = subset_dataset(build_dataset(args.data, "train", transforms["train"]), args.quick_limit_train, args.seed)
    valid_set = subset_dataset(build_dataset(args.data, "valid", transforms["valid"]), args.quick_limit_val, args.seed)
    classes = class_names_from_dataset(train_set)
    args.neighbor_groups = class_neighbor_groups(classes)
    args.components, args.class_component_indices = class_component_structure(classes)
    train_loader = grpo.make_loader(train_set, args, device, shuffle=True)
    valid_loader = grpo.make_loader(valid_set, args, device, shuffle=False)
    backbone = load_backbone(args.sa_checkpoint, device)
    backbone.eval()
    backbone.requires_grad_(False)
    config = GRPOSelectorConfig(
        num_slots=int(backbone.num_slots), slot_dim=int(backbone.slot_dim), num_classes=len(classes),
        hidden_dim=int(args.hidden_dim), policy_dim=int(args.policy_dim), max_steps=int(args.max_steps),
        dropout=float(args.dropout), min_steps=int(args.min_steps), early_exit_conf=float(args.early_exit_conf),
        policy_context_attention=True, first_step_num_heads=4,
        num_components=len(args.components) if args.component_mil_coef > 0 else 0,
    )
    model = SlotSelectorGRPO(config).to(device)
    ce_params, policy_params = ce_parameters(model), policy_parameters(model)
    if {id(p) for p in ce_params} & {id(p) for p in policy_params}:
        raise ValueError("CE and policy parameter groups overlap")
    warmup_optimizer = torch.optim.AdamW(ce_params, lr=float(args.lr), weight_decay=float(args.wd))
    policy_optimizer = torch.optim.AdamW(policy_params, lr=float(args.policy_lr), weight_decay=float(args.wd))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "args": vars(args), "config": asdict(config), "classes": classes,
        "uses_bbox_or_slothead": False, "slot_source": "raw_frozen_dinosaur_slots_and_attention",
        "loss": "train_ce + policy_loss",
        "class_reward": "true-class logit-margin increment",
        "balanced_margin_reward": "worst margin over class competitors sharing either composition component",
        "component_pair_reward": "best distinct-slot assignment to the two image-label components",
        "necessity_reward": "final margin minus margin after replaying the path without each selected slot",
        "novelty_reward": "relative attention concentration * positive new attention coverage",
        "stop": "min_steps AND confidence threshold AND remaining novelty threshold",
    }
    (out_dir / "experiment_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"device={device} train={len(train_set)} valid={len(valid_set)} classes={len(classes)} slots={config.num_slots}")
    print(f"warmup={args.warmup_epochs} freeze={args.freeze_classifier_after_warmup} component_pair_coef={args.component_pair_coef} component_mil_coef={args.component_mil_coef} balanced_coef={args.balanced_margin_coef} necessity_coef={args.necessity_coef} novelty_coef={args.novelty_coef} stop_novelty={args.novelty_stop_threshold}")

    best_acc, best_slots = -1.0, float("inf")
    history: list[dict] = []
    start = time.time()
    classifier_frozen = False
    for epoch in range(1, int(args.epochs) + 1):
        phase = "warmup" if epoch <= int(args.warmup_epochs) else "policy"
        if phase == "policy" and args.freeze_classifier_after_warmup and not classifier_frozen:
            freeze_classifier(model)
            classifier_frozen = True
            torch.save({"epoch": epoch - 1, "model_state_dict": model.state_dict(), "config": asdict(config)}, out_dir / "classifier_warmup.pt")
        optimizer = warmup_optimizer if phase == "warmup" else policy_optimizer
        train_metrics = run_epoch(model, backbone, train_loader, device, optimizer, args, phase)
        valid_metrics = run_epoch(model, backbone, valid_loader, device, optimizer, args, "valid")
        row = {"epoch": epoch, "phase": phase, "elapsed": time.strftime("%H:%M:%S", time.gmtime(time.time()-start)),
               **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"valid_{k}": v for k, v in valid_metrics.items()}}
        history.append(row)
        write_history(out_dir / "history_metrics.csv", history)
        print(f"epoch={epoch} phase={phase} train_acc={100*train_metrics['acc']:.2f} valid_acc={100*valid_metrics['acc']:.2f} valid_slots={valid_metrics['avg_selected']:.2f} stop={100*valid_metrics['stopped_rate']:.1f}% rem_nov={valid_metrics['final_max_remaining_novelty']:.3f}")
        if phase == "policy" and (valid_metrics["acc"] > best_acc or (valid_metrics["acc"] == best_acc and valid_metrics["avg_selected"] < best_slots)):
            best_acc, best_slots = valid_metrics["acc"], valid_metrics["avg_selected"]
            torch.save({"epoch": epoch, "valid_acc": best_acc, "valid_avg_selected": best_slots,
                        "model_state_dict": model.state_dict(), "config": asdict(config), "args": vars(args), "classes": classes},
                       out_dir / "selector_grpo_best.pt")
    (out_dir / "final_metrics.json").write_text(json.dumps({"best_valid_acc": best_acc, "best_valid_avg_selected": best_slots, "history": history}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
