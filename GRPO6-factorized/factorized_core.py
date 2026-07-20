"""Shared undirected factorized-MIL rewards and rollout utilities for GRPO6."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from selector_grpo import SlotSelectorGRPO


@torch.no_grad()
def encode(backbone, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.mlp(backbone.forward_dino(images))
    slots, _attention, _ = backbone.slot_attention(features)
    return slots.detach()


def load_rules(data: str, classes: list[str]):
    summary_path = Path(data) / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    by_class = {row["class_name"]: row for row in summary["pairs"]}
    missing = [name for name in classes if name not in by_class]
    if missing:
        raise ValueError(f"dataset summary has no fixed pair for classes: {missing}")
    objects = sorted(
        {by_class[name][role] for name in classes for role in ("object_a", "object_b")}
    )
    object_to_id = {name: index for index, name in enumerate(objects)}
    object_a_ids = torch.tensor(
        [object_to_id[by_class[name]["object_a"]] for name in classes], dtype=torch.long
    )
    object_b_ids = torch.tensor(
        [object_to_id[by_class[name]["object_b"]] for name in classes], dtype=torch.long
    )
    rules = {
        name: {
            "object_a": by_class[name]["object_a"],
            "object_b": by_class[name]["object_b"],
        }
        for name in classes
    }
    return objects, object_a_ids, object_b_ids, rules


def component_targets(
    labels: torch.Tensor,
    object_a_ids: torch.Tensor,
    object_b_ids: torch.Tensor,
    num_objects: int,
) -> torch.Tensor:
    targets = torch.zeros(
        labels.numel(), num_objects, dtype=torch.float32, device=labels.device
    )
    targets.scatter_(1, object_a_ids[labels, None], 1.0)
    targets.scatter_(1, object_b_ids[labels, None], 1.0)
    return targets


def mil_logits(slot_component_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature-controlled smooth max over slots, constant-preserving."""
    num_slots = slot_component_logits.size(1)
    return (
        torch.logsumexp(slot_component_logits * temperature, dim=1) / temperature
        - slot_component_logits.new_tensor(num_slots).log() / temperature
    )


def selected_support(
    component_prob: torch.Tensor,
    selected_mask: torch.Tensor,
    labels: torch.Tensor,
    object_a_ids: torch.Tensor,
    object_b_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return object-a, object-b, and undirected distinct-slot pair support."""
    batch, num_slots, _ = component_prob.shape
    rows = torch.arange(batch, device=component_prob.device)[:, None]
    slot_ids = torch.arange(num_slots, device=component_prob.device)[None, :]
    object_a_prob = component_prob[rows, slot_ids, object_a_ids[labels, None]]
    object_b_prob = component_prob[rows, slot_ids, object_b_ids[labels, None]]
    object_a_support = object_a_prob.masked_fill(~selected_mask, -1.0).amax(1).clamp_min(0.0)
    object_b_support = object_b_prob.masked_fill(~selected_mask, -1.0).amax(1).clamp_min(0.0)

    forward = object_a_prob[:, :, None] * object_b_prob[:, None, :]
    backward = object_b_prob[:, :, None] * object_a_prob[:, None, :]
    pair_prob = torch.maximum(forward, backward)
    valid_pair_mask = selected_mask[:, :, None] & selected_mask[:, None, :]
    same_slot = torch.eye(num_slots, dtype=torch.bool, device=component_prob.device)[None]
    pair_support = pair_prob.masked_fill(~valid_pair_mask | same_slot, -1.0).amax(dim=(1, 2)).clamp_min(0.0)
    return object_a_support, object_b_support, pair_support


def rule_pair_logits(
    component_prob: torch.Tensor,
    selected_mask: torch.Tensor,
    object_a_ids: torch.Tensor,
    object_b_ids: torch.Tensor,
) -> torch.Tensor:
    class_pair_logits = []
    for class_id in range(object_a_ids.numel()):
        labels = torch.full(
            (component_prob.size(0),), class_id, dtype=torch.long, device=component_prob.device
        )
        _object_a_support, _object_b_support, pair_support = selected_support(
            component_prob, selected_mask, labels, object_a_ids, object_b_ids
        )
        class_pair_logits.append(torch.log(pair_support + 1e-4))
    return torch.stack(class_pair_logits, dim=1)


def true_logit_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.masked_fill(F.one_hot(labels, logits.size(1)).bool(), -1e9)
    return true - torch.logsumexp(other, dim=1)


def group_advantage(
    reward: torch.Tensor,
    active_mask: torch.Tensor,
    batch: int,
    group: int,
    eps: float = 1e-4,
) -> torch.Tensor:
    reward_group = reward.detach().view(batch, group, -1)
    active_group = active_mask.detach().view(batch, group, -1)
    denominator = active_group.sum(1, keepdim=True).clamp_min(1.0)
    mean = (reward_group * active_group).sum(1, keepdim=True) / denominator
    centered = (reward_group - mean) * active_group
    scale = (centered.square().sum(1, keepdim=True) / denominator).sqrt().clamp_min(eps)
    return (centered / scale).reshape(batch * group, -1)


def ce_parameters(model: SlotSelectorGRPO) -> list[torch.nn.Parameter]:
    if model.component_head is None:
        raise ValueError("factorized training requires component_head")
    return [
        model.h0,
        *model.slot_embed.parameters(),
        *model.gru.parameters(),
        *model.classifier.parameters(),
        *model.component_head.parameters(),
    ]


def policy_parameters(model: SlotSelectorGRPO) -> list[torch.nn.Parameter]:
    parameters = [*model.query.parameters(), *model.key.parameters()]
    if model.first_step_query is not None:
        parameters.append(model.first_step_query)
    if model.first_step_attention is not None:
        parameters.extend(model.first_step_attention.parameters())
    if model.first_step_norm is not None:
        parameters.extend(model.first_step_norm.parameters())
    return parameters


def rollout(
    model: SlotSelectorGRPO,
    slots: torch.Tensor,
    labels: torch.Tensor,
    args,
    sample: bool,
    object_a_ids: torch.Tensor,
    object_b_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    slot_embeds = model.embed_slots(slots, None)
    component_prob = model.classify_components(slot_embeds).sigmoid().detach()
    batch, num_slots, _ = slot_embeds.shape
    hidden = model.initial_state(slot_embeds)
    selected_mask = torch.zeros(batch, num_slots, dtype=torch.bool, device=slots.device)
    active_mask = torch.ones(batch, dtype=torch.bool, device=slots.device)
    previous_object_a_support = slots.new_zeros(batch)
    previous_object_b_support = slots.new_zeros(batch)
    previous_pair_support = slots.new_zeros(batch)
    class_logits = model.classify(hidden)
    previous_class_margin = true_logit_margin(class_logits, labels).detach()
    log_prob_steps, active_steps = [], []
    object_a_reward_steps, object_b_reward_steps = [], []
    component_pair_reward_steps, class_margin_reward_steps = [], []

    for step in range(min(int(args.max_steps), num_slots)):
        policy_logits = model.slot_policy_logits(hidden.detach(), slot_embeds.detach(), selected_mask, step)
        distribution = Categorical(logits=policy_logits)
        action = distribution.sample() if sample else policy_logits.argmax(1)
        log_prob_steps.append(distribution.log_prob(action))
        active_steps.append(active_mask.to(slots.dtype))
        hidden, selected_mask = model.update_with_action(hidden, selected_mask, slot_embeds, action, active_mask)

        object_a_support, object_b_support, pair_support = selected_support(
            component_prob, selected_mask, labels, object_a_ids, object_b_ids
        )
        discount = float(args.rank_discount) ** step
        object_a_reward_steps.append((object_a_support - previous_object_a_support) * discount)
        object_b_reward_steps.append((object_b_support - previous_object_b_support) * discount)
        component_pair_reward_steps.append((pair_support - previous_pair_support) * discount)
        class_margin_now = true_logit_margin(model.classify(hidden), labels).detach()
        class_margin_reward_steps.append((class_margin_now - previous_class_margin) * discount)
        previous_object_a_support = object_a_support
        previous_object_b_support = object_b_support
        previous_pair_support = pair_support
        previous_class_margin = class_margin_now
        class_logits = model.classify(hidden)

    stack = lambda values: torch.stack(values, dim=1)
    return {
        "logits": class_logits,
        "log_probs": stack(log_prob_steps),
        "mask": stack(active_steps),
        "object_a": stack(object_a_reward_steps),
        "object_b": stack(object_b_reward_steps),
        "component_pair": stack(component_pair_reward_steps),
        "class_margin": stack(class_margin_reward_steps),
        "selected": selected_mask,
    }


@torch.no_grad()
def forced_metrics(
    model: SlotSelectorGRPO,
    slots: torch.Tensor,
    labels: torch.Tensor,
    top_ks: tuple[int, ...],
    object_a_ids: torch.Tensor,
    object_b_ids: torch.Tensor,
) -> dict[int, dict[str, torch.Tensor]]:
    slot_embeds = model.embed_slots(slots, None)
    component_prob = model.classify_components(slot_embeds).sigmoid()
    batch, num_slots, _ = slot_embeds.shape
    if max(top_ks) > num_slots:
        raise ValueError(f"top-k {max(top_ks)} exceeds available slots {num_slots}")
    hidden = model.initial_state(slot_embeds)
    selected_mask = torch.zeros(batch, num_slots, dtype=torch.bool, device=slots.device)
    active_mask = torch.ones(batch, dtype=torch.bool, device=slots.device)
    output = {}
    for step in range(max(top_ks)):
        action = model.slot_policy_logits(hidden, slot_embeds, selected_mask, step).argmax(1)
        hidden, selected_mask = model.update_with_action(hidden, selected_mask, slot_embeds, action, active_mask)
        k = step + 1
        if k in top_ks:
            object_a_support, object_b_support, pair_support = selected_support(
                component_prob, selected_mask, labels, object_a_ids, object_b_ids
            )
            output[k] = {
                "object_a": object_a_support,
                "object_b": object_b_support,
                "pair": pair_support,
                "pair_hit": (object_a_support >= 0.5) & (object_b_support >= 0.5),
                "correct": model.classify(hidden).argmax(1).eq(labels),
            }
    return output
