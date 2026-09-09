"""Shared undirected contrastive pair rewards and rollout utilities for GRPO6."""

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


def rule_pair_margin(
    component_prob: torch.Tensor,
    selected_mask: torch.Tensor,
    labels: torch.Tensor,
    object_a_ids: torch.Tensor,
    object_b_ids: torch.Tensor,
) -> torch.Tensor:
    """Contrast the true rule pair against other complete class pairs.

    Objects are intentionally not treated as negatives here because the dataset
    reuses objects across classes. The negative set is only the other full
    object-pair rules.
    """
    logits = rule_pair_logits(component_prob, selected_mask, object_a_ids, object_b_ids)
    return true_logit_margin(logits, labels)


def true_logit_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true = logits.gather(1, labels[:, None]).squeeze(1)
    other = logits.masked_fill(F.one_hot(labels, logits.size(1)).bool(), -1e9)
    return true - torch.logsumexp(other, dim=1)


def binary_auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    labels = labels.float()
    pos = labels.sum()
    neg = labels.numel() - pos
    if pos <= 0 or neg <= 0:
        return None
    sorted_scores, order = scores.sort()
    sorted_ranks = torch.arange(1, scores.numel() + 1, dtype=torch.float64)
    _unique_scores, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    rank_sums = torch.split(sorted_ranks, counts.tolist())
    average_ranks = torch.cat(
        [
            ranks.new_full((count,), float(ranks.mean()))
            for ranks, count in zip(rank_sums, counts.tolist())
        ]
    )
    ranks = torch.empty_like(scores, dtype=torch.float64)
    ranks[order] = average_ranks
    pos_rank_sum = ranks[labels.bool()].sum()
    auc = (pos_rank_sum - pos.double() * (pos.double() + 1.0) / 2.0) / (
        pos.double() * neg.double()
    )
    return float(auc)


def multiclass_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = logits.softmax(dim=-1).cpu()
    labels = labels.cpu()
    num_classes = probs.size(1)
    pred = probs.argmax(dim=-1)
    tp = torch.zeros(num_classes, dtype=torch.float64)
    fp = torch.zeros(num_classes, dtype=torch.float64)
    fn = torch.zeros(num_classes, dtype=torch.float64)
    for cls in range(num_classes):
        pred_cls = pred == cls
        true_cls = labels == cls
        tp[cls] = (pred_cls & true_cls).sum()
        fp[cls] = (pred_cls & ~true_cls).sum()
        fn[cls] = (~pred_cls & true_cls).sum()
    precision = tp / (tp + fp).clamp_min(1e-8)
    recall = tp / (tp + fn).clamp_min(1e-8)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1e-8)
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    aucs = [binary_auc(probs[:, cls], one_hot[:, cls]) for cls in range(num_classes)]
    valid_aucs = [auc for auc in aucs if auc is not None]
    micro_auc = binary_auc(probs.flatten(), one_hot.flatten())
    macro_auc = sum(valid_aucs) / len(valid_aucs) if valid_aucs else 0.0
    accuracy = pred.eq(labels).float().mean().item()
    return {
        "acc": float(accuracy),
        "accuracy": float(accuracy),
        "precision": float(precision.mean()),
        "recall": float(recall.mean()),
        "f1": float(f1.mean()),
        "auc": macro_auc,
        "macro_auc": macro_auc,
        "micro_auc": micro_auc if micro_auc is not None else 0.0,
    }


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
        raise ValueError("contrastive pair training requires component_head")
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
    previous_pair_margin = rule_pair_margin(
        component_prob, selected_mask, labels, object_a_ids, object_b_ids
    )
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
        pair_margin = rule_pair_margin(
            component_prob, selected_mask, labels, object_a_ids, object_b_ids
        )
        component_pair_reward_steps.append((pair_margin - previous_pair_margin) * discount)
        class_margin_now = true_logit_margin(model.classify(hidden), labels).detach()
        class_margin_reward_steps.append((class_margin_now - previous_class_margin) * discount)
        previous_object_a_support = object_a_support
        previous_object_b_support = object_b_support
        previous_pair_margin = pair_margin
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
            output[k] = {
                "pair_margin": rule_pair_margin(
                    component_prob, selected_mask, labels, object_a_ids, object_b_ids
                ),
                "logits": model.classify(hidden),
            }
            output[k]["correct"] = output[k]["logits"].argmax(1).eq(labels)
    return output
