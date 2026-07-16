from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GRPOSelectorConfig:
    num_slots: int
    slot_dim: int
    num_classes: int
    hidden_dim: int = 256
    policy_dim: int = 256
    max_steps: int = 0
    dropout: float = 0.1
    pos_dim: int = 0
    min_steps: int = 0
    early_exit_conf: float = 0.8
    decoupled_stop_policy: bool = False
    first_step_cross_attention: bool = False
    first_step_num_heads: int = 4
    policy_context_attention: bool = False
    num_components: int = 0

    def __post_init__(self) -> None:
        if self.max_steps <= 0:
            self.max_steps = self.num_slots
        if self.first_step_num_heads <= 0:
            raise ValueError("first_step_num_heads must be positive")
        if (self.first_step_cross_attention or self.policy_context_attention) and self.hidden_dim % self.first_step_num_heads != 0:
            raise ValueError("hidden_dim must be divisible by first_step_num_heads")


@dataclass
class SelectorOutput:
    logits: torch.Tensor
    selected_mask: torch.Tensor
    actions: torch.Tensor
    stopped: torch.Tensor


def true_class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    other_logits = logits.masked_fill(
        F.one_hot(labels, logits.size(1)).bool(),
        torch.finfo(logits.dtype).min,
    ).amax(dim=1)
    return true_logits - other_logits


class SlotSelectorGRPO(nn.Module):
    """Sequential hard slot selector trained with GRPO."""

    def __init__(self, cfg: GRPOSelectorConfig):
        super().__init__()
        self.cfg = cfg
        self.stop_idx = cfg.num_slots
        in_dim = cfg.slot_dim + cfg.pos_dim

        self.slot_embed = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, cfg.policy_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.policy_dim, cfg.hidden_dim),
        )
        self.h0 = nn.Parameter(torch.zeros(cfg.hidden_dim))
        self.gru = nn.GRUCell(cfg.hidden_dim, cfg.hidden_dim)

        if cfg.first_step_cross_attention or cfg.policy_context_attention:
            self.first_step_query = nn.Parameter(torch.zeros(1, 1, cfg.hidden_dim))
            self.first_step_attention = nn.MultiheadAttention(
                cfg.hidden_dim,
                cfg.first_step_num_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.first_step_norm = nn.LayerNorm(cfg.hidden_dim)
        else:
            self.first_step_query = None
            self.first_step_attention = None
            self.first_step_norm = None

        self.query = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.key = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.stop_key = nn.Parameter(torch.randn(cfg.hidden_dim) * 0.02)
        self.stop_bias = nn.Parameter(torch.zeros(()))
        self.stop_head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, 1),
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )
        self.component_head = nn.Linear(cfg.hidden_dim, cfg.num_components) if cfg.num_components > 0 else None

    def embed_slots(self, slots: torch.Tensor, slot_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.cfg.pos_dim > 0:
            if slot_pos is None:
                raise ValueError("slot_pos is required when GRPOSelectorConfig.pos_dim > 0")
            if slot_pos.size(-1) != self.cfg.pos_dim:
                raise ValueError(f"Expected slot_pos dim {self.cfg.pos_dim}, got {slot_pos.size(-1)}")
            slots = torch.cat([slots, slot_pos.to(dtype=slots.dtype)], dim=-1)
        return self.slot_embed(slots)

    def initial_state(self, slot_embeds: torch.Tensor) -> torch.Tensor:
        batch = slot_embeds.size(0)
        return self.h0.unsqueeze(0).expand(batch, -1)

    def action_state(self, h: torch.Tensor, slot_embeds: torch.Tensor, step: int) -> torch.Tensor:
        if self.first_step_attention is None:
            return h
        if step != 0 and not self.cfg.policy_context_attention:
            return h
        if self.first_step_query is None or self.first_step_norm is None:
            raise RuntimeError("first-step cross-attention layers are not initialized")
        learned_query = self.first_step_query.expand(slot_embeds.size(0), -1, -1)
        query = learned_query if step == 0 else h.unsqueeze(1)
        context, _ = self.first_step_attention(query, slot_embeds, slot_embeds, need_weights=False)
        if self.cfg.policy_context_attention and step != 0:
            return self.first_step_norm(h + context.squeeze(1))
        return self.first_step_norm(context.squeeze(1))

    def slot_policy_logits(
        self,
        h: torch.Tensor,
        slot_embeds: torch.Tensor,
        selected_mask: torch.Tensor,
        step: int = 0,
    ) -> torch.Tensor:
        action_h = self.action_state(h, slot_embeds, step)
        q = self.query(action_h).unsqueeze(1)
        k = self.key(slot_embeds)
        logits = (q * k).sum(dim=-1) * (h.size(-1) ** -0.5)
        return logits.masked_fill(selected_mask, torch.finfo(logits.dtype).min)

    def policy_logits(
        self,
        h: torch.Tensor,
        slot_embeds: torch.Tensor,
        selected_mask: torch.Tensor,
        step: int = 0,
    ) -> torch.Tensor:
        slot_logits = self.slot_policy_logits(h, slot_embeds, selected_mask, step)
        stop_logit = (self.query(h) * self.stop_key).sum(dim=-1, keepdim=True)
        stop_logit = stop_logit * (h.size(-1) ** -0.5) + self.stop_bias
        return torch.cat([slot_logits, stop_logit], dim=1)

    def update_with_action(
        self,
        h: torch.Tensor,
        selected_mask: torch.Tensor,
        slot_embeds: torch.Tensor,
        action: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, k, d = slot_embeds.shape
        slot_action = action.clamp(0, k - 1)
        chosen = slot_embeds.gather(1, slot_action.view(b, 1, 1).expand(-1, 1, d)).squeeze(1)
        do_select = active & (action < k) & ~selected_mask.gather(1, slot_action[:, None]).squeeze(1)
        h_next = self.gru(chosen, h)
        h = torch.where(do_select[:, None], h_next, h)
        selected_mask = selected_mask.clone()
        selected_mask.scatter_(1, slot_action[:, None], selected_mask.gather(1, slot_action[:, None]) | do_select[:, None])
        return h, selected_mask

    def classify(
        self,
        h: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(h)

    def classify_components(self, slot_embeds: torch.Tensor) -> torch.Tensor:
        if self.component_head is None:
            raise RuntimeError("component_head is disabled")
        return self.component_head(slot_embeds)

    def mask_order_logits(self, slot_embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, k, _ = slot_embeds.shape
        h = self.initial_state(slot_embeds)
        selected = torch.zeros(b, k, dtype=torch.bool, device=slot_embeds.device)
        active = torch.ones(b, dtype=torch.bool, device=slot_embeds.device)
        for idx in range(k):
            action = torch.full((b,), idx, dtype=torch.long, device=slot_embeds.device)
            should_select = mask[:, idx] & active
            h, selected = self.update_with_action(
                h,
                selected,
                slot_embeds,
                action,
                should_select,
            )
        return self.classify(h)

    @torch.no_grad()
    def forward_greedy(
        self,
        slots: torch.Tensor,
        slot_pos: Optional[torch.Tensor] = None,
        min_steps: Optional[int] = None,
        early_exit_conf: Optional[float] = None,
        confidence_early_exit: bool = False,
    ) -> SelectorOutput:
        slot_embeds = self.embed_slots(slots, slot_pos)
        b, k, _ = slot_embeds.shape
        min_steps = self.cfg.min_steps if min_steps is None else min_steps
        early_exit_conf = self.cfg.early_exit_conf if early_exit_conf is None else early_exit_conf

        h = self.initial_state(slot_embeds)
        selected = torch.zeros(b, k, dtype=torch.bool, device=slots.device)
        active = torch.ones(b, dtype=torch.bool, device=slots.device)
        stopped = torch.zeros(b, dtype=torch.bool, device=slots.device)
        actions = []
        logits = self.classify(h)
        for step in range(min(self.cfg.max_steps, k)):
            can_stop = selected.sum(dim=1) >= int(min_steps)
            if self.cfg.decoupled_stop_policy:
                stop_now = can_stop & (torch.sigmoid(self.stop_head(h).squeeze(-1)) >= 0.5)
                action_logits = self.slot_policy_logits(h, slot_embeds, selected, step)
                action = action_logits.argmax(dim=-1)
                action = torch.where(stop_now, torch.full_like(action, self.stop_idx), action)
            else:
                action_logits = self.policy_logits(h, slot_embeds, selected, step)
                action_logits[:, self.stop_idx] = torch.where(
                    can_stop,
                    action_logits[:, self.stop_idx],
                    torch.full_like(action_logits[:, self.stop_idx], torch.finfo(action_logits.dtype).min),
                )
                action = action_logits.argmax(dim=-1)
                stop_now = action == self.stop_idx
            select = active & ~stop_now
            h, selected = self.update_with_action(h, selected, slot_embeds, action, select)
            new_logits = self.classify(h)
            logits = torch.where(active[:, None], new_logits, logits)
            if confidence_early_exit:
                conf_stop = active & (selected.sum(dim=1) >= int(min_steps)) & (new_logits.softmax(dim=-1).amax(dim=1) >= early_exit_conf)
            else:
                conf_stop = torch.zeros_like(active)
            active = active & ~stop_now & ~conf_stop
            stopped = stopped | stop_now | conf_stop
            actions.append(action)
            if not active.any():
                break
        if actions:
            action_t = torch.stack(actions, dim=1)
        else:
            action_t = torch.empty(b, 0, dtype=torch.long, device=slots.device)
        return SelectorOutput(logits=logits, selected_mask=selected, actions=action_t, stopped=stopped)
