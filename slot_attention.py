from typing import Optional

import math
import torch
import torch.nn as nn


class SlotAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_slots: int,
        slot_dim: int,
        hidden_dim: int,
        routing_iters: int = 3,
        topk_patches: Optional[int] = None,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.routing_iters = routing_iters
        self.topk_patches = topk_patches

        self.norm_inputs = nn.LayerNorm(input_dim)
        self.W_q = nn.Parameter(torch.empty(slot_dim, slot_dim))
        self.W_k = nn.Parameter(torch.empty(input_dim, slot_dim))
        self.W_v = nn.Parameter(torch.empty(input_dim, slot_dim))

        for param in [self.W_q, self.W_k, self.W_v]:
            nn.init.xavier_uniform_(param)

        self.slot_loc = nn.Parameter(torch.zeros(1, slot_dim))
        self.slot_log_scale = nn.Parameter(torch.zeros(1, slot_dim))

        self.norm_slots = nn.LayerNorm(slot_dim)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, slot_dim),
        )
        self.register_buffer("scale", torch.tensor(slot_dim**-0.5))

    def forward(
        self,
        x: torch.Tensor,
        num_slots: Optional[int] = None,
        routing_iters: Optional[int] = None,
    ):
        # (b, n, c)
        x = self.norm_inputs(x)
        # (b, n, d)
        k = torch.einsum("bnc,cd->bnd", x, self.W_k) * self.scale
        v = torch.einsum("bnc,cd->bnd", x, self.W_v)

        if num_slots is None:
            num_slots = self.num_slots
        # (b, k, d)
        slots = self.slot_loc + self.slot_log_scale.exp() * torch.randn(
            x.shape[0], num_slots, self.slot_dim, device=x.device
        )

        if routing_iters is None:
            routing_iters = self.routing_iters

        for _ in range(routing_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            # (b, k, d)
            q = torch.einsum("bki,id->bkd", slots, self.W_q)
            # (b, k, n)
            agreement = torch.einsum("bkd,bnd->bkn", q, k)
            attn = agreement.softmax(dim=1) + 1e-8
            if self.topk_patches is not None and self.topk_patches > 0:
                topk = min(self.topk_patches, attn.shape[-1])
                topk_vals, topk_idx = attn.topk(topk, dim=-1)
                sparse_attn = torch.zeros_like(attn)
                attn = sparse_attn.scatter(-1, topk_idx, topk_vals)
            _attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)  # weighted mean
            # (b, k, d)
            slots = torch.einsum("bkn,bnd->bkd", _attn, v)
            # (b*k, d)
            slots = self.gru(
                slots.view(-1, self.slot_dim), slots_prev.view(-1, self.slot_dim)
            )
            slots = slots.view(-1, num_slots, self.slot_dim)
            slots = slots + self.mlp(slots)
        return slots, attn, agreement
