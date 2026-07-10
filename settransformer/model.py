from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProbeConfig:
    num_slots: int
    slot_dim: int
    num_classes: int
    hidden_dim: int = 256
    bottleneck_dim: int = 64
    num_heads: int = 4
    num_sab_layers: int = 2
    ff_dim: int = 512
    dropout: float = 0.15


def true_class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    other_logits = logits.masked_fill(
        F.one_hot(labels, logits.size(1)).bool(),
        torch.finfo(logits.dtype).min,
    ).amax(dim=1)
    return true_logits - other_logits


class SABlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        key_padding_mask = ~mask.bool()
        attn_out, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ff(x))


class DiscriminativeSetTransformer(nn.Module):
    def __init__(self, cfg: ProbeConfig):
        super().__init__()
        self.cfg = cfg
        self.slot_embed = nn.Sequential(
            nn.LayerNorm(cfg.slot_dim),
            nn.Linear(cfg.slot_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                SABlock(cfg.hidden_dim, cfg.num_heads, cfg.ff_dim, cfg.dropout)
                for _ in range(cfg.num_sab_layers)
            ]
        )
        self.pool_seed = nn.Parameter(torch.randn(1, 1, cfg.hidden_dim) * 0.02)
        self.pool_attn = nn.MultiheadAttention(
            cfg.hidden_dim,
            cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.pool_norm = nn.LayerNorm(cfg.hidden_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.bottleneck_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.bottleneck_dim, cfg.num_classes),
        )

    def forward(self, slots: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if slots.ndim != 3:
            raise ValueError(f"Expected slots with shape [B, K, D], got {tuple(slots.shape)}")
        b, k, _ = slots.shape
        if mask is None:
            mask = torch.ones(b, k, dtype=torch.bool, device=slots.device)
        else:
            mask = mask.to(device=slots.device, dtype=torch.bool)
        if mask.shape != (b, k):
            raise ValueError(f"Expected mask shape {(b, k)}, got {tuple(mask.shape)}")
        if not mask.any(dim=1).all():
            mask = mask.clone()
            empty = ~mask.any(dim=1)
            mask[empty, 0] = True

        x = self.slot_embed(slots)
        x = x.masked_fill(~mask[..., None], 0.0)
        for block in self.blocks:
            x = block(x, mask)
            x = x.masked_fill(~mask[..., None], 0.0)

        query = self.pool_seed.expand(b, -1, -1)
        pooled, _ = self.pool_attn(
            query,
            x,
            x,
            key_padding_mask=~mask,
            need_weights=False,
        )
        pooled = self.pool_norm(pooled.squeeze(1))
        return self.classifier(pooled)
