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
    dropout: float = 0.1


def true_class_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    other_logits = logits.masked_fill(
        F.one_hot(labels, logits.size(1)).bool(),
        torch.finfo(logits.dtype).min,
    ).amax(dim=1)
    return true_logits - other_logits


class MAB(nn.Module):
    def __init__(self, dim_q: int, dim_kv: int, dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.q_proj = nn.Linear(dim_q, dim)
        self.k_proj = nn.Linear(dim_kv, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        qh = self.q_proj(q)
        kh = self.k_proj(kv)
        attended, _ = self.attn(qh, kh, kh, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.norm1(qh + attended)
        return self.norm2(h + self.ff(h))


class SAB(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.mab = MAB(dim, dim, dim, num_heads, ff_dim, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        return self.mab(x, x, key_padding_mask=key_padding_mask)


class PMA(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.seed = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.seed, std=0.02)
        self.mab = MAB(dim, dim, dim, num_heads, ff_dim, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        seed = self.seed.expand(x.size(0), -1, -1)
        return self.mab(seed, x, key_padding_mask=key_padding_mask).squeeze(1)


class DiscriminativeSetTransformer(nn.Module):
    """Permutation-invariant slot-subset classifier with masked SAB/PMA.

    The model receives frozen slot vectors and a boolean subset mask. Unselected
    slots are removed through attention masks instead of being represented as
    zero vectors. Empty subsets are mapped to a learned empty token so selector
    reward code can score the pre-selection state without NaNs.
    """

    def __init__(self, cfg: ProbeConfig):
        super().__init__()
        self.cfg = cfg
        self.slot_proj = nn.Sequential(
            nn.LayerNorm(cfg.slot_dim),
            nn.Linear(cfg.slot_dim, cfg.bottleneck_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.bottleneck_dim, cfg.hidden_dim),
        )
        self.empty_slot = nn.Parameter(torch.zeros(cfg.hidden_dim))
        nn.init.normal_(self.empty_slot, std=0.02)
        self.sabs = nn.ModuleList(
            [SAB(cfg.hidden_dim, cfg.num_heads, cfg.ff_dim, cfg.dropout) for _ in range(cfg.num_sab_layers)]
        )
        self.pma = PMA(cfg.hidden_dim, cfg.num_heads, cfg.ff_dim, cfg.dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.hidden_dim),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def _effective_inputs(self, slots: torch.Tensor, subset_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        subset_mask = subset_mask.bool()
        x = self.slot_proj(slots)
        empty_rows = subset_mask.sum(dim=1) == 0
        if empty_rows.any():
            x = x.clone()
            x[empty_rows, 0] = self.empty_slot.to(dtype=x.dtype)
            subset_mask = subset_mask.clone()
            subset_mask[empty_rows, 0] = True
        return x, ~subset_mask

    def forward(self, slots: torch.Tensor, subset_mask: torch.Tensor) -> torch.Tensor:
        if slots.ndim != 3:
            raise ValueError(f"Expected slots [B, K, D], got {tuple(slots.shape)}")
        if subset_mask.shape != slots.shape[:2]:
            raise ValueError(f"Expected subset_mask {tuple(slots.shape[:2])}, got {tuple(subset_mask.shape)}")
        x, key_padding_mask = self._effective_inputs(slots, subset_mask)
        for sab in self.sabs:
            x = sab(x, key_padding_mask)
        pooled = self.pma(x, key_padding_mask)
        return self.head(pooled)
