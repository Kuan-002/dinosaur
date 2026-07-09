from typing import Tuple

import torch
import torch.nn as nn

from slot_attention import SlotAttention


class TransformerFeatureDecoder(nn.Module):
    def __init__(
        self,
        slot_dim: int,
        embed_dim: int,
        hidden_dim: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(slot_dim, embed_dim, bias=False),
            nn.LayerNorm(embed_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_blocks)
        self.output_proj = nn.Linear(embed_dim, embed_dim + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.output_proj(x)


class SlotAutoencoder(nn.Module):
    def __init__(
        self,
        embed_shape: Tuple[int, int] = (196, 768),
        hidden_dim: int = 2048,
        decoder_type: str = "mlp",
        num_blocks: int = 6,
        num_heads: int = 6,
        dropout: float = 0.0,
        num_slots: int = 6,
        slot_dim: int = 256,
        num_slot_heads: int = 1,
        routing_iters: int = 3,
        sa_topk_patches: int = 0,
        probabilistic: bool = False,
        proj_cov: bool = False,
    ):
        super().__init__()
        self.num_patches, self.embed_dim = embed_shape
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.probabilistic = probabilistic
        self.decoder_type = decoder_type

        self.dino = torch.hub.load("facebookresearch/dino:main", "dino_vitb16")
        self.dino.requires_grad_(False)

        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim, bias=False),
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        sa_kwargs = dict(
            input_dim=self.embed_dim,
            num_slots=num_slots,
            slot_dim=slot_dim,
            hidden_dim=slot_dim * 4,
            routing_iters=routing_iters,
            topk_patches=sa_topk_patches if sa_topk_patches > 0 else None,
        )

        self.slot_attention = SlotAttention(**sa_kwargs)

        self.pos_embed = nn.Parameter(0.02 * torch.randn(1, self.num_patches, slot_dim))
        if decoder_type == "mlp":
            self.decoder = nn.Sequential(
                nn.Linear(slot_dim, self.embed_dim, bias=False),
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, hidden_dim),
                nn.ReLU(),
                *[nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] * 2,
                nn.Linear(hidden_dim, self.embed_dim + 1),
            )
        elif decoder_type == "transformer":
            self.decoder = TransformerFeatureDecoder(
                slot_dim=slot_dim,
                embed_dim=self.embed_dim,
                hidden_dim=hidden_dim,
                num_blocks=num_blocks,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"Unknown decoder_type={decoder_type!r}. "
                "Expected 'mlp' or 'transformer'."
            )

    def forward_dino(self, x: torch.Tensor):
        self.dino.eval()
        x = self.dino.prepare_tokens(x)
        for block in self.dino.blocks:
            x = block(x)
        return x[:, 1:]  # remove CLS token

    def forward(self, x: torch.Tensor):
        x = self.forward_dino(x)
        x_target = x.clone().detach()
        # (b, num_patches, embd_dim)
        x = self.mlp(x)
        slots, attn, _ = self.slot_attention(x)
        # (b*num_slots, num_patches, slot_dim)
        x = slots.reshape(-1, self.slot_dim).unsqueeze(1).repeat(1, self.num_patches, 1)
        # (b*num_slots, num_patches, embd_dim + 1)
        x = self.decoder(x + self.pos_embed)
        x = x.reshape(-1, self.num_slots, self.num_patches, self.embed_dim + 1)
        # (b, num_slots, num_patches, embd_dim), (b, num_slots, num_patches, 1)
        recons, masks = x.split([self.embed_dim, 1], dim=-1)
        masks = masks.softmax(dim=1)
        # (b, num_patches, embd_dim)
        x = torch.sum(recons * masks, dim=1)
        loss = torch.mean((x - x_target) ** 2)
        return dict(loss=loss, decoder_attn=masks.squeeze(-1), slot_attn=attn)
