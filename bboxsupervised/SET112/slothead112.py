from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass
class Slothead112Config:
    slot_dim: int
    obj_dim: int
    geo_dim: int
    res_dim: int
    hidden_dim: int
    dropout: float
    num_categories: int

    @property
    def out_dim(self) -> int:
        return self.obj_dim + self.geo_dim + self.res_dim


class Slothead112Projector(nn.Module):
    def __init__(self, cfg: Slothead112Config):
        super().__init__()
        self.cfg = cfg
        self.projector = nn.Sequential(
            nn.LayerNorm(cfg.slot_dim),
            nn.Linear(cfg.slot_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.out_dim),
        )

    def split(self, slots: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        u = self.projector(slots)
        o = self.cfg.obj_dim
        g = self.cfg.geo_dim
        return u, u[..., :o], u[..., o:o + g], u[..., o + g:]

    def forward(self, slots: torch.Tensor, mode: str = "u") -> torch.Tensor:
        u, u_obj, u_geo, u_res = self.split(slots)
        if mode == "u":
            return u
        if mode == "obj":
            return u_obj
        if mode == "geo":
            return u_geo
        if mode == "res":
            return u_res
        if mode == "obj_geo":
            return torch.cat([u_obj, u_geo], dim=-1)
        if mode == "obj_res":
            return torch.cat([u_obj, u_res], dim=-1)
        if mode == "geo_res":
            return torch.cat([u_geo, u_res], dim=-1)
        raise ValueError(f"Unknown slothead mode: {mode}")


def load_slothead112(path: str | Path, device: torch.device) -> tuple[Slothead112Projector, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Slothead112Config(**ckpt["config"])
    model = Slothead112Projector(cfg).to(device)
    projector_state = {
        key.removeprefix("projector."): value
        for key, value in ckpt["model_state_dict"].items()
        if key.startswith("projector.")
    }
    model.projector.load_state_dict(projector_state)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model, ckpt


@torch.no_grad()
def project_slots112(projector: Slothead112Projector, slots: torch.Tensor, mode: str = "u") -> torch.Tensor:
    return projector(slots, mode=mode).detach()
