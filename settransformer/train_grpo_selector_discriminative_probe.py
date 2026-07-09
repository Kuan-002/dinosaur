#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import train_grpo_selector as grpo
from settransformer.model import DiscriminativeSetTransformer, ProbeConfig


def load_discriminative_reward_probe(path: str, device: torch.device) -> DiscriminativeSetTransformer:
    ckpt: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    cfg = ProbeConfig(**ckpt["probe_config"])
    probe = DiscriminativeSetTransformer(cfg).to(device)
    probe.load_state_dict(ckpt["model_state_dict"], strict=True)
    probe.eval()
    for param in probe.parameters():
        param.requires_grad = False
    return probe


def main() -> None:
    grpo.load_reward_probe = load_discriminative_reward_probe
    grpo.main()


if __name__ == "__main__":
    main()
