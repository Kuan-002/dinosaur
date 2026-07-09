#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from settransformer.experiment_common import add_common_args, normalized_slot_attention, run_experiment
from settransformer.model import true_class_margin


def single_slot_margins_train(model, slots: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    b, k, d = slots.shape
    eye = torch.eye(k, dtype=torch.bool, device=slots.device)
    masks = eye.unsqueeze(0).expand(b, -1, -1).reshape(b * k, k)
    flat_slots = slots[:, None].expand(-1, k, -1, -1).reshape(b * k, k, d)
    flat_labels = labels[:, None].expand(-1, k).reshape(b * k)
    return true_class_margin(model(flat_slots, masks), flat_labels).reshape(b, k)


def objectness_loss(model, slots: torch.Tensor, attn: torch.Tensor, labels: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    margins = single_slot_margins_train(model, slots, labels)
    p = normalized_slot_attention(attn).detach()
    entropy = -(p * p.clamp_min(1e-8).log()).sum(dim=-1) / math.log(p.size(-1))
    peak = p.amax(dim=-1)
    area = (p >= (peak[..., None] * args.objectness_threshold_rel)).float().mean(dim=-1)
    diffuse_badness = entropy.clamp(0.0, 1.0)
    too_large = F.relu(area - args.objectness_max_area) / max(1e-6, 1.0 - args.objectness_max_area)
    too_small = F.relu(args.objectness_min_area - area) / max(1e-6, args.objectness_min_area)
    badness = (diffuse_badness + too_large + too_small).detach()
    positive_margin = F.softplus(margins - args.objectness_margin_floor)
    loss = (positive_margin * badness).mean()
    return loss, {
        "objectness_badness": float(badness.mean().cpu()),
        "objectness_entropy": float(entropy.mean().cpu()),
        "objectness_area": float(area.mean().cpu()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment A: penalize high-margin diffuse/background-like slots.")
    add_common_args(parser, "checkpoints/settransformer/objectness_contrast")
    parser.add_argument("--extra_weight", type=float, default=0.25)
    parser.add_argument("--objectness_margin_floor", type=float, default=0.0)
    parser.add_argument("--objectness_threshold_rel", type=float, default=0.5)
    parser.add_argument("--objectness_min_area", type=float, default=0.015)
    parser.add_argument("--objectness_max_area", type=float, default=0.55)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        args,
        "objectness_contrast",
        objectness_loss,
        {
            "hypothesis": "Suppress diffuse/background-like slots when they receive high single-slot true-class margins.",
            "varied_factor": "L_objectness only; complement remains disabled.",
        },
    )


if __name__ == "__main__":
    main()
