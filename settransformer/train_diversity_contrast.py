#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from settransformer.experiment_common import add_common_args, pairwise_attention_overlap, run_experiment
from settransformer.model import true_class_margin


def single_slot_margins_train(model, slots: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    b, k, d = slots.shape
    eye = torch.eye(k, dtype=torch.bool, device=slots.device)
    masks = eye.unsqueeze(0).expand(b, -1, -1).reshape(b * k, k)
    flat_slots = slots[:, None].expand(-1, k, -1, -1).reshape(b * k, k, d)
    flat_labels = labels[:, None].expand(-1, k).reshape(b * k)
    return true_class_margin(model(flat_slots, masks), flat_labels).reshape(b, k)


def diversity_loss(model, slots: torch.Tensor, attn: torch.Tensor, labels: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    margins = single_slot_margins_train(model, slots, labels)
    b, k = margins.shape
    top_m = min(args.diversity_top_m, k)
    top_idx = margins.detach().topk(top_m, dim=1).indices
    overlaps = pairwise_attention_overlap(attn).detach()
    top_overlap = overlaps.gather(1, top_idx[:, :, None].expand(-1, -1, k)).gather(2, top_idx[:, None, :].expand(-1, top_m, -1))
    top_margins = margins.gather(1, top_idx)
    pair_weight = torch.sigmoid((top_margins[:, :, None] + top_margins[:, None, :]) * 0.5)
    eye = torch.eye(top_m, dtype=torch.bool, device=slots.device)
    valid_pairs = ~eye.unsqueeze(0)
    pair_penalty = F.relu(top_overlap - args.diversity_overlap_target) * pair_weight
    loss = pair_penalty[valid_pairs.expand_as(pair_penalty)].mean()
    return loss, {
        "diversity_top_overlap": float(top_overlap[valid_pairs.expand_as(top_overlap)].mean().cpu()),
        "diversity_top_margin": float(top_margins.mean().detach().cpu()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment B: discourage redundant overlapping top-margin slots.")
    add_common_args(parser, "checkpoints/settransformer/diversity_contrast")
    parser.add_argument("--extra_weight", type=float, default=0.35)
    parser.add_argument("--diversity_top_m", type=int, default=4)
    parser.add_argument("--diversity_overlap_target", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        args,
        "diversity_contrast",
        diversity_loss,
        {
            "hypothesis": "Top-margin slots should not all attend to the same anchor/background region.",
            "varied_factor": "L_diversity only; complement remains disabled.",
        },
    )


if __name__ == "__main__":
    main()
