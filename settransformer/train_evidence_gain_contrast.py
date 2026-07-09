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


def evidence_gain_loss(model, slots: torch.Tensor, attn: torch.Tensor, labels: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    b, k, d = slots.shape
    margins = single_slot_margins_train(model, slots, labels)
    best_idx = margins.detach().argmax(dim=1)
    base_mask = torch.zeros(b, k, dtype=torch.bool, device=slots.device)
    base_mask[torch.arange(b, device=slots.device), best_idx] = True
    base_margin = true_class_margin(model(slots, base_mask), labels)

    pair_masks = base_mask[:, None].expand(-1, k, -1).clone()
    pair_masks[:, torch.arange(k, device=slots.device), torch.arange(k, device=slots.device)] = True
    flat_slots = slots[:, None].expand(-1, k, -1, -1).reshape(b * k, k, d)
    flat_masks = pair_masks.reshape(b * k, k)
    flat_labels = labels[:, None].expand(-1, k).reshape(b * k)
    pair_margin = true_class_margin(model(flat_slots, flat_masks), flat_labels).reshape(b, k)
    gains = pair_margin - base_margin[:, None]

    overlaps = pairwise_attention_overlap(attn).detach()
    overlap_with_best = overlaps[torch.arange(b, device=slots.device), best_idx]
    candidate = overlap_with_best <= args.evidence_max_overlap
    candidate[torch.arange(b, device=slots.device), best_idx] = False
    valid = candidate.any(dim=1)
    if not bool(valid.any()):
        zero = slots.new_zeros(())
        return zero, {"evidence_valid_frac": 0.0, "evidence_best_gain": 0.0, "evidence_candidate_overlap": 0.0}
    masked_gains = gains.masked_fill(~candidate, torch.finfo(gains.dtype).min)
    best_nonoverlap_gain = masked_gains.max(dim=1).values
    loss = F.relu(args.evidence_gain_hinge - best_nonoverlap_gain[valid]).mean()
    candidate_overlap = overlap_with_best[candidate].mean()
    return loss, {
        "evidence_valid_frac": float(valid.float().mean().cpu()),
        "evidence_best_gain": float(best_nonoverlap_gain[valid].detach().mean().cpu()),
        "evidence_candidate_overlap": float(candidate_overlap.cpu()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment C: reward non-overlapping evidence gain after the best single slot.")
    add_common_args(parser, "checkpoints/settransformer/evidence_gain_contrast")
    parser.add_argument("--extra_weight", type=float, default=0.6)
    parser.add_argument("--evidence_max_overlap", type=float, default=0.35)
    parser.add_argument("--evidence_gain_hinge", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        args,
        "evidence_gain_contrast",
        evidence_gain_loss,
        {
            "hypothesis": "After the best single slot, a non-overlapping second slot should still add true-class margin.",
            "varied_factor": "L_evidence_gain only; complement remains disabled.",
        },
    )


if __name__ == "__main__":
    main()
