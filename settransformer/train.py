#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".cache" / "torch"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from misc_utils import seed_all
from settransformer.model import DiscriminativeSetTransformer, ProbeConfig, true_class_margin
from train_slot_classifier import build_dataset, build_transforms, load_backbone, subset_dataset


DEFAULT_DATA = (
    "/vol/biomedic3/kw1025/dinosaur/analysis/"
    "coco_top2_clean_scenes_anchor009_evidence005_10cls_450_150_150/classification_dataset"
)
DEFAULT_SA = (
    "/vol/biomedic3/kw1025/dinosaur/checkpoints/"
    "sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt"
)


def make_loader(dataset: Dataset, args: argparse.Namespace, device: torch.device, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=shuffle,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def class_names_from_dataset(dataset: Dataset) -> list[str]:
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return list(getattr(dataset, "classes", []))


@torch.no_grad()
def encode_slots(backbone, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    backbone.eval()
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.forward_dino(images)
    features = backbone.mlp(features)
    slots, _, _ = backbone.slot_attention(features)
    return slots.detach()


def random_size_mask(batch: int, num_slots: int, device: torch.device, min_size: int, max_size: int) -> torch.Tensor:
    lo = max(0, min(min_size, num_slots))
    hi = max(lo, min(max_size, num_slots))
    sizes = torch.randint(lo, hi + 1, (batch,), device=device)
    order = torch.rand(batch, num_slots, device=device).argsort(dim=1)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(num_slots, device=device).expand(batch, -1))
    return ranks < sizes[:, None]


def random_nontrivial_mask(batch: int, num_slots: int, device: torch.device) -> torch.Tensor:
    max_size = max(1, num_slots - 1)
    return random_size_mask(batch, num_slots, device, 1, max_size)


def all_single_masks(batch: int, num_slots: int, device: torch.device) -> torch.Tensor:
    eye = torch.eye(num_slots, dtype=torch.bool, device=device)
    return eye.unsqueeze(0).expand(batch, -1, -1)


def fixed_size_mask(batch: int, num_slots: int, size: int, device: torch.device) -> torch.Tensor:
    return random_size_mask(batch, num_slots, device, size, size)


def weighted_ce(logits: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor, smoothing: float) -> torch.Tensor:
    losses = F.cross_entropy(logits, labels, reduction="none", label_smoothing=smoothing)
    return (losses * weight).sum() / weight.sum().clamp_min(1e-6)


def permutation_consistency_loss(model, slots: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits_a = model(slots, mask)
    order = torch.rand(slots.size(0), slots.size(1), device=slots.device).argsort(dim=1)
    slots_b = slots.gather(1, order.unsqueeze(-1).expand(-1, -1, slots.size(-1)))
    mask_b = mask.gather(1, order)
    logits_b = model(slots_b, mask_b)
    return F.mse_loss(true_class_margin(logits_a, labels), true_class_margin(logits_b, labels))


def marginal_loss(model, slots: torch.Tensor, labels: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    b, k, d = slots.shape
    base_mask = random_size_mask(b, k, slots.device, 0, max(0, k - 1))
    base_logits = model(slots, base_mask)
    base_margin = true_class_margin(base_logits, labels)

    candidate_masks = base_mask[:, None].expand(-1, k, -1).clone()
    candidate_masks[:, torch.arange(k, device=slots.device), torch.arange(k, device=slots.device)] = True
    flat_slots = slots[:, None].expand(-1, k, -1, -1).reshape(b * k, k, d)
    flat_masks = candidate_masks.reshape(b * k, k)
    flat_labels = labels[:, None].expand(-1, k).reshape(b * k)
    candidate_margin = true_class_margin(model(flat_slots, flat_masks), flat_labels).reshape(b, k)
    gains = candidate_margin - base_margin[:, None]
    already_selected = base_mask
    gains_for_best = gains.masked_fill(already_selected, torch.finfo(gains.dtype).min)
    gains_for_worst = gains.masked_fill(already_selected, torch.finfo(gains.dtype).max)
    best_gain = gains_for_best.max(dim=1).values
    worst_gain = gains_for_worst.min(dim=1).values
    valid = (~already_selected).any(dim=1)
    loss = (
        F.relu(float(args.marginal_pos_hinge) - best_gain[valid]).mean()
        + F.relu(worst_gain[valid] - float(args.marginal_worst_max)).mean()
    )
    stats = {
        "marginal_best_gain": float(best_gain[valid].mean().detach().cpu()) if valid.any() else 0.0,
        "marginal_worst_gain": float(worst_gain[valid].mean().detach().cpu()) if valid.any() else 0.0,
        "marginal_spread": float((best_gain[valid] - worst_gain[valid]).mean().detach().cpu()) if valid.any() else 0.0,
    }
    return loss, stats


def compute_losses(model, slots: torch.Tensor, labels: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    b, k, d = slots.shape
    all_mask = torch.ones(b, k, dtype=torch.bool, device=slots.device)
    full_logits = model(slots, all_mask)
    l_full = F.cross_entropy(full_logits, labels, label_smoothing=args.label_smoothing)

    single_masks = all_single_masks(b, k, slots.device)
    single_logits = model(
        slots[:, None].expand(-1, k, -1, -1).reshape(b * k, k, d),
        single_masks.reshape(b * k, k),
    )
    l_single = F.cross_entropy(
        single_logits,
        labels[:, None].expand(-1, k).reshape(b * k),
        label_smoothing=args.label_smoothing,
    )

    subset_mask = random_size_mask(b, k, slots.device, args.min_subset_slots, args.max_subset_slots)
    subset_logits = model(slots, subset_mask)
    subset_size = subset_mask.sum(dim=1).clamp_min(1).to(slots.dtype)
    l_subset = weighted_ce(subset_logits, labels, 1.0 / subset_size, args.label_smoothing)

    if args.gamma_comp > 0:
        comp_mask = random_nontrivial_mask(b, k, slots.device)
        comp_margin = true_class_margin(model(slots, comp_mask), labels)
        complement_margin = true_class_margin(model(slots, ~comp_mask), labels)
        l_comp = F.relu(float(args.comp_hinge) - (comp_margin - complement_margin)).mean()
    else:
        l_comp = slots.new_zeros(())

    l_marginal, marginal_stats = marginal_loss(model, slots, labels, args)
    l_consistency = permutation_consistency_loss(model, slots, labels, subset_mask)

    total = (
        l_full
        + args.alpha_single * l_single
        + args.beta_subset * l_subset
        + args.gamma_comp * l_comp
        + args.delta_marginal * l_marginal
        + args.epsilon_consistency * l_consistency
    )
    stats = {
        "loss_full": float(l_full.detach().cpu()),
        "loss_single": float(l_single.detach().cpu()),
        "loss_subset": float(l_subset.detach().cpu()),
        "loss_comp": float(l_comp.detach().cpu()),
        "loss_marginal": float(l_marginal.detach().cpu()),
        "loss_consistency": float(l_consistency.detach().cpu()),
        **marginal_stats,
    }
    return total, stats


def train_epoch(model, backbone, loader, device, optimizer, args) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    seen = 0
    iterator = tqdm(loader, desc="train", mininterval=1.0) if sys.stdout.isatty() else loader
    for images, labels in iterator:
        labels = labels.to(device, non_blocking=device.type == "cuda")
        slots = encode_slots(backbone, images, device)
        loss, stats = compute_losses(model, slots, labels, args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        batch = labels.numel()
        seen += batch
        totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().cpu()) * batch
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + value * batch
    return {key: value / max(seen, 1) for key, value in totals.items()}


@torch.no_grad()
def eval_epoch(model, backbone, loader, device, args) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    seen = 0
    iterator = tqdm(loader, desc="valid", mininterval=1.0) if sys.stdout.isatty() else loader
    for images, labels in iterator:
        labels = labels.to(device, non_blocking=device.type == "cuda")
        slots = encode_slots(backbone, images, device)
        b, k, d = slots.shape
        all_mask = torch.ones(b, k, dtype=torch.bool, device=device)
        logits = model(slots, all_mask)
        pred = logits.argmax(dim=1)
        margin_all = true_class_margin(logits, labels)

        single_masks = all_single_masks(b, k, device)
        single_logits = model(
            slots[:, None].expand(-1, k, -1, -1).reshape(b * k, k, d),
            single_masks.reshape(b * k, k),
        )
        single_margins = true_class_margin(single_logits, labels[:, None].expand(-1, k).reshape(b * k)).reshape(b, k)
        single_range = single_margins.max(dim=1).values - single_margins.min(dim=1).values

        batch_stats = {
            "valid_loss": float(F.cross_entropy(logits, labels).detach().cpu()),
            "valid_all_acc": float((pred == labels).float().mean().cpu()),
            "valid_all_margin": float(margin_all.mean().cpu()),
            "valid_single_margin_std": float(single_margins.std(dim=1).mean().cpu()),
            "valid_single_margin_range": float(single_range.mean().cpu()),
        }
        size_margins = []
        for size in args.eval_subset_sizes:
            mask = fixed_size_mask(b, k, size, device)
            subset_logits = model(slots, mask)
            subset_margin = true_class_margin(subset_logits, labels)
            batch_stats[f"valid_subset_{size}_acc"] = float((subset_logits.argmax(dim=1) == labels).float().mean().cpu())
            batch_stats[f"valid_subset_{size}_margin"] = float(subset_margin.mean().cpu())
            size_margins.append((size, subset_margin))
        if len(size_margins) >= 2:
            batch_stats["valid_margin_1_to_2_gain"] = float((size_margins[1][1] - size_margins[0][1]).mean().cpu())
        batch_stats["valid_consistency_mse"] = float(permutation_consistency_loss(model, slots, labels, fixed_size_mask(b, k, min(2, k), device)).cpu())

        batch = labels.numel()
        seen += batch
        for key, value in batch_stats.items():
            totals[key] = totals.get(key, 0.0) + value * batch
    out = {key: value / max(seen, 1) for key, value in totals.items()}
    out["valid_selection_score"] = (
        out.get("valid_single_margin_range", 0.0)
        + max(0.0, out.get("valid_margin_1_to_2_gain", 0.0))
        - out.get("valid_consistency_mse", 0.0)
    )
    return out


def write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_ints(raw: str) -> list[int]:
    values = [int(part) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected comma-separated integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a discriminative bbox-free Set Transformer slot probe.")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", default=DEFAULT_SA)
    parser.add_argument("--output_dir", default="checkpoints/settransformer/discriminative_anchor_evidence")
    parser.add_argument("--input_res", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--bottleneck_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_sab_layers", type=int, default=2)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--min_subset_slots", type=int, default=1)
    parser.add_argument("--max_subset_slots", type=int, default=4)
    parser.add_argument("--eval_subset_sizes", type=parse_ints, default=parse_ints("1,2,3,4,8"))
    parser.add_argument("--alpha_single", type=float, default=1.0)
    parser.add_argument("--beta_subset", type=float, default=1.0)
    parser.add_argument("--gamma_comp", type=float, default=0.0)
    parser.add_argument("--delta_marginal", type=float, default=0.7)
    parser.add_argument("--epsilon_consistency", type=float, default=0.3)
    parser.add_argument("--comp_hinge", type=float, default=0.25)
    parser.add_argument("--marginal_pos_hinge", type=float, default=0.15)
    parser.add_argument("--marginal_worst_max", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--quick_limit_train", type=int, default=0)
    parser.add_argument("--quick_limit_val", type=int, default=0)
    parser.add_argument("--early_stop_patience", type=int, default=15)
    parser.add_argument("--early_stop_min_epochs", type=int, default=10)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_all(args.seed, False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tfm = build_transforms(args.input_res)
    train_set = subset_dataset(build_dataset(args.data, "train", tfm["train"]), args.quick_limit_train, args.seed)
    valid_set = subset_dataset(build_dataset(args.data, "valid", tfm["valid"]), args.quick_limit_val, args.seed)
    classes = class_names_from_dataset(train_set)
    if not classes:
        raise RuntimeError("Could not infer class names from the dataset.")
    train_loader = make_loader(train_set, args, device, shuffle=True)
    valid_loader = make_loader(valid_set, args, device, shuffle=False)
    backbone = load_backbone(args.checkpoint, device)

    cfg = ProbeConfig(
        num_slots=backbone.num_slots,
        slot_dim=backbone.slot_dim,
        num_classes=len(classes),
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        num_heads=args.num_heads,
        num_sab_layers=args.num_sab_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    model = DiscriminativeSetTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "args": vars(args),
        "probe_config": asdict(cfg),
        "classes": classes,
        "model_class": "settransformer.model.DiscriminativeSetTransformer",
        "selection_objective": "best checkpoint is selected by margin range/gain, not classification accuracy",
    }
    (out_dir / "probe_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"device={device} train={len(train_set)} valid={len(valid_set)} classes={len(classes)}")
    print(f"cfg={cfg}")
    print(f"trainable_params={sum(p.numel() for p in model.parameters()):,}")

    history: list[dict] = []
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch(model, backbone, train_loader, device, optimizer, args)
        valid_stats = eval_epoch(model, backbone, valid_loader, device, args)
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        row = {"epoch": epoch, "elapsed": elapsed, **{f"train_{k}": v for k, v in train_stats.items()}, **valid_stats}
        history.append(row)
        write_history(out_dir / "history_metrics.csv", history)

        score = valid_stats["valid_selection_score"]
        print(
            f"epoch={epoch} elapsed={elapsed} "
            f"score={score:.4f} acc={100 * valid_stats['valid_all_acc']:.2f} "
            f"single_range={valid_stats['valid_single_margin_range']:.4f} "
            f"gain1to2={valid_stats.get('valid_margin_1_to_2_gain', 0.0):.4f} "
            f"cons={valid_stats['valid_consistency_mse']:.6f}"
        )
        if score > best_score + args.early_stop_min_delta:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "args": vars(args),
                    "probe_config": asdict(cfg),
                    "classes": classes,
                    "epoch": epoch,
                    "valid_selection_score": score,
                    "valid_stats": valid_stats,
                    "model_state_dict": model.state_dict(),
                },
                out_dir / "settransformer_discriminative_best.pt",
            )
            print(f"saved {out_dir / 'settransformer_discriminative_best.pt'}")
        else:
            stale += 1
        if args.early_stop_patience > 0 and epoch >= args.early_stop_min_epochs and stale >= args.early_stop_patience:
            print(f"early_stop best_epoch={best_epoch} best_score={best_score:.4f}")
            break

    (out_dir / "final_metrics.json").write_text(
        json.dumps({"best_epoch": best_epoch, "best_valid_selection_score": best_score, "history": history}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
