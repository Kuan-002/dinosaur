from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".cache" / "torch"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from misc_utils import seed_all
from settransformer.model import DiscriminativeSetTransformer, ProbeConfig, true_class_margin
from settransformer.train import DEFAULT_DATA, DEFAULT_SA, compute_losses, eval_epoch, parse_ints
from train_slot_classifier import build_dataset, build_transforms, load_backbone, subset_dataset


ExtraLossFn = Callable[[DiscriminativeSetTransformer, torch.Tensor, torch.Tensor, torch.Tensor, argparse.Namespace], tuple[torch.Tensor, dict[str, float]]]


def add_common_args(parser: argparse.ArgumentParser, default_output_dir: str) -> None:
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", default=DEFAULT_SA)
    parser.add_argument("--output_dir", default=default_output_dir)
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


def class_names_from_dataset(dataset) -> list[str]:
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return list(getattr(dataset, "classes", []))


def make_loader(dataset, args: argparse.Namespace, device: torch.device, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=shuffle,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


@torch.no_grad()
def encode_slots_and_attn(backbone, images: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.forward_dino(images)
    features = backbone.mlp(features)
    slots, attn, _ = backbone.slot_attention(features)
    return slots.detach(), attn.detach()


def normalized_slot_attention(attn: torch.Tensor) -> torch.Tensor:
    p = attn.clamp_min(1e-8)
    return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def pairwise_attention_overlap(attn: torch.Tensor) -> torch.Tensor:
    p = normalized_slot_attention(attn)
    return torch.minimum(p[:, :, None, :], p[:, None, :, :]).sum(dim=-1)


@torch.no_grad()
def single_slot_margins(model: DiscriminativeSetTransformer, slots: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    b, k, d = slots.shape
    eye = torch.eye(k, dtype=torch.bool, device=slots.device)
    masks = eye.unsqueeze(0).expand(b, -1, -1).reshape(b * k, k)
    flat_slots = slots[:, None].expand(-1, k, -1, -1).reshape(b * k, k, d)
    flat_labels = labels[:, None].expand(-1, k).reshape(b * k)
    return true_class_margin(model(flat_slots, masks), flat_labels).reshape(b, k)


def train_epoch_with_extra(
    model: DiscriminativeSetTransformer,
    backbone,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    extra_loss_fn: ExtraLossFn,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    seen = 0
    iterator = tqdm(loader, desc="train", mininterval=1.0) if sys.stdout.isatty() else loader
    for images, labels in iterator:
        labels = labels.to(device, non_blocking=device.type == "cuda")
        slots, attn = encode_slots_and_attn(backbone, images, device)
        base_loss, base_stats = compute_losses(model, slots, labels, args)
        extra_loss, extra_stats = extra_loss_fn(model, slots, attn, labels, args)
        loss = base_loss + args.extra_weight * extra_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        batch = labels.numel()
        seen += batch
        stats = {
            "loss": float(loss.detach().cpu()),
            "base_loss": float(base_loss.detach().cpu()),
            "extra_loss": float(extra_loss.detach().cpu()),
            **base_stats,
            **extra_stats,
        }
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + value * batch
    return {key: value / max(seen, 1) for key, value in totals.items()}


def write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    args: argparse.Namespace,
    experiment_name: str,
    extra_loss_fn: ExtraLossFn,
    extra_meta: dict,
) -> None:
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
        "experiment_name": experiment_name,
        "extra_meta": extra_meta,
        "model_class": "settransformer.model.DiscriminativeSetTransformer",
        "base_objective": "no-comp SetTransformer objective plus one contrastive constraint",
    }
    (out_dir / "probe_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"experiment={experiment_name} device={device} train={len(train_set)} valid={len(valid_set)} classes={len(classes)}")
    print(f"cfg={cfg}")
    print(f"trainable_params={sum(p.numel() for p in model.parameters()):,}")

    history: list[dict] = []
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_stats = train_epoch_with_extra(model, backbone, train_loader, device, optimizer, args, extra_loss_fn)
        valid_stats = eval_epoch(model, backbone, valid_loader, device, args)
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        row = {"epoch": epoch, "elapsed": elapsed, **{f"train_{k}": v for k, v in train_stats.items()}, **valid_stats}
        history.append(row)
        write_history(out_dir / "history_metrics.csv", history)

        score = valid_stats["valid_selection_score"]
        print(
            f"epoch={epoch} elapsed={elapsed} score={score:.4f} "
            f"extra={train_stats.get('extra_loss', 0.0):.4f} "
            f"acc={100 * valid_stats['valid_all_acc']:.2f} "
            f"single_range={valid_stats['valid_single_margin_range']:.4f} "
            f"gain1to2={valid_stats.get('valid_margin_1_to_2_gain', 0.0):.4f}"
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
                    "experiment_name": experiment_name,
                    "model_state_dict": model.state_dict(),
                },
                out_dir / f"settransformer_{experiment_name}_best.pt",
            )
            print(f"saved {out_dir / f'settransformer_{experiment_name}_best.pt'}")
        else:
            stale += 1

        if args.early_stop_patience > 0 and epoch >= args.early_stop_min_epochs and stale >= args.early_stop_patience:
            print(f"early_stop best_epoch={best_epoch} best_score={best_score:.4f}")
            break

    (out_dir / "final_metrics.json").write_text(
        json.dumps({"best_epoch": best_epoch, "best_valid_selection_score": best_score, "history": history}, indent=2),
        encoding="utf-8",
    )
