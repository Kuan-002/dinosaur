#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault(
    "TORCH_HOME",
    str(Path(__file__).resolve().parent / ".cache" / "torch"),
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from misc_utils import seed_all
from train_slot_classifier import build_dataset, build_transforms, load_backbone, subset_dataset


@dataclass
class ProbeConfig:
    num_slots: int
    slot_dim: int
    num_classes: int
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 512
    dropout: float = 0.1


class SetTransformerSlotProbe(nn.Module):
    """Permutation-invariant classifier for arbitrary slot subsets.

    The model only receives slot vectors and a boolean subset mask. It has no
    slot index embedding and no access to attention salience or object labels.
    """

    def __init__(self, cfg: ProbeConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Sequential(
            nn.LayerNorm(cfg.slot_dim),
            nn.Linear(cfg.slot_dim, cfg.embed_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim),
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim, cfg.num_classes),
        )

    def forward(self, slots: torch.Tensor, subset_mask: torch.Tensor) -> torch.Tensor:
        if slots.ndim != 3:
            raise ValueError(f"Expected slots shape [B, K, D], got {tuple(slots.shape)}")
        if subset_mask.shape != slots.shape[:2]:
            raise ValueError(
                f"Expected subset_mask shape {tuple(slots.shape[:2])}, got {tuple(subset_mask.shape)}"
            )
        b = slots.size(0)
        subset_mask = subset_mask.bool()
        x = self.input_proj(slots)
        x = x * subset_mask.to(x.dtype).unsqueeze(-1)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        key_padding_mask = torch.cat(
            [
                torch.zeros(b, 1, dtype=torch.bool, device=slots.device),
                ~subset_mask,
            ],
            dim=1,
        )
        encoded = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.head(encoded[:, 0])


@dataclass
class EpochMetrics:
    loss: float
    acc: float
    by_size: dict[str, float]


def make_loader(dataset: Dataset, args, device: torch.device, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.bs,
        shuffle=shuffle,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )


def class_names_from_dataset(dataset) -> Optional[list[str]]:
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    classes = getattr(dataset, "classes", None)
    return list(classes) if classes is not None else None


@torch.no_grad()
def encode_slots(backbone, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    backbone.eval()
    images = images.to(device, non_blocking=device.type == "cuda")
    features = backbone.forward_dino(images)
    features = backbone.mlp(features)
    slots, _, _ = backbone.slot_attention(features)
    return slots.detach()


def random_size_mask(batch: int, num_slots: int, device: torch.device, min_slots: int, max_slots: int) -> torch.Tensor:
    lo = max(1, min(min_slots, num_slots))
    hi = max(lo, min(max_slots, num_slots))
    sizes = torch.randint(lo, hi + 1, (batch,), device=device)
    order = torch.rand(batch, num_slots, device=device).argsort(dim=1)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(num_slots, device=device).expand(batch, -1))
    return ranks < sizes[:, None]


def bernoulli_mask(batch: int, num_slots: int, device: torch.device, keep_prob: float, min_slots: int) -> torch.Tensor:
    mask = torch.rand(batch, num_slots, device=device) < keep_prob
    counts = mask.sum(dim=1)
    need_fix = counts < min_slots
    if need_fix.any():
        fix = random_size_mask(int(need_fix.sum()), num_slots, device, min_slots, min_slots)
        mask[need_fix] = fix
    return mask


def sample_subset_mask(batch: int, num_slots: int, device: torch.device, args) -> torch.Tensor:
    mode = random.choices(
        population=["all", "random_size", "dropout"],
        weights=[args.p_all, args.p_random_size, args.p_dropout],
        k=1,
    )[0]
    if mode == "all":
        return torch.ones(batch, num_slots, dtype=torch.bool, device=device)
    if mode == "dropout":
        return bernoulli_mask(batch, num_slots, device, args.slot_keep_prob, args.min_subset_slots)
    return random_size_mask(batch, num_slots, device, args.min_subset_slots, args.max_subset_slots)


def fixed_size_mask(batch: int, num_slots: int, size: int, device: torch.device) -> torch.Tensor:
    size = max(1, min(size, num_slots))
    order = torch.rand(batch, num_slots, device=device).argsort(dim=1)
    mask = torch.zeros(batch, num_slots, dtype=torch.bool, device=device)
    mask.scatter_(1, order[:, :size], True)
    return mask


def run_train_epoch(model, backbone, loader, device, optimizer, args) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    batches = tqdm(loader, desc="train", mininterval=1.0) if sys_stdout_is_tty() else loader
    for images, labels in batches:
        labels = labels.to(device, non_blocking=device.type == "cuda")
        slots = encode_slots(backbone, images, device)
        subset_mask = sample_subset_mask(slots.size(0), slots.size(1), device, args)
        logits = model(slots, subset_mask)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch = labels.numel()
        total += batch
        total_loss += float(loss.detach()) * batch
        total_correct += int((logits.argmax(dim=1) == labels).sum())
    return EpochMetrics(total_loss / max(total, 1), total_correct / max(total, 1), {})


@torch.no_grad()
def run_eval_epoch(model, backbone, loader, device, args) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    size_correct = {size: 0 for size in args.eval_subset_sizes}
    size_total = {size: 0 for size in args.eval_subset_sizes}
    batches = tqdm(loader, desc="valid", mininterval=1.0) if sys_stdout_is_tty() else loader
    for images, labels in batches:
        labels = labels.to(device, non_blocking=device.type == "cuda")
        slots = encode_slots(backbone, images, device)
        all_mask = torch.ones(slots.shape[:2], dtype=torch.bool, device=device)
        logits = model(slots, all_mask)
        loss = F.cross_entropy(logits, labels)
        batch = labels.numel()
        total += batch
        total_loss += float(loss.detach()) * batch
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        for size in args.eval_subset_sizes:
            subset_mask = fixed_size_mask(slots.size(0), slots.size(1), size, device)
            subset_logits = model(slots, subset_mask)
            size_correct[size] += int((subset_logits.argmax(dim=1) == labels).sum())
            size_total[size] += batch
    by_size = {
        f"subset_{size}_acc": size_correct[size] / max(size_total[size], 1)
        for size in args.eval_subset_sizes
    }
    return EpochMetrics(total_loss / max(total, 1), total_correct / max(total, 1), by_size)


def sys_stdout_is_tty() -> bool:
    import sys

    return sys.stdout.isatty()


def parse_subset_sizes(value: str) -> list[int]:
    sizes = [int(part) for part in value.split(",") if part.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("Expected a comma-separated list such as 1,2,3,8")
    return sizes


def write_history(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a frozen Set Transformer slot-subset reward probe.")
    parser.add_argument("--data", default="/vol/biomedic3/kw1025/dinosaur/dataset/classification_dataset_clean_600_200_200")
    parser.add_argument("--checkpoint", default="/vol/biomedic3/kw1025/dinosaur/checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt")
    parser.add_argument("--output_dir", default="./checkpoints/set_transformer_slot_probe")
    parser.add_argument("--input_res", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--ff_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min_subset_slots", type=int, default=1)
    parser.add_argument("--max_subset_slots", type=int, default=8)
    parser.add_argument("--slot_keep_prob", type=float, default=0.6)
    parser.add_argument("--p_all", type=float, default=0.25)
    parser.add_argument("--p_random_size", type=float, default=0.50)
    parser.add_argument("--p_dropout", type=float, default=0.25)
    parser.add_argument("--eval_subset_sizes", type=parse_subset_sizes, default=parse_subset_sizes("1,2,3,4,8"))
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--quick_limit_train", type=int, default=0)
    parser.add_argument("--quick_limit_val", type=int, default=0)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_min_epochs", type=int, default=5)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.001)
    args = parser.parse_args()

    weight_sum = args.p_all + args.p_random_size + args.p_dropout
    if weight_sum <= 0:
        raise ValueError("At least one subset sampling probability must be positive")
    if not 0.0 < args.slot_keep_prob <= 1.0:
        raise ValueError("--slot_keep_prob must be in (0, 1]")

    seed_all(args.seed, False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tfm = build_transforms(args.input_res)
    train_set = subset_dataset(build_dataset(args.data, "train", tfm["train"]), args.quick_limit_train, args.seed)
    valid_set = subset_dataset(build_dataset(args.data, "valid", tfm["valid"]), args.quick_limit_val, args.seed)
    classes = class_names_from_dataset(train_set)
    num_classes = len(classes) if classes is not None else 10
    train_loader = make_loader(train_set, args, device, shuffle=True)
    valid_loader = make_loader(valid_set, args, device, shuffle=False)

    backbone = load_backbone(args.checkpoint, device)
    cfg = ProbeConfig(
        num_slots=backbone.num_slots,
        slot_dim=backbone.slot_dim,
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    )
    model = SetTransformerSlotProbe(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "args": vars(args),
        "probe_config": asdict(cfg),
        "classes": classes,
        "slot_embedding_source": "frozen_sa",
        "reward_usage": "freeze this probe and compute rewards under torch.no_grad()",
    }
    (out_dir / "probe_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Using device={device} train={len(train_set)} valid={len(valid_set)} classes={num_classes}")
    print(
        "SetTransformerSlotProbe "
        f"num_slots={cfg.num_slots} slot_dim={cfg.slot_dim} embed_dim={cfg.embed_dim} "
        f"layers={cfg.num_layers} heads={cfg.num_heads}"
    )
    print(f"trainable params={sum(p.numel() for p in model.parameters()):,}")

    best_acc = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict] = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_train_epoch(model, backbone, train_loader, device, optimizer, args)
        valid_metrics = run_eval_epoch(model, backbone, valid_loader, device, args)
        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        row = {
            "epoch": epoch,
            "elapsed": elapsed,
            "train_loss": train_metrics.loss,
            "train_acc": train_metrics.acc,
            "valid_loss": valid_metrics.loss,
            "valid_all_acc": valid_metrics.acc,
            **valid_metrics.by_size,
        }
        history.append(row)
        write_history(out_dir / "history_metrics.csv", history)
        print(
            f"epoch={epoch} elapsed={elapsed} "
            f"train_acc={100 * train_metrics.acc:.2f} "
            f"valid_all_acc={100 * valid_metrics.acc:.2f} "
            + " ".join(f"{k}={100 * v:.2f}" for k, v in valid_metrics.by_size.items())
        )

        improved = valid_metrics.acc > best_acc + args.early_stop_min_delta
        if improved:
            best_acc = valid_metrics.acc
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "args": vars(args),
                    "probe_config": asdict(cfg),
                    "classes": classes,
                    "epoch": epoch,
                    "valid_all_acc": valid_metrics.acc,
                    "model_state_dict": model.state_dict(),
                },
                out_dir / "set_transformer_slot_probe_best.pt",
            )
            print(f"saved best probe: {out_dir / 'set_transformer_slot_probe_best.pt'}")
        else:
            stale_epochs += 1

        if (
            args.early_stop_patience > 0
            and epoch >= args.early_stop_min_epochs
            and stale_epochs >= args.early_stop_patience
        ):
            print(
                "early_stop: "
                f"best_epoch={best_epoch}, best_valid_all_acc={100 * best_acc:.2f}, "
                f"stale_epochs={stale_epochs}"
            )
            break

    (out_dir / "final_metrics.json").write_text(
        json.dumps({"best_valid_all_acc": best_acc, "best_epoch": best_epoch, "history": history}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
