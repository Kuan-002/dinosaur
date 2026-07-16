#!/usr/bin/env python3
"""Classification evaluation for AC80 selector checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "SET80") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "SET80"))

os.environ.setdefault("TORCH_HOME", str(REPO_ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from SET80.slothead80 import load_slothead80, project_slots80
from train_slot_classifier import build_dataset, build_transforms, load_backbone
from visualize_grpo_selector_paths import choose_device, load_selector


def parse_splits(raw: str) -> list[str]:
    splits = [part.strip() for part in raw.split(",") if part.strip()]
    if not splits:
        raise ValueError("--splits must contain at least one split")
    return ["valid" if split == "val" else split for split in splits]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint", default="selector_ac_best.pt")
    parser.add_argument("--splits", type=parse_splits, default=parse_splits("valid,test"))
    parser.add_argument("--data", default="")
    parser.add_argument("--sa_checkpoint", default="")
    parser.add_argument("--slothead_checkpoint", default="")
    parser.add_argument("--slothead_mode", default="")
    parser.add_argument("--input_res", type=int, default=0)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--min_steps", type=int, default=3)
    parser.add_argument("--early_exit_conf", type=float, default=0.85)
    parser.add_argument("--confidence_early_exit", action="store_true", default=True)
    parser.add_argument("--no_confidence_early_exit", dest="confidence_early_exit", action="store_false")
    parser.add_argument("--out_json", default="")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def meta_arg(meta: dict, name: str, default=None):
    return (meta.get("args") or {}).get(name, default)


@torch.no_grad()
def eval_split(
    split: str,
    dataset,
    loader,
    model,
    backbone,
    projector,
    device: torch.device,
    slothead_mode: str,
    min_steps: int,
    early_exit_conf: float,
    confidence_early_exit: bool,
) -> dict:
    total = 0
    correct = 0
    selected_sum = 0.0
    stopped_sum = 0.0
    conf_sum = 0.0
    true_prob_sum = 0.0
    selected_hist: dict[int, int] = {}
    rows = []
    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"ac80-{split}", mininterval=1.0)):
        labels = labels.to(device, non_blocking=device.type == "cuda")
        images = images.to(device, non_blocking=device.type == "cuda")
        features = backbone.forward_dino(images)
        features = backbone.mlp(features)
        raw_slots, _, _ = backbone.slot_attention(features)
        slots = project_slots80(projector, raw_slots.detach(), mode=slothead_mode)
        out = model.forward_greedy(
            slots,
            None,
            min_steps=min_steps,
            early_exit_conf=early_exit_conf,
            confidence_early_exit=confidence_early_exit,
        )
        probs = out.logits.softmax(dim=1)
        pred = probs.argmax(dim=1)
        selected_count = out.selected_mask.sum(dim=1)
        conf = probs.amax(dim=1)
        true_prob = probs.gather(1, labels[:, None]).squeeze(1)

        batch = labels.numel()
        total += batch
        correct += int((pred == labels).sum().item())
        selected_sum += float(selected_count.sum().detach().cpu())
        stopped_sum += float(out.stopped.to(torch.float32).sum().detach().cpu())
        conf_sum += float(conf.sum().detach().cpu())
        true_prob_sum += float(true_prob.sum().detach().cpu())
        for value in selected_count.detach().cpu().tolist():
            selected_hist[int(value)] = selected_hist.get(int(value), 0) + 1
        for offset in range(batch):
            rows.append(
                {
                    "split": split,
                    "index": batch_idx * loader.batch_size + offset,
                    "true": int(labels[offset].detach().cpu()),
                    "pred": int(pred[offset].detach().cpu()),
                    "correct": int(pred[offset] == labels[offset]),
                    "selected_count": int(selected_count[offset].detach().cpu()),
                    "stopped": int(out.stopped[offset].detach().cpu()),
                    "conf": float(conf[offset].detach().cpu()),
                    "true_prob": float(true_prob[offset].detach().cpu()),
                }
            )

    return {
        "split": split,
        "items": total,
        "acc": correct / max(total, 1),
        "avg_selected": selected_sum / max(total, 1),
        "stopped_rate": stopped_sum / max(total, 1),
        "avg_conf": conf_sum / max(total, 1),
        "true_prob": true_prob_sum / max(total, 1),
        "selected_hist": dict(sorted(selected_hist.items())),
        "rows": rows,
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    run_dir = Path(args.run_dir)
    meta, model, _checkpoint = load_selector(run_dir, args.checkpoint, device)
    data = args.data or meta_arg(meta, "data")
    sa_checkpoint = args.sa_checkpoint or meta_arg(meta, "sa_checkpoint")
    slothead_checkpoint = args.slothead_checkpoint or meta_arg(meta, "slothead_checkpoint")
    slothead_mode = args.slothead_mode or meta_arg(meta, "slothead_mode", "u")
    input_res = args.input_res or int(meta_arg(meta, "input_res", 224))
    if not data or not sa_checkpoint or not slothead_checkpoint:
        raise ValueError("Missing data, sa_checkpoint, or slothead_checkpoint; pass them explicitly.")

    tfm = build_transforms(input_res)
    backbone = load_backbone(sa_checkpoint, device)
    backbone.eval()
    backbone.requires_grad_(False)
    projector, _slothead_ckpt = load_slothead80(slothead_checkpoint, device)

    out_json = Path(args.out_json) if args.out_json else run_dir / f"ac80_classification_eval_min{args.min_steps}_p{args.early_exit_conf:.2f}.json"
    out_csv = out_json.with_suffix(".csv")
    summaries = []
    all_rows = []
    for split in args.splits:
        dataset = build_dataset(data, split, tfm["valid"])
        loader = DataLoader(
            dataset,
            batch_size=args.bs,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        result = eval_split(
            split,
            dataset,
            loader,
            model,
            backbone,
            projector,
            device,
            slothead_mode,
            int(args.min_steps),
            float(args.early_exit_conf),
            bool(args.confidence_early_exit),
        )
        all_rows.extend(result.pop("rows"))
        summaries.append(result)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": args.checkpoint,
        "slothead_mode": slothead_mode,
        "min_steps": args.min_steps,
        "early_exit_conf": args.early_exit_conf,
        "confidence_early_exit": args.confidence_early_exit,
        "splits": summaries,
        "csv": str(out_csv),
    }
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_rows(out_csv, all_rows)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
