#!/usr/bin/env python3
"""Audit candidate COCO pairs with frozen DINOSAUR attention-slot oracle metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from scripts.build_coco_compositional_pair10_dataset import load_coco_records
from scripts.evaluate_set_bbox import boxes_to_mask, make_heatmaps, slot_mask_mass, transform_boxes_to_input
from train_slot_classifier import build_transforms, load_backbone


class PairImages(Dataset):
    def __init__(self, coco_root: Path, records, transform):
        self.coco_root, self.records, self.transform = coco_root, records, transform

    def __len__(self): return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.coco_root / f"{record.source_split}2017" / record.file_name
        return self.transform(Image.open(path).convert("RGB")), index


@torch.no_grad()
def encode_attention(backbone, images, device):
    images = images.to(device)
    features = backbone.mlp(backbone.forward_dino(images))
    _slots, attention, _ = backbone.slot_attention(features)
    return attention.detach().cpu()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--coco_root", default=str(ROOT / "dataset/coco2017"))
    parser.add_argument("--sa_checkpoint", default=str(ROOT / "checkpoints/sa_coco_full_20260623_004920/checkpoint_best_mbo_i_slots.pt"))
    parser.add_argument("--evidences", default="person,dining table,car,chair")
    parser.add_argument("--max_pairs_per_evidence", type=int, default=12)
    parser.add_argument("--samples_per_pair", type=int, default=300)
    parser.add_argument("--final_pool_size", type=int, default=500)
    parser.add_argument("--thresholds", default="0.2,0.4")
    parser.add_argument("--threshold_rel", type=float, default=0.5)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    candidate_rows = list(csv.DictReader(open(args.candidate_csv, encoding="utf-8")))
    allowed_evidence = {item.strip() for item in args.evidences.split(",") if item.strip()}
    by_evidence = defaultdict(list)
    for row in candidate_rows:
        if row["shortlisted"] == "True" and row["evidence"] in allowed_evidence:
            by_evidence[row["evidence"]].append(row)
    pairs = []
    for evidence in sorted(by_evidence):
        rows = sorted(by_evidence[evidence], key=lambda row: -float(row["quality_score"]))
        pairs.extend((row["anchor"], evidence) for row in rows[: args.max_pairs_per_evidence])

    constraints = json.load(open(Path(args.candidate_csv).with_name("summary.json"), encoding="utf-8"))["constraints"]
    min_anchor = float(constraints["min_anchor_area"]); min_evidence = float(constraints["min_evidence_area"]); max_rank = int(constraints["max_rank"])
    coco_root = Path(args.coco_root); all_records, _ = load_coco_records(coco_root)
    boxes = {}
    for split in ("train", "val"):
        data = json.loads((coco_root / "annotations" / f"instances_{split}2017.json").read_text(encoding="utf-8"))
        categories = {int(row["id"]): row["name"] for row in data["categories"]}
        by_image = defaultdict(lambda: defaultdict(list))
        for ann in data["annotations"]:
            if not ann.get("iscrowd", 0): by_image[int(ann["image_id"])][categories[int(ann["category_id"])]].append(ann["bbox"])
        boxes[split] = by_image

    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    backbone = load_backbone(args.sa_checkpoint, device).eval(); backbone.requires_grad_(False)
    transform = build_transforms(224)["valid"]
    thresholds = [float(item) for item in args.thresholds.split(",")]
    summaries = []
    for anchor, evidence in pairs:
        candidates = []
        for record in all_records:
            ranks = {name: rank for rank, (name, _area) in enumerate(record.ranked, 1)}
            if (ranks.get(anchor, max_rank + 1) <= max_rank and ranks.get(evidence, max_rank + 1) <= max_rank
                    and record.ratios.get(anchor, 0.0) >= min_anchor and record.ratios.get(evidence, 0.0) >= min_evidence):
                candidates.append(record)
        candidates.sort(key=lambda record: (min(record.ratios[anchor], record.ratios[evidence]), record.ratios[anchor] + record.ratios[evidence]), reverse=True)
        pool = candidates[: args.final_pool_size]
        if len(pool) > args.samples_per_pair:
            # Evenly cover the quality-ranked final pool instead of auditing only its easiest head.
            indices = [round(i * (len(pool) - 1) / (args.samples_per_pair - 1)) for i in range(args.samples_per_pair)]
            audited = [pool[index] for index in indices]
        else:
            audited = pool
        loader = DataLoader(PairImages(coco_root, audited, transform), batch_size=args.bs, shuffle=False, num_workers=args.num_workers)
        counts = {threshold: defaultdict(float) for threshold in thresholds}
        total = 0
        for images, indices in tqdm(loader, desc=f"oracle {anchor}+{evidence}", mininterval=1.0):
            attention = encode_attention(backbone, images, device)
            for batch_index, record_index in enumerate(indices.tolist()):
                record = audited[record_index]
                heatmaps = make_heatmaps(attention[batch_index], 224)
                anchor_mask = boxes_to_mask(transform_boxes_to_input(boxes[record.source_split][record.image_id][anchor], record.width, record.height, 224), 224)
                evidence_mask = boxes_to_mask(transform_boxes_to_input(boxes[record.source_split][record.image_id][evidence], record.width, record.height, 224), 224)
                anchor_mass = [slot_mask_mass(heatmap, anchor_mask, args.threshold_rel) for heatmap in heatmaps]
                evidence_mass = [slot_mask_mass(heatmap, evidence_mask, args.threshold_rel) for heatmap in heatmaps]
                for threshold in thresholds:
                    anchor_slots = {index for index, mass in enumerate(anchor_mass) if mass >= threshold}
                    evidence_slots = {index for index, mass in enumerate(evidence_mass) if mass >= threshold}
                    distinct = any(a != e for a in anchor_slots for e in evidence_slots)
                    counts[threshold]["anchor"] += bool(anchor_slots)
                    counts[threshold]["evidence"] += bool(evidence_slots)
                    counts[threshold]["pair_any"] += bool(anchor_slots and evidence_slots)
                    counts[threshold]["pair_distinct"] += distinct
                    counts[threshold]["same_only"] += bool(anchor_slots and evidence_slots and not distinct)
                total += 1
        row = {"anchor": anchor, "evidence": evidence, "raw_candidates": len(candidates), "final_pool": len(pool), "audited": total}
        for threshold in thresholds:
            for name, value in counts[threshold].items(): row[f"{name}@{threshold:g}"] = value / max(total, 1)
        summaries.append(row)

    summaries.sort(key=lambda row: (-row.get("pair_distinct@0.4", 0.0), -row.get("pair_distinct@0.2", 0.0), row["evidence"], row["anchor"]))
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "slot_oracle.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    result = {"pairs": len(summaries), "samples_per_pair": args.samples_per_pair, "results": summaries}
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
