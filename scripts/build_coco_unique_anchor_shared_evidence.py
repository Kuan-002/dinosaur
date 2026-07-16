#!/usr/bin/env python3
"""Materialize a searched unique-anchor/shared-evidence COCO dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_coco_compositional_pair10_dataset import load_coco_records


FIELDS = [
    "split", "class_name", "class_idx", "source_split", "image_id", "file_name", "width", "height",
    "anchor_object", "evidence_object", "anchor_area_ratio", "evidence_area_ratio",
    "anchor_instance_count", "evidence_instance_count", "anchor_rank", "evidence_rank",
    "top1_object", "top2_object", "top1_area_ratio", "top2_area_ratio", "top_objects", "top_area_ratios",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search_report", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--coco_root", default=str(ROOT / "dataset" / "coco2017"))
    parser.add_argument("--train_per_class", type=int, default=300)
    parser.add_argument("--val_per_class", type=int, default=100)
    parser.add_argument("--test_per_class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--materialize", choices=["none", "symlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    report = json.loads(Path(args.search_report).read_text(encoding="utf-8"))
    constraints = report.get("constraints") or report.get("construction_filters")
    pairs = [(row["anchor"], row["evidence"], row["class_name"]) for row in report["pairs"]]
    if len({anchor for anchor, _, _ in pairs}) != len(pairs):
        raise ValueError("search report does not contain unique anchors")
    degrees = Counter(evidence for _, evidence, _ in pairs)
    if any(degree not in (2, 3) for degree in degrees.values()):
        raise ValueError(f"evidence degrees are not all 2--3: {degrees}")

    records, _ = load_coco_records(Path(args.coco_root))
    min_anchor = float(constraints.get("min_anchor_area_ratio", constraints.get("min_anchor_area")))
    min_evidence = float(constraints.get("min_evidence_area_ratio", constraints.get("min_evidence_area")))
    max_rank = int(constraints.get("max_object_rank", constraints.get("max_rank")))
    rows_by_class: dict[str, list[dict]] = defaultdict(list)
    ambiguous = 0
    for record in records:
        rank = {name: index for index, (name, _) in enumerate(record.ranked, start=1)}
        matches = [
            (anchor, evidence, name)
            for anchor, evidence, name in pairs
            if rank.get(anchor, max_rank + 1) <= max_rank
            and rank.get(evidence, max_rank + 1) <= max_rank
            and record.ratios.get(anchor, 0.0) >= min_anchor
            and record.ratios.get(evidence, 0.0) >= min_evidence
        ]
        if len(matches) != 1:
            ambiguous += int(len(matches) > 1)
            continue
        anchor, evidence, name = matches[0]
        top1 = record.ranked[0]
        top2 = record.ranked[1] if len(record.ranked) > 1 else ("", 0.0)
        rows_by_class[name].append({
            "class_name": name, "source_split": record.source_split, "image_id": record.image_id,
            "file_name": record.file_name, "width": record.width, "height": record.height,
            "anchor_object": anchor, "evidence_object": evidence,
            "anchor_area_ratio": f"{record.ratios[anchor]:.6f}", "evidence_area_ratio": f"{record.ratios[evidence]:.6f}",
            "anchor_instance_count": record.counts[anchor], "evidence_instance_count": record.counts[evidence],
            "anchor_rank": rank[anchor], "evidence_rank": rank[evidence],
            "top1_object": top1[0], "top2_object": top2[0],
            "top1_area_ratio": f"{top1[1]:.6f}", "top2_area_ratio": f"{top2[1]:.6f}",
            "top_objects": ";".join(obj for obj, _ in record.ranked[:5]),
            "top_area_ratios": ";".join(f"{obj}:{area:.6f}" for obj, area in record.ranked[:5]),
        })

    required = args.train_per_class + args.val_per_class + args.test_per_class
    missing = {name: len(rows_by_class[name]) for _, _, name in pairs if len(rows_by_class[name]) < required}
    if missing:
        raise ValueError(f"not enough clean candidates for {required} samples per class: {missing}")
    rng = random.Random(args.seed)
    class_names = sorted(name for _, _, name in pairs)
    class_to_idx = {name: index for index, name in enumerate(class_names)}
    selected = []
    for name in class_names:
        candidates = list(rows_by_class[name])
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda row: (min(float(row["anchor_area_ratio"]), float(row["evidence_area_ratio"])),
                             float(row["anchor_area_ratio"]) + float(row["evidence_area_ratio"])),
            reverse=True,
        )
        chosen = candidates[:required]
        rng.shuffle(chosen)
        offset = 0
        for split, count in (("train", args.train_per_class), ("val", args.val_per_class), ("test", args.test_per_class)):
            for row in chosen[offset:offset + count]:
                row = dict(row); row["split"] = split; row["class_idx"] = class_to_idx[name]
                selected.append(row)
            offset += count

    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output directory is not empty: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    write_csv(out / "all_candidates.csv", [row for name in class_names for row in rows_by_class[name]], FIELDS[1:])
    write_csv(out / "balanced_samples.csv", selected, FIELDS)
    summary = {
        "source": "COCO train2017 + val2017", "search_report": str(Path(args.search_report)),
        "train_per_class": args.train_per_class, "val_per_class": args.val_per_class, "test_per_class": args.test_per_class,
        "unique_anchors": True, "anchor_evidence_vocab_disjoint": True, "one_evidence_per_class": True,
        "evidence_degrees": dict(sorted(degrees.items())), "ambiguous_candidate_count_dropped": ambiguous,
        "candidate_counts": {name: len(rows_by_class[name]) for name in class_names},
        "classes": class_names, "class_to_idx": class_to_idx,
        "pairs": [{"class_name": name, "anchor": anchor, "evidence": evidence} for anchor, evidence, name in pairs],
        "filters": {"min_anchor_area_ratio": min_anchor, "min_evidence_area_ratio": min_evidence, "max_object_rank": max_rank},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.materialize != "none":
        root = out / "classification_dataset"; root.mkdir()
        (root / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
        (root / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2) + "\n", encoding="utf-8")
        metadata = []
        for row in selected:
            source = (Path(args.coco_root) / f"{row['source_split']}2017" / row["file_name"]).resolve()
            destination = root / row["split"] / row["class_name"] / f"{row['source_split']}_{int(row['image_id']):012d}_{row['file_name']}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if args.materialize == "symlink": os.symlink(source, destination)
            else: shutil.copy2(source, destination)
            meta = dict(row); meta["relative_path"] = str(destination.relative_to(root)); metadata.append(meta)
        write_csv(root / "metadata.csv", metadata, FIELDS + ["relative_path"])
        shutil.copy2(out / "summary.json", root / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
