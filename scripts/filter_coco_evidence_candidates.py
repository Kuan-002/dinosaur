#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def parse_bbox(value: str) -> tuple[float, float, float, float]:
    parts = value.strip().strip("[]").split(",")
    if len(parts) != 4:
        raise ValueError(f"bad bbox: {value!r}")
    return tuple(float(p.strip()) for p in parts)  # type: ignore[return-value]


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def split_semicolon(value: str) -> list[str]:
    return [x for x in value.split(";") if x]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: Iterable[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"unknown materialize mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter COCO rule candidates to large, separated evidence objects."
    )
    parser.add_argument(
        "--input_dir",
        default="analysis/coco_and_scene_dataset/coco_scene_guidelines_10_v2_no_overlap",
        help="Directory containing database_samples_no_overlap.csv and objects.csv.",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--coco_root", default="dataset/coco")
    parser.add_argument("--min_area_ratio", type=float, default=0.10)
    parser.add_argument("--min_evidence_objects", type=int, default=2)
    parser.add_argument("--max_evidence_objects", type=int, default=3)
    parser.add_argument("--max_pairwise_bbox_iou", type=float, default=0.35)
    parser.add_argument("--train_per_class", type=int, default=0, help="0 keeps all filtered train rows.")
    parser.add_argument("--val_per_class", type=int, default=0, help="0 keeps all filtered val rows.")
    parser.add_argument(
        "--split_mode",
        choices=["preserve", "random"],
        default="preserve",
        help="preserve keeps existing split labels; random resplits filtered rows per class.",
    )
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument(
        "--materialize",
        choices=["none", "symlink", "copy"],
        default="none",
        help="Optionally create classification_dataset/{train,val}/class_name image tree.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    sample_rows = read_rows(input_dir / "database_samples_no_overlap.csv")
    object_rows = read_rows(input_dir / "objects.csv")

    objects_by_image_class: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in object_rows:
        key = (row["source_split"], row["image_id"], row["class_name"])
        objects_by_image_class[key].append(
            {
                "annotation_id": row["annotation_id"],
                "area": float(row["area"]),
                "bbox": parse_bbox(row["bbox"]),
            }
        )

    filtered: list[dict[str, str]] = []
    reject_reasons = Counter()
    for row in sample_rows:
        image_area = float(row["width"]) * float(row["height"])
        evidence_objects = sorted(set(split_semicolon(row["evidence_objects"])))
        if not (args.min_evidence_objects <= len(evidence_objects) <= args.max_evidence_objects):
            reject_reasons["evidence_count"] += 1
            continue

        selected: list[tuple[str, dict[str, object], float]] = []
        for class_name in evidence_objects:
            anns = objects_by_image_class.get((row["source_split"], row["image_id"], class_name), [])
            if not anns:
                continue
            best = max(anns, key=lambda ann: float(ann["area"]))
            area_ratio = float(best["area"]) / image_area
            if area_ratio >= args.min_area_ratio:
                selected.append((class_name, best, area_ratio))

        if len(selected) < args.min_evidence_objects:
            reject_reasons["small_evidence_area"] += 1
            continue

        too_overlapped = False
        for i in range(len(selected)):
            for j in range(i + 1, len(selected)):
                iou = bbox_iou(
                    selected[i][1]["bbox"],  # type: ignore[arg-type]
                    selected[j][1]["bbox"],  # type: ignore[arg-type]
                )
                if iou > args.max_pairwise_bbox_iou:
                    too_overlapped = True
                    break
            if too_overlapped:
                break
        if too_overlapped:
            reject_reasons["bbox_overlap"] += 1
            continue

        out = dict(row)
        out["large_evidence_objects"] = ";".join(item[0] for item in selected)
        out["large_evidence_area_ratios"] = ";".join(f"{item[0]}:{item[2]:.4f}" for item in selected)
        filtered.append(out)

    rng = random.Random(args.seed)
    balanced: list[dict[str, str]] = []
    if args.split_mode == "preserve":
        for split, limit in [("train", args.train_per_class), ("val", args.val_per_class)]:
            by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in filtered:
                if row["split"] == split:
                    by_class[row["database_scene"]].append(row)
            for class_name in sorted(by_class):
                rows = [dict(row) for row in by_class[class_name]]
                rng.shuffle(rows)
                if limit > 0:
                    rows = rows[:limit]
                balanced.extend(rows)
    else:
        by_class = defaultdict(list)
        for row in filtered:
            by_class[row["database_scene"]].append(row)
        for class_name in sorted(by_class):
            rows = [dict(row) for row in by_class[class_name]]
            rng.shuffle(rows)
            val_n = args.val_per_class if args.val_per_class > 0 else 0
            train_n = args.train_per_class if args.train_per_class > 0 else max(0, len(rows) - val_n)
            val_rows = rows[:val_n]
            train_rows = rows[val_n : val_n + train_n]
            for row in train_rows:
                row["split"] = "train"
            for row in val_rows:
                row["split"] = "val"
            balanced.extend(train_rows)
            balanced.extend(val_rows)

    fieldnames = list(filtered[0].keys()) if filtered else list(sample_rows[0].keys())
    write_rows(out_dir / "filtered_candidates.csv", filtered, fieldnames)
    write_rows(out_dir / "balanced_samples.csv", balanced, fieldnames)

    classes = sorted({row["database_scene"] for row in balanced})
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    summary = {
        "input_dir": str(input_dir),
        "min_area_ratio": args.min_area_ratio,
        "min_evidence_objects": args.min_evidence_objects,
        "max_evidence_objects": args.max_evidence_objects,
        "max_pairwise_bbox_iou": args.max_pairwise_bbox_iou,
        "split_mode": args.split_mode,
        "reject_reasons": dict(reject_reasons),
        "filtered_counts": dict(Counter(row["database_scene"] for row in filtered)),
        "balanced_counts": {
            split: dict(Counter(row["database_scene"] for row in balanced if row["split"] == split))
            for split in ["train", "val"]
        },
        "classes": classes,
        "materialize": args.materialize,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if args.materialize != "none":
        root = out_dir / "classification_dataset"
        for name in ["classes.txt", "class_to_idx.json", "dataset_info.json"]:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
        (root / "classes.txt").write_text("\n".join(classes) + "\n")
        (root / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2) + "\n")
        (root / "dataset_info.json").write_text(
            json.dumps({"classes": classes, "class_to_idx": class_to_idx}, indent=2) + "\n"
        )
        shutil.copy2(input_dir / "scene_rule_specs.json", root / "scene_rule_specs.json")
        coco_root = Path(args.coco_root)
        materialized_rows = []
        for row in balanced:
            src = (coco_root / f"{row['source_split']}2017" / row["file_name"]).resolve()
            if not src.exists():
                raise FileNotFoundError(f"COCO image not found: {src}")
            dst_name = f"{row['source_split']}_{int(row['image_id']):012d}_{row['file_name']}"
            dst = root / row["split"] / row["database_scene"] / dst_name
            make_link_or_copy(src, dst, args.materialize)
            meta = dict(row)
            meta["class_name"] = row["database_scene"]
            meta["class_idx"] = str(class_to_idx[row["database_scene"]])
            meta["relative_path"] = str(dst.relative_to(root))
            materialized_rows.append(meta)
        meta_fields = list(materialized_rows[0].keys()) if materialized_rows else fieldnames
        write_rows(root / "metadata.csv", materialized_rows, meta_fields)
        shutil.copy2(out_dir / "summary.json", root / "summary.json")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
