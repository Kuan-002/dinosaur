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


def read_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def materialize_image(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"unknown materialize mode: {mode}")


def pair_label(a: str, b: str, ordered: bool) -> str:
    if ordered:
        return f"{a}__{b}"
    x, y = sorted([a, b])
    return f"{x}__{y}"


def collect_split_rows(coco_root: Path, source_split: str, min_area_ratio: float, ordered_pair: bool) -> list[dict[str, str]]:
    data = read_json(coco_root / "annotations" / f"instances_{source_split}2017.json")
    categories = {int(cat["id"]): cat["name"] for cat in data["categories"]}
    images = {int(img["id"]): img for img in data["images"]}
    area_by_image_class: dict[int, Counter[str]] = defaultdict(Counter)
    count_by_image_class: dict[int, Counter[str]] = defaultdict(Counter)

    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        image_id = int(ann["image_id"])
        class_name = categories[int(ann["category_id"])]
        area_by_image_class[image_id][class_name] += float(ann.get("area", 0.0))
        count_by_image_class[image_id][class_name] += 1

    rows: list[dict[str, str]] = []
    for image_id, info in images.items():
        image_area = float(info["width"]) * float(info["height"])
        if image_area <= 0:
            continue
        ranked = [
            (name, area / image_area, area, count_by_image_class[image_id][name])
            for name, area in area_by_image_class.get(image_id, {}).items()
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        if len(ranked) < 2:
            continue
        first, second = ranked[0], ranked[1]
        if first[1] < min_area_ratio or second[1] < min_area_ratio:
            continue
        top_objects = [first[0], second[0]]
        label = pair_label(first[0], second[0], ordered_pair)
        rows.append(
            {
                "source_split": source_split,
                "image_id": str(image_id),
                "file_name": info["file_name"],
                "width": str(info["width"]),
                "height": str(info["height"]),
                "class_name": label,
                "top1_object": first[0],
                "top2_object": second[0],
                "top1_area_ratio": f"{first[1]:.6f}",
                "top2_area_ratio": f"{second[1]:.6f}",
                "top1_total_area": f"{first[2]:.3f}",
                "top2_total_area": f"{second[2]:.3f}",
                "top1_instance_count": str(first[3]),
                "top2_instance_count": str(second[3]),
                "top_objects": ";".join(top_objects),
                "top_area_ratios": f"{first[0]}:{first[1]:.6f};{second[0]}:{second[1]:.6f}",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a COCO object-pair classification dataset from images whose top-2 object categories both occupy enough area."
    )
    parser.add_argument("--coco_root", default="dataset/coco2017")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--min_area_ratio", type=float, default=0.10)
    parser.add_argument("--min_class_count", type=int, default=100)
    parser.add_argument("--train_per_class", type=int, default=0, help="0 keeps all remaining rows after val split.")
    parser.add_argument("--val_per_class", type=int, default=50)
    parser.add_argument("--max_classes", type=int, default=0, help="0 keeps all pair classes passing min_class_count.")
    parser.add_argument("--ordered_pair", action="store_true", help="Keep largest-second-largest order in the class label.")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--materialize", choices=["none", "symlink", "copy"], default="none")
    args = parser.parse_args()

    coco_root = Path(args.coco_root)
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)

    rows = []
    for source_split in ["train", "val"]:
        rows.extend(collect_split_rows(coco_root, source_split, args.min_area_ratio, args.ordered_pair))

    class_counts = Counter(row["class_name"] for row in rows)
    kept_classes = [name for name, count in class_counts.items() if count >= args.min_class_count]
    kept_classes.sort(key=lambda name: (-class_counts[name], name))
    if args.max_classes > 0:
        kept_classes = kept_classes[: args.max_classes]
    kept_class_set = set(kept_classes)
    candidate_rows = [row for row in rows if row["class_name"] in kept_class_set]

    balanced: list[dict[str, str]] = []
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidate_rows:
        by_class[row["class_name"]].append(row)
    for class_name in kept_classes:
        class_rows = [dict(row) for row in by_class[class_name]]
        rng.shuffle(class_rows)
        val_n = min(args.val_per_class, len(class_rows))
        train_pool = class_rows[val_n:]
        if args.train_per_class > 0:
            train_pool = train_pool[: args.train_per_class]
        for row in train_pool:
            row["split"] = "train"
        for row in class_rows[:val_n]:
            row["split"] = "val"
        balanced.extend(train_pool)
        balanced.extend(class_rows[:val_n])

    classes = kept_classes
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    for row in balanced:
        row["class_idx"] = str(class_to_idx[row["class_name"]])

    fieldnames = [
        "split",
        "class_name",
        "class_idx",
        "source_split",
        "image_id",
        "file_name",
        "width",
        "height",
        "top1_object",
        "top2_object",
        "top1_area_ratio",
        "top2_area_ratio",
        "top1_total_area",
        "top2_total_area",
        "top1_instance_count",
        "top2_instance_count",
        "top_objects",
        "top_area_ratios",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "all_candidates.csv", candidate_rows, [name for name in fieldnames if name not in {"split", "class_idx"}])
    write_rows(out_dir / "balanced_samples.csv", balanced, fieldnames)

    summary = {
        "source": "COCO train2017 + val2017 merged",
        "label_definition": "top-2 object category pair by summed same-category annotation area",
        "ordered_pair": args.ordered_pair,
        "min_area_ratio": args.min_area_ratio,
        "min_class_count": args.min_class_count,
        "train_per_class": args.train_per_class,
        "val_per_class": args.val_per_class,
        "max_classes": args.max_classes,
        "raw_candidate_count": len(rows),
        "kept_candidate_count": len(candidate_rows),
        "raw_pair_counts_top30": dict(class_counts.most_common(30)),
        "kept_pair_counts": {name: class_counts[name] for name in classes},
        "balanced_counts": {
            split: dict(Counter(row["class_name"] for row in balanced if row["split"] == split))
            for split in ["train", "val"]
        },
        "classes": classes,
        "materialize": args.materialize,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    if args.materialize != "none":
        root = out_dir / "classification_dataset"
        root.mkdir(parents=True, exist_ok=True)
        (root / "classes.txt").write_text("\n".join(classes) + "\n")
        (root / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2) + "\n")
        (root / "dataset_info.json").write_text(
            json.dumps({"classes": classes, "class_to_idx": class_to_idx}, indent=2) + "\n"
        )
        materialized = []
        for row in balanced:
            src = (coco_root / f"{row['source_split']}2017" / row["file_name"]).resolve()
            if not src.exists():
                raise FileNotFoundError(f"COCO image not found: {src}")
            dst_name = f"{row['source_split']}_{int(row['image_id']):012d}_{row['file_name']}"
            dst = root / row["split"] / row["class_name"] / dst_name
            materialize_image(src, dst, args.materialize)
            meta = dict(row)
            meta["relative_path"] = str(dst.relative_to(root))
            materialized.append(meta)
        write_rows(root / "metadata.csv", materialized, fieldnames + ["relative_path"])
        shutil.copy2(out_dir / "summary.json", root / "summary.json")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
