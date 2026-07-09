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


DEFAULT_CONCEPTS = [
    {"name": "dining_table_scene", "anchors": ["dining table"], "description": "dominant dining table"},
    {"name": "couch_living_scene", "anchors": ["couch"], "description": "dominant couch"},
    {"name": "bedroom_scene", "anchors": ["bed"], "description": "dominant bed"},
    {"name": "motorcycle_scene", "anchors": ["motorcycle"], "description": "dominant motorcycle"},
    {"name": "horse_scene", "anchors": ["horse"], "description": "dominant horse"},
    {"name": "bench_outdoor_scene", "anchors": ["bench"], "description": "dominant bench"},
    {"name": "road_vehicle_scene", "anchors": ["car", "bus", "truck"], "description": "dominant road vehicle"},
    {"name": "kitchen_appliance_scene", "anchors": ["oven", "refrigerator", "microwave", "toaster"], "description": "dominant kitchen appliance"},
    {"name": "pet_scene", "anchors": ["cat", "dog"], "description": "dominant cat or dog"},
    {"name": "computer_media_scene", "anchors": ["laptop", "keyboard", "tv"], "description": "dominant computer or TV"},
]


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


def concept_for_top1(top_object: str, concepts: list[dict]) -> dict | None:
    matches = [concept for concept in concepts if top_object in set(concept["anchors"])]
    if len(matches) > 1:
        raise ValueError(f"ambiguous concept match for top1={top_object}: {[m['name'] for m in matches]}")
    return matches[0] if matches else None


def collect_candidates(coco_root: Path, min_top1_area_ratio: float, concepts: list[dict]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source_split in ["train", "val"]:
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

        for image_id, info in images.items():
            image_area = float(info["width"]) * float(info["height"])
            if image_area <= 0:
                continue
            ranked = [
                (name, area / image_area, area, count_by_image_class[image_id][name])
                for name, area in area_by_image_class.get(image_id, {}).items()
            ]
            ranked.sort(key=lambda item: (-item[1], item[0]))
            if not ranked:
                continue
            top1 = ranked[0]
            if top1[1] < min_top1_area_ratio:
                continue
            concept = concept_for_top1(top1[0], concepts)
            if concept is None:
                continue
            top2 = ranked[1] if len(ranked) > 1 else ("", 0.0, 0.0, 0)
            rows.append(
                {
                    "source_split": source_split,
                    "image_id": str(image_id),
                    "file_name": info["file_name"],
                    "width": str(info["width"]),
                    "height": str(info["height"]),
                    "class_name": concept["name"],
                    "concept_description": concept.get("description", ""),
                    "top1_object": top1[0],
                    "top2_object": top2[0],
                    "top1_area_ratio": f"{top1[1]:.6f}",
                    "top2_area_ratio": f"{top2[1]:.6f}",
                    "top1_total_area": f"{top1[2]:.3f}",
                    "top2_total_area": f"{top2[2]:.3f}",
                    "top1_instance_count": str(top1[3]),
                    "top2_instance_count": str(top2[3]),
                    "top_objects": ";".join([item[0] for item in ranked[:5]]),
                    "top_area_ratios": ";".join(f"{item[0]}:{item[1]:.6f}" for item in ranked[:5]),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a 10-class COCO dataset from dominant top-1 object anchors.")
    parser.add_argument("--coco_root", default="dataset/coco2017")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--min_top1_area_ratio", type=float, default=0.07)
    parser.add_argument("--train_per_class", type=int, default=600)
    parser.add_argument("--val_per_class", type=int, default=200)
    parser.add_argument("--test_per_class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--materialize", choices=["none", "symlink", "copy"], default="none")
    args = parser.parse_args()

    coco_root = Path(args.coco_root)
    out_dir = Path(args.out_dir)
    rng = random.Random(args.seed)
    concepts = DEFAULT_CONCEPTS
    classes = [concept["name"] for concept in concepts]
    class_to_idx = {name: idx for idx, name in enumerate(classes)}

    candidates = collect_candidates(coco_root, args.min_top1_area_ratio, concepts)
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_class[row["class_name"]].append(row)

    required = args.train_per_class + args.val_per_class + args.test_per_class
    missing = {name: len(by_class[name]) for name in classes if len(by_class[name]) < required}
    if missing:
        raise ValueError(f"not enough candidates for requested split total={required}: {missing}")

    balanced: list[dict[str, str]] = []
    for class_name in classes:
        rows = [dict(row) for row in by_class[class_name]]
        rng.shuffle(rows)
        offset = 0
        for split, count in [("train", args.train_per_class), ("val", args.val_per_class), ("test", args.test_per_class)]:
            for row in rows[offset : offset + count]:
                row["split"] = split
                row["class_idx"] = str(class_to_idx[class_name])
                balanced.append(row)
            offset += count

    fieldnames = [
        "split",
        "class_name",
        "class_idx",
        "concept_description",
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
    write_rows(out_dir / "all_candidates.csv", candidates, [name for name in fieldnames if name not in {"split", "class_idx"}])
    write_rows(out_dir / "balanced_samples.csv", balanced, fieldnames)

    summary = {
        "source": "COCO train2017 + val2017 merged",
        "label_definition": "10 scene-anchor classes from dominant top-1 object category/group",
        "min_top1_area_ratio": args.min_top1_area_ratio,
        "train_per_class": args.train_per_class,
        "val_per_class": args.val_per_class,
        "test_per_class": args.test_per_class,
        "candidate_counts": {name: len(by_class[name]) for name in classes},
        "balanced_counts": {
            split: dict(Counter(row["class_name"] for row in balanced if row["split"] == split))
            for split in ["train", "val", "test"]
        },
        "classes": classes,
        "class_to_idx": class_to_idx,
        "concepts": concepts,
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
