#!/usr/bin/env python3
"""Build a 10-class COCO dataset with disjoint fixed two-anchor pairs.

Each class is defined by exactly two anchor object categories.  No object
category is reused by any other class.  The two anchors must both be present
with sufficient image-area ratio, but they do not have to be the image top-2
objects; samples are still selected by largest anchor coverage first.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


DUAL_ANCHOR_PAIRS = [
    ("dining_table_pizza_scene", "dining table", "pizza"),
    ("bed_cat_scene", "bed", "cat"),
    ("person_refrigerator_scene", "person", "refrigerator"),
    ("bowl_broccoli_scene", "bowl", "broccoli"),
    ("laptop_tv_scene", "laptop", "tv"),
    ("potted_plant_vase_scene", "potted plant", "vase"),
    ("chair_couch_scene", "chair", "couch"),
    ("cup_sandwich_scene", "cup", "sandwich"),
    ("car_truck_scene", "car", "truck"),
    ("sink_toilet_scene", "sink", "toilet"),
]


FIELDNAMES = [
    "split",
    "class_name",
    "class_idx",
    "source_split",
    "image_id",
    "file_name",
    "width",
    "height",
    "anchor_a_object",
    "anchor_b_object",
    "anchor_a_area_ratio",
    "anchor_b_area_ratio",
    "anchor_a_instance_count",
    "anchor_b_instance_count",
    "anchor_a_rank",
    "anchor_b_rank",
    "top1_object",
    "top2_object",
    "top1_area_ratio",
    "top2_area_ratio",
    "top_objects",
    "top_area_ratios",
]


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
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


def candidate_strength(row: dict[str, str]) -> tuple[float, float, float, float, int]:
    a = float(row["anchor_a_area_ratio"])
    b = float(row["anchor_b_area_ratio"])
    return (min(a, b), a + b, float(row["top2_area_ratio"]), max(a, b), int(row["image_id"]))


def collect_candidates(
    coco_root: Path,
    pairs: list[tuple[str, str, str]],
    min_anchor_area_ratio: float,
) -> tuple[list[dict[str, str]], int]:
    used: Counter[str] = Counter()
    for _class_name, a, b in pairs:
        used[a] += 1
        used[b] += 1
    repeated = sorted(name for name, count in used.items() if count > 1)
    if repeated:
        raise ValueError(f"dual-anchor objects must be globally disjoint; repeated objects: {repeated}")
    rows: list[dict[str, str]] = []
    ambiguous = 0
    for source_split in ["train", "val"]:
        data = read_json(coco_root / "annotations" / f"instances_{source_split}2017.json")
        categories = {int(cat["id"]): str(cat["name"]) for cat in data["categories"]}
        images = {int(img["id"]): img for img in data["images"]}
        area_by_image_class: dict[int, Counter[str]] = defaultdict(Counter)
        count_by_image_class: dict[int, Counter[str]] = defaultdict(Counter)
        for ann in data["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            image_id = int(ann["image_id"])
            name = categories[int(ann["category_id"])]
            area_by_image_class[image_id][name] += float(ann.get("area", 0.0))
            count_by_image_class[image_id][name] += 1

        for image_id, info in images.items():
            image_area = float(info["width"]) * float(info["height"])
            if image_area <= 0:
                continue
            ratios = {name: area / image_area for name, area in area_by_image_class.get(image_id, {}).items()}
            ranked = sorted(ratios.items(), key=lambda item: (-item[1], item[0]))
            if len(ranked) < 2:
                continue
            top1, top2 = ranked[0], ranked[1]
            rank_by_object = {name: idx + 1 for idx, (name, _ratio) in enumerate(ranked)}
            matches = []
            for class_name, anchor_a, anchor_b in pairs:
                if ratios.get(anchor_a, 0.0) > min_anchor_area_ratio and ratios.get(anchor_b, 0.0) > min_anchor_area_ratio:
                    matches.append((class_name, anchor_a, anchor_b))
            if len(matches) > 1:
                ambiguous += 1
                continue
            if not matches:
                continue
            class_name, anchor_a, anchor_b = matches[0]
            rows.append(
                {
                    "source_split": source_split,
                    "image_id": str(image_id),
                    "file_name": str(info["file_name"]),
                    "width": str(info["width"]),
                    "height": str(info["height"]),
                    "class_name": class_name,
                    "anchor_a_object": anchor_a,
                    "anchor_b_object": anchor_b,
                    "anchor_a_area_ratio": f"{ratios[anchor_a]:.6f}",
                    "anchor_b_area_ratio": f"{ratios[anchor_b]:.6f}",
                    "anchor_a_instance_count": str(count_by_image_class[image_id][anchor_a]),
                    "anchor_b_instance_count": str(count_by_image_class[image_id][anchor_b]),
                    "anchor_a_rank": str(rank_by_object[anchor_a]),
                    "anchor_b_rank": str(rank_by_object[anchor_b]),
                    "top1_object": top1[0],
                    "top2_object": top2[0],
                    "top1_area_ratio": f"{top1[1]:.6f}",
                    "top2_area_ratio": f"{top2[1]:.6f}",
                    "top_objects": ";".join(item[0] for item in ranked[:5]),
                    "top_area_ratios": ";".join(f"{item[0]}:{item[1]:.6f}" for item in ranked[:5]),
                }
            )
    return rows, ambiguous


def split_rows(
    rows_by_class: dict[str, list[dict[str, str]]],
    classes: list[str],
    class_to_idx: dict[str, int],
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    seed: int,
) -> list[dict[str, str]]:
    required = train_per_class + val_per_class + test_per_class
    missing = {name: len(rows_by_class[name]) for name in classes if len(rows_by_class[name]) < required}
    if missing:
        raise ValueError(f"not enough dual-anchor candidates for requested split total={required}: {missing}")
    rng = random.Random(seed)
    balanced: list[dict[str, str]] = []
    for class_name in classes:
        rows = [dict(row) for row in rows_by_class[class_name]]
        rng.shuffle(rows)
        rows.sort(key=candidate_strength, reverse=True)
        offset = 0
        for split, count in [("train", train_per_class), ("val", val_per_class), ("test", test_per_class)]:
            for row in rows[offset : offset + count]:
                row["split"] = split
                row["class_idx"] = str(class_to_idx[class_name])
                balanced.append(row)
            offset += count
    return balanced


def write_dataset(root: Path, coco_root: Path, out_dir: Path, classes: list[str], class_to_idx: dict[str, int], rows: list[dict[str, str]], mode: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (root / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2) + "\n", encoding="utf-8")
    (root / "dataset_info.json").write_text(json.dumps({"classes": classes, "class_to_idx": class_to_idx}, indent=2) + "\n", encoding="utf-8")
    materialized = []
    for row in rows:
        src = (coco_root / f"{row['source_split']}2017" / row["file_name"]).resolve()
        if not src.exists():
            raise FileNotFoundError(src)
        dst_name = f"{row['source_split']}_{int(row['image_id']):012d}_{row['file_name']}"
        dst = root / row["split"] / row["class_name"] / dst_name
        materialize_image(src, dst, mode)
        meta = dict(row)
        meta["relative_path"] = str(dst.relative_to(root))
        materialized.append(meta)
    write_rows(root / "metadata.csv", materialized, FIELDNAMES + ["relative_path"])
    shutil.copy2(out_dir / "summary.json", root / "summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco_root", default=str(REPO_ROOT / "dataset/coco2017"))
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--min_anchor_area_ratio", type=float, default=0.015)
    parser.add_argument("--train_per_class", type=int, default=300)
    parser.add_argument("--val_per_class", type=int, default=100)
    parser.add_argument("--test_per_class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--materialize", choices=["none", "symlink", "copy"], default="symlink")
    args = parser.parse_args()

    coco_root = Path(args.coco_root)
    out_dir = Path(args.out_dir)
    classes = [name for name, _a, _b in DUAL_ANCHOR_PAIRS]
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    candidates, ambiguous = collect_candidates(coco_root, DUAL_ANCHOR_PAIRS, float(args.min_anchor_area_ratio))
    rows_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        rows_by_class[row["class_name"]].append(row)
    balanced = split_rows(
        rows_by_class,
        classes,
        class_to_idx,
        int(args.train_per_class),
        int(args.val_per_class),
        int(args.test_per_class),
        int(args.seed),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "all_candidates.csv", candidates, [name for name in FIELDNAMES if name not in {"split", "class_idx"}])
    write_rows(out_dir / "balanced_samples.csv", balanced, FIELDNAMES)
    pair_by_class = {name: {"anchor_a": a, "anchor_b": b} for name, a, b in DUAL_ANCHOR_PAIRS}
    summary = {
        "source": "COCO train2017 + val2017 merged",
        "label_definition": "10 fixed dual-anchor object pairs; both objects are class-defining anchors and no object is reused across classes",
        "selection_strategy": "largest dual-anchor objects first",
        "min_anchor_area_ratio": float(args.min_anchor_area_ratio),
        "train_per_class": int(args.train_per_class),
        "val_per_class": int(args.val_per_class),
        "test_per_class": int(args.test_per_class),
        "ambiguous_candidate_count_dropped": ambiguous,
        "candidate_counts": {name: len(rows_by_class[name]) for name in classes},
        "balanced_counts": {
            split: dict(Counter(row["class_name"] for row in balanced if row["split"] == split))
            for split in ["train", "val", "test"]
        },
        "mean_selected_area_ratio": {
            name: {
                "anchor_a": sum(float(row["anchor_a_area_ratio"]) for row in balanced if row["class_name"] == name) / max(1, sum(1 for row in balanced if row["class_name"] == name)),
                "anchor_b": sum(float(row["anchor_b_area_ratio"]) for row in balanced if row["class_name"] == name) / max(1, sum(1 for row in balanced if row["class_name"] == name)),
            }
            for name in classes
        },
        "classes": classes,
        "class_to_idx": class_to_idx,
        "fixed_dual_anchors": pair_by_class,
        "materialize": args.materialize,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.materialize != "none":
        write_dataset(out_dir / "classification_dataset", coco_root, out_dir, classes, class_to_idx, balanced, args.materialize)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
