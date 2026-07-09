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
    {
        "name": "dining_table_scene",
        "anchors": ["dining table"],
        "evidence": ["person", "cup", "bowl", "pizza", "sandwich", "cake", "wine glass", "bottle", "fork", "knife", "spoon", "hot dog", "donut"],
        "description": "dominant dining table plus person/tableware/food",
    },
    {
        "name": "bed_context_scene",
        "anchors": ["bed"],
        "evidence": ["person", "cat", "dog", "teddy bear", "suitcase", "chair"],
        "description": "dominant bed plus person, animal, or nearby object",
    },
    {
        "name": "road_vehicle_scene",
        "anchors": ["car", "bus", "truck"],
        "evidence": ["person", "traffic light", "stop sign", "parking meter", "fire hydrant"],
        "description": "dominant road vehicle plus road/person context",
    },
    {
        "name": "large_animal_scene",
        "anchors": ["horse", "cow", "sheep", "elephant", "giraffe", "zebra"],
        "evidence": ["person"],
        "description": "dominant large animal plus human context",
    },
    {
        "name": "bench_context_scene",
        "anchors": ["bench"],
        "evidence": ["person", "cat", "dog", "bird", "bicycle", "potted plant", "chair", "teddy bear", "umbrella", "book", "suitcase"],
        "description": "dominant bench plus person, animal, or nearby object",
    },
    {
        "name": "couch_context_scene",
        "anchors": ["couch"],
        "evidence": ["person", "cat", "dog", "chair", "teddy bear", "suitcase"],
        "description": "dominant couch plus person, animal, or nearby object",
    },
    {
        "name": "device_interaction_scene",
        "anchors": ["laptop", "keyboard", "tv"],
        "evidence": ["person", "cat", "dog", "chair", "book", "mouse", "cell phone", "remote"],
        "description": "dominant electronic device plus user or interaction object",
    },
    {
        "name": "sports_equipment_scene",
        "anchors": ["surfboard", "skateboard", "tennis racket", "kite", "snowboard", "skis", "baseball bat", "sports ball", "frisbee"],
        "evidence": ["person"],
        "description": "dominant sports equipment plus participant",
    },
    {
        "name": "motorcycle_person_scene",
        "anchors": ["motorcycle"],
        "evidence": ["person"],
        "description": "dominant motorcycle plus person",
    },
    {
        "name": "kitchen_appliance_context_scene",
        "anchors": ["oven", "refrigerator", "microwave"],
        "evidence": ["person", "bottle", "pizza", "bowl", "chair", "cat", "dog", "cake", "broccoli", "carrot", "sandwich", "cup", "sink"],
        "description": "dominant kitchen appliance plus food, person, animal, or nearby object",
    },
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


def validate_concepts(concepts: list[dict]) -> None:
    anchor_to_class: dict[str, str] = {}
    for concept in concepts:
        for anchor in concept["anchors"]:
            if anchor in anchor_to_class:
                raise ValueError(f"anchor {anchor!r} appears in multiple classes: {anchor_to_class[anchor]!r} and {concept['name']!r}")
            anchor_to_class[anchor] = concept["name"]

    errors = []
    all_anchors = set(anchor_to_class)
    for concept in concepts:
        overlap = sorted(set(concept["evidence"]) & all_anchors)
        if overlap:
            errors.append(f"{concept['name']}: evidence contains anchor object(s): {', '.join(overlap)}")
    if errors:
        raise ValueError("invalid concept rules; an anchor object must not be used as evidence:\n" + "\n".join(errors))


def collect_candidates(
    coco_root: Path,
    min_anchor_area_ratio: float,
    min_evidence_area_ratio: float,
    concepts: list[dict],
) -> tuple[list[dict[str, str]], int]:
    rows: list[dict[str, str]] = []
    ambiguous = 0
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
            ratios = {name: area / image_area for name, area in area_by_image_class.get(image_id, {}).items()}
            ranked = sorted(ratios.items(), key=lambda item: (-item[1], item[0]))
            if len(ranked) < 2:
                continue
            top1, top2 = ranked[0], ranked[1]
            if top1[1] <= min_evidence_area_ratio or top2[1] <= min_evidence_area_ratio:
                continue
            top2_objects = {top1[0], top2[0]}
            matches = []
            for concept in concepts:
                anchors = top2_objects & set(concept["anchors"])
                evidence = top2_objects & set(concept["evidence"])
                if not anchors or not evidence:
                    continue
                if max(ratios[obj] for obj in anchors) <= min_anchor_area_ratio:
                    continue
                if max(ratios[obj] for obj in evidence) <= min_evidence_area_ratio:
                    continue
                if anchors and evidence:
                    matches.append(concept)
            if len(matches) > 1:
                ambiguous += 1
                continue
            if not matches:
                continue
            concept = matches[0]
            anchor_obj = sorted(top2_objects & set(concept["anchors"]))[0]
            evidence_obj = sorted(top2_objects & set(concept["evidence"]))[0]
            rows.append(
                {
                    "source_split": source_split,
                    "image_id": str(image_id),
                    "file_name": info["file_name"],
                    "width": str(info["width"]),
                    "height": str(info["height"]),
                    "class_name": concept["name"],
                    "concept_description": concept.get("description", ""),
                    "anchor_object": anchor_obj,
                    "evidence_object": evidence_obj,
                    "anchor_area_ratio": f"{ratios[anchor_obj]:.6f}",
                    "evidence_area_ratio": f"{ratios[evidence_obj]:.6f}",
                    "anchor_instance_count": str(count_by_image_class[image_id][anchor_obj]),
                    "evidence_instance_count": str(count_by_image_class[image_id][evidence_obj]),
                    "top1_object": top1[0],
                    "top2_object": top2[0],
                    "top1_area_ratio": f"{top1[1]:.6f}",
                    "top2_area_ratio": f"{top2[1]:.6f}",
                    "top_objects": ";".join([item[0] for item in ranked[:5]]),
                    "top_area_ratios": ";".join(f"{item[0]}:{item[1]:.6f}" for item in ranked[:5]),
                }
            )
    return rows, ambiguous


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean 10-class COCO top-2-object scene dataset.")
    parser.add_argument("--coco_root", default="dataset/coco2017")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--min_top2_area_ratio", type=float, default=None, help="Deprecated alias that sets both anchor and evidence thresholds.")
    parser.add_argument("--min_anchor_area_ratio", type=float, default=0.09)
    parser.add_argument("--min_evidence_area_ratio", type=float, default=0.05)
    parser.add_argument("--train_per_class", type=int, default=300)
    parser.add_argument("--val_per_class", type=int, default=100)
    parser.add_argument("--test_per_class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--materialize", choices=["none", "symlink", "copy"], default="none")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    coco_root = Path(args.coco_root)
    rng = random.Random(args.seed)
    concepts = DEFAULT_CONCEPTS
    validate_concepts(concepts)
    if args.min_top2_area_ratio is not None:
        args.min_anchor_area_ratio = args.min_top2_area_ratio
        args.min_evidence_area_ratio = args.min_top2_area_ratio
    classes = [concept["name"] for concept in concepts]
    class_to_idx = {name: idx for idx, name in enumerate(classes)}

    candidates, ambiguous = collect_candidates(coco_root, args.min_anchor_area_ratio, args.min_evidence_area_ratio, concepts)
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
        "anchor_object",
        "evidence_object",
        "anchor_area_ratio",
        "evidence_area_ratio",
        "anchor_instance_count",
        "evidence_instance_count",
        "top1_object",
        "top2_object",
        "top1_area_ratio",
        "top2_area_ratio",
        "top_objects",
        "top_area_ratios",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "all_candidates.csv", candidates, [name for name in fieldnames if name not in {"split", "class_idx"}])
    write_rows(out_dir / "balanced_samples.csv", balanced, fieldnames)

    summary = {
        "source": "COCO train2017 + val2017 merged",
        "label_definition": "10 semantically separated classes; top-2 object categories uniquely match one concept, with a stronger anchor area threshold",
        "min_anchor_area_ratio": args.min_anchor_area_ratio,
        "min_evidence_area_ratio": args.min_evidence_area_ratio,
        "ambiguous_candidate_count_dropped": ambiguous,
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
        (root / "dataset_info.json").write_text(json.dumps({"classes": classes, "class_to_idx": class_to_idx}, indent=2) + "\n")
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
