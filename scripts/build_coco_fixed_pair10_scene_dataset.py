#!/usr/bin/env python3
"""Build a 10-class COCO dataset with one fixed anchor/evidence pair per class.

This is a stricter variant of the top-2 clean scene dataset.  Each class is
defined by exactly one anchor category and exactly one evidence category, so
role diagnostics are not confounded by multiple evidence object types inside a
single class.
"""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_coco_top2_clean10_scene_dataset import (  # noqa: E402
    collect_candidates,
    materialize_image,
    validate_concepts,
    write_rows,
)


FIXED_PAIR_CONCEPTS = [
    {
        "name": "dining_table_person_scene",
        "anchors": ["dining table"],
        "evidence": ["person"],
        "description": "fixed pair: dining table anchor plus person evidence",
    },
    {
        "name": "motorcycle_person_scene",
        "anchors": ["motorcycle"],
        "evidence": ["person"],
        "description": "fixed pair: motorcycle anchor plus person evidence",
    },
    {
        "name": "couch_person_scene",
        "anchors": ["couch"],
        "evidence": ["person"],
        "description": "fixed pair: couch anchor plus person evidence",
    },
    {
        "name": "umbrella_person_scene",
        "anchors": ["umbrella"],
        "evidence": ["person"],
        "description": "fixed pair: umbrella anchor plus person evidence",
    },
    {
        "name": "chair_person_scene",
        "anchors": ["chair"],
        "evidence": ["person"],
        "description": "fixed pair: chair anchor plus person evidence",
    },
    {
        "name": "horse_person_scene",
        "anchors": ["horse"],
        "evidence": ["person"],
        "description": "fixed pair: horse anchor plus person evidence",
    },
    {
        "name": "bed_person_scene",
        "anchors": ["bed"],
        "evidence": ["person"],
        "description": "fixed pair: bed anchor plus person evidence",
    },
    {
        "name": "bench_person_scene",
        "anchors": ["bench"],
        "evidence": ["person"],
        "description": "fixed pair: bench anchor plus person evidence",
    },
    {
        "name": "car_person_scene",
        "anchors": ["car"],
        "evidence": ["person"],
        "description": "fixed pair: car anchor plus person evidence",
    },
    {
        "name": "surfboard_person_scene",
        "anchors": ["surfboard"],
        "evidence": ["person"],
        "description": "fixed pair: surfboard anchor plus person evidence",
    },
]


FIELDNAMES = [
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


def candidate_strength(row: dict[str, str]) -> tuple[float, float, float, float, int]:
    anchor = float(row["anchor_area_ratio"])
    evidence = float(row["evidence_area_ratio"])
    top2 = float(row["top2_area_ratio"])
    return (min(anchor, evidence), anchor + evidence, top2, max(anchor, evidence), int(row["image_id"]))


def read_concepts(path: str | None) -> list[dict]:
    if path is None:
        return FIXED_PAIR_CONCEPTS
    with Path(path).open(encoding="utf-8") as f:
        concepts = json.load(f)
    for concept in concepts:
        if len(concept.get("anchors", [])) != 1 or len(concept.get("evidence", [])) != 1:
            raise ValueError("fixed-pair concepts must have exactly one anchor and exactly one evidence category")
    return concepts


def split_rows(
    rows_by_class: dict[str, list[dict[str, str]]],
    classes: list[str],
    class_to_idx: dict[str, int],
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    seed: int,
) -> list[dict[str, str]]:
    rng = random.Random(seed)
    balanced: list[dict[str, str]] = []
    required = train_per_class + val_per_class + test_per_class
    missing = {name: len(rows_by_class[name]) for name in classes if len(rows_by_class[name]) < required}
    if missing:
        raise ValueError(f"not enough fixed-pair candidates for requested split total={required}: {missing}")

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


def write_classification_dataset(
    root: Path,
    coco_root: Path,
    out_dir: Path,
    classes: list[str],
    class_to_idx: dict[str, int],
    balanced: list[dict[str, str]],
    materialize: str,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (root / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2) + "\n", encoding="utf-8")
    (root / "dataset_info.json").write_text(
        json.dumps({"classes": classes, "class_to_idx": class_to_idx}, indent=2) + "\n",
        encoding="utf-8",
    )
    materialized = []
    for row in balanced:
        src = (coco_root / f"{row['source_split']}2017" / row["file_name"]).resolve()
        if not src.exists():
            raise FileNotFoundError(f"COCO image not found: {src}")
        dst_name = f"{row['source_split']}_{int(row['image_id']):012d}_{row['file_name']}"
        dst = root / row["split"] / row["class_name"] / dst_name
        materialize_image(src, dst, materialize)
        meta = dict(row)
        meta["relative_path"] = str(dst.relative_to(root))
        materialized.append(meta)
    write_rows(root / "metadata.csv", materialized, FIELDNAMES + ["relative_path"])
    shutil.copy2(out_dir / "summary.json", root / "summary.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco_root", default=str(REPO_ROOT / "dataset/coco2017"))
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--concepts_json", default=None)
    parser.add_argument("--min_anchor_area_ratio", type=float, default=0.055)
    parser.add_argument("--min_evidence_area_ratio", type=float, default=0.04)
    parser.add_argument("--train_per_class", type=int, default=300)
    parser.add_argument("--val_per_class", type=int, default=100)
    parser.add_argument("--test_per_class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--materialize", choices=["none", "symlink", "copy"], default="symlink")
    args = parser.parse_args()

    coco_root = Path(args.coco_root)
    out_dir = Path(args.out_dir)
    concepts = read_concepts(args.concepts_json)
    validate_concepts(concepts)
    classes = [concept["name"] for concept in concepts]
    class_to_idx = {name: idx for idx, name in enumerate(classes)}

    candidates, ambiguous = collect_candidates(
        coco_root=coco_root,
        min_anchor_area_ratio=float(args.min_anchor_area_ratio),
        min_evidence_area_ratio=float(args.min_evidence_area_ratio),
        concepts=concepts,
    )
    rows_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        rows_by_class[row["class_name"]].append(row)

    balanced = split_rows(
        rows_by_class=rows_by_class,
        classes=classes,
        class_to_idx=class_to_idx,
        train_per_class=int(args.train_per_class),
        val_per_class=int(args.val_per_class),
        test_per_class=int(args.test_per_class),
        seed=int(args.seed),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "all_candidates.csv", candidates, [name for name in FIELDNAMES if name not in {"split", "class_idx"}])
    write_rows(out_dir / "balanced_samples.csv", balanced, FIELDNAMES)

    pair_by_class = {
        concept["name"]: {
            "anchor": concept["anchors"][0],
            "evidence": concept["evidence"][0],
        }
        for concept in concepts
    }
    summary = {
        "source": "COCO train2017 + val2017 merged",
        "label_definition": "10 fixed anchor/evidence top-2 object pairs; one anchor category and one evidence category per class",
        "selection_strategy": "largest fixed-pair objects first",
        "min_anchor_area_ratio": float(args.min_anchor_area_ratio),
        "min_evidence_area_ratio": float(args.min_evidence_area_ratio),
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
                "anchor": sum(float(row["anchor_area_ratio"]) for row in balanced if row["class_name"] == name) / max(1, sum(1 for row in balanced if row["class_name"] == name)),
                "evidence": sum(float(row["evidence_area_ratio"]) for row in balanced if row["class_name"] == name) / max(1, sum(1 for row in balanced if row["class_name"] == name)),
            }
            for name in classes
        },
        "classes": classes,
        "class_to_idx": class_to_idx,
        "fixed_pairs": pair_by_class,
        "concepts": concepts,
        "materialize": args.materialize,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.materialize != "none":
        write_classification_dataset(
            root=out_dir / "classification_dataset",
            coco_root=coco_root,
            out_dir=out_dir,
            classes=classes,
            class_to_idx=class_to_idx,
            balanced=balanced,
            materialize=args.materialize,
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
