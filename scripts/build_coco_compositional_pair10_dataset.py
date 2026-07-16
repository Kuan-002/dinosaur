#!/usr/bin/env python3
"""Build a COCO compositional-pair classification dataset.

Each class is an unordered pair of COCO object categories.  Unlike the older
fixed-pair datasets, objects are deliberately reused across classes: every
selected object should normally have graph degree 2 or 3 (hard maximum 4).
Consequently, neither member of a pair uniquely determines the class.

The builder has two phases:

1. Aggregate COCO instance areas per image and build all sufficiently large
   object-pair candidate sets.
2. Search for a graph with ``num_classes`` edges whose clean (unambiguous)
   candidate count is large enough for every edge and whose vertex degrees
   satisfy the reuse constraints.

An image is clean for an edge only when it matches exactly one selected edge.
Bounding boxes/areas are used only to construct and audit the dataset; the
materialized classification dataset contains ordinary image/class folders.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
Pair = tuple[str, str]


FIELDNAMES = [
    "split",
    "class_name",
    "class_idx",
    "source_split",
    "image_id",
    "file_name",
    "width",
    "height",
    "object_a",
    "object_b",
    "object_a_area_ratio",
    "object_b_area_ratio",
    "object_a_instance_count",
    "object_b_instance_count",
    "object_a_rank",
    "object_b_rank",
    "anchor_object",
    "evidence_object",
    "anchor_area_ratio",
    "evidence_area_ratio",
    "top1_object",
    "top2_object",
    "top1_area_ratio",
    "top2_area_ratio",
    "top_objects",
    "top_area_ratios",
]


@dataclass(frozen=True)
class ImageRecord:
    source_split: str
    image_id: int
    file_name: str
    width: int
    height: int
    ratios: dict[str, float]
    counts: dict[str, int]
    ranked: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class SearchResult:
    edges: tuple[Pair, ...]
    clean_counts: dict[Pair, int]
    raw_counts: dict[Pair, int]
    degrees: dict[str, int]
    score: float
    feasible: bool


def canonical_pair(a: str, b: str) -> Pair:
    if a == b:
        raise ValueError(f"pair members must differ: {a!r}")
    return (a, b) if a < b else (b, a)


def slug(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return value or "object"


def class_name(pair: Pair) -> str:
    return f"{slug(pair[0])}__{slug(pair[1])}_scene"


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
        raise ValueError(f"unknown materialization mode: {mode}")


def parse_object_list(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def load_coco_records(coco_root: Path) -> tuple[list[ImageRecord], list[str]]:
    records: list[ImageRecord] = []
    category_names: set[str] = set()
    for source_split in ("train", "val"):
        annotation_path = coco_root / "annotations" / f"instances_{source_split}2017.json"
        data = read_json(annotation_path)
        categories = {int(row["id"]): str(row["name"]) for row in data["categories"]}
        category_names.update(categories.values())
        images = {int(row["id"]): row for row in data["images"]}
        area_by_image: dict[int, Counter[str]] = defaultdict(Counter)
        count_by_image: dict[int, Counter[str]] = defaultdict(Counter)
        for ann in data["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            image_id = int(ann["image_id"])
            name = categories[int(ann["category_id"])]
            area_by_image[image_id][name] += float(ann.get("area", 0.0))
            count_by_image[image_id][name] += 1

        for image_id, areas in area_by_image.items():
            info = images[image_id]
            image_area = float(info["width"]) * float(info["height"])
            if image_area <= 0:
                continue
            ratios = {name: float(area) / image_area for name, area in areas.items()}
            ranked = tuple(sorted(ratios.items(), key=lambda item: (-item[1], item[0])))
            records.append(
                ImageRecord(
                    source_split=source_split,
                    image_id=image_id,
                    file_name=str(info["file_name"]),
                    width=int(info["width"]),
                    height=int(info["height"]),
                    ratios=ratios,
                    counts=dict(count_by_image[image_id]),
                    ranked=ranked,
                )
            )
    records.sort(key=lambda row: (row.source_split, row.image_id))
    return records, sorted(category_names)


def qualifying_pairs(
    record: ImageRecord,
    min_object_area_ratio: float,
    min_larger_object_area_ratio: float,
    min_pair_area_ratio: float,
    max_object_rank: int,
    excluded_objects: set[str],
) -> list[Pair]:
    eligible = [
        (name, ratio)
        for rank, (name, ratio) in enumerate(record.ranked, start=1)
        if rank <= max_object_rank and ratio >= min_object_area_ratio and name not in excluded_objects
    ]
    pairs: list[Pair] = []
    for idx, (a, area_a) in enumerate(eligible):
        for b, area_b in eligible[idx + 1 :]:
            if max(area_a, area_b) < min_larger_object_area_ratio:
                continue
            if area_a + area_b < min_pair_area_ratio:
                continue
            pairs.append(canonical_pair(a, b))
    return pairs


def build_pair_index(
    records: list[ImageRecord],
    args: argparse.Namespace,
) -> tuple[dict[Pair, int], dict[Pair, int]]:
    raw_counts: Counter[Pair] = Counter()
    pair_bits: dict[Pair, int] = defaultdict(int)
    excluded = parse_object_list(args.exclude_objects)
    for index, record in enumerate(records):
        pairs = qualifying_pairs(
            record,
            min_object_area_ratio=float(args.min_object_area_ratio),
            min_larger_object_area_ratio=float(args.min_larger_object_area_ratio),
            min_pair_area_ratio=float(args.min_pair_area_ratio),
            max_object_rank=int(args.max_object_rank),
            excluded_objects=excluded,
        )
        bit = 1 << index
        for pair in pairs:
            raw_counts[pair] += 1
            pair_bits[pair] |= bit
    return dict(raw_counts), dict(pair_bits)


def selected_degrees(edges: Iterable[Pair]) -> dict[str, int]:
    degrees: Counter[str] = Counter()
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    return dict(degrees)


def exactly_once_bits(edges: Iterable[Pair], pair_bits: dict[Pair, int]) -> int:
    once = 0
    multiple = 0
    for edge in edges:
        bits = pair_bits[edge]
        new_multiple = multiple | (once & bits)
        once = (once ^ bits) & ~new_multiple
        multiple = new_multiple
    return once


def clean_counts_for_edges(edges: tuple[Pair, ...], pair_bits: dict[Pair, int]) -> dict[Pair, int]:
    once = exactly_once_bits(edges, pair_bits)
    return {edge: (pair_bits[edge] & once).bit_count() for edge in edges}


def state_score(
    edges: tuple[Pair, ...],
    raw_counts: dict[Pair, int],
    pair_bits: dict[Pair, int],
    target_clean: int,
    min_degree: int,
    preferred_max_degree: int,
    max_degree: int,
) -> tuple[float, dict[Pair, int], dict[str, int], bool]:
    degrees = selected_degrees(edges)
    clean_counts = clean_counts_for_edges(edges, pair_bits)
    low_degree = sum(max(0, min_degree - degree) for degree in degrees.values())
    over_degree = sum(max(0, degree - max_degree) for degree in degrees.values())
    degree_four = sum(max(0, degree - preferred_max_degree) for degree in degrees.values())
    deficits = sum(max(0, target_clean - count) for count in clean_counts.values())
    minimum = min(clean_counts.values(), default=0)
    mean_clean = sum(clean_counts.values()) / max(len(clean_counts), 1)
    mean_raw = sum(raw_counts[edge] for edge in edges) / max(len(edges), 1)
    # Constraint violations dominate count/quality terms, while the softer
    # degree-four penalty expresses the requested preference for degree 2--3.
    score = (
        -10_000_000.0 * over_degree
        -2_000_000.0 * low_degree
        -20_000.0 * deficits
        -5_000.0 * degree_four
        +1_000.0 * minimum
        +10.0 * mean_clean
        +math.log1p(mean_raw)
    )
    feasible = over_degree == 0 and low_degree == 0 and deficits == 0
    return score, clean_counts, degrees, feasible


def random_initial_state(edges: list[Pair], num_classes: int, rng: random.Random) -> tuple[Pair, ...]:
    # Weight toward abundant edges without making initialization deterministic.
    sample_pool = edges[: min(len(edges), max(num_classes * 5, num_classes))]
    if len(sample_pool) < num_classes:
        raise ValueError(f"only {len(sample_pool)} pair candidates remain for {num_classes} classes")
    return tuple(sorted(rng.sample(sample_pool, num_classes)))


def search_graph(
    raw_counts: dict[Pair, int],
    pair_bits: dict[Pair, int],
    args: argparse.Namespace,
) -> SearchResult:
    required = int(args.train_per_class) + int(args.val_per_class) + int(args.test_per_class)
    target_clean = required + int(args.candidate_reserve)
    candidate_edges = [edge for edge, count in raw_counts.items() if count >= required]
    candidate_edges.sort(key=lambda edge: (-raw_counts[edge], edge))
    if len(candidate_edges) < int(args.num_classes):
        raise ValueError(
            f"only {len(candidate_edges)} raw pairs have at least {required} candidates; "
            "relax area/rank thresholds"
        )

    rng = random.Random(int(args.seed))
    selected_set: set[Pair]
    current = random_initial_state(candidate_edges, int(args.num_classes), rng)
    current_eval = state_score(
        current,
        raw_counts,
        pair_bits,
        target_clean,
        int(args.min_object_degree),
        int(args.preferred_max_object_degree),
        int(args.max_object_degree),
    )
    best = current
    best_eval = current_eval
    cache: dict[tuple[Pair, ...], tuple[float, dict[Pair, int], dict[str, int], bool]] = {current: current_eval}

    iterations = int(args.search_iterations)
    restarts = max(1, int(args.search_restarts))
    steps_per_restart = max(1, iterations // restarts)
    for restart in range(restarts):
        if restart:
            current = random_initial_state(candidate_edges, int(args.num_classes), rng)
            current_eval = cache.get(current) or state_score(
                current,
                raw_counts,
                pair_bits,
                target_clean,
                int(args.min_object_degree),
                int(args.preferred_max_object_degree),
                int(args.max_object_degree),
            )
            cache[current] = current_eval
        for step in range(steps_per_restart):
            selected_set = set(current)
            removed = rng.choice(current)
            replacement = rng.choice(candidate_edges)
            if replacement in selected_set:
                continue
            proposal = tuple(sorted((selected_set - {removed}) | {replacement}))
            proposal_eval = cache.get(proposal)
            if proposal_eval is None:
                proposal_eval = state_score(
                    proposal,
                    raw_counts,
                    pair_bits,
                    target_clean,
                    int(args.min_object_degree),
                    int(args.preferred_max_object_degree),
                    int(args.max_object_degree),
                )
                if len(cache) < int(args.search_cache_size):
                    cache[proposal] = proposal_eval
            temperature = max(1.0, 250_000.0 * (1.0 - step / steps_per_restart))
            delta = proposal_eval[0] - current_eval[0]
            if delta >= 0 or rng.random() < math.exp(max(-50.0, delta / temperature)):
                current, current_eval = proposal, proposal_eval
            if proposal_eval[0] > best_eval[0]:
                best, best_eval = proposal, proposal_eval
            if best_eval[3] and min(best_eval[1].values()) >= target_clean:
                # Keep searching briefly is unnecessary: the objective has
                # already enforced the hard constraints and reserve target.
                break
        if best_eval[3]:
            break

    return SearchResult(
        edges=best,
        clean_counts=best_eval[1],
        raw_counts={edge: raw_counts[edge] for edge in best},
        degrees=best_eval[2],
        score=best_eval[0],
        feasible=best_eval[3],
    )


def load_fixed_edges(path: Path) -> tuple[Pair, ...]:
    data = read_json(path)
    rows = data.get("pairs", data) if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ValueError("pairs JSON must be a list or an object containing a 'pairs' list")
    edges = []
    for row in rows:
        if isinstance(row, dict):
            a = row.get("object_a", row.get("anchor"))
            b = row.get("object_b", row.get("evidence"))
        else:
            a, b = row
        if not a or not b:
            raise ValueError(f"invalid pair entry: {row!r}")
        edges.append(canonical_pair(str(a), str(b)))
    if len(set(edges)) != len(edges):
        raise ValueError("pairs JSON contains duplicate edges")
    return tuple(sorted(edges))


def orient_edges(
    edges: tuple[Pair, ...],
    records: list[ImageRecord],
    pair_bits: dict[Pair, int],
) -> dict[Pair, tuple[str, str]]:
    once = exactly_once_bits(edges, pair_bits)
    oriented: dict[Pair, tuple[str, str]] = {}
    for edge in edges:
        clean_bits = pair_bits[edge] & once
        area_sums = {edge[0]: 0.0, edge[1]: 0.0}
        count = 0
        bits = clean_bits
        while bits:
            low = bits & -bits
            index = low.bit_length() - 1
            record = records[index]
            area_sums[edge[0]] += record.ratios[edge[0]]
            area_sums[edge[1]] += record.ratios[edge[1]]
            count += 1
            bits ^= low
        if count and area_sums[edge[1]] > area_sums[edge[0]]:
            oriented[edge] = (edge[1], edge[0])
        else:
            oriented[edge] = edge
    return oriented


def rows_for_graph(
    records: list[ImageRecord],
    edges: tuple[Pair, ...],
    pair_bits: dict[Pair, int],
    orientations: dict[Pair, tuple[str, str]],
) -> list[dict[str, str]]:
    once = exactly_once_bits(edges, pair_bits)
    edge_by_index: dict[int, Pair] = {}
    for edge in edges:
        bits = pair_bits[edge] & once
        while bits:
            low = bits & -bits
            index = low.bit_length() - 1
            edge_by_index[index] = edge
            bits ^= low

    rows: list[dict[str, str]] = []
    for index, edge in sorted(edge_by_index.items()):
        record = records[index]
        rank_by_object = {name: rank for rank, (name, _ratio) in enumerate(record.ranked, start=1)}
        a, b = edge
        anchor, evidence = orientations[edge]
        top1 = record.ranked[0]
        top2 = record.ranked[1] if len(record.ranked) > 1 else ("", 0.0)
        rows.append(
            {
                "class_name": class_name(edge),
                "source_split": record.source_split,
                "image_id": str(record.image_id),
                "file_name": record.file_name,
                "width": str(record.width),
                "height": str(record.height),
                "object_a": a,
                "object_b": b,
                "object_a_area_ratio": f"{record.ratios[a]:.6f}",
                "object_b_area_ratio": f"{record.ratios[b]:.6f}",
                "object_a_instance_count": str(record.counts[a]),
                "object_b_instance_count": str(record.counts[b]),
                "object_a_rank": str(rank_by_object[a]),
                "object_b_rank": str(rank_by_object[b]),
                "anchor_object": anchor,
                "evidence_object": evidence,
                "anchor_area_ratio": f"{record.ratios[anchor]:.6f}",
                "evidence_area_ratio": f"{record.ratios[evidence]:.6f}",
                "top1_object": top1[0],
                "top2_object": top2[0],
                "top1_area_ratio": f"{top1[1]:.6f}",
                "top2_area_ratio": f"{top2[1]:.6f}",
                "top_objects": ";".join(name for name, _ratio in record.ranked[:5]),
                "top_area_ratios": ";".join(f"{name}:{ratio:.6f}" for name, ratio in record.ranked[:5]),
            }
        )
    return rows


def candidate_strength(row: dict[str, str]) -> tuple[float, float, float, int]:
    a = float(row["object_a_area_ratio"])
    b = float(row["object_b_area_ratio"])
    return (min(a, b), a + b, max(a, b), int(row["image_id"]))


def balanced_split(
    candidates: list[dict[str, str]],
    classes: list[str],
    class_to_idx: dict[str, int],
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    required_by_split = {
        "train": int(args.train_per_class),
        "val": int(args.val_per_class),
        "test": int(args.test_per_class),
    }
    required = sum(required_by_split.values())
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        by_class[row["class_name"]].append(row)
    missing = {name: len(by_class[name]) for name in classes if len(by_class[name]) < required}
    if missing:
        raise ValueError(f"not enough clean candidates for total={required}: {missing}")

    rng = random.Random(int(args.seed))
    balanced: list[dict[str, str]] = []
    pool_size = required + int(args.candidate_reserve)
    for name in classes:
        rows = [dict(row) for row in by_class[name]]
        # Retain a high-quality pool, then randomize before splitting so that
        # train/val/test do not correspond to different area quantiles.
        rng.shuffle(rows)
        rows.sort(key=candidate_strength, reverse=True)
        rows = rows[: min(len(rows), pool_size)]
        rng.shuffle(rows)
        chosen = rows[:required]
        offset = 0
        for split, count in required_by_split.items():
            for row in chosen[offset : offset + count]:
                row["split"] = split
                row["class_idx"] = str(class_to_idx[name])
                balanced.append(row)
            offset += count
    return balanced


def write_classification_dataset(
    root: Path,
    coco_root: Path,
    out_dir: Path,
    classes: list[str],
    class_to_idx: dict[str, int],
    rows: list[dict[str, str]],
    mode: str,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    (root / "class_to_idx.json").write_text(json.dumps(class_to_idx, indent=2) + "\n", encoding="utf-8")
    (root / "dataset_info.json").write_text(
        json.dumps({"classes": classes, "class_to_idx": class_to_idx}, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata: list[dict[str, str]] = []
    for row in rows:
        src = (coco_root / f"{row['source_split']}2017" / row["file_name"]).resolve()
        if not src.exists():
            raise FileNotFoundError(f"COCO image not found: {src}")
        dst_name = f"{row['source_split']}_{int(row['image_id']):012d}_{row['file_name']}"
        dst = root / row["split"] / row["class_name"] / dst_name
        materialize_image(src, dst, mode)
        meta = dict(row)
        meta["relative_path"] = str(dst.relative_to(root))
        metadata.append(meta)
    write_rows(root / "metadata.csv", metadata, FIELDNAMES + ["relative_path"])
    shutil.copy2(out_dir / "summary.json", root / "summary.json")


def validate_args(args: argparse.Namespace) -> None:
    if args.num_classes <= 0:
        raise ValueError("--num_classes must be positive")
    if args.min_object_degree < 2:
        raise ValueError("--min_object_degree must be at least 2 to prevent singleton shortcuts")
    if not args.min_object_degree <= args.preferred_max_object_degree <= args.max_object_degree:
        raise ValueError("object degree constraints must satisfy min <= preferred_max <= max")
    if args.max_object_rank < 2:
        raise ValueError("--max_object_rank must be at least 2")
    required = args.train_per_class + args.val_per_class + args.test_per_class
    if required <= 0:
        raise ValueError("requested split sizes must have a positive total")
    if args.candidate_reserve < 0:
        raise ValueError("--candidate_reserve cannot be negative")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco_root", default=str(REPO_ROOT / "dataset/coco2017"))
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--pairs_json", default="", help="Optional fixed graph; skip automatic graph search.")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--train_per_class", type=int, default=300)
    parser.add_argument("--val_per_class", type=int, default=100)
    parser.add_argument("--test_per_class", type=int, default=100)
    parser.add_argument("--candidate_reserve", type=int, default=100)
    parser.add_argument("--min_object_area_ratio", type=float, default=0.03)
    parser.add_argument("--min_larger_object_area_ratio", type=float, default=0.06)
    parser.add_argument("--min_pair_area_ratio", type=float, default=0.12)
    parser.add_argument("--max_object_rank", type=int, default=3)
    parser.add_argument("--min_object_degree", type=int, default=2)
    parser.add_argument("--preferred_max_object_degree", type=int, default=3)
    parser.add_argument("--max_object_degree", type=int, default=4)
    parser.add_argument("--exclude_objects", default="", help="Comma-separated COCO category names.")
    parser.add_argument("--search_iterations", type=int, default=120000)
    parser.add_argument("--search_restarts", type=int, default=24)
    parser.add_argument("--search_cache_size", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--materialize", choices=["none", "symlink", "copy"], default="none")
    parser.add_argument("--allow_infeasible_graph", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    validate_args(args)
    return args


def main() -> None:
    args = parse_args()
    coco_root = Path(args.coco_root)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output directory is not empty: {out_dir}; pass --overwrite to replace outputs")

    records, category_names = load_coco_records(coco_root)
    raw_counts, pair_bits = build_pair_index(records, args)
    if args.pairs_json:
        edges = load_fixed_edges(Path(args.pairs_json))
        missing_edges = [edge for edge in edges if edge not in pair_bits]
        if missing_edges:
            raise ValueError(f"fixed pairs have no qualifying candidates: {missing_edges}")
        clean_counts = clean_counts_for_edges(edges, pair_bits)
        degrees = selected_degrees(edges)
        required = args.train_per_class + args.val_per_class + args.test_per_class
        target_clean = required + args.candidate_reserve
        feasible = (
            len(edges) == args.num_classes
            and min(degrees.values()) >= args.min_object_degree
            and max(degrees.values()) <= args.max_object_degree
            and min(clean_counts.values()) >= target_clean
        )
        result = SearchResult(
            edges=edges,
            clean_counts=clean_counts,
            raw_counts={edge: raw_counts.get(edge, 0) for edge in edges},
            degrees=degrees,
            score=0.0,
            feasible=feasible,
        )
    else:
        result = search_graph(raw_counts, pair_bits, args)

    if not result.feasible and not args.allow_infeasible_graph:
        diagnostic = {
            "message": "no graph met all constraints; relax reserve/area/rank constraints or increase search iterations",
            "best_edges": [list(edge) for edge in result.edges],
            "clean_counts": {" + ".join(edge): result.clean_counts[edge] for edge in result.edges},
            "raw_counts": {" + ".join(edge): result.raw_counts[edge] for edge in result.edges},
            "degrees": result.degrees,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "infeasible_search_report.json").write_text(
            json.dumps(diagnostic, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(diagnostic, indent=2))
        raise RuntimeError(diagnostic["message"])

    orientations = orient_edges(result.edges, records, pair_bits)
    candidates = rows_for_graph(records, result.edges, pair_bits, orientations)
    classes = [class_name(edge) for edge in result.edges]
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    balanced = balanced_split(candidates, classes, class_to_idx, args)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "all_candidates.csv", candidates, [name for name in FIELDNAMES if name not in {"split", "class_idx"}])
    write_rows(out_dir / "balanced_samples.csv", balanced, FIELDNAMES)
    graph_json = {
        "pairs": [
            {
                "class_name": class_name(edge),
                "object_a": edge[0],
                "object_b": edge[1],
                "anchor": orientations[edge][0],
                "evidence": orientations[edge][1],
                "raw_candidates": result.raw_counts[edge],
                "clean_candidates": result.clean_counts[edge],
            }
            for edge in result.edges
        ],
        "object_degrees": result.degrees,
    }
    (out_dir / "selected_graph.json").write_text(json.dumps(graph_json, indent=2) + "\n", encoding="utf-8")

    summary = {
        "source": "COCO train2017 + val2017 merged",
        "label_definition": "compositional object-pair classes with reused objects; clean images match exactly one selected edge",
        "num_source_images": len(records),
        "num_coco_categories": len(category_names),
        "num_classes": len(classes),
        "train_per_class": int(args.train_per_class),
        "val_per_class": int(args.val_per_class),
        "test_per_class": int(args.test_per_class),
        "candidate_reserve": int(args.candidate_reserve),
        "filters": {
            "min_object_area_ratio": float(args.min_object_area_ratio),
            "min_larger_object_area_ratio": float(args.min_larger_object_area_ratio),
            "min_pair_area_ratio": float(args.min_pair_area_ratio),
            "max_object_rank": int(args.max_object_rank),
            "excluded_objects": sorted(parse_object_list(args.exclude_objects)),
        },
        "degree_constraints": {
            "minimum": int(args.min_object_degree),
            "preferred_maximum": int(args.preferred_max_object_degree),
            "hard_maximum": int(args.max_object_degree),
        },
        "graph_feasible": result.feasible,
        "graph_score": result.score,
        "object_degrees": result.degrees,
        "classes": classes,
        "class_to_idx": class_to_idx,
        "pairs": graph_json["pairs"],
        "balanced_counts": {
            split: dict(Counter(row["class_name"] for row in balanced if row["split"] == split))
            for split in ("train", "val", "test")
        },
        "mean_selected_area_ratio": {
            name: {
                "object_a": sum(float(row["object_a_area_ratio"]) for row in balanced if row["class_name"] == name)
                / max(1, sum(row["class_name"] == name for row in balanced)),
                "object_b": sum(float(row["object_b_area_ratio"]) for row in balanced if row["class_name"] == name)
                / max(1, sum(row["class_name"] == name for row in balanced)),
            }
            for name in classes
        },
        "materialize": args.materialize,
        "construction_only_annotations": "COCO object category, instance area, and rank; classification training need not read metadata",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if args.materialize != "none":
        write_classification_dataset(
            out_dir / "classification_dataset",
            coco_root,
            out_dir,
            classes,
            class_to_idx,
            balanced,
            args.materialize,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
