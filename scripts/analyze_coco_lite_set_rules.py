#!/usr/bin/env python3
"""Audit COCO for balanced 2x2 set-valued anchor/evidence rules.

The audit treats the two largest annotated object categories in an image as
the only class-defining objects.  One must belong to a two-category anchor
set and the other to a two-category evidence set.  Areas are aggregated over
all non-crowd instances of a category before thresholds are applied.

This script does not materialize a dataset.  It reports individual feasible
2x2 rules and searches for a complete 2-by-N grid of disjoint category
groups, which is the direct 10-class extension when N=5.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_directed_pair_counts(
    coco_root: Path,
    min_anchor_area_ratio: float,
    min_evidence_area_ratio: float,
) -> tuple[dict[tuple[str, str], int], list[str], int]:
    """Count top-2 pairs, oriented by which member passes anchor threshold."""
    counts: Counter[tuple[str, str]] = Counter()
    category_names: set[str] = set()
    eligible_images = 0
    for split in ("train", "val"):
        with (coco_root / "annotations" / f"instances_{split}2017.json").open(encoding="utf-8") as f:
            data = json.load(f)
        categories = {int(row["id"]): str(row["name"]) for row in data["categories"]}
        category_names.update(categories.values())
        images = {int(row["id"]): row for row in data["images"]}
        area_by_image: dict[int, Counter[str]] = defaultdict(Counter)
        for ann in data["annotations"]:
            if ann.get("iscrowd", 0):
                continue
            area_by_image[int(ann["image_id"])][categories[int(ann["category_id"])]] += float(ann.get("area", 0.0))

        for image_id, areas in area_by_image.items():
            info = images[image_id]
            image_area = float(info["width"]) * float(info["height"])
            if image_area <= 0:
                continue
            ranked = sorted(
                ((name, area / image_area) for name, area in areas.items()),
                key=lambda item: (-item[1], item[0]),
            )
            if len(ranked) < 2 or ranked[1][1] <= min_evidence_area_ratio:
                continue
            eligible_images += 1
            (first, first_area), (second, second_area) = ranked[:2]
            if first_area > min_anchor_area_ratio and second_area > min_evidence_area_ratio:
                counts[(first, second)] += 1
            if second_area > min_anchor_area_ratio and first_area > min_evidence_area_ratio:
                counts[(second, first)] += 1
    return dict(counts), sorted(category_names), eligible_images


def balanced_capacity(
    values: list[int],
    variant_min_fraction: float,
    variant_max_fraction: float,
) -> int:
    """Largest total satisfying both per-variant fraction bounds."""
    if not values:
        return 0
    total = sum(values)
    if variant_min_fraction > 0:
        total = min(total, int(min(values) / variant_min_fraction))
    # F(T) is monotone. Starting above the optimum and repeatedly replacing T
    # by F(T) cannot skip a feasible value, and avoids a non-monotone binary
    # search caused by floor(max_fraction * T).
    while total > 0:
        capped = sum(min(value, int(variant_max_fraction * total)) for value in values)
        if capped >= total:
            return total
        total = capped
    return 0


def enumerate_rules(
    counts: dict[tuple[str, str], int],
    categories: list[str],
    required: int,
    min_variant_fraction: float,
    max_variant_fraction: float,
) -> list[dict]:
    variant_floor = int(required * min_variant_fraction + 0.999999)
    rules: list[dict] = []
    for a1, a2 in combinations(categories, 2):
        shared_evidence = [
            evidence
            for evidence in categories
            if evidence not in {a1, a2}
            and counts.get((a1, evidence), 0) >= variant_floor
            and counts.get((a2, evidence), 0) >= variant_floor
        ]
        for e1, e2 in combinations(shared_evidence, 2):
            values = [
                counts[(a1, e1)], counts[(a1, e2)],
                counts[(a2, e1)], counts[(a2, e2)],
            ]
            capacity = balanced_capacity(values, min_variant_fraction, max_variant_fraction)
            if capacity < required:
                continue
            rules.append(
                {
                    "anchors": [a1, a2],
                    "evidence": [e1, e2],
                    "variant_counts": {
                        f"{a1} + {e1}": values[0],
                        f"{a1} + {e2}": values[1],
                        f"{a2} + {e1}": values[2],
                        f"{a2} + {e2}": values[3],
                    },
                    "raw_total": sum(values),
                    "balanced_capacity": capacity,
                    "minimum_variant_count": min(values),
                }
            )
    rules.sort(
        key=lambda row: (
            row["balanced_capacity"],
            row["minimum_variant_count"],
            row["raw_total"],
        ),
        reverse=True,
    )
    return rules


def grid_score(
    anchor_groups: list[tuple[str, str]],
    evidence_groups: list[tuple[str, str]],
    rule_lookup: dict[tuple[tuple[str, str], tuple[str, str]], dict],
    required: int,
) -> tuple[int, int, int]:
    capacities = [
        rule_lookup.get((anchors, evidence), {}).get("balanced_capacity", 0)
        for anchors in anchor_groups
        for evidence in evidence_groups
    ]
    return (
        sum(capacity >= required for capacity in capacities),
        min(capacities, default=0),
        sum(min(capacity, required) for capacity in capacities),
    )


def maximum_disjoint_pairs(
    pairs: set[tuple[str, str]],
    forbidden: set[str],
    target: int,
) -> list[tuple[str, str]]:
    """Return an exact maximum category-disjoint subset, capped at target."""
    usable = {pair for pair in pairs if not (set(pair) & forbidden)}
    vertices = sorted({name for pair in usable for name in pair})
    index = {name: idx for idx, name in enumerate(vertices)}
    neighbors = [0] * len(vertices)
    for left, right in usable:
        i, j = index[left], index[right]
        neighbors[i] |= 1 << j
        neighbors[j] |= 1 << i
    memo: dict[int, tuple[tuple[str, str], ...]] = {}

    def solve(mask: int) -> tuple[tuple[str, str], ...]:
        if not mask:
            return ()
        cached = memo.get(mask)
        if cached is not None:
            return cached
        low = mask & -mask
        i = low.bit_length() - 1
        remaining = mask ^ low
        best: tuple[tuple[str, str], ...] = ()
        choices = neighbors[i] & remaining
        # Try matched branches first: dense feasible graphs then reach the
        # requested target without exploring the exponentially large skip tree.
        while choices:
            other_low = choices & -choices
            j = other_low.bit_length() - 1
            tail = solve(remaining ^ other_low)
            candidate = ((vertices[i], vertices[j]),) + tail
            if len(candidate) > len(best):
                best = candidate[:target]
                if len(best) >= target:
                    memo[mask] = best
                    return best
            choices ^= other_low
        skipped = solve(remaining)
        if len(skipped) > len(best):
            best = skipped
        memo[mask] = best
        return best

    return list(solve((1 << len(vertices)) - 1))


def exact_two_by_n_grid(
    rules: list[dict],
    evidence_group_count: int,
    required: int,
) -> dict:
    """Exactly test a 2-by-N grid by intersecting group neighborhoods."""
    neighbors: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    lookup: dict[tuple[tuple[str, str], tuple[str, str]], dict] = {}
    for row in rules:
        anchors = tuple(row["anchors"])
        evidence = tuple(row["evidence"])
        neighbors[anchors].add(evidence)
        lookup[(anchors, evidence)] = row

    anchor_groups = sorted(neighbors)
    best_anchors: tuple[tuple[str, str], tuple[str, str]] | None = None
    best_evidence: list[tuple[str, str]] = []
    tested = 0
    for first, second in combinations(anchor_groups, 2):
        if set(first) & set(second):
            continue
        tested += 1
        common = neighbors[first] & neighbors[second]
        selected = maximum_disjoint_pairs(common, set(first) | set(second), evidence_group_count)
        if len(selected) > len(best_evidence):
            best_anchors = (first, second)
            best_evidence = selected
        if len(selected) >= evidence_group_count:
            break

    if best_anchors is None:
        return {"feasible": False, "reason": "no two disjoint anchor groups"}
    cells = []
    for anchors in best_anchors:
        for evidence in best_evidence:
            row = lookup[(anchors, tuple(sorted(evidence)))]
            cells.append(
                {
                    "anchors": list(anchors),
                    "evidence": list(evidence),
                    "balanced_capacity": row["balanced_capacity"],
                    "variant_counts": row["variant_counts"],
                }
            )
    return {
        "feasible": len(best_evidence) >= evidence_group_count,
        "feasible_cells": 2 * len(best_evidence),
        "total_cells": 2 * evidence_group_count,
        "anchor_groups": [list(group) for group in best_anchors],
        "evidence_groups": [list(group) for group in best_evidence],
        "cells": cells,
        "anchor_group_pairs_tested": tested,
        "search_is_exact": True,
    }


def exact_n_by_two_grid(
    rules: list[dict],
    anchor_group_count: int,
) -> dict:
    """Exactly test an N-by-2 grid by intersecting reverse neighborhoods."""
    neighbors: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    lookup: dict[tuple[tuple[str, str], tuple[str, str]], dict] = {}
    for row in rules:
        anchors = tuple(row["anchors"])
        evidence = tuple(row["evidence"])
        neighbors[evidence].add(anchors)
        lookup[(anchors, evidence)] = row

    evidence_groups = sorted(neighbors)
    best_evidence: tuple[tuple[str, str], tuple[str, str]] | None = None
    best_anchors: list[tuple[str, str]] = []
    tested = 0
    for first, second in combinations(evidence_groups, 2):
        if set(first) & set(second):
            continue
        tested += 1
        common = neighbors[first] & neighbors[second]
        selected = maximum_disjoint_pairs(common, set(first) | set(second), anchor_group_count)
        if len(selected) > len(best_anchors):
            best_evidence = (first, second)
            best_anchors = selected
        if len(selected) >= anchor_group_count:
            break

    if best_evidence is None:
        return {"feasible": False, "reason": "no two disjoint evidence groups"}
    cells = []
    for anchors in best_anchors:
        for evidence in best_evidence:
            row = lookup[(tuple(sorted(anchors)), evidence)]
            cells.append(
                {
                    "anchors": list(anchors),
                    "evidence": list(evidence),
                    "balanced_capacity": row["balanced_capacity"],
                    "variant_counts": row["variant_counts"],
                }
            )
    return {
        "feasible": len(best_anchors) >= anchor_group_count,
        "feasible_cells": 2 * len(best_anchors),
        "total_cells": 2 * anchor_group_count,
        "anchor_groups": [list(group) for group in best_anchors],
        "evidence_groups": [list(group) for group in best_evidence],
        "cells": cells,
        "evidence_group_pairs_tested": tested,
        "search_is_exact": True,
    }


def search_grid(
    rules: list[dict],
    categories: list[str],
    anchor_group_count: int,
    evidence_group_count: int,
    required: int,
    iterations: int,
    seed: int,
) -> dict:
    if anchor_group_count == 2:
        return exact_two_by_n_grid(rules, evidence_group_count, required)
    if evidence_group_count == 2:
        return exact_n_by_two_grid(rules, anchor_group_count)
    rule_lookup = {
        (tuple(row["anchors"]), tuple(row["evidence"])): row
        for row in rules
    }
    rng = random.Random(seed)
    total_groups = anchor_group_count + evidence_group_count
    needed_categories = total_groups * 2
    if needed_categories > len(categories):
        raise ValueError("grid needs more distinct categories than COCO provides")

    # Restrict random proposals to categories that occur in at least one
    # feasible rule in the corresponding role.
    anchor_pool = sorted({name for row in rules for name in row["anchors"]})
    evidence_pool = sorted({name for row in rules for name in row["evidence"]})
    if len(anchor_pool) < anchor_group_count * 2 or len(evidence_pool) < evidence_group_count * 2:
        return {"feasible": False, "reason": "too few categories occur in feasible individual rules"}

    best_state = None
    best_score = (-1, -1, -1)
    for _ in range(iterations):
        anchors_flat = rng.sample(anchor_pool, anchor_group_count * 2)
        used = set(anchors_flat)
        available_evidence = [name for name in evidence_pool if name not in used]
        if len(available_evidence) < evidence_group_count * 2:
            continue
        evidence_flat = rng.sample(available_evidence, evidence_group_count * 2)
        anchor_groups = [tuple(sorted(anchors_flat[i:i + 2])) for i in range(0, len(anchors_flat), 2)]
        evidence_groups = [tuple(sorted(evidence_flat[i:i + 2])) for i in range(0, len(evidence_flat), 2)]
        score = grid_score(anchor_groups, evidence_groups, rule_lookup, required)
        if score > best_score:
            best_score = score
            best_state = (anchor_groups, evidence_groups)
    if best_state is None:
        return {"feasible": False, "reason": "search produced no disjoint state"}
    anchor_groups, evidence_groups = best_state
    cells = []
    for anchors in anchor_groups:
        for evidence in evidence_groups:
            rule = rule_lookup.get((anchors, evidence))
            cells.append(
                {
                    "anchors": list(anchors),
                    "evidence": list(evidence),
                    "balanced_capacity": rule["balanced_capacity"] if rule else 0,
                    "variant_counts": rule["variant_counts"] if rule else {},
                }
            )
    return {
        "feasible": best_score[0] == anchor_group_count * evidence_group_count,
        "feasible_cells": best_score[0],
        "total_cells": anchor_group_count * evidence_group_count,
        "minimum_cell_capacity": best_score[1],
        "anchor_groups": [list(group) for group in anchor_groups],
        "evidence_groups": [list(group) for group in evidence_groups],
        "cells": cells,
        "random_search_iterations": iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco_root", default=str(REPO_ROOT / "dataset/coco2017"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--min_anchor_area_ratio", type=float, default=0.09)
    parser.add_argument("--min_evidence_area_ratio", type=float, default=0.05)
    parser.add_argument("--required_per_class", type=int, default=500)
    parser.add_argument("--min_variant_fraction", type=float, default=0.15)
    parser.add_argument("--max_variant_fraction", type=float, default=0.35)
    parser.add_argument("--anchor_group_count", type=int, default=2)
    parser.add_argument("--evidence_group_count", type=int, default=5)
    parser.add_argument("--grid_search_iterations", type=int, default=500_000)
    parser.add_argument("--top_rules", type=int, default=100)
    parser.add_argument("--seed", type=int, default=8)
    args = parser.parse_args()

    counts, categories, eligible_images = load_directed_pair_counts(
        Path(args.coco_root),
        float(args.min_anchor_area_ratio),
        float(args.min_evidence_area_ratio),
    )
    rules = enumerate_rules(
        counts,
        categories,
        int(args.required_per_class),
        float(args.min_variant_fraction),
        float(args.max_variant_fraction),
    )
    grid = search_grid(
        rules,
        categories,
        int(args.anchor_group_count),
        int(args.evidence_group_count),
        int(args.required_per_class),
        int(args.grid_search_iterations),
        int(args.seed),
    )
    report = {
        "source": "COCO train2017 + val2017 merged",
        "sample_rule": "top-2 annotated categories only; aggregated non-crowd instance area",
        "min_anchor_area_ratio": float(args.min_anchor_area_ratio),
        "min_evidence_area_ratio": float(args.min_evidence_area_ratio),
        "required_per_class": int(args.required_per_class),
        "variant_fraction_range": [float(args.min_variant_fraction), float(args.max_variant_fraction)],
        "eligible_top2_images": eligible_images,
        "feasible_individual_rule_count": len(rules),
        "top_individual_rules": rules[: int(args.top_rules)],
        "grid_search": grid,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "top_individual_rules"}, indent=2))


if __name__ == "__main__":
    main()
