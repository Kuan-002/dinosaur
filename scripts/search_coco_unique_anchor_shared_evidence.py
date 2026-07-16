#!/usr/bin/env python3
"""Search COCO for 10 unique anchors paired with evidence reused 2--3 times."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_coco_compositional_pair10_dataset import load_coco_records


Edge = tuple[str, str]  # (anchor, evidence)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco_root", default=str(ROOT / "dataset" / "coco2017"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--required_per_class", type=int, default=1000)
    parser.add_argument("--candidate_reserve", type=int, default=50)
    parser.add_argument("--min_anchor_area_ratio", type=float, default=0.02)
    parser.add_argument("--min_evidence_area_ratio", type=float, default=0.015)
    parser.add_argument("--max_object_rank", type=int, default=4)
    parser.add_argument("--min_evidence_degree", type=int, default=2)
    parser.add_argument("--max_evidence_degree", type=int, default=3)
    parser.add_argument("--evidence_pool_size", type=int, default=24)
    parser.add_argument("--restarts", type=int, default=80)
    parser.add_argument("--steps_per_restart", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--exclude_objects", default="")
    parser.add_argument("--allow_cross_role", action="store_true")
    return parser.parse_args()


def build_edge_index(records, args: argparse.Namespace):
    raw: Counter[Edge] = Counter()
    bits: dict[Edge, int] = defaultdict(int)
    area_sums: Counter[Edge] = Counter()
    excluded = {item.strip() for item in args.exclude_objects.split(",") if item.strip()}
    for index, record in enumerate(records):
        ranked = [
            (name, ratio)
            for rank, (name, ratio) in enumerate(record.ranked, start=1)
            if rank <= args.max_object_rank and name not in excluded
        ]
        anchors = [(name, ratio) for name, ratio in ranked if ratio >= args.min_anchor_area_ratio]
        evidences = [(name, ratio) for name, ratio in ranked if ratio >= args.min_evidence_area_ratio]
        image_bit = 1 << index
        for anchor, anchor_area in anchors:
            for evidence, evidence_area in evidences:
                if anchor == evidence:
                    continue
                edge = (anchor, evidence)
                raw[edge] += 1
                bits[edge] |= image_bit
                area_sums[edge] += min(anchor_area, evidence_area)
    mean_min_area = {edge: area_sums[edge] / raw[edge] for edge in raw}
    return dict(raw), dict(bits), mean_min_area


def clean_counts(edges: tuple[Edge, ...], edge_bits: dict[Edge, int]) -> dict[Edge, int]:
    once = 0
    multiple = 0
    for edge in edges:
        present = edge_bits[edge]
        multiple |= once & present
        once = (once ^ present) & ~multiple
    return {edge: (edge_bits[edge] & once).bit_count() for edge in edges}


def degree_patterns(num_classes: int, minimum: int, maximum: int) -> list[tuple[int, ...]]:
    out = []
    for groups in range(math.ceil(num_classes / maximum), num_classes // minimum + 1):
        def visit(prefix: tuple[int, ...], remaining: int) -> None:
            if len(prefix) == groups:
                if remaining == 0:
                    out.append(prefix)
                return
            for degree in range(minimum, maximum + 1):
                if degree <= remaining:
                    visit(prefix + (degree,), remaining - degree)
        visit((), num_classes)
    return sorted(set(out))


def evaluate(edges: tuple[Edge, ...], edge_bits: dict[Edge, int], target: int):
    counts = clean_counts(edges, edge_bits)
    values = list(counts.values())
    deficit = sum(max(0, target - value) for value in values)
    minimum = min(values)
    mean = sum(values) / len(values)
    spread = max(values) - minimum
    score = -1_000_000 * deficit + 10_000 * minimum + 10 * mean - spread
    return score, counts


def weighted_choice(rng: random.Random, rows: list[tuple[str, int]]) -> str:
    weights = [max(1.0, count**1.5) for _, count in rows]
    return rng.choices([name for name, _ in rows], weights=weights, k=1)[0]


def initial_state(
    rng: random.Random,
    evidence_pool: list[str],
    patterns: list[tuple[int, ...]],
    anchors_by_evidence: dict[str, list[tuple[str, int]]],
    allow_cross_role: bool,
) -> tuple[Edge, ...] | None:
    pattern = rng.choice(patterns)
    evidences = rng.sample(evidence_pool, len(pattern))
    rng.shuffle(evidences)
    used_anchors: set[str] = set() if allow_cross_role else set(evidences)
    edges = []
    for evidence, degree in zip(evidences, pattern):
        candidates = [(anchor, count) for anchor, count in anchors_by_evidence[evidence] if anchor not in used_anchors]
        if len(candidates) < degree:
            return None
        for _ in range(degree):
            anchor = weighted_choice(rng, candidates)
            edges.append((anchor, evidence))
            used_anchors.add(anchor)
            candidates = [(name, count) for name, count in candidates if name != anchor]
    return tuple(sorted(edges))


def search(raw, edge_bits, args: argparse.Namespace):
    target = args.required_per_class + args.candidate_reserve
    eligible = {edge: count for edge, count in raw.items() if count >= target}
    anchors_by_evidence: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (anchor, evidence), count in eligible.items():
        anchors_by_evidence[evidence].append((anchor, count))
    for evidence in anchors_by_evidence:
        anchors_by_evidence[evidence].sort(key=lambda item: (-item[1], item[0]))
    evidence_pool = sorted(
        (
            evidence
            for evidence, candidates in anchors_by_evidence.items()
            if len(candidates) >= args.min_evidence_degree
        ),
        key=lambda evidence: (-len(anchors_by_evidence[evidence]), -sum(v for _, v in anchors_by_evidence[evidence]), evidence),
    )[: args.evidence_pool_size]
    patterns = degree_patterns(args.num_classes, args.min_evidence_degree, args.max_evidence_degree)
    rng = random.Random(args.seed)
    best_edges = None
    best_eval = (-float("inf"), {})
    for _restart in range(args.restarts):
        current = initial_state(rng, evidence_pool, patterns, anchors_by_evidence, args.allow_cross_role)
        if current is None:
            continue
        current_eval = evaluate(current, edge_bits, target)
        if current_eval[0] > best_eval[0]:
            best_edges, best_eval = current, current_eval
        for step in range(args.steps_per_restart):
            index = rng.randrange(len(current))
            old_anchor, evidence = current[index]
            used_anchors = {edge[0] for edge in current}
            evidences = {edge[1] for edge in current}
            candidates = [
                (anchor, count)
                for anchor, count in anchors_by_evidence[evidence]
                if (args.allow_cross_role or anchor not in evidences)
                and (anchor == old_anchor or anchor not in used_anchors)
            ]
            if len(candidates) < 2:
                continue
            new_anchor = weighted_choice(rng, candidates)
            if new_anchor == old_anchor:
                continue
            proposal = list(current)
            proposal[index] = (new_anchor, evidence)
            proposal_t = tuple(sorted(proposal))
            proposal_eval = evaluate(proposal_t, edge_bits, target)
            temperature = max(1.0, 2_000_000.0 * (1.0 - step / args.steps_per_restart))
            delta = proposal_eval[0] - current_eval[0]
            if delta >= 0 or rng.random() < math.exp(max(-50.0, delta / temperature)):
                current, current_eval = proposal_t, proposal_eval
            if proposal_eval[0] > best_eval[0]:
                best_edges, best_eval = proposal_t, proposal_eval
    if best_edges is None:
        raise RuntimeError("could not initialize a valid anchor/evidence assignment")
    return best_edges, best_eval[1], evidence_pool, target


def main() -> None:
    args = parse_args()
    records, categories = load_coco_records(Path(args.coco_root))
    raw, edge_bits, mean_min_area = build_edge_index(records, args)
    edges, counts, evidence_pool, target = search(raw, edge_bits, args)
    degrees = Counter(evidence for _anchor, evidence in edges)
    minimum_clean = min(counts.values())
    feasible = minimum_clean >= args.required_per_class
    result = {
        "source": "COCO train2017 + val2017",
        "num_source_images": len(records),
        "num_categories": len(categories),
        "constraints": {
            "num_classes": args.num_classes,
            "unique_anchors": True,
            "anchor_evidence_vocab_disjoint": not args.allow_cross_role,
            "one_evidence_per_class": True,
            "evidence_degree": [args.min_evidence_degree, args.max_evidence_degree],
            "required_per_class": args.required_per_class,
            "candidate_reserve": args.candidate_reserve,
            "target_clean_candidates": target,
            "min_anchor_area_ratio": args.min_anchor_area_ratio,
            "min_evidence_area_ratio": args.min_evidence_area_ratio,
            "max_object_rank": args.max_object_rank,
        },
        "feasible": feasible,
        "reserve_feasible": minimum_clean >= target,
        "minimum_clean_candidates": minimum_clean,
        "mean_clean_candidates": sum(counts.values()) / len(counts),
        "evidence_degrees": dict(sorted(degrees.items())),
        "cross_role_objects": sorted({anchor for anchor, _ in edges} & {evidence for _, evidence in edges}),
        "pairs": [
            {
                "class_name": f"{anchor.replace(' ', '_')}__{evidence.replace(' ', '_')}_scene",
                "anchor": anchor,
                "evidence": evidence,
                "raw_candidates": raw[(anchor, evidence)],
                "clean_candidates": counts[(anchor, evidence)],
                "mean_min_object_area_ratio": mean_min_area[(anchor, evidence)],
            }
            for anchor, evidence in edges
        ],
        "search": {
            "seed": args.seed,
            "restarts": args.restarts,
            "steps_per_restart": args.steps_per_restart,
            "evidence_pool": evidence_pool,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
