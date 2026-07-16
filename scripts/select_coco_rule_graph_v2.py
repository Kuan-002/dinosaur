#!/usr/bin/env python3
"""Select an oracle-qualified 8--10 class shared-evidence rule graph."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_coco_compositional_pair10_dataset import load_coco_records


def clean_counts(edges, bits):
    once = 0; multiple = 0
    for edge in edges:
        present = bits[edge]; multiple |= once & present; once = (once ^ present) & ~multiple
    return {edge: (bits[edge] & once).bit_count() for edge in edges}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle_csv", required=True)
    parser.add_argument("--candidate_summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--required_clean", type=int, default=500)
    parser.add_argument("--min_oracle_distinct_04", type=float, default=0.70)
    parser.add_argument("--iterations", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=8)
    args = parser.parse_args()

    oracle_rows = list(csv.DictReader(open(args.oracle_csv, encoding="utf-8")))
    oracle = {(row["anchor"], row["evidence"]): row for row in oracle_rows}
    allowed = {
        edge: row for edge, row in oracle.items()
        if float(row["pair_distinct@0.4"]) >= args.min_oracle_distinct_04 and int(row["final_pool"]) >= args.required_clean
    }
    by_evidence = defaultdict(list)
    for edge in allowed: by_evidence[edge[1]].append(edge)
    evidence_pool = sorted(evidence for evidence, edges in by_evidence.items() if len(edges) >= 2)
    if len(evidence_pool) < 3:
        raise RuntimeError(f"only {len(evidence_pool)} evidence groups have >=2 oracle-qualified pairs: {evidence_pool}")

    constraints = json.load(open(args.candidate_summary, encoding="utf-8"))["constraints"]
    records, _ = load_coco_records(Path(constraints["coco_root"]))
    min_anchor = float(constraints["min_anchor_area"]); min_evidence = float(constraints["min_evidence_area"]); max_rank = int(constraints["max_rank"])
    bits = defaultdict(int); raw = Counter()
    allowed_set = set(allowed)
    for index, record in enumerate(records):
        ranks = {name: rank for rank, (name, _area) in enumerate(record.ranked, 1)}
        image_bit = 1 << index
        for edge in allowed_set:
            anchor, evidence = edge
            if (ranks.get(anchor, max_rank + 1) <= max_rank and ranks.get(evidence, max_rank + 1) <= max_rank
                    and record.ratios.get(anchor, 0.0) >= min_anchor and record.ratios.get(evidence, 0.0) >= min_evidence):
                bits[edge] |= image_bit; raw[edge] += 1

    rng = random.Random(args.seed); best = None; best_score = -float("inf"); best_counts = None
    possible_group_counts = [
        count for count in range(3, len(evidence_pool) + 1)
        if 2 * count <= args.num_classes <= 3 * count
    ]
    if not possible_group_counts:
        raise RuntimeError(
            f"cannot express {args.num_classes} classes with evidence degree 2--3 "
            f"from {len(evidence_pool)} eligible groups"
        )
    for _ in range(args.iterations):
        group_count = rng.choice(possible_group_counts)
        evidences = rng.sample(evidence_pool, group_count)
        # Random 2/3 degrees that sum to the requested number of classes.
        degrees = [2] * group_count
        extra = args.num_classes - sum(degrees)
        if extra < 0 or extra > group_count: continue
        for index in rng.sample(range(group_count), extra): degrees[index] = 3
        evidence_set = set(evidences); used_anchors = set(); edges = []; valid = True
        for evidence, degree in sorted(zip(evidences, degrees), key=lambda item: len(by_evidence[item[0]])):
            candidates = [edge for edge in by_evidence[evidence] if edge[0] not in evidence_set and edge[0] not in used_anchors]
            if len(candidates) < degree: valid = False; break
            weights = [max(1e-6, float(oracle[edge]["pair_distinct@0.4"])) ** 4 for edge in candidates]
            chosen = []
            for _pick in range(degree):
                edge = rng.choices(candidates, weights=weights, k=1)[0]
                chosen.append(edge); chosen_anchor = edge[0]; used_anchors.add(chosen_anchor)
                keep = [(candidate, weight) for candidate, weight in zip(candidates, weights) if candidate[0] != chosen_anchor]
                candidates = [item[0] for item in keep]; weights = [item[1] for item in keep]
            edges.extend(chosen)
        if not valid or len(edges) != args.num_classes: continue
        edges = tuple(sorted(edges)); counts = clean_counts(edges, bits); minimum = min(counts.values())
        deficit = sum(max(0, args.required_clean - value) for value in counts.values())
        oracle_min = min(float(oracle[edge]["pair_distinct@0.4"]) for edge in edges)
        oracle_mean = sum(float(oracle[edge]["pair_distinct@0.4"]) for edge in edges) / len(edges)
        # Once every class reaches the requested capacity, extra examples are only
        # a weak tie-breaker: the graph is selected primarily for slot visibility.
        capacity_score = min(minimum, args.required_clean)
        score = -1_000_000 * deficit + 10 * capacity_score + 100000 * oracle_min + 10000 * oracle_mean
        if score > best_score:
            best, best_counts, best_score = edges, counts, score
    if best is None: raise RuntimeError("no graph could be initialized")
    evidence_degrees = Counter(evidence for _anchor, evidence in best)
    result = {
        "feasible": min(best_counts.values()) >= args.required_clean,
        "num_classes": len(best), "required_clean": args.required_clean,
        "minimum_clean": min(best_counts.values()), "evidence_degrees": dict(sorted(evidence_degrees.items())),
        "pairs": [{
            "class_name": f"{anchor.replace(' ', '_')}__{evidence.replace(' ', '_')}_scene",
            "anchor": anchor, "evidence": evidence, "raw_candidates": raw[(anchor, evidence)],
            "clean_candidates": best_counts[(anchor, evidence)],
            "oracle_distinct_02": float(oracle[(anchor, evidence)]["pair_distinct@0.2"]),
            "oracle_distinct_04": float(oracle[(anchor, evidence)]["pair_distinct@0.4"]),
            "oracle_same_only_04": float(oracle[(anchor, evidence)]["same_only@0.4"]),
        } for anchor, evidence in best],
        "search": {"iterations": args.iterations, "seed": args.seed, "minimum_oracle_distinct_04": args.min_oracle_distinct_04},
        "construction_filters": {"min_anchor_area": min_anchor, "min_evidence_area": min_evidence, "max_rank": max_rank},
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
