#!/usr/bin/env python3
"""Mine directed COCO anchor/evidence pairs for a balanced slot-selection dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_coco_compositional_pair10_dataset import load_coco_records


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(fraction * (len(values) - 1)))]


def max_iou_and_min_distance(boxes_a, boxes_b, width: int, height: int):
    best_iou, min_distance = 0.0, float("inf")
    diagonal = math.hypot(width, height)
    for ax, ay, aw, ah in boxes_a:
        for bx, by, bw, bh in boxes_b:
            iw = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
            ih = max(0.0, min(ay + ah, by + bh) - max(ay, by))
            intersection = iw * ih
            union = aw * ah + bw * bh - intersection
            best_iou = max(best_iou, intersection / max(union, 1.0))
            distance = math.hypot((ax + aw / 2) - (bx + bw / 2), (ay + ah / 2) - (by + bh / 2))
            min_distance = min(min_distance, distance / max(diagonal, 1.0))
    return best_iou, min_distance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco_root", default=str(ROOT / "dataset/coco2017"))
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--min_anchor_area", type=float, default=0.02)
    parser.add_argument("--min_evidence_area", type=float, default=0.015)
    parser.add_argument("--max_rank", type=int, default=3)
    parser.add_argument("--min_candidates", type=int, default=650)
    parser.add_argument("--min_top2_both_rate", type=float, default=0.80)
    parser.add_argument("--max_overlap_rate", type=float, default=0.95)
    args = parser.parse_args()

    coco_root = Path(args.coco_root)
    records, _categories = load_coco_records(coco_root)
    boxes = {}
    for split in ("train", "val"):
        data = json.loads((coco_root / "annotations" / f"instances_{split}2017.json").read_text(encoding="utf-8"))
        categories = {int(row["id"]): row["name"] for row in data["categories"]}
        by_image = defaultdict(lambda: defaultdict(list))
        for ann in data["annotations"]:
            if not ann.get("iscrowd", 0):
                by_image[int(ann["image_id"])][categories[int(ann["category_id"])]].append(ann["bbox"])
        boxes[split] = by_image

    stats = defaultdict(lambda: {
        "count": 0, "top2_both": 0, "anchor_rank": Counter(), "evidence_rank": Counter(),
        "anchor_area": [], "evidence_area": [], "min_area": [], "iou": [], "distance": [],
    })
    for record in records:
        eligible = [(rank, name, area) for rank, (name, area) in enumerate(record.ranked, 1) if rank <= args.max_rank]
        for anchor_rank, anchor, anchor_area in eligible:
            if anchor_area < args.min_anchor_area:
                continue
            for evidence_rank, evidence, evidence_area in eligible:
                if anchor == evidence or evidence_area < args.min_evidence_area:
                    continue
                iou, distance = max_iou_and_min_distance(
                    boxes[record.source_split][record.image_id][anchor],
                    boxes[record.source_split][record.image_id][evidence],
                    record.width,
                    record.height,
                )
                row = stats[(anchor, evidence)]
                row["count"] += 1
                row["top2_both"] += int(anchor_rank <= 2 and evidence_rank <= 2)
                row["anchor_rank"][anchor_rank] += 1; row["evidence_rank"][evidence_rank] += 1
                row["anchor_area"].append(anchor_area); row["evidence_area"].append(evidence_area)
                row["min_area"].append(min(anchor_area, evidence_area)); row["iou"].append(iou); row["distance"].append(distance)

    output_rows = []
    for (anchor, evidence), value in stats.items():
        count = value["count"]
        if count < args.min_candidates:
            continue
        top2_rate = value["top2_both"] / count
        overlap_rate = sum(item > 0.1 for item in value["iou"]) / count
        row = {
            "anchor": anchor, "evidence": evidence, "candidates": count,
            "top2_both_rate": top2_rate,
            "anchor_rank3_rate": value["anchor_rank"][3] / count,
            "evidence_rank3_rate": value["evidence_rank"][3] / count,
            "anchor_area_q10": percentile(value["anchor_area"], 0.1),
            "anchor_area_median": percentile(value["anchor_area"], 0.5),
            "evidence_area_q10": percentile(value["evidence_area"], 0.1),
            "evidence_area_median": percentile(value["evidence_area"], 0.5),
            "min_area_q10": percentile(value["min_area"], 0.1),
            "bbox_overlap_rate_iou_gt_01": overlap_rate,
            "bbox_iou_median": percentile(value["iou"], 0.5),
            "center_distance_median": percentile(value["distance"], 0.5),
        }
        row["shortlisted"] = bool(top2_rate >= args.min_top2_both_rate and overlap_rate <= args.max_overlap_rate)
        # Count and weakest-object area dominate; distance is a soft preference.
        row["quality_score"] = (
            2.0 * min(1.0, count / 1000.0) + 3.0 * top2_rate
            + 10.0 * min(row["min_area_q10"], 0.1) + min(row["center_distance_median"], 0.3)
            - 0.5 * overlap_rate
        )
        output_rows.append(row)
    output_rows.sort(key=lambda row: (-row["shortlisted"], -row["quality_score"], row["anchor"], row["evidence"]))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "pair_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0])); writer.writeheader(); writer.writerows(output_rows)
    summary = {
        "constraints": vars(args), "num_source_images": len(records), "eligible_pairs": len(output_rows),
        "shortlisted_pairs": sum(row["shortlisted"] for row in output_rows),
        "top_shortlist": [row for row in output_rows if row["shortlisted"]][:100],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("num_source_images", "eligible_pairs", "shortlisted_pairs")}, indent=2))


if __name__ == "__main__":
    main()
