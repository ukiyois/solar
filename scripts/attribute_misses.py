"""Why GT instances are missed (single-view d0 candidates), split by annotator count and area.

Categories (per GT instance and annotator record, at the given operating point):
  matched                 final prediction IoU > .5
  query_ok_low_score      some query mask IoU > .5 but its score is below the threshold
  query_ok_lost           some query IoU > .5 and score passes, lost in ownership/mean-prob filter
  near_miss_0.3_0.5       best query IoU in (.3, .5]
  weak_0.1_0.3            best query IoU in (.1, .3]
  no_query                no query with IoU > .1

Input: a fold-0 dump of scripts/dump_tta_candidates.py; the categories come from
scripts/sweep_recall_tta.attribute. Output: JSON {group: {category: count}} for single-annotator and
multi-annotator observations (one or several annotator records), overall and per GT area bin; the printed
table shows the shares per group.

usage: python -m scripts.attribute_misses --candidates runs/tta/<name>/val --output <attribution.json>
"""
import argparse
import json
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from scripts.sweep_recall_tta import attribute, ground_truth, load


def run(job):
    """Pool worker: attribute() cases of one image, each tagged with the image's number of annotator records."""
    path, records, score, mask = job
    crops, binaries = load(path)
    cases = attribute(crops, binaries, records, score, mask)
    for case in cases:
        case["annotators"] = len(records)
    return cases


def area_bin(area):
    """Label of the GT area bin (pixels)."""
    return "<500" if area < 500 else "500-1500" if area < 1500 else "1500-5000" if area < 5000 else ">=5000"


def main():
    """Attribute the misses on every fold-0 image, write the counts and print the shares per group."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--score", type=float, default=.15)
    parser.add_argument("--mask", type=float, default=.725)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    truth = ground_truth()
    jobs = [(Path(args.candidates) / f"{Path(n).stem}.npz", r, args.score, args.mask) for n, r in sorted(truth.items())]
    with Pool(args.workers) as pool:
        cases = [c for part in pool.map(run, jobs) for c in part]
    rename = {"query_ok_score_below_threshold": "query_ok_low_score", "query_ok_lost_in_ownership_or_mean_prob": "query_ok_lost",
              "best_query_iou_0.3_0.5": "near_miss_0.3_0.5", "best_query_iou_0.1_0.3": "weak_0.1_0.3",
              "no_query_iou_above_0.1": "no_query"}
    table = defaultdict(Counter)
    for c in cases:
        group = "single-annotator" if c["annotators"] == 1 else "multi-annotator"
        category = rename.get(c["category"], c["category"])
        table[group][category] += 1
        table[f"{group} {area_bin(c['area'])}"][category] += 1
    order = ["matched", "query_ok_low_score", "query_ok_lost", "near_miss_0.3_0.5", "weak_0.1_0.3", "no_query"]
    Path(args.output).write_text(json.dumps({k: dict(v) for k, v in table.items()}, indent=1))
    print(f"{'group':34s} " + " ".join(f"{o[:12]:>12s}" for o in order) + "   total")
    for group in sorted(table):
        row = table[group]
        total = sum(row.values())
        print(f"{group:34s} " + " ".join(f"{row[o] / total:12.1%}" for o in order) + f"   {total}")


if __name__ == "__main__":
    main()
