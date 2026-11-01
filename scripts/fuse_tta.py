"""Test-time-augmentation fusion of dumped candidates and submission export (CPU).

Input: one folder per checkpoint from scripts/dump_tta_candidates.py; the final submission uses ONE
checkpoint (final-A) x 8 dihedral views. Every (checkpoint, view) pair is one voter. Candidates are
clustered across voters by IoU > .5 at probability .5 (at most one member per voter), member
probabilities are averaged and the fused score is the summed member score divided by the number of
voters ('members' rule). A fused instance is kept if it has >= votes members, score >= score and,
after binarization at `mask`, wins its pixels in the score-ordered pixel competition of
solar_filaments.inference (min area 32 px, >= 50 % of its area retained). Validation reports the
official PQ over a grid; --test exports the CSV with the first value of each option.

usage (final submission, votes 1/8, score .10, mask .40):
  python -m scripts.fuse_tta --folders runs/tta/final_a/test --votes 0.125 --scores 0.10 --masks 0.40 \
      --test --workers 4 --output deliverables/submission.csv
"""
import argparse
import csv
import itertools
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from scripts.sweep_recall_tta import fuse, ground_truth, load, ownership
from solar_filaments.metrics import aggregate_pq, official_pq

VIEWS = ["d0", "d1", "d2", "d3", "f0", "f1", "f2", "f3"]


def pooled_candidates(stem, folders, views):
    """views: one list shared by every folder, or a list of per-folder lists."""
    per_folder = views if views and isinstance(views[0], (list, tuple)) else [views] * len(folders)
    crops = []
    for index, (folder, own) in enumerate(zip(folders, per_folder)):
        for crop in load(Path(folder) / f"{stem}.npz")[0]:
            if crop["variant"] in own:
                crops.append({**crop, "variant": f"{index}:{crop['variant']}"})
    voters = [f"{i}:{v}" for i, own in enumerate(per_folder) for v in own]
    return crops, voters


def evaluate(job):
    stem, records, folders, views, grid = job
    crops, voters = pooled_candidates(stem, folders, views)
    fused = fuse(crops, voters, "members")
    rows = {}
    for votes, score, mask in grid:
        predictions = [p["rle"] for p in ownership(fused, score, mask, min_votes=votes)]
        rows[(votes, score, mask)] = [{**official_pq(predictions, truth), "record_id": rid} for rid, truth in records]
    return rows


def predict(job):
    stem, folders, views, (votes, score, mask) = job
    crops, voters = pooled_candidates(stem, folders, views)
    return stem, ownership(fuse(crops, voters, "members"), score, mask, min_votes=votes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", nargs="+", required=True, help="Candidate dumps, one per model")
    parser.add_argument("--views", nargs="+", default=VIEWS)
    parser.add_argument("--folder-views", nargs="+",
                        help="Per-folder comma-separated view lists (same order as --folders); overrides --views")
    parser.add_argument("--output", required=True)
    parser.add_argument("--votes", type=float, nargs="+", default=[.25, .375, .5],
                        help="Minimum votes as a fraction of all voters")
    parser.add_argument("--scores", type=float, nargs="+", default=[.08, .1, .12, .15, .2])
    parser.add_argument("--masks", type=float, nargs="+", default=[.5, .55, .6, .65, .7])
    parser.add_argument("--test", action="store_true", help="Export CSV with the first value of each option")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.folder_views:
        if len(args.folder_views) != len(args.folders):
            parser.error("--folder-views needs one entry per folder")
        args.views = [v.split(",") for v in args.folder_views]
    voters = sum(len(v) for v in args.views) if isinstance(args.views[0], list) else len(args.folders) * len(args.views)
    votes = sorted({max(1, int(round(f * voters))) for f in args.votes})
    if args.test:
        setting = (votes[0], args.scores[0], args.masks[0])
        manifest = json.loads((Path(args.folders[0]) / "manifest.json").read_text())
        stems = [Path(n).stem for n in manifest["files"]]
        with Pool(args.workers) as pool:
            results = dict(pool.map(predict, [(s, args.folders, args.views, setting) for s in stems]))
        count = 0
        with open(args.output, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["filament_id", "segmentation_rle"])
            writer.writeheader()
            for stem in stems:
                for number, p in enumerate(results[stem], 1):
                    writer.writerow({"filament_id": f"{stem}_{number}",
                                     "segmentation_rle": p["rle"]["counts"].decode("ascii")})
                    count += 1
        print(json.dumps({"images": len(stems), "instances": count, "setting": setting, "voters": voters,
                          "empty_images": [s for s in stems if not results[s]]}))
        return
    grid = list(itertools.product(votes, args.scores, args.masks))
    truth = ground_truth()
    jobs = [(Path(name).stem, records, args.folders, args.views, grid) for name, records in sorted(truth.items())]
    totals = defaultdict(list)
    with Pool(args.workers) as pool:
        for rows in pool.imap_unordered(evaluate, jobs):
            for key, value in rows.items():
                totals[key].extend(value)
    table = sorted(({"votes": k[0], "score": k[1], "mask": k[2], **aggregate_pq(v)} for k, v in totals.items()),
                   key=lambda r: -r["pq"])
    Path(args.output).write_text(json.dumps({"folders": args.folders, "voters": voters, "table": table}, indent=1))
    for row in table[:12]:
        print(f"votes={row['votes']:>2} score={row['score']:.2f} mask={row['mask']:.2f} PQ={row['pq']:.4f} "
              f"SQ={row['sq']:.4f} RQ={row['rq']:.4f} TP={row['tp']} FP={row['fp']} FN={row['fn']}")


if __name__ == "__main__":
    main()
