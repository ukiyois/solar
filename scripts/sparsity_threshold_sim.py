"""Optimal operating point if the hidden test labels are sparser than fold-0 labels.

Fold-0 truth is thinned per annotator record to a fraction r of its instances,
either at random (omissions independent of the filament) or smallest-first
(annotators skipping small/faint filaments). Fused TTA predictions are scored for
a grid of score thresholds; reports the best threshold per scenario and what the
current setting loses. Same fusion/ownership as the submission.

Scenarios per rate r: random_<r> (--repeats seeded draws, pooled) and smallest_<r>; random_1.0 is the
unthinned truth. Output JSON: per scenario the best score threshold, its PQ and the PQ at score .12, plus the
PQ / precision / recall table. The defaults (votes .111, mask .65, summary at score .12) differ from the final
submission setting (votes 1/8, score .10, mask .40).

usage: python -m scripts.sparsity_threshold_sim --folders runs/tta/<name>/val \
           --folder-views d0,d1,d2,d3,f0,f1,f2,f3 --output <sparsity.json>
"""
import argparse
import itertools
import json
import random
from multiprocessing import Pool
from pathlib import Path

from pycocotools import mask as mu

from scripts.fuse_tta import pooled_candidates
from scripts.sweep_recall_tta import fuse, ground_truth, ownership
from solar_filaments.metrics import aggregate_pq, official_pq


def thin(truth, rate, mode, rng):
    """Keep round(rate x n) of a record's n GT instances (at least 1 if n > 0).

    mode "random" keeps a random subset; mode "smallest" keeps the largest instances, i.e. drops the smallest.
    """
    keep = max(1, round(rate * len(truth))) if truth else 0
    if mode == "random":
        return rng.sample(truth, keep)
    order = sorted(truth, key=lambda r: -float(mu.area(r)))
    return order[:keep]


def run(job):
    """Pool worker: official PQ rows of one fold-0 image for every (scenario, score threshold).

    job = (stem, records, folders, views, votes, scores, mask, scenarios, seed). The random generator is seeded
    by (seed, image, repeat), so every score threshold is scored against the same thinned truth. Returns
    {(scenario name, score): [official_pq rows]}; the repeats of a scenario share one key.
    """
    stem, records, folders, views, votes, scores, mask, scenarios, seed = job
    crops, voters = pooled_candidates(stem, folders, views)
    fused = fuse(crops, voters, "members")
    rows = {}
    for s in scores:
        predictions = [p["rle"] for p in ownership(fused, s, mask, min_votes=votes)]
        for name, rate, mode, repeat in scenarios:
            rng = random.Random(f"{seed}:{stem}:{repeat}")
            for rid, truth in records:
                rows.setdefault((name, s), []).append(official_pq(predictions, thin(truth, rate, mode, rng)))
    return rows


def main():
    """Score every thinning scenario over the score grid and report the best threshold per scenario."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", nargs="+", required=True)
    parser.add_argument("--folder-views", nargs="+", required=True)
    parser.add_argument("--votes", type=float, default=.111)
    parser.add_argument("--mask", type=float, default=.65)
    parser.add_argument("--scores", type=float, nargs="+", default=[.1, .12, .15, .18, .21, .25, .3, .35, .4])
    parser.add_argument("--rates", type=float, nargs="+", default=[1.0, .85, .69])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    views = [v.split(",") for v in args.folder_views]
    votes = max(1, int(round(args.votes * sum(len(v) for v in views))))
    # (name, rate, mode, repeat): random thinning drawn --repeats times when rate < 1; no smallest-first at rate 1
    scenarios = [(f"{mode}_{rate}", rate, mode, k) for rate in args.rates for mode in ("random", "smallest")
                 for k in range(args.repeats if mode == "random" and rate < 1 else 1) if not (rate == 1 and mode == "smallest")]
    truth = ground_truth()
    jobs = [(Path(n).stem, recs, args.folders, views, votes, args.scores, args.mask, scenarios, 2026)
            for n, recs in sorted(truth.items())]
    totals = {}
    with Pool(args.workers) as pool:
        for rows in pool.imap_unordered(run, jobs):
            for key, value in rows.items():
                totals.setdefault(key, []).extend(value)
    table = {}
    for (name, s), value in sorted(totals.items()):
        a = aggregate_pq(value)
        table.setdefault(name, {})[s] = {k: round(a[k], 4) for k in ("pq", "precision", "recall")}
    summary = {name: {"best_score": max(t, key=lambda s: t[s]["pq"]), "best_pq": max(r["pq"] for r in t.values()),
                      "pq_at_.12": t.get(.12, {}).get("pq")} for name, t in table.items()}
    Path(args.output).write_text(json.dumps({"votes": votes, "mask": args.mask, "summary": summary, "table":
                                             {k: {str(s): v for s, v in t.items()} for k, t in table.items()}}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
