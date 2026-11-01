"""Does fold-0 PQ depend on near-in-time training observations? (CV -> leaderboard gap check)

For every fold-0 validation image: the smallest time gap to any training-fold
observation (any site). Test images: the same gap to any train image. Fused TTA
predictions (same rule as the submission) are scored per validation image and
aggregated for 'near' (< --hours) and 'far' images, plus a reweighting of the two
groups to the test set's near/far mix.
"""
import argparse
import datetime as dt
import json
from multiprocessing import Pool
from pathlib import Path

from scripts.fuse_tta import pooled_candidates
from scripts.sweep_recall_tta import fuse, ground_truth, ownership
from solar_filaments.data import physical_fold
from solar_filaments.metrics import aggregate_pq, official_pq

ROOT = Path("data/raw/MAGFiLO_1.0_Kaggle_2026")


def stamp(name):
    return dt.datetime.strptime(name[:14], "%Y%m%d%H%M%S")


def gaps(targets, pool):
    times = sorted(stamp(n) for n in pool)
    result = {}
    for name in targets:
        t = stamp(name)
        result[name] = min(abs((t - u).total_seconds()) for u in times) / 3600
    return result


def score(job):
    stem, records, folders, views, setting = job
    crops, voters = pooled_candidates(stem, folders, views)
    votes, s, m = setting
    predictions = [p["rle"] for p in ownership(fuse(crops, voters, "members"), s, m, min_votes=votes)]
    return stem, [official_pq(predictions, truth) for _, truth in records]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", nargs="+", required=True)
    parser.add_argument("--folder-views", nargs="+", required=True)
    parser.add_argument("--votes", type=float, default=.111, help="fraction of voters")
    parser.add_argument("--score", type=float, default=.12)
    parser.add_argument("--mask", type=float, default=.65)
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    train = sorted(p.name for p in (ROOT / "train/train_images").iterdir())
    test = sorted(p.name for p in (ROOT / "test/test_images").iterdir())
    val = [n for n in train if physical_fold(n, 5, 2026) == 0]
    fit = [n for n in train if physical_fold(n, 5, 2026) != 0]
    val_gap, test_gap = gaps(val, fit), gaps(test, train)
    views = [v.split(",") for v in args.folder_views]
    voters = sum(len(v) for v in views)
    setting = (max(1, int(round(args.votes * voters))), args.score, args.mask)
    truth = ground_truth()
    jobs = [(Path(name).stem, records, args.folders, views, setting) for name, records in sorted(truth.items())]
    with Pool(args.workers) as pool:
        rows = dict(pool.map(score, jobs))
    near = [r for name in val if val_gap[name] < args.hours for r in rows[Path(name).stem]]
    far = [r for name in val if val_gap[name] >= args.hours for r in rows[Path(name).stem]]
    test_near = sum(g < args.hours for g in test_gap.values()) / len(test_gap)
    summary = {
        "hours": args.hours, "setting": setting, "voters": voters,
        "val_images_near": sum(val_gap[n] < args.hours for n in val), "val_images": len(val),
        "test_images_near": sum(g < args.hours for g in test_gap.values()), "test_images": len(test),
        "pq_all": aggregate_pq(near + far)["pq"], "pq_near": aggregate_pq(near)["pq"], "pq_far": aggregate_pq(far)["pq"],
        "near_detail": {k: aggregate_pq(near)[k] for k in ("sq", "rq", "tp", "fp", "fn")},
        "far_detail": {k: aggregate_pq(far)[k] for k in ("sq", "rq", "tp", "fp", "fn")},
    }
    summary["pq_test_mix"] = test_near * summary["pq_near"] + (1 - test_near) * summary["pq_far"]
    # Breakdown by annotator records per physical image (1, 2 or 3 independent versions).
    by_count = {}
    for name in val:
        by_count.setdefault(len(rows[Path(name).stem]), []).extend(rows[Path(name).stem])
    summary["by_records_per_image"] = {
        k: {"images": sum(len(rows[Path(n).stem]) == k for n in val), **{m: aggregate_pq(v)[m] for m in
            ("pq", "sq", "rq", "tp", "fp", "fn")}, "gt_per_record": (aggregate_pq(v)["tp"] + aggregate_pq(v)["fn"]) / len(v),
            "pred_per_record": (aggregate_pq(v)["tp"] + aggregate_pq(v)["fp"]) / len(v)}
        for k, v in sorted(by_count.items())}
    Path(args.output).write_text(json.dumps({"summary": summary, "val_gap_hours": val_gap,
                                             "test_gap_hours": test_gap}, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
