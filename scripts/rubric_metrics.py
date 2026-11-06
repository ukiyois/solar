"""Final-judging metrics on fold 0, computed as in the organizers' self-evaluation notebook
(kaggle.com/code/azimahmadzadeh/self-evaluation-notebook):

* Panoptic Quality per annotator-image record (IoU > 0.5 hits, pooled TP/FP/FN),
* the distribution of IoU and of Dice over every overlapping (score > 0) GT/prediction pair,
* the one-to-many (one GT overlapping n predictions) and many-to-one (one prediction overlapping
  n GT filaments) relations,

plus the semantic Dice of each record: torchmetrics.segmentation's dice_score (the functional form of
DiceScore; index input, filament class only) on the binary filament/background masks, i.e. the union
of the record's GT instances against the union of the predicted instances. It is cross-checked
against the same quantity computed on RLEs, and also reported pooled over all records (global).

Predictions are the fused TTA candidates of scripts/dump_tta_candidates.py, selected exactly as for
a submission (member fusion + pixel ownership with the given votes / score / mask thresholds).

Uncertainty: 95 % percentile intervals from a bootstrap over observations (all annotator records of
an observation are resampled together, since they share one image). --compare takes the
per_record.json of another run on the same observations and reports paired differences.

Outputs in --output-dir: rubric_metrics.json (summary, setting and bootstrap), per_record.json (one row of
additive statistics per record; read by --compare and scripts/morphology_figures) and rubric_distributions.png.

usage: python -m scripts.rubric_metrics --folders runs/tta_s1a/val --folder-views d0,d1,d2,d3,f0,f1,f2,f3 \
           --votes 1 --score .10 --mask .40 --output-dir docs/figures/rubric_fold0 \
           [--compare docs/figures/rubric_fold0_v7/per_record.json]
"""
import argparse
import json
import warnings
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mu
from torchmetrics.functional.segmentation import dice_score

from scripts.fuse_tta import predict
from scripts.sweep_recall_tta import ground_truth

# average="micro" is passed explicitly; torchmetrics still warns about its future default.
warnings.filterwarnings("ignore", message=".*dice_score metric currently defaults.*")


def torchmetrics_dice(union_gt, union_pred, shape=(2048, 2048)):
    """torchmetrics Dice of the filament class for one binary mask pair (None = empty mask)."""
    def index(rle):
        return torch.from_numpy(mu.decode(rle).astype(np.int64)) if rle is not None else torch.zeros(shape, dtype=torch.long)
    return float(dice_score(index(union_pred)[None], index(union_gt)[None], num_classes=2, include_background=False,
                            input_format="index", average="micro", aggregation_level="samplewise")[0])


def record_rows(job):
    """Pool worker: one row per annotator record of a fold-0 image.

    job = (stem, [(record id, GT RLEs)], folders, views, (votes, score, mask)). A row holds the record and image
    ids, the instance counts, the (n_gt, n_pred) IoU matrix and the semantic Dice of the GT union against the
    prediction union, computed on RLEs (two empty masks give 1) and with torchmetrics, plus the intersection
    and the two union areas in pixels.
    """
    stem, records, folders, views, setting = job
    _, kept = predict((stem, folders, views, setting))
    predictions = [p["rle"] for p in kept]
    union_pred = mu.merge(predictions) if predictions else None
    pred_area = float(mu.area(union_pred)) if union_pred is not None else 0.
    rows = []
    for record_id, truth in records:
        n_gt, n_pred = len(truth), len(predictions)
        iou = (np.asarray(mu.iou(predictions, truth, [0] * n_gt)).reshape(n_pred, n_gt).T
               if n_gt and n_pred else np.zeros((n_gt, n_pred)))
        union_gt = mu.merge(truth) if truth else None
        gt_area = float(mu.area(union_gt)) if union_gt is not None else 0.
        inter = (float(mu.area(mu.merge([union_gt, union_pred], intersect=True)))
                 if union_gt is not None and union_pred is not None else 0.)
        dice = 2 * inter / (gt_area + pred_area) if gt_area + pred_area else 1.0
        rows.append({"record": record_id, "image": stem, "n_gt": n_gt, "n_pred": n_pred, "iou": iou,
                     "semantic_dice": dice, "torchmetrics_dice": torchmetrics_dice(union_gt, union_pred),
                     "intersection": inter, "gt_area": gt_area, "pred_area": pred_area})
    return rows


def record_stats(row):
    """Additive statistics of one record: PQ counts at IoU > .5 and the sums over overlapping pairs."""
    iou, n_gt, n_pred = row["iou"], row["n_gt"], row["n_pred"]
    if n_gt == 0 or n_pred == 0:
        return {"tp_iou": 0., "tp": 0, "fp": n_pred, "fn": n_gt, "pairs": 0, "pair_iou_sum": 0., "pair_dice_sum": 0.}
    hit, pairs = iou > .5, iou[iou > 0]
    return {"tp_iou": float(iou[hit].sum()), "tp": int(hit.sum()), "fp": int((hit.sum(0) == 0).sum()),
            "fn": int((hit.sum(1) == 0).sum()), "pairs": int(pairs.size), "pair_iou_sum": float(pairs.sum()),
            "pair_dice_sum": float((2 * pairs / (1 + pairs)).sum())}


# Additive per-record statistics: summing them over (resampled) observations yields every bootstrapped metric.
COLUMNS = ("tp_iou", "tp", "fp", "fn", "pairs", "pair_iou_sum", "pair_dice_sum", "torchmetrics_dice", "intersection",
           "areas", "records")


def observation_table(records, images):
    """One row of summed COLUMNS per observation (records of the same image added together)."""
    index = {image: i for i, image in enumerate(images)}
    table = np.zeros((len(images), len(COLUMNS)))
    for r in records:
        if r["image"] in index:
            values = {**r, "areas": r["gt_area"] + r["pred_area"], "records": 1}
            table[index[r["image"]]] += [values[c] for c in COLUMNS]
    return table


def metrics_from_sums(sums):
    """PQ / SQ / RQ, mean pair IoU and Dice and the semantic Dice from (..., len(COLUMNS)) column sums."""
    c = {name: sums[..., i] for i, name in enumerate(COLUMNS)}
    denominator = c["tp"] + .5 * c["fp"] + .5 * c["fn"]
    return {"pq": c["tp_iou"] / denominator, "sq": c["tp_iou"] / np.maximum(c["tp"], 1), "rq": c["tp"] / denominator,
            "pair_iou_mean": c["pair_iou_sum"] / c["pairs"], "pair_dice_mean": c["pair_dice_sum"] / c["pairs"],
            "semantic_dice_mean": c["torchmetrics_dice"] / c["records"],
            "semantic_dice_global": 2 * c["intersection"] / c["areas"]}


def bootstrap(records, other=None, samples=4000, seed=2026):
    """95 % intervals over resampled observations; with `other`, paired differences (this run - other)."""
    images = sorted({r["image"] for r in records} & ({r["image"] for r in other} if other else {r["image"] for r in records}))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(images), (samples, len(images)))
    mine = metrics_from_sums(observation_table(records, images)[draws].sum(1))
    interval = lambda values: [round(float(q), 4) for q in np.percentile(values, [2.5, 97.5])]
    result = {"observations": len(images), "samples": samples, "ci95": {k: interval(v) for k, v in mine.items()}}
    if other:
        theirs = metrics_from_sums(observation_table(other, images)[draws].sum(1))
        full = {k: float(v) for k, v in metrics_from_sums(observation_table(records, images).sum(0)).items()}
        base = {k: float(v) for k, v in metrics_from_sums(observation_table(other, images).sum(0)).items()}
        result["paired_difference"] = {k: {"difference": round(full[k] - base[k], 4), "ci95": interval(mine[k] - theirs[k]),
                                           "p_not_better": round(float(np.mean(mine[k] - theirs[k] <= 0)), 4)}
                                       for k in ("pq", "pair_dice_mean", "semantic_dice_mean")}
    return result


def summarize(rows):
    """Pool the record rows into the rubric summary.

    PQ / SQ / RQ from TP (IoU > .5), FP and FN pooled over records; mean and quantiles of IoU and of Dice
    (2 IoU / (1 + IoU)) over every overlapping (IoU > 0) GT/prediction pair; semantic Dice per record (mean,
    median) and pooled over records (global), with the largest torchmetrics-vs-RLE difference; degree counts
    (predictions overlapping each GT instance, GT instances overlapping each prediction). Returns (summary,
    pair IoUs, pair Dice, per-record torchmetrics Dice, GT degree Counter, prediction degree Counter).
    """
    tp_iou, fp, fn = [], 0, 0
    pair_iou, gt_degree, pred_degree = [], Counter(), Counter()
    for row in rows:
        iou, n_gt, n_pred = row["iou"], row["n_gt"], row["n_pred"]
        if n_gt == 0:
            fp += n_pred
            pred_degree[0] += n_pred
            continue
        if n_pred == 0:
            fn += n_gt
            gt_degree[0] += n_gt
            continue
        hit = iou > .5
        tp_iou += iou[hit].tolist()
        fp += int((hit.sum(0) == 0).sum())
        fn += int((hit.sum(1) == 0).sum())
        pair_iou += iou[iou > 0].tolist()
        gt_degree.update((iou > 0).sum(1).tolist())
        pred_degree.update((iou > 0).sum(0).tolist())
    pair_iou = np.asarray(pair_iou)
    pair_dice = 2 * pair_iou / (1 + pair_iou)
    tp = len(tp_iou)
    pq = sum(tp_iou) / (tp + .5 * fp + .5 * fn) if tp + fp + fn else 0.
    semantic = np.asarray([r["torchmetrics_dice"] for r in rows])
    rle_semantic = np.asarray([r["semantic_dice"] for r in rows])
    inter, areas = sum(r["intersection"] for r in rows), sum(r["gt_area"] + r["pred_area"] for r in rows)
    return {
        "records": len(rows), "images": len({r["image"] for r in rows}),
        "pq": round(pq, 4), "sq": round(float(np.mean(tp_iou)), 4) if tp else 0., "rq": round(tp / (tp + .5 * fp + .5 * fn), 4),
        "tp": tp, "fp": fp, "fn": fn,
        "pair_iou_mean": round(float(pair_iou.mean()), 4), "pair_dice_mean": round(float(pair_dice.mean()), 4),
        "pair_iou_quantiles_10_25_50_75_90": [round(float(q), 3) for q in np.quantile(pair_iou, [.1, .25, .5, .75, .9])],
        "pair_dice_quantiles_10_25_50_75_90": [round(float(q), 3) for q in np.quantile(pair_dice, [.1, .25, .5, .75, .9])],
        "semantic_dice_mean": round(float(semantic.mean()), 4),
        "semantic_dice_median": round(float(np.median(semantic)), 4),
        "semantic_dice_global": round(2 * inter / areas, 4) if areas else 1.,
        "semantic_dice_torchmetrics_vs_rle_max_abs_diff": float(np.abs(semantic - rle_semantic).max()),
        "one_to_many_gt_degree": dict(sorted(gt_degree.items())),
        "many_to_one_pred_degree": dict(sorted(pred_degree.items())),
    }, pair_iou, pair_dice, semantic, gt_degree, pred_degree


def plot(pair_iou, pair_dice, semantic, gt_degree, pred_degree, out, title):
    """Save <out>/rubric_distributions.png: pair IoU, pair Dice and per-record semantic Dice histograms, and the
    many-to-one (left, n GT : 1 prediction) / one-to-many (right, 1 GT : n predictions) bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def histogram(ax, scores, label, color):
        counts, edges = np.histogram(scores, bins=np.linspace(0, 1, 51))
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, counts / max(1, counts.sum()), width=.02, edgecolor="black", alpha=.75, color=color)
        ax.axvline(scores.mean(), color="crimson", linestyle="--", linewidth=1.5, label=f"Mean = {scores.mean():.3f}")
        ax.set_xlabel(label)
        ax.set_ylabel("Probability")
        ax.grid(axis="y", linestyle="--")
        ax.legend(fontsize=8)

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.6))
    histogram(axes[0], pair_iou, "IoU (overlapping GT/prediction pairs)", "navy")
    axes[0].set_title("IoU distribution")
    histogram(axes[1], pair_dice, "Dice (overlapping GT/prediction pairs)", "teal")
    axes[1].set_title("Dice distribution")
    histogram(axes[2], semantic, "torchmetrics DiceScore of the filament mask per record", "slategray")
    axes[2].set_title("Semantic Dice per annotator-image record")
    ax = axes[3]
    bars = [(degree + .5, count, "crimson" if degree == 0 else "forestgreen") for degree, count in gt_degree.items()]
    bars += [(-(degree + .5), count, "darkorange" if degree == 0 else "steelblue") for degree, count in pred_degree.items()]
    for x, count, color in bars:
        ax.bar(x, count, width=.8, color=color, edgecolor="black", alpha=.75)
        ax.text(x, count, str(count), ha="center", va="bottom", fontsize=7)
    ticks = sorted(x for x, _, _ in bars)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(abs(t) - .5)}:1" if t < 0 else f"1:{int(t - .5)}" for t in ticks], rotation=45, fontsize=8)
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_ylim(0, 1.4 * max(count for _, count, _ in bars))
    ax.set_title("Many-to-one (left: GT per prediction) / one-to-many (right)", fontsize=10)
    ax.set_ylabel("Frequency")
    ax.grid(axis="y", linestyle="--")
    ax.legend(handles=[
        Patch(facecolor="forestgreen", label=f"GT hits: {sum(c for d, c in gt_degree.items() if d)}"),
        Patch(facecolor="crimson", label=f"GT misses (1:0): {gt_degree.get(0, 0)}"),
        Patch(facecolor="steelblue", label=f"Prediction hits: {sum(c for d, c in pred_degree.items() if d)}"),
        Patch(facecolor="darkorange", label=f"Prediction misses (0:1): {pred_degree.get(0, 0)}")],
        fontsize=7, ncol=2, loc="upper center")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out / "rubric_distributions.png", dpi=130)
    plt.close(fig)


def main():
    """Score fold 0 at the given setting and write the JSON files and the figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", nargs="+", required=True)
    parser.add_argument("--folder-views", nargs="+", required=True)
    parser.add_argument("--votes", type=int, default=1)
    parser.add_argument("--score", type=float, default=.10)
    parser.add_argument("--mask", type=float, default=.40)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--title", default="Fold-0 validation, final judging metrics")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compare", help="per_record.json of another run: paired bootstrap differences")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    views = [v.split(",") for v in args.folder_views]
    truth = ground_truth()
    jobs = [(Path(n).stem, recs, args.folders, views, (args.votes, args.score, args.mask)) for n, recs in sorted(truth.items())]
    rows = []
    with Pool(args.workers) as pool:
        for part in pool.imap_unordered(record_rows, jobs):
            rows += part
    summary, pair_iou, pair_dice, semantic, gt_degree, pred_degree = summarize(rows)
    summary["setting"] = {"folders": args.folders, "views": args.folder_views, "votes": args.votes,
                          "score": args.score, "mask": args.mask}
    records = [{k: v for k, v in r.items() if k != "iou"} | record_stats(r) for r in rows]
    other = json.loads(Path(args.compare).read_text()) if args.compare else None
    summary["bootstrap"] = bootstrap(records, other)
    (out / "rubric_metrics.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    (out / "per_record.json").write_text(json.dumps(records, indent=1), encoding="utf-8", newline="\n")
    plot(pair_iou, pair_dice, semantic, gt_degree, pred_degree, out, args.title)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
