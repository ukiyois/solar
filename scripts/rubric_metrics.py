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

usage: python -m scripts.rubric_metrics --folders runs/tta_s1a/val --folder-views d0,d1,d2,d3,f0,f1,f2,f3 \
           --votes 1 --score .10 --mask .40 --output-dir docs/figures/rubric_fold0
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


def summarize(rows):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", nargs="+", required=True)
    parser.add_argument("--folder-views", nargs="+", required=True)
    parser.add_argument("--votes", type=int, default=1)
    parser.add_argument("--score", type=float, default=.10)
    parser.add_argument("--mask", type=float, default=.40)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--title", default="Fold-0 validation, final judging metrics")
    parser.add_argument("--output-dir", required=True)
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
    (out / "rubric_metrics.json").write_text(json.dumps(summary, indent=1))
    (out / "per_record.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k != "iou"} | {"tp": int((r["iou"] > .5).any(1).sum()) if r["n_pred"] else 0}
         for r in rows], indent=1))
    plot(pair_iou, pair_dice, semantic, gt_degree, pred_degree, out, args.title)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
