"""Quantitative morphology of filament masks: annotations vs predictions (CPU).

Per instance: area, number of connected parts (>= 32 px), elongation (length / width of the minimum-area
rectangle of the largest part), maximum inscribed width (twice the largest distance-transform value),
solidity (area / convex-hull area of the largest part; low = barbed or curved) and enclosed holes.
Populations: fold-0 annotations (every annotator record), fold-0 predictions of the TTA candidates
(submission setting) and the predictions of a submission CSV on the test images.

usage: python -m scripts.shape_statistics --folders runs/tta_s1a/val --folder-views d0,d1,d2,d3,f0,f1,f2,f3 \
           --csv deliverables/submission_single_ffinal_a_m40.csv --output-dir docs/figures/shape
"""
import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mu

from scripts.fuse_tta import predict
from scripts.sweep_recall_tta import ground_truth

FEATURES = ("area", "parts", "elongation", "width", "solidity", "holes")


def describe(rle):
    """Shape descriptors of one instance mask (pycocotools RLE), as defined in the module docstring.

    area counts every pixel and parts the 8-connected components with >= 32 px; the other descriptors use
    the largest component: elongation = long / short side of the minimum-area rectangle of its outer contour
    (short side clamped to >= 1 px), width from the L2 distance transform, solidity against the convex hull of
    the outer contour, and holes = inner contours of the two-level contour hierarchy.
    """
    mask = mu.decode(rle).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sizes = stats[1:, cv2.CC_STAT_AREA]
    largest = (labels == 1 + int(np.argmax(sizes))).astype(np.uint8)
    contours, hierarchy = cv2.findContours(largest, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    outer = max((c for c, h in zip(contours, hierarchy[0]) if h[3] < 0), key=cv2.contourArea)
    _, (w, h), _ = cv2.minAreaRect(outer)
    hull = cv2.contourArea(cv2.convexHull(outer))
    return {"area": int(mask.sum()), "parts": int((sizes >= 32).sum()),
            "elongation": max(w, h) / max(1., min(w, h)),
            "width": 2 * float(cv2.distanceTransform(largest, cv2.DIST_L2, 5).max()),
            "solidity": float(largest.sum()) / max(1., hull),
            "holes": int((hierarchy[0][:, 3] >= 0).sum())}


def fold0_job(job):
    """Pool worker for one fold-0 image: (descriptors of every GT instance of every record, of the predictions).

    The predictions follow the submission rule and are counted once per image, the annotations once per record.
    """
    stem, records, folders, views, setting = job
    _, kept = predict((stem, folders, views, setting))
    truth = [describe(rle) for _, rles in records for rle in rles]
    return truth, [describe(p["rle"]) for p in kept]


def csv_job(counts):
    """Descriptors of one submission instance, given its RLE counts string (2048 x 2048 mask)."""
    return describe({"size": [2048, 2048], "counts": counts.encode("ascii")})


def summarize(rows):
    """Mean and 10/50/90 % quantiles of every descriptor, plus the shares of multi-part and of holed instances."""
    table = {"instances": len(rows)}
    for key in FEATURES:
        values = np.asarray([r[key] for r in rows], dtype=float)
        table[key] = {"mean": round(float(values.mean()), 3),
                      "quantiles_10_50_90": [round(float(q), 2) for q in np.quantile(values, [.1, .5, .9])]}
    table["multi_part_fraction"] = round(float(np.mean([r["parts"] > 1 for r in rows])), 4)
    table["with_holes_fraction"] = round(float(np.mean([r["holes"] > 0 for r in rows])), 4)
    return table


def plot(populations, out):
    """Save <out>/shape_statistics.png: normalized histograms of area, width, elongation and solidity per
    population, and the shares of instances with one part, several parts and holes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("area", "Area (px, log scale)", np.logspace(2, 4.7, 40), True),
              ("width", "Maximum inscribed width (px)", np.linspace(0, 60, 41), False),
              ("elongation", "Elongation (length / width)", np.linspace(1, 14, 40), False),
              ("solidity", "Solidity (area / convex hull)", np.linspace(0, 1, 41), False)]
    colors = {"fold-0 annotations": "black", "fold-0 predictions": "tab:blue", "test predictions": "tab:orange"}
    fig, axes = plt.subplots(1, 5, figsize=(26, 4.6))
    for ax, (key, label, bins, log) in zip(axes, panels):
        for name, rows in populations.items():
            values = np.asarray([r[key] for r in rows], dtype=float)
            weights = np.full(len(values), 1 / max(1, len(values)))
            ax.hist(values, bins=bins, weights=weights, histtype="step", linewidth=1.8, color=colors[name],
                    label=f"{name} (median {np.median(values):.3g})")
        if log:
            ax.set_xscale("log")
        ax.set_xlabel(label)
        ax.set_ylabel("Fraction of instances")
        ax.grid(axis="y", linestyle="--")
        ax.legend(fontsize=7)
    ax = axes[4]
    names = list(populations)
    for i, name in enumerate(names):
        rows = populations[name]
        fractions = [np.mean([r["parts"] == 1 for r in rows]), np.mean([r["parts"] > 1 for r in rows]),
                     np.mean([r["holes"] > 0 for r in rows])]
        ax.bar(np.arange(3) + (i - 1) * .27, fractions, width=.27, color=colors[name], label=name)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["one part", "several parts", "has holes"])
    ax.set_ylabel("Fraction of instances")
    ax.legend(fontsize=7)
    ax.grid(axis="y", linestyle="--")
    fig.suptitle("Filament shape statistics: annotations vs predictions (fold-0 twin of the final recipe; final-A on test)")
    fig.tight_layout()
    fig.savefig(out / "shape_statistics.png", dpi=120)
    plt.close(fig)


def main():
    """Describe the three populations, write shape_statistics.json and draw the figure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", nargs="+", required=True)
    parser.add_argument("--folder-views", nargs="+", required=True)
    parser.add_argument("--votes", type=int, default=1)
    parser.add_argument("--score", type=float, default=.10)
    parser.add_argument("--mask", type=float, default=.40)
    parser.add_argument("--csv", required=True, help="test submission CSV")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    views = [v.split(",") for v in args.folder_views]
    jobs = [(Path(n).stem, recs, args.folders, views, (args.votes, args.score, args.mask))
            for n, recs in sorted(ground_truth().items())]
    with open(args.csv, newline="", encoding="utf-8") as stream:
        test_rles = [row["segmentation_rle"] for row in csv.DictReader(stream)]
    with Pool(args.workers) as pool:
        fold0 = pool.map(fold0_job, jobs)
        test = pool.map(csv_job, test_rles, chunksize=16)
    populations = {"fold-0 annotations": [r for truth, _ in fold0 for r in truth],
                   "fold-0 predictions": [r for _, kept in fold0 for r in kept],
                   "test predictions": test}
    summary = {name: summarize(rows) for name, rows in populations.items()}
    (out / "shape_statistics.json").write_text(json.dumps(summary, indent=1))
    plot(populations, out)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
