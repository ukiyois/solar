"""Close-up morphology examples on fold 0: outline agreement, multi-part instances, missed small filaments.

Rows (chosen deterministically, evenly spaced along a sorted list, not hand-picked):
  1. matched filaments (best IoU > .5, GT area >= 1500 px) at the 20/40/60/80th percentile of IoU,
     annotator outline (yellow) against the predicted outline (cyan);
  2. predicted instances made of >= 2 parts (>= 32 px each, 8-connectivity, as in shape_statistics),
     evenly spaced by area, with the polygons of the annotator record that overlaps them most;
  3. missed small filaments (GT area < 400 px, no prediction with IoU > .5), evenly spaced by area,
     with any prediction in the window.
Predictions are the fused 8-view TTA candidates at the submission setting.

usage: python -m scripts.morphology_details --folders runs/tta_s1a/val --folder-views d0,d1,d2,d3,f0,f1,f2,f3 \
           --output docs/figures/morphology_details.png [--paper-output paper/figures/morphology_details.pdf]
--paper-output also draws the same examples as a compact single-column figure (one short tag per tile).
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mu

from scripts.fuse_tta import predict
from scripts.sweep_recall_tta import ground_truth

ROOT = Path("data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images")
# outline colors (RGB): annotator yellow, prediction cyan; every panel is drawn at DISPLAY x DISPLAY px
GT_COLOR, PRED_COLOR, DISPLAY = (255, 214, 0), (0, 200, 255), 320


def parts(rle):
    """Number of 8-connected parts with >= 32 px in a mask (counted as in shape_statistics)."""
    _, _, stats, _ = cv2.connectedComponentsWithStats(mu.decode(rle).astype(np.uint8), connectivity=8)
    return int((stats[1:, cv2.CC_STAT_AREA] >= 32).sum())


def collect(job):
    """Pool worker: the example candidates of one fold-0 image for the three rows.

    job = (stem, [(record id, GT RLEs)], folders, views, (votes, score, mask)). Returns (matched, missed,
    multipart): matched GT instances (best IoU > .5, area >= 1500 px) with their best prediction; missed small
    GT instances (best IoU <= .5, area < 400 px) with all predictions of the image; and predictions with >= 2
    parts together with the GT instances of the record that overlaps them most. GT instances are collected
    per annotator record, so one filament can appear once per record.
    """
    stem, records, folders, views, setting = job
    _, kept = predict((stem, folders, views, setting))
    predictions = [p["rle"] for p in kept]
    matched, missed, multipart = [], [], []
    for record_id, truth in records:
        if not truth:
            continue
        iou = (np.asarray(mu.iou(predictions, truth, [0] * len(truth))).reshape(len(predictions), len(truth))
               if predictions else np.zeros((0, len(truth))))
        for j, gt in enumerate(truth):
            area = float(mu.area(gt))
            best = float(iou[:, j].max()) if len(predictions) else 0.
            if best > .5 and area >= 1500:
                matched.append({"stem": stem, "gt": gt, "pred": predictions[int(iou[:, j].argmax())], "iou": best,
                                "gt_area": area})
            elif best <= .5 and area < 400:
                missed.append({"stem": stem, "gt": gt, "iou": best, "gt_area": area, "preds": predictions})
    for prediction in predictions:
        count = parts(prediction)
        if count >= 2:
            overlaps = [(float(mu.area(mu.merge([mu.merge(t), prediction], intersect=True))), t) for _, t in records if t]
            truth = max(overlaps, key=lambda o: o[0])[1] if overlaps else []
            multipart.append({"stem": stem, "pred": prediction, "parts": count, "area": float(mu.area(prediction)),
                              "truth": truth})
    return matched, missed, multipart


def spread(items, count):
    """`count` items at evenly spaced interior positions of a sorted list (both ends excluded); all if few."""
    if len(items) <= count:
        return items
    return [items[int(round(q * (len(items) - 1)))] for q in np.linspace(.0, 1., count + 2)[1:-1]]


def window(rle, pad=40, minimum=192):
    """Square window around a mask: bounding box + `pad` px per side, at least `minimum` px, inside the image.

    Returns (y0, x0, side) in pixels of the 2048 x 2048 image.
    """
    x, y, w, h = mu.toBbox(rle)
    side = int(min(2048, max(max(w, h) + 2 * pad, minimum)))
    cx, cy = x + w / 2, y + h / 2
    x0 = int(np.clip(cx - side / 2, 0, 2048 - side))
    y0 = int(np.clip(cy - side / 2, 0, 2048 - side))
    return y0, x0, side


def panel(stem, focus, outlines, widen=0):
    """Crop around `focus`, resize to DISPLAY px, draw (rle, color, thickness + widen) outlines; returns (RGB, side)."""
    image = np.asarray(Image.open(ROOT / f"{stem}.jpeg").convert("L"))
    y0, x0, side = window(focus)
    crop = cv2.resize(image[y0:y0 + side, x0:x0 + side], (DISPLAY, DISPLAY), interpolation=cv2.INTER_CUBIC)
    canvas = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    for rle, color, thickness in outlines:
        mask = mu.decode(rle)[y0:y0 + side, x0:x0 + side]
        if not mask.any():
            continue
        mask = cv2.resize(mask, (DISPLAY, DISPLAY), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, color, thickness + widen)
    return canvas, side


def draw(rows, output, paper=False):
    """Draw the 3 x 4 grid of close-ups and save it to `output`.

    rows: [(row label, short row tag, [(stem, focus RLE, outlines, title, tag), ...])]. The report version has
    a title per panel; paper=True gives a 3.5 inch single-column figure with a short tag per tile and wider
    outlines, saved at 300 dpi.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    size = (3.5, 2.72) if paper else (15.5, 12.4)
    fig, axes = plt.subplots(3, 4, figsize=size, gridspec_kw={"wspace": .03, "hspace": .03} if paper else None)
    for r, (label, short, examples) in enumerate(rows):
        for c in range(4):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(examples):
                continue
            stem, focus, outlines, title, tag = examples[c]
            canvas, side = panel(stem, focus, outlines, widen=2 if paper else 0)
            ax.imshow(canvas)
            if paper:
                ax.text(.03, .97, tag, transform=ax.transAxes, fontsize=5.5, color="white", va="top",
                        bbox={"facecolor": "black", "alpha": .55, "pad": 1., "linewidth": 0})
            else:
                ax.set_title(f"{title}\n{stem} · {side} px window", fontsize=9)
        if paper:
            axes[r, 0].text(-.06, .5, short, transform=axes[r, 0].transAxes, fontsize=6, rotation=90,
                            va="center", ha="right")
        else:
            fig.text(.006, .97 * (1 - (r + .5) / 3), label, fontsize=11, fontweight="bold", va="center",
                     rotation=90, ha="left")
    if paper:
        fig.subplots_adjust(left=.045, right=1, top=1, bottom=0)
    else:
        fig.suptitle("Fold 0, final recipe (8-view TTA, submission setting): annotator outline yellow, prediction cyan",
                     fontsize=12)
        fig.tight_layout(rect=(.025, 0, 1, .97))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300 if paper else 110)
    plt.close(fig)


def main():
    """Collect the fold-0 examples, draw the figure(s) and write the example counts to <output stem>.json."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", nargs="+", required=True)
    parser.add_argument("--folder-views", nargs="+", required=True)
    parser.add_argument("--votes", type=int, default=1)
    parser.add_argument("--score", type=float, default=.10)
    parser.add_argument("--mask", type=float, default=.40)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--paper-output", help="compact single-column version (PDF or PNG)")
    args = parser.parse_args()
    views = [v.split(",") for v in args.folder_views]
    jobs = [(Path(n).stem, recs, args.folders, views, (args.votes, args.score, args.mask))
            for n, recs in sorted(ground_truth().items())]
    with Pool(args.workers) as pool:
        results = pool.map(collect, jobs)
    matched = sorted((m for r in results for m in r[0]), key=lambda m: m["iou"])
    missed = sorted((m for r in results for m in r[1]), key=lambda m: m["gt_area"])
    multipart = sorted((m for r in results for m in r[2]), key=lambda m: m["area"])
    rows = [("Matched filaments\n(IoU quantiles .2/.4/.6/.8)", "matched",
             [(m["stem"], m["gt"], [(m["gt"], GT_COLOR, 2), (m["pred"], PRED_COLOR, 2)],
               f"IoU {m['iou']:.2f} · GT {m['gt_area']:,.0f} px → pred {mu.area(m['pred']):,.0f} px", f"IoU {m['iou']:.2f}")
              for m in [matched[int(round(q * (len(matched) - 1)))] for q in (.2, .4, .6, .8)]]),
            ("Predictions in\nseveral parts", "several parts",
             [(m["stem"], m["pred"], [*[(t, GT_COLOR, 2) for t in m["truth"]], (m["pred"], PRED_COLOR, 2)],
               f"1 prediction, {m['parts']} parts · {m['area']:,.0f} px", f"{m['parts']} parts")
              for m in spread(multipart, 4)]),
            ("Missed small\nfilaments (< 400 px)", "missed, < 400 px",
             [(m["stem"], m["gt"], [*[(p, PRED_COLOR, 2) for p in m["preds"]], (m["gt"], GT_COLOR, 2)],
               f"GT {m['gt_area']:,.0f} px · best IoU {m['iou']:.2f}", f"{m['gt_area']:,.0f} px")
              for m in spread(missed, 4)])]
    draw(rows, args.output)
    if args.paper_output:
        draw(rows, args.paper_output, paper=True)
    summary = {"matched_pairs_gt_area_ge_1500": len(matched), "multipart_predictions": len(multipart),
               "missed_small_gt_lt_400": len(missed)}
    Path(args.output).with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
