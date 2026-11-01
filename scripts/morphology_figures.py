"""Morphology figures for the report: H-alpha image, annotator ground truth and predicted instances.

fold-0 mode: predictions are fused TTA candidates (submission setting) and every annotator record of the
image is drawn; images are chosen from rubric_metrics' per_record.json (best / median / hardest by
semantic Dice among images with several filaments) unless --stems is given.
test mode (--csv): predictions are read from a submission CSV (no ground truth exists).

Each figure: top row full disk (image | ground truth | prediction), bottom row the same 512 px window
around the largest predicted filament, so thin spines, barbs and boundaries are visible.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mu

ROOT = Path("data/raw/MAGFiLO_1.0_Kaggle_2026")
COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
          (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255)]


def draw(image, masks, styles):
    """RGB overlay: each mask's outline (thickness 3 at 2048 px) in its color."""
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for mask, color in zip(masks, styles):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(canvas, contours, -1, color, 3)
    return canvas


def window(masks, size=512, shape=(2048, 2048)):
    if not masks:
        return slice(768, 1280), slice(768, 1280)
    largest = max(masks, key=lambda m: m.sum())
    ys, xs = np.where(largest)
    cy, cx = int(ys.mean()), int(xs.mean())
    y0 = int(np.clip(cy - size // 2, 0, shape[0] - size))
    x0 = int(np.clip(cx - size // 2, 0, shape[1] - size))
    return slice(y0, y0 + size), slice(x0, x0 + size)


def figure(stem, image, truth_records, predictions, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gt_masks, gt_colors = [], []
    for k, (_, rles) in enumerate(truth_records):
        for rle in rles:
            gt_masks.append(mu.decode(rle).astype(bool))
            gt_colors.append([(255, 255, 0), (0, 255, 255), (255, 0, 255)][k % 3])
    pred_masks = [mu.decode(r).astype(bool) for r in predictions]
    pred_colors = [COLORS[i % len(COLORS)] for i in range(len(pred_masks))]
    rows = [(slice(None), slice(None)), window(pred_masks or gt_masks)]
    has_truth = bool(truth_records)
    columns = 3 if has_truth else 2
    fig, axes = plt.subplots(2, columns, figsize=(6 * columns, 12))
    overlays = [cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)]
    if has_truth:
        overlays.append(draw(image, gt_masks, gt_colors))
    overlays.append(draw(image, pred_masks, pred_colors))
    names = ["H-alpha observation"] + ([f"Ground truth ({len(truth_records)} annotator record(s): "
                                         "yellow / cyan / magenta)"] if has_truth else []) + \
            [f"Prediction ({len(pred_masks)} instances, one color each)"]
    for r, (ys, xs) in enumerate(rows):
        for c, (overlay, name) in enumerate(zip(overlays, names)):
            ax = axes[r, c]
            ax.imshow(overlay[ys, xs] if r else cv2.resize(overlay, (1024, 1024), interpolation=cv2.INTER_AREA))
            ax.set_title(name if r == 0 else f"512 px window around the largest filament", fontsize=10)
            ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out / f"morphology_{stem}.png", dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", nargs="*", default=[])
    parser.add_argument("--folder-views", nargs="*", default=[])
    parser.add_argument("--votes", type=int, default=1)
    parser.add_argument("--score", type=float, default=.10)
    parser.add_argument("--mask", type=float, default=.40)
    parser.add_argument("--per-record", help="rubric_metrics per_record.json, to choose fold-0 images")
    parser.add_argument("--stems", nargs="*")
    parser.add_argument("--csv", help="submission CSV (test mode)")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--label", default="")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.csv:
        by_image = defaultdict(list)
        with open(args.csv, newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                stem = row["filament_id"].rsplit("_", 1)[0]
                by_image[stem].append({"size": [2048, 2048], "counts": row["segmentation_rle"].encode("ascii")})
        # Default: the busiest, a median and the sparsest predicted observation (a spread of cases).
        ranked = sorted(by_image, key=lambda s: (-len(by_image[s]), s))
        stems = args.stems or [ranked[0], ranked[len(ranked) // 2], ranked[-1]][:args.count]
        tags = [""] * len(stems) if args.stems else ["most instances", "median instance count", "fewest instances"]
        for stem, tag in zip(stems, tags):
            path = next((ROOT / "test/test_images").glob(f"{stem}.*"))
            image = np.asarray(Image.open(path).convert("L"))
            figure(stem, image, [], by_image[stem], out,
                   f"Test observation {stem} ({tag}){': ' + args.label if args.label else ''}")
            print("wrote", out / f"morphology_{stem}.png")
        return
    from scripts.fuse_tta import predict
    from scripts.sweep_recall_tta import ground_truth
    truth = {Path(n).stem: recs for n, recs in ground_truth().items()}
    if args.stems:
        stems = args.stems
    else:
        rows = json.loads(Path(args.per_record).read_text())
        per_image = defaultdict(list)
        for r in rows:
            if r["n_gt"] >= 4:
                per_image[r["image"]].append(r["semantic_dice"])
        ranked = sorted(per_image, key=lambda s: -np.mean(per_image[s]))
        stems = [ranked[0], ranked[len(ranked) // 2], ranked[-1]][:args.count]
    views = [v.split(",") for v in args.folder_views]
    for stem, tag in zip(stems, ["best", "median", "hardest"] if not args.stems else [""] * len(stems)):
        _, kept = predict((stem, args.folders, views, (args.votes, args.score, args.mask)))
        image = np.asarray(Image.open(ROOT / "train/train_images" / f"{stem}.jpeg").convert("L"))
        figure(stem, image, truth[stem], [p["rle"] for p in kept], out,
               f"Fold-0 validation {stem} ({tag} semantic Dice among images with >= 4 filaments)"
               f"{': ' + args.label if args.label else ''}")
        print("wrote", out / f"morphology_{stem}.png")


if __name__ == "__main__":
    main()
