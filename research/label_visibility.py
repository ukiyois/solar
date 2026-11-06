"""Label visibility: annotator agreement and model errors against H-alpha observables (CPU).

Per annotated filament:
  contrast  mean relative darkness I/B - 1 inside the polygon, with B the on-disk background of
            solar_filaments.model.limb_flattened (normalized convolution, sigma ~32 native px). For an absorbing
            cloud above the chromosphere this is the cloud-model contrast (S/I0 - 1)(1 - exp(-tau)) < 0.
  area      native pixels;  mu = cosine of the heliocentric angle of the centroid (disk: I > .25 x 99th percentile)
Contrast and area are also profiled against mu over all annotated filaments (centre-to-limb behaviour), and the
contrast is expressed in units of the observation's quiet-Sun texture sigma (robust per-pixel spread of I/B - 1
on the inner disk), a visibility measure.
Annotator agreement (observations with several records): share of the other records that contain an instance
with IoU > .5 to it. Model status (fold 0, the fold-0 model at the submission setting): matched (IoU > .5), near
miss (best IoU in (.1, .5]) or absent (best IoU <= .1). Model vs human: the model scored against every fold-0
record and every ordered pair of records on the same observations scored the same way (Hungarian matching,
PQ at IoU thresholds .3-.6, error profile at .5). A paired bootstrap over the fold-0 observations with several
records gives 95 % intervals of both PQs at IoU > .5 and of their difference. --replot only redraws the
figure from an existing --output JSON.

usage (repository root; --folder is a fold-0 dump of scripts/dump_tta_candidates.py, see README):
  python research/label_visibility.py --folder runs/tta/<fold-0 model>/val
writes research/label_visibility.json and docs/figures/visibility.png / .pdf
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mu
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from scripts.fuse_tta import VIEWS, predict  # noqa: E402
from solar_filaments.data import physical_fold  # noqa: E402
from solar_filaments.model import limb_flattened  # noqa: E402

ROOT = REPOSITORY / "data/raw/MAGFiLO_1.0_Kaggle_2026"
SETTING = (1, .10, .40)  # votes, score, mask threshold of the submission
CONTRAST_EDGES = [-1, -.20, -.14, -.10, -.07, -.04, 1]
AREA_EDGES = [0, 500, 1500, 5000, 1e9]
MU_EDGES = [0, .5, .7, .85, 1.01]
CONTRAST_SIGMA_EDGES = [-100, 1, 2, 3, 5, 100]  # -C in units of the observation's quiet-Sun texture sigma
THRESHOLDS = (.3, .4, .5, .6)  # IoU thresholds of the model-vs-human PQ comparison


def records_by_observation():
    """{image stem: [(record id, [RLE per instance]), ...]} for every training observation (all folds), by stem."""
    data = json.loads((ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text(encoding="utf-8"))
    by_record = defaultdict(list)
    for annotation in data["annotations"]:
        rles = mu.frPyObjects(annotation["segmentation"], 2048, 2048)
        by_record[str(annotation["image_id"])].append(mu.merge(rles) if isinstance(rles, list) else rles)
    by_image = defaultdict(list)
    for record in data["images"]:
        by_image[Path(record["file_name"]).stem].append((str(record["id"]), by_record[str(record["id"])]))
    return dict(sorted(by_image.items()))


def half(a):
    """2 x 2 mean pooling of a 2048 x 2048 array to 1024 x 1024."""
    return a.reshape(1024, 2, 1024, 2).mean((1, 3))


def texture(stem):
    """Quiet-Sun texture: robust (MAD) standard deviation of I/B - 1 per 1024-px pixel on the inner disk (r < .9 R),
    on the same map as the contrast. Fibrils, mottles and noise set it; filaments cover too little to matter."""
    image = np.asarray(Image.open(ROOT / "train/train_images" / f"{stem}.jpeg").convert("L"))
    small = half(image.astype(np.float32))
    flat = (limb_flattened(torch.from_numpy(small / 255)[None, None], factor=4, sigma=4.0)[0, 0] / 4).numpy()
    disk = small > .25 * np.quantile(small[::4, ::4], .99)
    ys, xs = np.nonzero(disk)
    cy, cx, radius = ys.mean(), xs.mean(), np.sqrt(disk.sum() / np.pi)
    yy, xx = np.mgrid[:1024, :1024]
    values = flat[np.hypot(yy - cy, xx - cx) < .9 * radius]
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def observables(stem, rles):
    """Contrast, area and disk position of each instance mask of one observation; None for empty masks.

    At 1024 px (2 x 2 mean); factor 4 / sigma 4 keeps the native background scale. contrast = mean of I/B - 1
    inside the mask, area = native pixels (4 per 1024-px pixel), mu = cosine of the heliocentric angle of the
    mask centroid (disk centre and radius from the thresholded disk area).
    """
    image = np.asarray(Image.open(ROOT / "train/train_images" / f"{stem}.jpeg").convert("L"))
    small = half(image.astype(np.float32))
    flat = (limb_flattened(torch.from_numpy(small / 255)[None, None], factor=4, sigma=4.0)[0, 0] / 4).numpy()
    disk = small > .25 * np.quantile(small[::4, ::4], .99)
    ys, xs = np.nonzero(disk)
    cy, cx, radius = ys.mean(), xs.mean(), np.sqrt(disk.sum() / np.pi)
    rows = []
    for rle in rles:
        m = half(mu.decode(rle).astype(np.float32)) >= .5
        if not m.any():
            rows.append(None)
            continue
        yy, xx = np.nonzero(m)
        r = min(1., float(np.hypot(yy.mean() - cy, xx.mean() - cx) / radius))
        rows.append({"contrast": float(flat[m].mean()), "area": int(4 * m.sum()), "mu": float(np.sqrt(1 - r * r))})
    return rows


def iou_matrix(predictions, truth):
    """(n_predictions, n_truth) IoU matrix of two RLE lists; zeros when either list is empty."""
    if not predictions or not truth:
        return np.zeros((len(predictions), len(truth)))
    return np.asarray(mu.iou(predictions, truth, [0] * len(truth))).reshape(len(predictions), len(truth))


def compare(predictions, truth):
    """PQ counts at several IoU thresholds and the matched / near-miss / absent profile at .5.

    For each threshold t: (TP, matched IoU sum, FP, FN) from a one-to-one Hungarian matching of the pairs with
    IoU > t (below .5, IoU matches are no longer unique). "gt" / "pred" count the reference / predicted
    instances whose best IoU is > .5, in (.1, .5] or <= .1.
    """
    iou = iou_matrix(predictions, truth)
    row = {"pq": {}}
    for t in THRESHOLDS:
        tp, iou_sum = 0, 0.
        if iou.size:
            weights = np.where(iou > t, iou, 0.)
            r, c = linear_sum_assignment(-weights)
            hits = weights[r, c] > t
            tp, iou_sum = int(hits.sum()), float(weights[r, c][hits].sum())
        row["pq"][t] = (tp, iou_sum, len(predictions) - tp, len(truth) - tp)
    best_gt = iou.max(0) if len(predictions) else np.zeros(len(truth))
    best_pred = iou.max(1) if len(truth) else np.zeros(len(predictions))
    profile = lambda best: [int((best > .5).sum()), int(((best > .1) & (best <= .5)).sum()), int((best <= .1).sum())]
    row["gt"], row["pred"] = profile(best_gt), profile(best_pred)
    return row


def summary(pairs):
    """Pool compare() rows: PQ at every threshold and the shares of matched / near-miss / absent reference
    instances and of matched / near-miss / unsupported predictions (each share list ends with the count)."""
    out = {"pairs": len(pairs)}
    for t in THRESHOLDS:
        tp, iou, fp, fn = (sum(p["pq"][t][k] for p in pairs) for k in range(4))
        out[f"pq@{t}"] = round(iou / max(1e-9, tp + .5 * fp + .5 * fn), 4)
    gt, pred = np.sum([p["gt"] for p in pairs], 0), np.sum([p["pred"] for p in pairs], 0)
    out["reference_matched_nearmiss_absent"] = (gt / gt.sum()).round(3).tolist() + [int(gt.sum())]
    out["predicted_matched_nearmiss_unsupported"] = (pred / pred.sum()).round(3).tolist() + [int(pred.sum())]
    return out


def paired_pq_bootstrap(model_pairs, human_pairs, images, samples=4000, seed=2026):
    """95 % intervals of PQ(model vs record), PQ(record vs record) and their difference at IoU > .5,
    resampling observations (all pairs of an observation together)."""
    images = sorted(images)
    index = {image: i for i, image in enumerate(images)}

    def table(pairs):
        sums = np.zeros((len(images), 4))  # TP-IoU sum, TP, FP, FN
        for p in pairs:
            if p["image"] in index:
                tp, iou, fp, fn = p["pq"][.5]
                sums[index[p["image"]]] += (iou, tp, fp, fn)
        return sums
    draws = np.random.default_rng(seed).integers(0, len(images), (samples, len(images)))
    pq = lambda s: s[..., 0] / (s[..., 1] + .5 * s[..., 2] + .5 * s[..., 3])
    model, human = pq(table(model_pairs)[draws].sum(1)), pq(table(human_pairs)[draws].sum(1))
    interval = lambda values: [round(float(q), 4) for q in np.percentile(values, [2.5, 97.5])]
    return {"observations": len(images), "samples": samples, "model_pq_ci95": interval(model),
            "human_pq_ci95": interval(human), "difference_ci95": interval(model - human),
            "p_model_not_better": round(float(np.mean(model - human <= 0)), 4)}


def profile(rows, key, edges):
    """Per bin: count, median contrast and median area (bins [lo, hi) of row[key]; empty bins skipped)."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = [r for r in rows if lo <= r[key] < hi]
        if selected:
            out.append([f"[{lo:g},{hi:g})", len(selected), round(float(np.median([r["contrast"] for r in selected])), 3),
                        int(np.median([r["area"] for r in selected]))])
    return out


def binned(rows, key, edges, value):
    """Per non-empty bin [lo, hi) of row[key]: [label, count, value(rows of the bin)]."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = [r for r in rows if lo <= r[key] < hi]
        if selected:
            out.append([f"[{lo:g},{hi:g})", len(selected), value(selected)])
    return out


def figure(result, path):
    """Single-column (3.5 in) two-panel figure: annotator agreement and the model's matched / missed (absent)
    shares against contrast and area bins, +-1 binomial standard error; saved as <path>.png and <path>.pdf."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.linewidth": .6, "lines.linewidth": 1.1, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.88), sharey=True, gridspec_kw={"width_ratios": [6, 4]})
    # contrast ticks: centres of the inner CONTRAST_EDGES bins (outer bins open-ended); area ticks: AREA_EDGES ranges
    panels = (("contrast", ["<-.20", "-.17", "-.12", "-.085", "-.055", ">-.04"], "contrast $C = I/B - 1$"),
              ("area", ["<500", "500-1.5k", "1.5k-5k", ">5k"], "area (px)"))
    # +-1 binomial standard error sqrt(p(1-p)/n) per bin
    error = lambda p, n: np.sqrt(np.asarray(p) * (1 - np.asarray(p)) / np.asarray(n))
    for ax, (key, labels, xlabel) in zip(axes, panels):
        human, model = result[f"human_agreement_vs_{key}"], result[f"model_status_vs_{key}"]
        x = np.arange(len(human))
        series = (([row[2] for row in human], [row[1] for row in human], "-o", "black", "annotator agreement"),
                  ([row[2][0] for row in model], [row[1] for row in model], "-s", "tab:blue", "model: matched"),
                  ([row[2][2] for row in model], [row[1] for row in model], "--^", "tab:orange", "model: missed"))
        for values, counts, style, color, label in series:
            ax.errorbar(x, values, yerr=error(values, counts), fmt=style, color=color, ms=2.5, capsize=1.5,
                        elinewidth=.6, capthick=.6, label=label)
        ax.set_xticks(x, labels, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1)
        ax.grid(alpha=.3, linewidth=.4)
    axes[0].set_ylabel("fraction of filaments")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=6, frameon=False,
               handlelength=1.8, columnspacing=1.0)
    fig.tight_layout(pad=.3, w_pad=.5, rect=(0, 0, 1, .91))
    for suffix in (".png", ".pdf"):
        fig.savefig(str(path) + suffix, dpi=200)
    plt.close(fig)


def main():
    """Measure every annotated instance, score the model and the annotator pairs, write the JSON and the figure."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--folder", help="fold-0 dump of scripts/dump_tta_candidates.py (required unless --replot)")
    parser.add_argument("--output", default=str(REPOSITORY / "research/label_visibility.json"))
    parser.add_argument("--figure", default=str(REPOSITORY / "docs/figures/visibility"), help="path without suffix")
    parser.add_argument("--replot", action="store_true", help="only redraw the figure from an existing --output")
    args = parser.parse_args()
    if args.replot:
        return figure(json.loads(Path(args.output).read_text()), args.figure)
    if not args.folder:
        parser.error("--folder is required")
    torch.set_num_threads(1)
    observations = records_by_observation()
    fold0 = {stem for stem in observations if physical_fold(f"{stem}.jpeg", 5, 2026) == 0}

    all_rows, human_rows, human_pairs, sigma = [], [], [], {}
    multi = {stem for stem, records in observations.items() if len(records) > 1}
    for i, (stem, records) in enumerate(observations.items()):
        sigma[stem] = texture(stem)
        for a, (_, truth) in enumerate(records):
            others = [t for b, (_, t) in enumerate(records) if b != a]
            human_pairs += [{"image": stem, **compare(truth, t)} for t in others]
            for rle, row in zip(truth, observables(stem, truth)):
                if row is None:
                    continue
                row["contrast_sigma"] = -row["contrast"] / sigma[stem]
                all_rows.append(row)
                if others:  # agreement needs another record of the same observation
                    agree = [float(len(o) > 0 and iou_matrix([rle], o).max() > .5) for o in others]
                    human_rows.append({**row, "agreement": float(np.mean(agree))})
        if (i + 1) % 100 == 0:
            print(f"observations {i + 1}/{len(observations)}", flush=True)

    model_rows, model_pairs = [], []
    for i, stem in enumerate(sorted(fold0)):
        records = observations[stem]
        _, kept = predict((stem, [args.folder], [VIEWS], SETTING))
        predictions = [p["rle"] for p in kept]
        for _, truth in records:
            model_pairs.append({"image": stem, "records": len(records), **compare(predictions, truth)})
            if not truth:
                continue
            best = iou_matrix(predictions, truth).max(0) if predictions else np.zeros(len(truth))
            for b, row in zip(best, observables(stem, truth)):
                if row is not None:
                    status = "matched" if b > .5 else "near" if b > .1 else "absent"
                    model_rows.append({**row, "status": status, "contrast_sigma": -row["contrast"] / sigma[stem]})
        if (i + 1) % 40 == 0:
            print(f"fold-0 observations {i + 1}/{len(fold0)}", flush=True)

    agreement = lambda rows: round(float(np.mean([r["agreement"] for r in rows])), 3)
    status = lambda rows: [round(float(np.mean([r["status"] == k for r in rows])), 3) for k in ("matched", "near", "absent")]
    multi0 = {p["image"] for p in model_pairs if p["records"] > 1}
    c, g = np.array([r["contrast"] for r in human_rows]), np.array([r["agreement"] for r in human_rows])
    a = np.log([r["area"] for r in human_rows])
    absent = np.array([r["status"] == "absent" for r in model_rows])
    result = {
        "counts": {"annotated_instances": len(all_rows), "multi_record_observations": len(multi),
                   "multi_record_instances": len(human_rows),
                   "fold0_observations": len(fold0), "fold0_instances": len(model_rows),
                   "median_contrast_annotated": round(float(np.median(c)), 3)},
        "human_agreement_vs_contrast": binned(human_rows, "contrast", CONTRAST_EDGES, agreement),
        "human_agreement_vs_area": binned(human_rows, "area", AREA_EDGES, agreement),
        "human_agreement_vs_mu": binned(human_rows, "mu", MU_EDGES, agreement),
        "annotated_count_contrast_area_vs_mu": profile(all_rows, "mu", MU_EDGES),
        "quiet_sun_texture_sigma_p10_p50_p90": np.quantile(list(sigma.values()), [.1, .5, .9]).round(4).tolist(),
        "annotated_share_within_1_2_3_sigma": [round(float(np.mean([r["contrast_sigma"] < k for r in all_rows])), 3)
                                               for k in (1, 2, 3)],
        "human_agreement_vs_contrast_sigma": binned(human_rows, "contrast_sigma", CONTRAST_SIGMA_EDGES, agreement),
        "model_status_vs_contrast_sigma": binned(model_rows, "contrast_sigma", CONTRAST_SIGMA_EDGES, status),
        "model_status_vs_contrast": binned(model_rows, "contrast", CONTRAST_EDGES, status),
        "model_status_vs_area": binned(model_rows, "area", AREA_EDGES, status),
        "model_status_vs_mu": binned(model_rows, "mu", MU_EDGES, status),
        "spearman_agreement": {"contrast": round(float(spearmanr(c, g)[0]), 3), "log_area": round(float(spearmanr(a, g)[0]), 3),
                               "contrast_vs_log_area": round(float(spearmanr(c, a)[0]), 3)},
        "spearman_model_absent": {"contrast": round(float(spearmanr([r["contrast"] for r in model_rows], absent)[0]), 3),
                                  "log_area": round(float(spearmanr(np.log([r["area"] for r in model_rows]), absent)[0]), 3)},
        "model_vs_fold0_records": summary(model_pairs),
        "model_vs_fold0_records_multi_record_observations": summary([p for p in model_pairs if p["image"] in multi0]),
        "human_vs_human_fold0_multi_record_observations": summary([p for p in human_pairs if p["image"] in fold0]),
        "human_vs_human_all_multi_record_observations": summary(human_pairs),
        "model_vs_human_bootstrap": paired_pq_bootstrap([p for p in model_pairs if p["image"] in multi0],
                                                        [p for p in human_pairs if p["image"] in fold0], multi0),
        "status_legend": "matched / near miss / absent shares of the reference instances in each bin",
    }
    Path(args.output).write_text(json.dumps(result, indent=1), encoding="utf-8", newline="\n")
    figure(result, args.figure)
    print(json.dumps({k: result[k] for k in ("counts", "model_vs_fold0_records_multi_record_observations",
                                             "human_vs_human_fold0_multi_record_observations")}, indent=1))


if __name__ == "__main__":
    main()
