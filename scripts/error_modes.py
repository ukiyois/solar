"""Where does the single-model TTA prediction lose PQ? (fold 0, CPU)

For the submission setting, every FN ground-truth instance and every FP prediction is
assigned to a failure mode:
  FN: near-miss   - a kept prediction overlaps it with 0.1 < IoU <= 0.5 (extent / split-merge)
      low-score   - not near-missed, but a fused candidate below the selection rule has IoU > 0.5
      no-candidate- no kept prediction with IoU > 0.1 and no fused candidate with IoU > 0.5
  FP: near-miss   - overlaps a GT instance of that record with 0.1 < IoU <= 0.5
      unlabeled   - no GT instance of that record with IoU > 0.1 (annotator did not draw it / hallucination)
FN instances on multi-annotator images are also tagged by agreement: the fraction of the
image's other records that contain an instance with IoU > 0.5 to it (minority-drawn = < .5).
The quantiles of the matched (TP) IoUs measure the SQ headroom.

Output JSON: TP / FP / FN, PQ and SQ from these counts, TP IoU quantiles and every count: fn_*, fp_* and the
nm_* details of the FN near misses (mean ratios = nm_area_ratio_sum / fn_near_miss and
nm_<kind>_*_ratio_sum / nm_<kind>_n).

usage: python -m scripts.error_modes --folders runs/tta/<name>/val --folder-views d0,d1,d2,d3,f0,f1,f2,f3 \
           --output docs/analysis/error_modes_fold0.json
"""
import argparse
import json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from scripts.fuse_tta import pooled_candidates
from scripts.sweep_recall_tta import full_rle, fuse, ground_truth, ownership


def candidate_rles(fused, mask_threshold, min_score=.02):
    """Full-image RLEs of the fused candidates before selection: score >= min_score, >= 32 px at mask_threshold.

    They decide whether a missed GT instance had a correct candidate ("low-score") or none ("no-candidate").
    """
    out = []
    for c in fused:
        if c["score"] < min_score:
            continue
        fg = (c["prob"] >= mask_threshold).astype(np.uint8)
        if fg.sum() >= 32:
            out.append(full_rle(fg, c["box"]))
    return out


def run(job):
    """Pool worker: failure-mode counts and matched IoUs of one fold-0 image.

    job = (stem, [(record id, GT RLEs)], folders, views, votes, score, mask); predictions follow the submission
    rule and every record is scored against them separately. An FN near miss is also described by its
    best-overlapping prediction: too_small (precision > .7, recall < .6), too_big (recall > .7, precision < .6)
    or shifted_or_both; whether 2+ predictions or 2+ GT instances are involved (IoU > .05); and the length and
    width ratios prediction / GT (length = skeleton pixels in the joint bounding box, width = area / length).
    FN area classes: small < 800 px, medium < 3000 px, large. Returns (Counter, list of TP IoUs).
    """
    stem, records, folders, views, votes, score, mask = job
    crops, voters = pooled_candidates(stem, folders, views)
    fused = fuse(crops, voters, "members")
    preds = [p["rle"] for p in ownership(fused, score, mask, min_votes=votes)]
    cands = candidate_rles(fused, mask)
    stats = Counter()
    tp_ious = []
    truths = [truth for _, truth in records]
    for r, truth in enumerate(truths):
        if not truth:
            stats["fp_unlabeled"] += len(preds)
            continue
        iou = np.asarray(mu.iou(preds, truth, [0] * len(truth))) if preds else np.zeros((0, len(truth)))
        ciou = np.asarray(mu.iou(cands, truth, [0] * len(truth))) if cands else np.zeros((0, len(truth)))
        matched_gt = (iou > .5).any(0) if len(preds) else np.zeros(len(truth), bool)
        matched_pred = (iou > .5).any(1) if len(preds) else np.zeros(0, bool)
        if len(preds):
            tp_ious += iou[iou > .5].tolist()
        others = [t for k, t in enumerate(truths) if k != r and t]
        for g in np.where(~matched_gt)[0]:
            best_pred = iou[:, g].max() if len(preds) else 0.
            if best_pred > .1:
                # near miss: direction of the extent error and split structure
                k = int(iou[:, g].argmax())
                inter = float(mu.area(mu.merge([preds[k], truth[g]], intersect=True)))
                p_area, g_area = float(mu.area(preds[k])), float(mu.area(truth[g]))
                precision, recall = inter / max(p_area, 1), inter / max(g_area, 1)
                kind = ("too_small" if precision > .7 and recall < .6 else "too_big" if recall > .7 and precision < .6
                        else "shifted_or_both")
                stats[f"nm_{kind}"] += 1
                pieces = int((iou[:, g] > .05).sum())
                stats["nm_split_2plus_preds" if pieces >= 2 else "nm_single_pred"] += 1
                gts_hit = int((iou[k] > .05).sum())
                stats["nm_pred_spans_2plus_gt" if gts_hit >= 2 else "nm_pred_spans_1_gt"] += 1
                stats["nm_area_ratio_sum"] += p_area / max(g_area, 1)
                # length vs width: skeleton length of each mask inside their joint bounding box
                from skimage.morphology import skeletonize
                pm, gm = mu.decode(preds[k]).astype(bool), mu.decode(truth[g]).astype(bool)
                ys, xs = np.where(pm | gm)
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                pl = max(int(skeletonize(pm[y0:y1, x0:x1]).sum()), 1)
                gl = max(int(skeletonize(gm[y0:y1, x0:x1]).sum()), 1)
                stats[f"nm_{kind}_len_ratio_sum"] += pl / gl
                stats[f"nm_{kind}_width_ratio_sum"] += (p_area / pl) / max(g_area / gl, 1e-6)
                stats[f"nm_{kind}_n"] += 1
            best_cand = ciou[:, g].max() if len(cands) else 0.
            mode = "near_miss" if best_pred > .1 else ("low_score" if best_cand > .5 else "no_candidate")
            stats[f"fn_{mode}"] += 1
            area = float(mu.area(truth[g]))
            stats[f"fn_{mode}_area_{'small' if area < 800 else 'medium' if area < 3000 else 'large'}"] += 1
            if others:
                agree = np.mean([float(np.asarray(mu.iou([truth[g]], t, [0] * len(t))).max() > .5) for t in others])
                stats[f"fn_{mode}_{'minority' if agree < .5 else 'majority'}"] += 1
        for p in np.where(~matched_pred)[0]:
            stats["fp_near_miss" if iou[p].max() > .1 else "fp_unlabeled"] += 1
        stats["tp"] += int(matched_pred.sum())
    return stats, tp_ious


def main():
    """Count the failure modes over all fold-0 images and write the JSON summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", nargs="+", required=True)
    parser.add_argument("--folder-views", nargs="+", required=True)
    parser.add_argument("--votes", type=int, default=1)
    parser.add_argument("--score", type=float, default=.10)
    parser.add_argument("--mask", type=float, default=.40)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    views = [v.split(",") for v in args.folder_views]
    truth = ground_truth()
    jobs = [(Path(n).stem, recs, args.folders, views, args.votes, args.score, args.mask) for n, recs in sorted(truth.items())]
    total, ious = Counter(), []
    with Pool(args.workers) as pool:
        for stats, tp in pool.imap_unordered(run, jobs):
            total.update(stats)
            ious += tp
    ious = np.array(ious)
    fn = sum(v for k, v in total.items() if k in ("fn_near_miss", "fn_low_score", "fn_no_candidate"))
    fp = total["fp_near_miss"] + total["fp_unlabeled"]
    summary = {"tp": total["tp"], "fp": fp, "fn": fn,
               "pq": float(ious.sum() / (total["tp"] + .5 * fp + .5 * fn)) if len(ious) else 0,
               "sq": float(ious.mean()) if len(ious) else 0,
               "tp_iou_quantiles": [round(float(q), 3) for q in np.quantile(ious, [.1, .25, .5, .75, .9])] if len(ious) else [],
               "tp_iou_below_0.6": int((ious < .6).sum()),
               "counts": dict(sorted(total.items()))}
    Path(args.output).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
