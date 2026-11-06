"""CPU post-processing over dumped query crops: TTA instance fusion, threshold
sweeps, miss attribution, and test submission export.

Input: a folder with one <image stem>.npz per image from scripts/dump_tta_candidates.py (FP16 probability
crops of the candidate queries of every dihedral view, see load()). All images are 2048 x 2048 px.

Ownership follows solar_filaments.inference.instance_predictions exactly
(score-ordered pixel competition on probability*score, min area, retained
fraction). Fusion clusters candidates from different dihedral views by binary
IoU, then averages their probability crops.

The default mode sweeps view sets x fusion modes x votes x score / mask / retain thresholds over the fold-0
validation images, scores every annotator record with the official PQ and attributes the misses of the
single-view baseline (view d0, score .15, mask .725). --test exports a submission CSV with the first value
of every option. fuse() and ownership() are shared with scripts/fuse_tta.py, which wrote the final submission.

usage (repository root): python -m scripts.sweep_recall_tta --candidates runs/tta/<name>/val --output <sweep.json>
"""
import argparse
from collections import Counter, defaultdict
import csv
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pycocotools import mask as mu

from solar_filaments.data import FilamentDataset
from solar_filaments.metrics import aggregate_pq, official_pq

SIZE = (2048, 2048)  # every competition image (and every archive frame) is 2048 x 2048 px
# View sets compared by the sweep: d<k> = rotation by k * 90 degrees, f<k> = flipped and rotated.
SETS = {
    "single": ["d0"],
    "flip4": ["d0", "f0", "d2", "f2"],
    "rot4": ["d0", "d1", "d2", "d3"],
    "dihedral8": ["d0", "d1", "d2", "d3", "f0", "f1", "f2", "f3"],
}


def full_rle(binary, box):
    """COCO RLE of a binary crop pasted into an empty 2048 x 2048 canvas at box = (y0, y1, x0, x1).

    Box ends are exclusive. Returns the pycocotools dict {"size": [2048, 2048], "counts": bytes}.
    """
    y0, y1, x0, x1 = box
    canvas = np.zeros(SIZE, dtype=np.uint8)
    canvas[y0:y1, x0:x1] = binary
    rle = mu.encode(np.asfortranarray(canvas))
    return rle


def load(path):
    """Read one candidate dump (<image stem>.npz) of scripts/dump_tta_candidates.py.

    The archive holds `meta`, a JSON list with one entry per stored query, and one FP16 probability crop per
    entry with a box, stored under the entry's key "<view>_<i>". Returns (crops, binaries). crops: dicts
    {variant (view name), query, score, box = (y0, y1, x0, x1) in full-image pixels, prob (float32 crop)}.
    binaries: the low-score d0 queries kept only as a binary mask, dicts {query, score, rle} with a
    pycocotools RLE; they are used for miss attribution only.
    """
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    crops, binaries = [], []
    for m in meta:
        if "rle" in m:
            binaries.append({"query": m["query"], "score": m["score"],
                             "rle": {"size": list(SIZE), "counts": m["rle"].encode("ascii")}})
        else:
            crops.append({"variant": m["variant"], "query": m["query"], "score": m["score"],
                          "box": tuple(m["box"]), "prob": z[m["key"]].astype(np.float32)})
    return crops, binaries


def fuse(crops, variants, mode, cluster_threshold=.5, cluster_iou=.5):
    """Fuse the candidates of several views (voters) into one candidate list, highest score first.

    Candidates of the views in `variants` are visited in descending score order. Each one joins the cluster
    whose representative (its first, highest-scoring member) overlaps it most, with IoU > cluster_iou between
    the masks binarized at probability cluster_threshold, unless that cluster already has a member from the
    same view; then the next-best cluster above the IoU threshold is tried, otherwise a new cluster starts.
    So each view votes at most once per cluster. The fused probability is the member sum on the union box
    divided by the number of views (mode "all": absent views count as zero) or by the number of members
    (mode "members": mean over the members). The fused score is the summed member score divided by the number
    of views in both modes, which down-weights instances found by few views; "votes" is the member count.
    With a single view the candidates are only filtered and sorted (no "votes" key).
    """
    pool = sorted((c for c in crops if c["variant"] in variants), key=lambda c: -c["score"])
    if len(variants) == 1:
        return pool
    rles = [full_rle((c["prob"] >= cluster_threshold).astype(np.uint8), c["box"]) for c in pool]
    clusters = []  # [representative index, members]
    for i, c in enumerate(pool):
        if not clusters:
            clusters.append([i, [i]])
            continue
        reps = [rles[k[0]] for k in clusters]
        ious = np.asarray(mu.iou([rles[i]], reps, [0] * len(reps)))[0]
        best = None
        for j in np.argsort(-ious):
            if ious[j] <= cluster_iou:
                break
            if all(pool[m]["variant"] != c["variant"] for m in clusters[j][1]):
                best = j
                break
        if best is None:
            clusters.append([i, [i]])
        else:
            clusters[best][1].append(i)
    n = len(variants)
    fused = []
    for _, members in clusters:
        boxes = np.array([pool[m]["box"] for m in members])
        y0, x0 = boxes[:, 0].min(), boxes[:, 2].min()
        y1, x1 = boxes[:, 1].max(), boxes[:, 3].max()
        total = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        for m in members:
            a, b, c_, d = pool[m]["box"]
            total[a - y0:b - y0, c_ - x0:d - x0] += pool[m]["prob"]
        scores = sum(pool[m]["score"] for m in members)
        denominator = n if mode == "all" else len(members)
        fused.append({"variant": "fused", "score": scores / n, "box": (int(y0), int(y1), int(x0), int(x1)),
                      "prob": total / denominator, "votes": len(members)})
    return sorted(fused, key=lambda c: -c["score"])


def ownership(candidates, score_threshold, mask_threshold, min_area=32, retain=.5, min_votes=1):
    """Select instances and resolve overlaps by pixel competition (the rule of instance_predictions).

    `candidates` must be sorted by descending score, as fuse() returns them. A candidate takes part if its
    score >= score_threshold and votes >= min_votes, its mask binarized at mask_threshold has >= min_area
    pixels, and its score times the mean foreground probability still passes score_threshold. Each pixel goes
    to the candidate with the highest probability x rescored score (ties keep the earlier candidate). A
    candidate then keeps only the pixels it won and is dropped if fewer than min_area pixels or less than
    `retain` of its binarized area remain. Returns non-overlapping instances in candidate order as dicts
    {rle (full-image pycocotools RLE), score (rescored), area (retained pixels)}.
    """
    winning = np.zeros(SIZE, dtype=np.float32)
    labels = np.zeros(SIZE, dtype=np.int32)
    kept = {}
    for index, c in enumerate(candidates):
        if c["score"] < score_threshold or c.get("votes", 1) < min_votes:
            continue
        foreground = c["prob"] >= mask_threshold
        area = int(foreground.sum())
        if area < min_area:
            continue
        score = c["score"] * float(c["prob"][foreground].mean())
        if score < score_threshold:
            continue
        y0, y1, x0, x1 = c["box"]
        confidence = c["prob"] * score
        # numpy views: writing into the windows updates the full-image canvases in place
        window_w, window_l = winning[y0:y1, x0:x1], labels[y0:y1, x0:x1]
        replace = foreground & (confidence > window_w)
        window_w[replace] = confidence[replace]
        window_l[replace] = index + 1
        kept[index + 1] = (area, score, c["box"])
    result = []
    for label, (area, score, box) in kept.items():
        y0, y1, x0, x1 = box
        mask = labels[y0:y1, x0:x1] == label
        retained = int(mask.sum())
        if retained < min_area or retained / area < retain:
            continue
        result.append({"rle": full_rle(mask.astype(np.uint8), box), "score": score, "area": retained})
    return result


def ground_truth():
    """Fold-0 validation labels as {file name: [(record id, [RLE per annotated filament]), ...]}.

    Fold 0 of the 5 physical-image folds with seed 2026, the training default. Each annotator record is one
    entry, and several records can share one observation (file). The polygons of an instance are merged into
    one pycocotools RLE. The data root is relative, so run from the repository root.
    """
    dataset = FilamentDataset("data/raw/MAGFiLO_1.0_Kaggle_2026", "val", 0, 5, 2026,
                              geometry=False, image_uint8=True, decode_masks=False)
    by_file = defaultdict(list)
    for record in dataset.records:
        truth = []
        for annotation in dataset.annotations[str(record["id"])]:
            rles = mu.frPyObjects(annotation["segmentation"], record["height"], record["width"])
            truth.append(mu.merge(rles) if isinstance(rles, list) else rles)
        by_file[record["file_name"]].append((str(record["id"]), truth))
    return by_file


def configurations(args):
    """Expand the option grid into (set, mode, min_votes, score, mask, retain) tuples.

    A single-view set is evaluated without fusion (mode "single") and with one vote only.
    """
    configs = []
    for name in args.sets:
        modes = ["single"] if len(SETS[name]) == 1 else args.modes
        for mode, votes, score, mask, retain in itertools.product(
                modes, args.min_votes, args.scores, args.masks, args.retain):
            if mode == "single" and votes > 1:
                continue
            configs.append((name, mode, votes, score, mask, retain))
    return configs


def evaluate_image(job):
    """Pool worker: score every configuration on one validation image.

    job = (dump path, [(record id, GT RLEs)], configurations). Fusion runs once per (set, mode) and is reused
    for all thresholds. Returns ({configuration: [official_pq row per record]}, attribute() cases).
    """
    path, records, configs = job
    crops, binaries = load(path)
    fused_cache = {}
    rows = defaultdict(list)
    for config in configs:
        name, mode, votes, score, mask, retain = config
        key = (name, mode)
        if key not in fused_cache:
            fused_cache[key] = fuse(crops, SETS[name], mode)
        predictions = [p["rle"] for p in ownership(fused_cache[key], score, mask, retain=retain, min_votes=votes)]
        for record_id, truth in records:
            rows[config].append({**official_pq(predictions, truth), "record_id": record_id, "file": path.stem})
    attribution = attribute(crops, binaries, records)
    return dict(rows), attribution


def attribute(crops, binaries, records, score_threshold=.15, mask_threshold=.725):
    """Why each GT is missed by the single-view baseline operating point.

    The baseline is view d0 alone with ownership() at score_threshold / mask_threshold. Each GT instance of
    each record gets one category: matched (a kept prediction has IoU > .5); query_ok_score_below_threshold
    (d0 queries reach IoU > .5, but all of them score below the threshold); query_ok_lost_in_ownership_or_mean_prob
    (such a query passes the score threshold but is removed by the area filters, the mean-probability rescoring
    or the pixel competition); otherwise the bin of the best d0 query IoU (.3-.5, .1-.3, <= .1). The d0 queries
    include the low-score binaries, which keep the probability threshold they were dumped at. best_tta_iou is
    the best IoU of any other-view candidate binarized at mask_threshold. Returns one dict per GT instance.
    """
    d0 = [c for c in crops if c["variant"] == "d0"]
    final = [p["rle"] for p in ownership(sorted(d0, key=lambda c: -c["score"]), score_threshold, mask_threshold)]
    queries = [(c["score"], full_rle((c["prob"] >= mask_threshold).astype(np.uint8), c["box"])) for c in d0]
    queries += [(b["score"], b["rle"]) for b in binaries]
    tta = [full_rle((c["prob"] >= mask_threshold).astype(np.uint8), c["box"]) for c in crops if c["variant"] != "d0"]
    cases = []
    for record_id, truth in records:
        areas = mu.area(truth)
        matched = (np.asarray(mu.iou(final, truth, [0] * len(truth))) > .5).any(0) if final else np.zeros(len(truth), bool)
        q_iou = np.asarray(mu.iou([q[1] for q in queries], truth, [0] * len(truth)))
        q_score = np.array([q[0] for q in queries])
        t_iou = np.asarray(mu.iou(tta, truth, [0] * len(truth))).max(0) if tta else np.zeros(len(truth))
        for j in range(len(truth)):
            best = float(q_iou[:, j].max())
            if matched[j]:
                category = "matched"
            elif best > .5:
                good = q_iou[:, j] > .5
                category = ("query_ok_score_below_threshold" if q_score[good].max() < score_threshold
                            else "query_ok_lost_in_ownership_or_mean_prob")
            elif best > .3:
                category = "best_query_iou_0.3_0.5"
            elif best > .1:
                category = "best_query_iou_0.1_0.3"
            else:
                category = "no_query_iou_above_0.1"
            cases.append({"record_id": record_id, "gt": j, "area": float(areas[j]), "category": category,
                          "best_query_iou": best, "best_query_score": float(q_score[q_iou[:, j].argmax()]),
                          "best_tta_iou": float(t_iou[j])})
    return cases


def area_bin(area):
    """Label of the GT area bin (pixels) used in the attribution tables."""
    return "<500" if area < 500 else "500-1500" if area < 1500 else "1500-5000" if area < 5000 else ">=5000"


def run_validation(args):
    """Sweep every configuration over the fold-0 validation images and write the JSON report to --output.

    Report keys: summary (one row per configuration with pooled PQ / SQ / RQ and counts, best PQ first);
    attribution (baseline miss categories overall and by GT area, plus the oracle recalls of the d0 queries and
    of all views); cases (one attribute() row per GT instance); records (per-record PQ rows of the
    --keep-records best configurations and of the baseline). With --partial only the images whose dump
    already exists are scored.
    """
    truth = ground_truth()
    folder = Path(args.candidates)
    configs = configurations(args)
    jobs = [(folder / f"{Path(name).stem}.npz", records, configs) for name, records in sorted(truth.items())]
    missing = [str(j[0]) for j in jobs if not j[0].exists()]
    if missing and args.partial:
        jobs = [j for j in jobs if j[0].exists()]
    elif missing:
        raise FileNotFoundError(f"{len(missing)} candidate files missing, e.g. {missing[0]}")
    totals, cases = defaultdict(list), []
    with Pool(args.workers) as pool:
        for n, (rows, attribution) in enumerate(pool.imap_unordered(evaluate_image, jobs)):
            for config, value in rows.items():
                totals[config].extend(value)
            cases.extend(attribution)
            if (n + 1) % 20 == 0:
                print(json.dumps({"images": n + 1}), flush=True)
    summary = sorted(({"set": c[0], "mode": c[1], "min_votes": c[2], "score": c[3], "mask": c[4], "retain": c[5],
                       **aggregate_pq(rows)} for c, rows in totals.items()), key=lambda r: -r["pq"])
    categories = Counter(c["category"] for c in cases)
    by_area = defaultdict(Counter)
    for c in cases:
        by_area[area_bin(c["area"])][c["category"]] += 1
    misses = [c for c in cases if c["category"] != "matched"]
    attribution = {
        "operating_point": {"set": "single", "score": .15, "mask": .725},
        "gt_instances": len(cases), "categories": dict(categories),
        "by_gt_area": {k: dict(v) for k, v in sorted(by_area.items())},
        "missed_with_any_tta_view_iou_above_0.5": sum(c["best_tta_iou"] > .5 for c in misses),
        "oracle_recall_single_view_queries": sum(c["best_query_iou"] > .5 for c in cases) / len(cases),
        "oracle_recall_any_view": sum(max(c["best_query_iou"], c["best_tta_iou"]) > .5 for c in cases) / len(cases),
    }
    keep = {(r["set"], r["mode"], r["min_votes"], r["score"], r["mask"], r["retain"]) for r in summary[:args.keep_records]}
    records = {"|".join(map(str, c)): rows for c, rows in totals.items()
               if c in keep or (c[0] == "single" and c[3] == .15 and c[4] == .725 and c[5] == .5)}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "attribution": attribution, "cases": cases,
                               "records": records}, indent=1))
    best_by_set = {}
    for row in summary:
        best_by_set.setdefault((row["set"], row["mode"]), row)
    baseline = next(r for r in summary if r["set"] == "single" and r["score"] == .15 and r["mask"] == .725
                    and r["retain"] == .5)
    print(json.dumps({"baseline": baseline, "best_by_set": list(best_by_set.values()),
                      "attribution": attribution}, indent=1))


def predict_image(job):
    """Pool worker: apply one configuration to one test image; returns (stem, ownership() instances)."""
    path, set_name, mode, votes, score, mask, retain = job
    crops, _ = load(path)
    predictions = ownership(fuse(crops, SETS[set_name], mode), score, mask, retain=retain, min_votes=votes)
    return path.stem, predictions


def run_test(args):
    """Write a submission CSV with the first value of every option.

    Images follow the dump's manifest.json. Each row is filament_id = "<image stem>_<n>" (n counts from 1 per
    image, in ownership order) and segmentation_rle = the COCO compressed RLE string of the full-size mask.
    """
    folder = Path(args.candidates)
    manifest = json.loads((folder / "manifest.json").read_text())
    stems = [Path(name).stem for name in manifest["files"]]
    jobs = [(folder / f"{stem}.npz", args.sets[0], args.modes[0], args.min_votes[0], args.scores[0],
             args.masks[0], args.retain[0]) for stem in stems]
    with Pool(args.workers) as pool:
        results = dict(pool.map(predict_image, jobs))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["filament_id", "segmentation_rle"])
        writer.writeheader()
        for stem in stems:
            for number, p in enumerate(results[stem], 1):
                writer.writerow({"filament_id": f"{stem}_{number}", "segmentation_rle": p["rle"]["counts"].decode("ascii")})
                count += 1
    print(json.dumps({"images": len(stems), "images_without_predictions": [s for s in stems if not results[s]],
                      "instances": count, "output": str(out)}))


def main():
    """Command-line entry point: validation sweep, or CSV export with --test."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test", action="store_true", help="Export submission CSV for the first value of every option")
    parser.add_argument("--sets", nargs="+", default=list(SETS))
    parser.add_argument("--modes", nargs="+", default=["all", "members"])
    parser.add_argument("--min-votes", type=int, nargs="+", default=[1])
    parser.add_argument("--scores", type=float, nargs="+", default=[.05, .08, .1, .12, .15, .2, .25])
    parser.add_argument("--masks", type=float, nargs="+", default=[.65, .7, .725, .75, .8])
    parser.add_argument("--retain", type=float, nargs="+", default=[.5])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--keep-records", type=int, default=10, help="Store per-record rows of the top configs")
    parser.add_argument("--partial", action="store_true", help="Smoke test on the candidate files present so far")
    args = parser.parse_args()
    run_test(args) if args.test else run_validation(args)


if __name__ == "__main__":
    main()
