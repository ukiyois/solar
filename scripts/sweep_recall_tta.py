"""CPU post-processing over dumped query crops: TTA instance fusion, threshold
sweeps, miss attribution, and test submission export.

Ownership follows solar_filaments.inference.instance_predictions exactly
(score-ordered pixel competition on probability*score, min area, retained
fraction). Fusion clusters candidates from different dihedral views by binary
IoU, then averages their probability crops.
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

SIZE = (2048, 2048)
SETS = {
    "single": ["d0"],
    "flip4": ["d0", "f0", "d2", "f2"],
    "rot4": ["d0", "d1", "d2", "d3"],
    "dihedral8": ["d0", "d1", "d2", "d3", "f0", "f1", "f2", "f3"],
}


def full_rle(binary, box):
    y0, y1, x0, x1 = box
    canvas = np.zeros(SIZE, dtype=np.uint8)
    canvas[y0:y1, x0:x1] = binary
    rle = mu.encode(np.asfortranarray(canvas))
    return rle


def load(path):
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
    """Why each GT is missed by the single-view baseline operating point."""
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
    return "<500" if area < 500 else "500-1500" if area < 1500 else "1500-5000" if area < 5000 else ">=5000"


def run_validation(args):
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
    path, set_name, mode, votes, score, mask, retain = job
    crops, _ = load(path)
    predictions = ownership(fuse(crops, SETS[set_name], mode), score, mask, retain=retain, min_votes=votes)
    return path.stem, predictions


def run_test(args):
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
