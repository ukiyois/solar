"""Do annotated filaments, and the model's errors, lie on magnetic polarity inversion lines? (CPU, analysis only)

Filaments form above polarity inversion lines (PILs) of the photospheric field. This analysis pairs every training
observation with a line-of-sight magnetogram and measures the distance of each H-alpha instance to the nearest PIL.
Magnetograms are NOT used for training or inference (rule 1: inference reads only the H-alpha test image).

Pairs (research/magnetogram_pairs.csv, 707 training observations): 612 GONG QuickReduce bqa 10-minute averages from
the same site (610 within 15 min, 2 at 16-18 min) and 95 SDO/HMI hmi.M_720s magnetograms (92 within 12 min,
3 at 19-60 min). Both are mapped north-up in image orientation: GONG flips rows; HMI (CROTA2 ~ 180) is 4 x 4
block-averaged and flips columns (verified by the plage / |B| correlation). Off-disk pixels (r > .98 R) are zeroed,
B_LOS is smoothed at sigma = 18.5 Mm and PIL pixels are where the dilated >= +0.5 G and <= -0.5 G regions touch
(r < .97 R). Each instance is mapped through the two limb fits (centre, radius) and summarized by the median
projected PIL distance of up to 400 of its pixels. Null: 500 random on-disk points (r < .95 R) per observation.
Populations: every annotated filament (by cloud-model contrast, by disk position mu and, on multi-record
observations, by annotator agreement), and on fold 0 the predictions at the submission setting (matched / near
miss / unsupported against any record) and the missed filaments (best IoU <= .1). The centre-to-limb rows compare
filaments with random points of the same mu: a decline shared by both is a property of the magnetograms.

usage (repository root; FITS files as listed in the pairs file):
  python research/magnetogram_pil.py --folder runs/tta/<fold-0 model>/val --gong-dir data/gong_bqa/fits \
      --hmi-dir data/hmi_720s/fits
writes research/magnetogram_pil.json
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from astropy.io import fits
from PIL import Image
from pycocotools import mask as mu
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates, maximum_filter

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from scripts.fuse_tta import VIEWS, predict  # noqa: E402
from solar_filaments.data import physical_fold  # noqa: E402
from solar_filaments.model import limb_flattened  # noqa: E402

ROOT = REPOSITORY / "data/raw/MAGFiLO_1.0_Kaggle_2026"
SETTING = (1, .10, .40)  # votes, score, mask threshold of the submission
RSUN_MM, SIGMA_MM, THRESHOLD_G = 696.0, 18.5, .5
RNG = np.random.default_rng(0)


def load_magnetogram(pair, directories):
    """North-up B_LOS (row 0 = north) on a ~2 arcsec grid and its disk geometry (cy, cx, radius) in pixels."""
    with fits.open(directories[pair["source"]] / pair["magnetogram_file"]) as hdul:
        hdu = next(h for h in hdul if h.data is not None)
        header, data = hdu.header, hdu.data.astype(np.float32)
    if pair["source"] == "GONG":
        data = data[::-1].copy()
        cy, cx = (data.shape[0] - 1) - (header["FNDLMBYC"] - 1), header["FNDLMBXC"] - 1
        radius = .5 * (header["FNDLMBMI"] + header["FNDLMBMA"])
        return np.where(data <= -700., 0., data), (cy, cx, radius)  # -700 G and below: missing data
    blocks = data.reshape(1024, 4, 1024, 4)
    valid = np.isfinite(blocks).sum((1, 3))
    data = np.where(valid > 0, np.nansum(blocks, (1, 3)) / np.maximum(valid, 1), 0.).astype(np.float32)[:, ::-1].copy()
    cy, cx = (float(pair["hmi_crpix2"]) - 1) / 4 - .375, (float(pair["hmi_crpix1"]) - 1) / 4 - .375
    radius = float(pair["hmi_rsun_obs"]) / float(pair["hmi_cdelt1"]) / 4
    return data, (cy, (data.shape[1] - 1) - cx, radius)


def pil_distance(pair, directories):
    data, (cy, cx, radius) = load_magnetogram(pair, directories)
    yy, xx = np.mgrid[:data.shape[0], :data.shape[1]]
    rr = np.hypot(yy - cy, xx - cx) / radius
    mm_per_px = RSUN_MM / radius
    smooth = gaussian_filter(np.where(rr < .98, data, 0.), SIGMA_MM / mm_per_px)
    pil = maximum_filter(smooth >= THRESHOLD_G, 3) & maximum_filter(smooth <= -THRESHOLD_G, 3) & (rr < .97)
    distance = distance_transform_edt(~pil) * mm_per_px if pil.any() else np.full(data.shape, np.inf)
    return distance, (cy, cx, radius), float(pil[rr < .97].mean())


def halpha(name):
    image = np.asarray(Image.open(ROOT / "train/train_images" / name).convert("L")).astype(np.float32)
    disk = image > .25 * np.quantile(image[::8, ::8], .99)
    ys, xs = np.nonzero(disk)
    small = image.reshape(1024, 2, 1024, 2).mean((1, 3))
    flat = (limb_flattened(torch.from_numpy(small / 255)[None, None], factor=4, sigma=4.0)[0, 0] / 4).numpy()
    return (ys.mean(), xs.mean(), np.sqrt(disk.sum() / np.pi)), flat


def instance_stats(rle, hgeo, mgeo, distance, flat):
    ys, xs = np.nonzero(mu.decode(rle))
    area = len(ys)
    if area > 400:
        pick = RNG.choice(area, 400, replace=False)
        ys, xs = ys[pick], xs[pick]
    (hy, hx, hr), (my, mx, mr) = hgeo, mgeo
    u, v = (xs - hx) / hr, (ys - hy) / hr
    d = map_coordinates(distance, [my + v * mr, mx + u * mr], order=0)
    r = min(1., float(np.hypot(v.mean(), u.mean())))
    return {"pil_mm": float(np.median(d)), "area": area, "contrast": float(flat[ys // 2, xs // 2].mean()),
            "mu": float(np.sqrt(1 - r * r))}


def null_distances(hgeo, mgeo, distance, n=500):
    """(mu, PIL distance) of n uniform random on-disk points (r < .95 R)."""
    angle, radius = RNG.uniform(0, 2 * np.pi, n), np.sqrt(RNG.uniform(0, .95 ** 2, n))
    (_, _, _), (my, mx, mr) = hgeo, mgeo
    d = map_coordinates(distance, [my + radius * np.sin(angle) * mr, mx + radius * np.cos(angle) * mr], order=0)
    return list(zip(np.sqrt(1 - radius ** 2).tolist(), d.tolist()))


def iou_matrix(predictions, truth):
    if not predictions or not truth:
        return np.zeros((len(predictions), len(truth)))
    return np.asarray(mu.iou(predictions, truth, [0] * len(truth))).reshape(len(predictions), len(truth))


def summary(values):
    v = np.asarray(values)
    if len(v) == 0:
        return {"n": 0}
    return {"n": int(len(v)), "median_mm": round(float(np.median(v)), 1), "within_10mm": round(float((v <= 10).mean()), 3),
            "within_20mm": round(float((v <= 20).mean()), 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--folder", required=True, help="fold-0 dump of scripts/dump_tta_candidates.py")
    parser.add_argument("--pairs", default=str(REPOSITORY / "research/magnetogram_pairs.csv"))
    parser.add_argument("--gong-dir", default=str(REPOSITORY / "data/gong_bqa/fits"))
    parser.add_argument("--hmi-dir", default=str(REPOSITORY / "data/hmi_720s/fits"))
    parser.add_argument("--output", default=str(REPOSITORY / "research/magnetogram_pil.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    directories = {"GONG": Path(args.gong_dir), "HMI": Path(args.hmi_dir)}
    pairs = {p["ha_filename"]: p for p in csv.DictReader(open(args.pairs, encoding="utf-8"))}
    data = json.loads((ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text(encoding="utf-8"))
    by_record = defaultdict(list)
    for annotation in data["annotations"]:
        rles = mu.frPyObjects(annotation["segmentation"], 2048, 2048)
        by_record[str(annotation["image_id"])].append(mu.merge(rles) if isinstance(rles, list) else rles)
    by_image = defaultdict(list)
    for record in data["images"]:
        by_image[record["file_name"]].append(by_record[str(record["id"])])

    annotated, null, model_rows, pil_cover, skipped = [], [], [], [], []
    names = sorted(n for n in by_image if n in pairs)
    for i, name in enumerate(names):
        source = pairs[name]["source"]
        try:
            distance, mgeo, pil_fraction = pil_distance(pairs[name], directories)
        except (TypeError, OSError, ValueError) as error:  # unreadable file: skip and report
            skipped.append([name, f"{type(error).__name__}: {str(error)[:60]}"])
            continue
        pil_cover.append(pil_fraction)
        hgeo, flat = halpha(name)
        null += [(source, m, d) for m, d in null_distances(hgeo, mgeo, distance)]
        records = by_image[name]
        for a, truth in enumerate(records):
            others = [t for b, t in enumerate(records) if b != a]
            for rle in truth:
                row = {"source": source, **instance_stats(rle, hgeo, mgeo, distance, flat)}
                if others:
                    row["agreement"] = float(np.mean([len(o) > 0 and iou_matrix([rle], o).max() > .5 for o in others]))
                annotated.append(row)
        if physical_fold(name, 5, 2026) == 0:
            _, kept = predict((Path(name).stem, [args.folder], [VIEWS], SETTING))
            predictions = [p["rle"] for p in kept]
            union = [r for truth in records for r in truth]
            best = iou_matrix(predictions, union).max(1) if union else np.zeros(len(predictions))
            for b, rle in zip(best, predictions):
                kind = "pred_matched" if b > .5 else "pred_near_miss" if b > .1 else "pred_unsupported"
                model_rows.append({"kind": kind, "source": source, **instance_stats(rle, hgeo, mgeo, distance, flat)})
            for truth in records:
                best = iou_matrix(predictions, truth).max(0) if predictions else np.zeros(len(truth))
                for b, rle in zip(best, truth):
                    if b <= .1:
                        model_rows.append({"kind": "gt_missed", "source": source, **instance_stats(rle, hgeo, mgeo, distance, flat)})
        if (i + 1) % 100 == 0:
            print(f"observations {i + 1}/{len(names)}", flush=True)

    result = {"observations": len(names) - len(skipped), "skipped_unreadable": skipped,
              "setting": f"sigma {SIGMA_MM} Mm, threshold {THRESHOLD_G} G", "pil_pixel_fraction_mean": round(float(np.mean(pil_cover)), 4)}
    for source in ("GONG", "HMI", "ALL"):
        pick = lambda r, s=source: s == "ALL" or r["source"] == s
        result[f"{source}_null_random_disk_points"] = summary([d for s, _, d in null if source in ("ALL", s)])
        result[f"{source}_annotated_filaments"] = summary([r["pil_mm"] for r in annotated if pick(r)])
        for lo, hi in ((-1, -.10), (-.10, -.07), (-.07, 1)):
            result[f"{source}_annotated_contrast_{lo}_{hi}"] = summary([r["pil_mm"] for r in annotated if pick(r) and lo <= r["contrast"] < hi])
        for lo, hi in ((0, .34), (.34, .67), (.67, 1.01)):
            result[f"{source}_annotated_agreement_{lo}_{hi}"] = summary(
                [r["pil_mm"] for r in annotated if pick(r) and "agreement" in r and lo <= r["agreement"] < hi])
    for kind in ("pred_matched", "pred_near_miss", "pred_unsupported", "gt_missed"):
        result[f"fold0_{kind}"] = summary([r["pil_mm"] for r in model_rows if r["kind"] == kind])
    within = lambda key: result[key]["within_20mm"]
    result["enrichment_within_20mm"] = round(within("ALL_annotated_filaments") / within("ALL_null_random_disk_points"), 2)
    # centre-to-limb: if the share falls for filaments and random points alike, the magnetograms lose PILs, not filaments
    for lo, hi in ((.85, 1.01), (.6, .85), (0, .6)):
        tag = f"mu_{lo}_{hi}"
        result[f"{tag}_annotated"] = summary([r["pil_mm"] for r in annotated if lo <= r["mu"] < hi])
        result[f"{tag}_random"] = summary([d for _, m, d in null if lo <= m < hi])
        result[f"{tag}_enrichment_within_20mm"] = round(within(f"{tag}_annotated") / within(f"{tag}_random"), 2)
    Path(args.output).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k.startswith(("ALL_null", "ALL_annotated_filaments", "fold0", "enrich", "mu_"))},
                     indent=1))


if __name__ == "__main__":
    main()
