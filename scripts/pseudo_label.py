"""Pseudo-label unlabelled GONG archive observations with a TTA teacher (Noisy Student, GPU).

Teacher = one or more fold-0 checkpoints x dihedral views, fused exactly like a submission (member
fusion + pixel ownership, scripts/sweep_recall_tta.py). Frames observed on a hidden-test day are
never labelled (excluded_days); with <output>/exclude_fold.txt the fold's validation days are dropped
too. Writes a directory with the competition layout so FilamentDataset reads it:
  <output>/train/train_images/*.jpeg   (symlinks to the cleaned archive images)
  <output>/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
Instances come from the same member fusion + pixel ownership used for
submissions; masks become polygons (cv2 contours). Images without any
instance are left out. Resumable through <output>/pseudo.jsonl.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mu

from scripts.dump_tta_candidates import candidates, load_model
from scripts.sweep_recall_tta import fuse, ownership


def polygons(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = []
    for contour in contours:
        points = contour.reshape(-1, 2).astype(np.float64) + .5  # pixel centers -> COCO coordinates
        if len(points) >= 3:
            result.append([round(float(v), 2) for v in points.reshape(-1)])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--views", nargs="+", default=["d0", "d2", "f0", "f2"])
    parser.add_argument("--votes", type=int, required=True)
    parser.add_argument("--score", type=float, default=.12)
    parser.add_argument("--mask", type=float, default=.65)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    device = torch.device("cuda")
    models = [load_model("configs/local.yaml", c, device)[0] for c in args.checkpoints]
    voters = [f"{m}:{v}" for m in range(len(models)) for v in args.views]
    out = Path(args.output)
    (out / "train").mkdir(parents=True, exist_ok=True)
    done_path = out / f"pseudo_{args.shard}.jsonl"
    done = {json.loads(l)["file_name"] for l in done_path.read_text().splitlines()} if done_path.exists() else set()
    skip = excluded_days(out)
    files = [f for f in sorted(Path(args.images).glob("*.jpeg")) if f.name[:8] not in skip][args.shard::args.shards]
    with done_path.open("a", encoding="utf-8") as log:
        for number, path in enumerate(files):
            if path.name in done:
                continue
            started = time.perf_counter()
            image = torch.from_numpy(np.asarray(Image.open(path).convert("L")).copy()).float()[None, None].div(255).to(device)
            crops = []
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                for m, model in enumerate(models):
                    for view in args.views:
                        k, flip = int(view[1]), view[0] == "f"
                        for item in candidates(model, image, k, flip, .02, .2):
                            item["prob"] = item["prob"].astype(np.float32)
                            crops.append({**item, "variant": f"{m}:{view}"})
            kept = ownership(fuse(crops, voters, "members"), args.score, args.mask, min_votes=args.votes)
            annotations = []
            for p in kept:
                poly = polygons(mu.decode(p["rle"]))
                if poly:
                    annotations.append({"segmentation": poly, "score": round(p["score"], 4), "area": p["area"]})
            log.write(json.dumps({"file_name": path.name, "annotations": annotations,
                                  "seconds": round(time.perf_counter() - started, 2)}) + "\n")
            log.flush()
            if (number + 1) % 50 == 0:
                print(json.dumps({"shard": args.shard, "done": number + 1, "of": len(files)}), flush=True)


def excluded_days(output, data_root="data/raw/MAGFiLO_1.0_Kaggle_2026"):
    """Observation days (YYYYMMDD) whose archive frames must not be pseudo-labelled.

    Always the hidden-test days: the host allows only the provided test frames at
    inference and no self-training on test images, so training never sees
    near-duplicates of a test observation. If <output>/exclude_fold.txt names a fold,
    that fold's validation days too, so a fold-0 experiment is scored under the same
    treatment the test set gets (no transductive frames of the scored days).
    """
    root = Path(data_root)
    days = {p.name[:8] for p in (root / "test/test_images").iterdir()}
    marker = Path(output) / "exclude_fold.txt"
    if marker.exists():
        from solar_filaments.data import physical_fold
        fold = int(marker.read_text().strip())
        days |= {p.name[:8] for p in (root / "train/train_images").iterdir() if physical_fold(p.name, 5, 2026) == fold}
    return days


def assemble(output, images):
    """Merge shard logs into the competition-format JSON (run once after all shards).
    Frames from excluded_days(output) are left out."""
    out = Path(output)
    rows = [json.loads(l) for f in sorted(out.glob("pseudo_*.jsonl")) for l in f.read_text().splitlines()]
    days = excluded_days(out)
    excluded = sum(row["file_name"][:8] in days for row in rows)
    rows = [row for row in rows if row["file_name"][:8] not in days]
    link = out / "train/train_images"
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(Path(images).resolve())
    data = {"info": {"description": "GONG archive pseudo labels"}, "licenses": [], "categories": [{"id": 1, "name": "filament"}],
            "images": [], "annotations": []}
    for row in rows:
        if not row["annotations"]:
            continue
        image_id = "pseudo-" + Path(row["file_name"]).stem
        data["images"].append({"file_name": row["file_name"], "height": 2048, "width": 2048, "id": image_id})
        for i, a in enumerate(row["annotations"]):
            data["annotations"].append({"segmentation": a["segmentation"], "image_id": image_id, "id": f"{image_id}-{i}",
                                        "category_id": 1, "iscrowd": 0, "area": a["area"], "score": a["score"]})
    (out / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_text(json.dumps(data))
    print(json.dumps({"images_with_instances": len(data["images"]), "images_total": len(rows),
                      "instances": len(data["annotations"]), "excluded_test_day_images": excluded}))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "assemble":
        assemble(sys.argv[2], sys.argv[3])
    else:
        main()
