"""Dump per-query native probability crops of the 8 dihedral test-time-augmentation views (GPU).

One forward per (image, view). Each view is the image rotated by k*90 degrees (d<k>) or flipped and
rotated (f<k>); probability maps are mapped back to the original orientation and stored as FP16
bounding-box crops, so fusion and threshold choices (scripts/sweep_recall_tta.py) run on CPU without
rerunning the model. View "d0" additionally keeps every low-score query as an RLE (miss attribution).
An optional suffix s<size> sets the backbone view size of that view (multi-scale TTA; not used by the
final submission).

Output: one <image stem>.npz per image. `meta` is a JSON list with one entry per stored query (key, variant =
view, query index, score, and either box = (y0, y1, x0, x1) or rle = COCO counts string); every box entry has
an FP16 crop under its key. Crops are kept for queries with score >= --keep-score and cover the bounding box
of the pixels with probability >= --crop-probability. manifest.json records the checkpoint, split, views,
thresholds and file order. Existing .npz files are skipped, so an interrupted run resumes; --shard / --shards
split the images across processes.

usage (final submission):
  python -m scripts.dump_tta_candidates --checkpoint weights/final_a.pt --split test --output runs/tta/final_a/test
  python -m scripts.dump_tta_candidates --checkpoint <fold-0 model> --split val --output runs/tta/<name>/val
"""
import argparse
import json
from pathlib import Path
import re
import time

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch

from solar_filaments.config import checkpoint_config, model_config, read_config
from solar_filaments.data import FilamentDataset
from solar_filaments.masks import fine_mask_logits
from solar_filaments.model import build_model

DIHEDRAL = ["d0", "d1", "d2", "d3", "f0", "f1", "f2", "f3"]


def dihedral(image, k, flip):
    """View d<k> or f<k>: rotate the last two axes by k * 90 degrees, then flip horizontally if `flip`."""
    image = torch.rot90(image, k, (-2, -1))
    return image.flip(-1) if flip else image


def undo_dihedral(maps, k, flip):
    """Inverse of dihedral(): map per-query maps of a view back to the original image orientation."""
    if flip:
        maps = maps.flip(-1)
    return torch.rot90(maps, -k, (-2, -1))


def load_model(config_path, checkpoint, device):
    """Checkpoint architecture with the local operational settings (DCN backend, tiling) of config_path."""
    local = model_config(read_config(config_path))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    operational = ("context_size", "dcn_backend", "detail_tile", "detail_halo", "backbone_checkpointing", "detail_checkpointing")
    config = checkpoint_config(saved["model_config"], **{key: getattr(local, key) for key in operational})
    model, _ = build_model(config, load_backbone=False)
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval(), saved


@torch.no_grad()
def candidates(model, image, k, flip, keep_score, crop_probability, binary_rest=None, chunk=8, context_size=None):
    """Queries scoring >= keep_score keep FP16 probability crops (support >= crop_probability); with
    binary_rest, every other query keeps only its RLE at that probability threshold (miss attribution).
    Score = P(filament) * sigmoid(quality), as in solar_filaments.inference.

    Native-resolution masks are rendered `chunk` queries at a time to bound GPU memory. Returns dicts
    {query, score, box, prob (numpy FP16 crop)} or {query, score, rle}; a query without any pixel at
    crop_probability is dropped."""
    output = model(dihedral(image, k, flip), tiled_detail=True, context_size=context_size)
    scores = (output["pred_logits"][0].float().softmax(-1)[:, 0] *
              output["pred_quality"][0].float().sigmoid())
    ids = torch.arange(len(scores), device=scores.device) if binary_rest else torch.where(scores >= keep_score)[0]
    result = []
    for start in range(0, len(ids), chunk):
        part = ids[start:start + chunk]
        probability = undo_dihedral(fine_mask_logits(output, 0, part).float().sigmoid(), k, flip)
        for query, prob in zip(part.tolist(), probability):
            if float(scores[query]) < keep_score:
                rle = mask_utils.encode(np.asfortranarray((prob >= binary_rest).cpu().numpy().astype(np.uint8)))
                result.append({"query": query, "score": float(scores[query]), "rle": rle["counts"].decode("ascii")})
                continue
            support = prob >= crop_probability
            if not support.any():
                continue
            ys = torch.where(support.any(1))[0]
            xs = torch.where(support.any(0))[0]
            y0, y1, x0, x1 = int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1
            result.append({"query": query, "score": float(scores[query]), "box": (y0, y1, x0, x1),
                           "prob": prob[y0:y1, x0:x1].half().cpu().numpy()})
    del output
    return result


def main():
    """Dump the candidates of every view for each image of the split (or of --images), one .npz per image."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local.yaml", help="local operational settings (backend, tiling)")
    parser.add_argument("--train-config", default="configs/v9_ffinal_a.yaml", help="data root and fold definition")
    parser.add_argument("--checkpoint", default="weights/final_a.pt")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", nargs="+", default=DIHEDRAL)
    parser.add_argument("--keep-score", type=float, default=0.02)
    parser.add_argument("--crop-probability", type=float, default=0.2)
    parser.add_argument("--attribution-mask", type=float, default=0.725,
                        help="d0 queries below --keep-score are stored as RLE at this probability")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--images", help="Dump every *.jpeg/*.jpg in this directory (or listed, one name per line, "
                                         "in --images-list) instead of a competition split")
    parser.add_argument("--images-list", help="File with image names inside --images (default: all)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda")
    model, saved = load_model(args.config, args.checkpoint, device)
    document = read_config(args.train_config)["train"]
    if args.images:
        folder = Path(args.images)
        if args.images_list:
            names = [line.strip() for line in Path(args.images_list).read_text().splitlines() if line.strip()]
            paths = [folder / name for name in names]
        else:
            paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"})
    elif args.split == "val":
        dataset = FilamentDataset(document["data_root"], "val", document["fold"], document["folds"], document["seed"],
                                  geometry=False, image_uint8=True, decode_masks=False)
        names = sorted({record["file_name"] for record in dataset.records})
        paths = [Path(document["data_root"]) / "train/train_images" / name for name in names]
    else:
        folder = Path(document["data_root"]) / "test/test_images"
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    paths = paths[:args.limit] if args.limit else paths
    paths = paths[args.shard::args.shards]
    views = []
    for variant in args.variants:
        parsed = re.fullmatch(r"([df])([0-3])(?:s(\d+))?", variant)
        if parsed is None:
            raise ValueError(f"Bad view {variant}: expected d<k> or f<k> (k = 0..3), optionally s<size>")
        views.append((variant, int(parsed.group(2)), parsed.group(1) == "f",
                      int(parsed.group(3)) if parsed.group(3) else None))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / ("manifest.json" if args.shards == 1 else f"manifest_{args.shard}.json")).write_text(json.dumps({
        "checkpoint": args.checkpoint, "epoch": saved.get("epoch"), "validation": saved.get("validation"),
        "split": args.split, "variants": args.variants, "keep_score": args.keep_score,
        "crop_probability": args.crop_probability, "files": [p.name for p in paths]}, indent=2))
    for number, path in enumerate(paths):
        target = out / f"{path.stem}.npz"
        if target.exists():
            continue
        start = time.perf_counter()
        image = torch.from_numpy(np.asarray(Image.open(path).convert("L")).copy()).float()[None, None].div(255).to(device)
        arrays, meta = {}, []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for variant, k, flip, size in views:
                found = candidates(model, image, k, flip, args.keep_score, args.crop_probability,
                                   args.attribution_mask if variant == "d0" else None, context_size=size)
                torch.cuda.empty_cache()
                for i, item in enumerate(found):
                    key = f"{variant}_{i}"
                    if "prob" in item:
                        arrays[key] = item.pop("prob")
                    meta.append({"key": key, "variant": variant, **item})
        # write then rename, so an interrupted run never leaves a truncated .npz that a resume would skip
        temporary = target.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, meta=np.array(json.dumps(meta)), **arrays)
        temporary.replace(target)
        print(json.dumps({"n": number + 1, "of": len(paths), "file": path.name, "candidates": len(meta),
                          "seconds": round(time.perf_counter() - start, 2)}), flush=True)


if __name__ == "__main__":
    main()
