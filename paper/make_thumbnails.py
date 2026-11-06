"""Thumbnails for the architecture figure (paper/figures/arch_*.png).

One test observation, its three native input channels (computed exactly as in solar_filaments/model.py) and the
instance masks of a submission file, filled in distinct colours.

usage (repository root): python paper/make_thumbnails.py --csv <submission.csv> [--stem <test image stem>]
Without --stem, the test observation with the largest total predicted area is used.
Outputs (SIZE x SIZE px): arch_input.png (the image), arch_contrast.png (local contrast 4 (I - 15 x 15 mean)),
arch_darkness.png (limb-flattened relative darkness) and arch_output.png (instances dilated by 9 x 9 px).
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mu
from torch.nn import functional as F

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from solar_filaments.model import limb_flattened  # noqa: E402

TEST = REPOSITORY / "data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images"
OUT = REPOSITORY / "paper/figures"
SIZE = 400
COLORS = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
          (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255)]


def save(array, name, low, high):
    """Map `array` linearly from [low, high] to 8-bit grey (clipped) and save it as OUT/name, SIZE x SIZE px."""
    scaled = np.clip((np.asarray(array, np.float32) - low) / (high - low), 0, 1)
    image = Image.fromarray((scaled * 255).astype(np.uint8))
    image.resize((SIZE, SIZE), Image.LANCZOS).save(OUT / name)


def main():
    """Write the four thumbnails of one test observation to paper/figures."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--csv", required=True)
    parser.add_argument("--stem")
    args = parser.parse_args()
    instances = defaultdict(list)
    with open(args.csv, encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            instances[row["filament_id"].rsplit("_", 1)[0]].append(
                {"size": [2048, 2048], "counts": row["segmentation_rle"].encode("ascii")})
    stem = args.stem or max(instances, key=lambda s: sum(mu.area(r) for r in instances[s]))
    image = np.asarray(Image.open(TEST / f"{stem}.jpeg").convert("L"), dtype=np.float32) / 255
    x = torch.from_numpy(image)[None, None]
    OUT.mkdir(parents=True, exist_ok=True)
    save(image, "arch_input.png", 0, 1)
    save((4 * (x - F.avg_pool2d(x, 15, 1, 7)))[0, 0], "arch_contrast.png", -.25, .25)
    save(limb_flattened(x)[0, 0], "arch_darkness.png", -.9, .5)
    overlay = np.repeat(image[..., None] * 255, 3, -1)
    for index, rle in enumerate(instances[stem]):
        # widened by a few pixels so that thin instances stay visible at thumbnail scale
        mask = cv2.dilate(mu.decode(rle), np.ones((9, 9), np.uint8)).astype(bool)
        overlay[mask] = COLORS[index % len(COLORS)]
    Image.fromarray(overlay.astype(np.uint8)).resize((SIZE, SIZE), Image.LANCZOS).save(OUT / "arch_output.png")
    print(f"{stem}: {len(instances[stem])} instances -> {OUT}/arch_*.png")


if __name__ == "__main__":
    main()
