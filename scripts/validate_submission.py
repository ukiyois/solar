"""Mechanical checks of a submission CSV before uploading."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--test-dir", default="data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images")
    args = parser.parse_args()
    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    tests = {p.stem for p in Path(args.test_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    assert rows and list(rows[0]) == ["filament_id", "segmentation_rle"], "bad header"
    ids = [r["filament_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate filament_id"
    by_image = defaultdict(list)
    for r in rows:
        stem = r["filament_id"].rsplit("_", 1)[0]
        assert stem in tests, f"unknown test image {stem}"
        rle = {"size": [2048, 2048], "counts": r["segmentation_rle"].encode("ascii")}
        area = int(mu.area(rle))
        assert area > 0, f"empty mask {r['filament_id']}"
        by_image[stem].append(rle)
    overlapping = 0
    for stem, rles in by_image.items():
        total = np.zeros((2048, 2048), np.uint8)
        for rle in rles:
            total += mu.decode(rle)
        overlapping += int((total > 1).sum())
    areas = [int(mu.area({"size": [2048, 2048], "counts": r["segmentation_rle"].encode("ascii")})) for r in rows]
    print(json.dumps({"rows": len(rows), "images_with_predictions": len(by_image), "test_images": len(tests),
                      "images_without_predictions": sorted(tests - set(by_image)), "overlapping_pixels": overlapping,
                      "area_min_median_max": [min(areas), int(np.median(areas)), max(areas)],
                      "sha256": hashlib.sha256(Path(args.csv).read_bytes()).hexdigest()}))
    assert overlapping == 0, "instances overlap"


if __name__ == "__main__":
    main()
