"""Measure apparent filament geometry using segmentation polygons only."""

import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
OUTPUT = Path(__file__).with_name("filament_geometry.json")


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        f"p{p}": round(float(np.percentile(values, p)), 3)
        for p in (10, 25, 50, 75, 90, 95, 99)
    }


data = json.loads(SOURCE.read_text(encoding="utf-8"))
rows = []
for annotation in data["annotations"]:
    polygons = [np.asarray(p, dtype=np.float64).reshape(-1, 2)
                for p in annotation["segmentation"] if len(p) >= 6]
    if not polygons:
        continue
    vertices = np.concatenate(polygons)
    origin = np.floor(vertices.min(axis=0)).astype(int) - 3
    end = np.ceil(vertices.max(axis=0)).astype(int) + 3
    width, height = (end - origin + 1).tolist()
    local_polygons = [(p - origin).reshape(-1).tolist() for p in polygons]
    rles = mask_utils.frPyObjects(local_polygons, height, width)
    rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
    foreground = mask_utils.decode(rle).astype(bool)
    if not foreground.any():
        continue
    skeleton = skeletonize(foreground)
    # Twice the inscribed radius at skeleton points is a width proxy in pixels.
    apparent_width = 2.0 * distance_transform_edt(foreground)[skeleton]
    box_sides = cv2.minAreaRect(vertices.astype(np.float32))[1]
    long_side, short_side = sorted(box_sides, reverse=True)
    rows.append({
        "annotation_id": annotation["id"],
        "area_pixels": int(foreground.sum()),
        "oriented_box_long_side_pixels": float(long_side),
        "oriented_box_short_side_pixels": float(short_side),
        "oriented_box_aspect_ratio": float(long_side / max(short_side, 1e-6)),
        "skeleton_pixels": int(skeleton.sum()),
        "median_skeleton_width_pixels": float(np.median(apparent_width)),
        "p10_skeleton_width_pixels": float(np.percentile(apparent_width, 10)),
    })

metrics = [key for key in rows[0] if key != "annotation_id"]
widths = np.asarray([r["median_skeleton_width_pixels"] for r in rows])
summary = {
    "annotations_measured": len(rows),
    "statistics": {key: quantiles([r[key] for r in rows]) for key in metrics},
    "median_instance_width_at_input_sizes": {
        str(size): round(float(np.median(widths) * size / 2048), 3)
        for size in (2048, 1536, 1024, 768, 512)
    },
    "fraction_of_instances_with_median_width_below_native_pixels": {
        str(limit): round(float((widths < limit).mean()), 4)
        for limit in (4, 8, 12, 16)
    },
    "interpretation": (
        "All 8199 annotator-instance records are weighted equally; duplicated physical "
        "filaments across annotators are retained. Width means twice the Euclidean "
        "distance to background at a rasterized mask skeleton, not a measured plasma "
        "diameter. Oriented box aspect ratio includes bends and branches. Only "
        "segmentation polygons are read; provided bbox/spine/category are not used."
    ),
}
OUTPUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
