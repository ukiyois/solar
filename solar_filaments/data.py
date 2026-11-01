"""MAGFiLO records, physical-image folds, augmentation and the pseudo-label mixture.

Every annotator record (one annotator's drawing of one observation) is one training sample; folds are
assigned per physical image (file name), so all records of an observation fall in the same fold.
Geometry targets (packed masks, skeleton anchors, ROIs) are precomputed by scripts/prepare_data.py.
"""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch
from torch.utils.data import Dataset
from .geometry import CACHE_VERSION, derive_geometry, transform_geometry


def physical_fold(filename, folds=5, seed=2026):
    digest = hashlib.sha256(f"{seed}:{filename}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % folds


def rotate_record(image, annotations, angle, scale, flip):
    """Arbitrary rotation/scale about the image center, applied identically to
    the image (bilinear) and to the exact polygon vertices (then rasterized).

    COCO polygon coordinates are pixel-edge based; OpenCV maps pixel centers,
    hence the .5 shifts. Instances rotated completely out of frame are dropped.
    """
    import cv2
    h, w = image.shape
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, scale)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if flip:
        rotated = rotated[:, ::-1]
    masks = []
    for annotation in annotations:
        polygons = []
        for polygon in annotation["segmentation"]:
            points = np.asarray(polygon, np.float64).reshape(-1, 2) - .5
            points = points @ matrix[:, :2].T + matrix[:, 2] + .5
            if flip:
                points[:, 0] = w - points[:, 0]
            polygons.append(points.reshape(-1).tolist())
        rles = mask_utils.frPyObjects(polygons, h, w)
        mask = mask_utils.decode(mask_utils.merge(rles) if isinstance(rles, list) else rles).astype(bool)
        if mask.any():
            masks.append(mask)
    masks = np.stack(masks) if masks else np.zeros((0, h, w), dtype=bool)
    return np.ascontiguousarray(rotated), masks


def decode_annotations(annotations, height, width):
    masks = []
    for annotation in annotations:
        rles = mask_utils.frPyObjects(annotation["segmentation"], height, width)
        rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
        masks.append(mask_utils.decode(rle).astype(bool))
    return np.stack(masks) if masks else np.zeros((0, height, width), dtype=bool)


class FilamentDataset(Dataset):
    """split: 'train' / 'val' (fold `fold` of `folds` by physical image) or 'all'.

    augment: random quarter turn + flip of the cached geometry; with rotate_probability > 0 an arbitrary
    rotation and scale (1 +- scale_jitter) about the image center is applied instead, to the image and
    to the exact polygon vertices (then rasterized).
    """

    def __init__(self, root, split, fold=0, folds=5, seed=2026, augment=False,
                 cache_dir=None, require_cache=False, geometry=True, image_uint8=False, decode_masks=True,
                 rotate_probability=0.0, scale_jitter=0.0):
        self.root, self.augment = Path(root), augment
        self.rotate_probability, self.scale_jitter = float(rotate_probability), float(scale_jitter)
        self.geometry, self.image_uint8 = geometry, image_uint8
        self.decode_masks = decode_masks
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.require_cache = require_cache
        self.seed = seed
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        payload = (self.root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_bytes()
        self.fingerprint = hashlib.sha256(payload + f"geometry-v{CACHE_VERSION}".encode()).hexdigest()[:16]
        data = json.loads(payload)
        self.annotations = defaultdict(list)
        for annotation in data["annotations"]:
            self.annotations[str(annotation["image_id"])].append({"segmentation": annotation["segmentation"]})
        self.records = [record for record in data["images"]
                        if split == "all" or (physical_fold(record["file_name"], folds, seed) == fold) == (split == "val")]
        if not self.records:
            raise ValueError("Empty data split")

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch.fill_(epoch)

    def cache_path(self, record):
        key = hashlib.sha256(str(record["id"]).encode()).hexdigest()[:20]
        return self.cache_dir / self.fingerprint / f"{key}.npz" if self.cache_dir else None

    def prepare(self, index):
        record = self.records[index]
        path = self.cache_path(record)
        if path is not None and path.exists():
            return path
        masks = decode_annotations(self.annotations[str(record["id"])], record["height"], record["width"])
        geometry = derive_geometry(masks)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(f".{os.getpid()}.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **geometry)
            temporary.replace(path)
        return path

    def __getitem__(self, index):
        record = self.records[index]
        image = np.asarray(Image.open(self.root / "train/train_images" / record["file_name"]).convert("L"))
        rotated = None
        if self.geometry and self.augment and self.rotate_probability > 0:
            rng = random.Random(self.seed + 7919 * int(self.epoch) + 104729 * index)
            if rng.random() < self.rotate_probability:
                angle = rng.uniform(0., 360.)
                scale = 1 + rng.uniform(-self.scale_jitter, self.scale_jitter)
                image, masks = rotate_record(image, self.annotations[str(record["id"])], angle, scale, rng.random() < .5)
                rotated = derive_geometry(masks)
        if rotated is not None:
            target = {key: torch.from_numpy(value.copy()) for key, value in rotated.items()}
        elif self.geometry:
            path = self.cache_path(record)
            if path is not None and path.exists():
                with np.load(path, allow_pickle=False) as cached:
                    values = dict(cached)
            else:
                if self.require_cache:
                    raise FileNotFoundError("Prepare geometry first: python scripts/prepare_data.py --config configs/flash_large.yaml")
                values = derive_geometry(decode_annotations(self.annotations[str(record["id"])], *image.shape))
            if self.augment:
                rng = random.Random(self.seed + 1000003 * int(self.epoch) + index)
                rotation, flip = rng.randrange(4), rng.random() < 0.5
                image = np.rot90(image, rotation)
                if flip:
                    image = image[:, ::-1]
                values = transform_geometry(values, rotation, flip)
            target = {key: torch.from_numpy(value.copy()) for key, value in values.items()}
        elif self.decode_masks:
            target = {"masks": torch.from_numpy(decode_annotations(self.annotations[str(record["id"])], *image.shape))}
        else:
            target = {}
        target.update(file_name=record["file_name"], record_id=str(record["id"]), image_size=image.shape)
        tensor = torch.from_numpy(image.copy()).unsqueeze(0)
        return (tensor if self.image_uint8 else tensor.float() / 255), target


class MixedDataset(Dataset):
    """All labelled samples plus `extra_per_epoch` pseudo-labelled samples per epoch.

    The extra subset is a fresh seeded permutation each epoch, so an epoch costs
    len(labelled) + extra_per_epoch samples however large the external pool is.
    """

    def __init__(self, labelled, extras, extra_per_epoch, seed=2026):
        self.labelled, self.extras = labelled, extras
        self.extra_per_epoch = min(int(extra_per_epoch), sum(len(e) for e in extras))
        self.offsets = np.cumsum([0] + [len(e) for e in extras])
        self.seed = seed
        self.epoch = torch.zeros((), dtype=torch.int64).share_memory_()
        self.records = labelled.records

    def __len__(self):
        return len(self.labelled) + self.extra_per_epoch

    def set_epoch(self, epoch):
        self.epoch.fill_(epoch)
        for dataset in (self.labelled, *self.extras):
            dataset.set_epoch(epoch)

    def __getitem__(self, index):
        if index < len(self.labelled):
            return self.labelled[index]
        generator = torch.Generator().manual_seed(self.seed + 7907 * int(self.epoch))
        chosen = int(torch.randperm(int(self.offsets[-1]), generator=generator)[index - len(self.labelled)])
        source = int(np.searchsorted(self.offsets, chosen, side="right") - 1)
        return self.extras[source][chosen - int(self.offsets[source])]


def collate(batch):
    return torch.stack([item[0] for item in batch]), [item[1] for item in batch]
