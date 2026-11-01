"""Segmentation-only geometry cached on CPU. Four native pixels per byte."""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
import torch
from torch.nn import functional as F

CACHE_VERSION = 4


def pack_masks(masks):
    return (masks[:, 0::2, 0::2].astype(np.uint8) |
            (masks[:, 0::2, 1::2].astype(np.uint8) << 1) |
            (masks[:, 1::2, 0::2].astype(np.uint8) << 2) |
            (masks[:, 1::2, 1::2].astype(np.uint8) << 3))


def unpack_masks(packed):
    phases = torch.stack([((packed >> bit) & 1) for bit in range(4)], 1)
    return F.pixel_shuffle(phases.float(), 2)[:, 0].bool()


def target_at_points(target, points):
    if "packed_masks" not in target:
        masks = target["masks"]
        h, w = masks.shape[-2:]
        x = (points[:, 0] * w).long().clamp(0, w - 1)
        y = (points[:, 1] * h).long().clamp(0, h - 1)
        return masks[:, y, x].float()
    packed = target["packed_masks"]
    h2, w2 = packed.shape[-2:]
    x = (points[:, 0] * (2 * w2)).long().clamp(0, 2 * w2 - 1)
    y = (points[:, 1] * (2 * h2)).long().clamp(0, 2 * h2 - 1)
    return ((packed[:, y // 2, x // 2] >> (2 * (y % 2) + x % 2)) & 1).float()


def _coordinates(binary, count, rng, size):
    y, x = np.nonzero(binary)
    if not len(x):
        return np.full((count, 2), 0.5, np.float32)
    chosen = rng.integers(len(x), size=count)
    return np.stack(((x[chosen] + 0.5) / size[1], (y[chosen] + 0.5) / size[0]), -1).astype(np.float32)


def derive_geometry(masks, anchors=512, roi_size=128):
    n, h, w = masks.shape
    if h % 4 or w % 4:
        raise ValueError("Geometry cache expects dimensions divisible by four")
    rng = np.random.default_rng(2026)
    auxiliary = np.zeros((3, h, w), bool)
    positive, skeleton_points, edge_points, rois, roi_masks, roi_skeletons, roi_distances = [], [], [], [], [], [], []
    for mask in masks:
        y, x = np.nonzero(mask)
        if not len(x):
            raise ValueError("Empty annotated polygon; inspect the source record")
        y0, y1, x0, x1 = max(0, y.min() - 2), min(h, y.max() + 3), max(0, x.min() - 2), min(w, x.max() + 3)
        crop = mask[y0:y1, x0:x1]
        skeleton = np.zeros_like(mask)
        skeleton[y0:y1, x0:x1] = skeletonize(crop)
        edge = np.zeros_like(mask)
        edge[y0:y1, x0:x1] = ndi.binary_dilation(crop) ^ ndi.binary_erosion(crop)
        auxiliary[0] |= mask
        auxiliary[1] |= skeleton
        auxiliary[2] |= edge
        positive.append(_coordinates(mask, anchors, rng, (h, w)))
        skeleton_points.append(_coordinates(skeleton, anchors, rng, (h, w)))
        edge_points.append(_coordinates(edge, anchors, rng, (h, w)))
        side = max(32, int(max(y.max() - y.min() + 1, x.max() - x.min() + 1)) + 24)
        cy, cx = (y.min() + y.max() + 1) / 2, (x.min() + x.max() + 1) / 2
        roi = np.array([(cx - side / 2) / w, (cy - side / 2) / h,
                        (cx + side / 2) / w, (cy + side / 2) / h], np.float32)
        rois.append(roi)
        axis = (torch.arange(roi_size) + 0.5) / roi_size
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        grid = torch.stack((roi[0] + xx * (roi[2] - roi[0]), roi[1] + yy * (roi[3] - roi[1])), -1)
        sampled = F.grid_sample(torch.from_numpy(mask[None, None].astype(np.float32)), grid[None] * 2 - 1,
                                mode="nearest", align_corners=False)[0, 0].numpy() > 0
        roi_masks.append(sampled)
        roi_skeletons.append(skeletonize(sampled))
        distance = ndi.distance_transform_edt(~sampled) - ndi.distance_transform_edt(sampled)
        roi_distances.append(np.clip(distance / 4, -1, 1))
    packed = pack_masks(masks)
    area4 = masks.reshape(n, h // 4, 4, w // 4, 4).sum((2, 4)).astype(np.uint8)
    aux2 = auxiliary.reshape(3, h // 2, 2, w // 2, 2).max((2, 4)).astype(np.uint8)
    def stacked(values, shape, dtype):
        return np.stack(values).astype(dtype) if n else np.zeros((0, *shape), dtype)
    return {"packed_masks": packed, "occupancy4": area4, "auxiliary": aux2,
            "positive_points": stacked(positive, (anchors, 2), np.float32),
            "skeleton_points": stacked(skeleton_points, (anchors, 2), np.float32),
            "edge_points": stacked(edge_points, (anchors, 2), np.float32),
            "rois": stacked(rois, (4,), np.float32),
            "roi_masks": stacked(roi_masks, (roi_size, roi_size), np.uint8),
            "roi_skeletons": stacked(roi_skeletons, (roi_size, roi_size), np.uint8),
            "roi_distance": stacked(roi_distances, (roi_size, roi_size), np.float16)}


def transform_geometry(geometry, rotation, flip):
    result = {k: v.copy() for k, v in geometry.items()}
    packed = np.rot90(result["packed_masks"], rotation, axes=(-2, -1))
    phases = np.rot90(np.arange(4).reshape(2, 2), rotation)
    if flip:
        packed, phases = packed[..., ::-1], phases[:, ::-1]
    transformed = np.zeros_like(packed)
    for new_bit, old_bit in enumerate(phases.flatten()):
        transformed |= (((packed >> int(old_bit)) & 1) << new_bit).astype(np.uint8)
    result["packed_masks"] = transformed
    for key in ("occupancy4", "auxiliary", "roi_masks", "roi_skeletons", "roi_distance"):
        value = np.rot90(result[key], rotation, axes=(-2, -1))
        result[key] = (value[..., ::-1] if flip else value).copy()
    def transform_xy(xy):
        xy = xy.copy()
        for _ in range(rotation):
            xy = np.stack((xy[..., 1], 1 - xy[..., 0]), -1)
        if flip:
            xy[..., 0] = 1 - xy[..., 0]
        return xy.copy()
    for key in ("positive_points", "skeleton_points", "edge_points"):
        result[key] = transform_xy(result[key])
    boxes = result["rois"]
    a, b = transform_xy(boxes[:, :2]), transform_xy(boxes[:, 2:])
    result["rois"] = np.concatenate((np.minimum(a, b), np.maximum(a, b)), -1)
    return result
