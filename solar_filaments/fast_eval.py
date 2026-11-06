"""Exact official PQ on the GPU for many operating points from one forward pass.

`owned_masks` reproduces inference.instance_predictions (score-ordered pixel
competition on probability*score, minimum area, retained fraction) without
RLE encoding. `pq_counts` uses exact pixel IoU with the organizer rule IoU>0.5,
so aggregated TP/FP/FN/IoU equal metrics.official_pq on the same masks.
"""
import torch
from pycocotools import mask as mask_utils

from .masks import fine_mask_logits


@torch.no_grad()
def query_probabilities(output, index=0, min_score=.05, chunk=8, original_size=None, dtype=torch.float16):
    """Native probabilities (FP16 by default) for queries whose class*quality score >= min_score.

    Returns (scores K, probabilities K x H x W) in descending score order; original_size crops the padding.
    Rendering once at the lowest score threshold of a sweep serves every operating point.
    """
    scores = (output["pred_logits"][index].float().softmax(-1)[:, 0] *
              output["pred_quality"][index].float().sigmoid())
    ids = torch.where(scores >= min_score)[0]
    ids = ids[scores[ids].argsort(descending=True)]
    h, w = original_size or output["image_size"]
    probabilities = []
    for start in range(0, len(ids), chunk):
        part = ids[start:start + chunk]
        probabilities.append(fine_mask_logits(output, index, part)[:, :h, :w].float().sigmoid().to(dtype))
    if probabilities:
        probabilities = torch.cat(probabilities)
    else:
        probabilities = scores.new_zeros((0, h, w), dtype=dtype)
    return scores[ids], probabilities


@torch.no_grad()
def owned_masks(scores, probabilities, score_threshold, mask_threshold, min_area=32, min_retained_fraction=.5):
    """Boolean instance masks and scores after exclusive pixel ownership.

    Same rule as inference.instance_predictions at one (score_threshold, mask_threshold) pair. scores and
    probabilities come from query_probabilities (descending score). Returns (K x H x W bool masks, final
    scores).
    """
    h, w = probabilities.shape[-2:]
    winning = probabilities.new_zeros((h, w), dtype=torch.float32)
    labels = torch.full((h, w), -1, dtype=torch.int32, device=probabilities.device)
    kept = []
    for j in range(len(scores)):  # already sorted by descending score
        if float(scores[j]) < score_threshold:
            break
        probability = probabilities[j].float()
        foreground = probability >= mask_threshold
        area = int(foreground.sum())
        if area < min_area:
            continue
        score = float(scores[j]) * float(probability[foreground].mean())
        if score < score_threshold:
            continue
        confidence = probability * score
        replace = foreground & (confidence > winning)
        winning[replace] = confidence[replace]
        labels[replace] = j
        kept.append((j, area, score))
    masks, result_scores = [], []
    for j, area, score in kept:
        mask = labels == j
        retained = int(mask.sum())
        if retained < min_area or retained / area < min_retained_fraction:
            continue
        masks.append(mask)
        result_scores.append(score)
    if masks:
        return torch.stack(masks), result_scores
    return torch.zeros((0, h, w), dtype=torch.bool, device=probabilities.device), []


def decode_truth(annotations, height, width, device):
    """GT instance masks of one annotator record (COCO polygons) as an N x H x W bool tensor on `device`."""
    masks = []
    for annotation in annotations:
        rles = mask_utils.frPyObjects(annotation["segmentation"], height, width)
        rle = mask_utils.merge(rles) if isinstance(rles, list) else rles
        masks.append(torch.from_numpy(mask_utils.decode(rle)).to(device, non_blocking=True).bool())
    if masks:
        return torch.stack(masks)
    return torch.zeros((0, height, width), dtype=torch.bool, device=device)


@torch.no_grad()
def pq_counts(prediction, truth, chunk=8):
    """Organizer PQ terms: TP iff IoU strictly > .5 (one-to-one by construction).

    prediction: K x H x W bool, truth: N x H x W bool, on the same device. Returns this record's matched IoU
    sum and TP/FP/FN counts, the terms of metrics.official_pq.
    """
    k, n = len(prediction), len(truth)
    if k == 0 or n == 0:
        return {"iou_sum": 0.0, "tp": 0, "fp": k, "fn": n}
    g = truth.flatten(1).float()
    area_g = g.sum(-1)
    inter = []
    for start in range(0, k, chunk):
        inter.append(prediction[start:start + chunk].flatten(1).float() @ g.T)
    inter = torch.cat(inter)
    area_p = prediction.flatten(1).sum(-1).float()
    iou = inter / (area_p[:, None] + area_g[None] - inter).clamp_min(1)
    matched = iou > .5
    return {"iou_sum": float(iou[matched].sum()), "tp": int(matched.sum()),
            "fp": int((~matched.any(1)).sum()), "fn": int((~matched.any(0)).sum())}


def summarize(rows):
    """Pool per-record pq_counts like metrics.aggregate_pq: PQ = sum of TP IoUs / (TP + FP/2 + FN/2)."""
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    iou = sum(r["iou_sum"] for r in rows)
    denominator = tp + .5 * fp + .5 * fn
    return {"pq": iou / denominator if denominator else 0.0, "sq": iou / tp if tp else 0.0,
            "rq": tp / denominator if denominator else 0.0, "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0}
