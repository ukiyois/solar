"""Inference-aligned sampled panoptic forward, with straight-through gradients.

Discrete decisions have their inference values in the forward pass. Backward
uses smooth local approximations; this is a biased gradient estimator, not a
claim that the official discrete PQ metric is differentiable. Native-pixel
importance sampling still makes areas and IoUs finite-sample estimates.

Selected by losses.FilamentCriterion(pq_surrogate='straight_through'). The final recipe uses the soft
surrogate in losses.py instead: this variant lowered fold-0 PQ (docs/TECHNICAL_REPORT.md, section 10).
"""
import torch
import math
from scipy.optimize import linear_sum_assignment


def straight_through(soft, hard):
    """Straight-through estimator (Bengio et al., 2013): the value of `hard`, the gradient of `soft`."""
    return soft + (hard.to(soft.dtype) - soft).detach()


def sampled_panoptic_terms(logits, confidence, truth, weights, image_area,
                           score_threshold=.35, iou_threshold=.5,
                           score_temperature=.05, iou_temperature=.1,
                           mask_temperature=.5, owner_temperature=.15,
                           min_area=32, min_retained_fraction=.5, mask_threshold=.5):
    """PQ numerator and denominator of one annotator record: hard decisions forward, soft gradients backward.

    logits: Q x P native mask logits at P importance-sampled pixels; confidence: Q query scores
    (P(filament) * sigmoid(quality) * mean foreground probability); truth: N x P GT labels; weights: P
    inverse-proposal weights (losses.native_area_points); image_area: native pixel count, which turns
    weighted sample sums into pixel areas. The forward pass applies the rule of
    inference.instance_predictions: admission by score and min_area, exclusive pixel ownership by the
    highest probability * confidence, and the min_area / min_retained_fraction filter. Hungarian matching
    after ownership maximizes the number of IoU > iou_threshold matches first. Backward uses the soft
    detection, ownership, mask and TP sigmoids of the soft surrogate with the given temperatures.
    Returns (numerator = sum of TP IoUs, denominator = (#kept queries + #GT) / 2, diagnostics); the caller
    sums both over the batch before dividing.
    """
    probability = logits.float().sigmoid()
    weight = weights.float()[None]
    detection_soft = ((confidence - score_threshold) / score_temperature).sigmoid()
    owner_logits = (probability.clamp_min(1e-8).log() +
                    confidence.detach().clamp_min(1e-8).log()[:, None] +
                    detection_soft.detach().clamp_min(1e-8).log()[:, None]) / owner_temperature
    centered = logits.float() - math.log(mask_threshold / (1 - mask_threshold))
    soft_owned = (centered / mask_temperature).sigmoid() * owner_logits.softmax(0)
    with torch.no_grad():
        foreground = centered >= 0
        # Each draw has weight 1/(N_cells * p(cell)); its mean estimates
        # native image occupancy. Do not normalize by sampled weight sum.
        pixel_scale = float(image_area) / logits.shape[-1]
        raw_area = (foreground * weight).sum(-1) * pixel_scale
        admitted = (confidence >= score_threshold) & (raw_area >= min_area)
        valid = foreground & admitted[:, None]
        priority = (probability * confidence[:, None]).masked_fill(~valid, -1.)
        winner = priority.argmax(0)
        hard_owned = (torch.arange(len(logits), device=logits.device)[:, None] == winner[None]) & valid.any(0)[None]
        retained_area = (hard_owned * weight).sum(-1) * pixel_scale
        kept = admitted & (retained_area >= min_area) & (
            retained_area / raw_area.clamp_min(1e-8) >= min_retained_fraction)
        # Inference leaves holes after rejecting an instance; do not reassign.
        hard_owned = hard_owned & kept[:, None]
    owned = straight_through(soft_owned, hard_owned)
    detected = straight_through(detection_soft, kept)
    y = truth.float()
    intersection = (owned * weight) @ y.T
    union = (owned * weight).sum(-1, keepdim=True) + (y * weight).sum(-1)[None] - intersection
    iou = intersection / union.clamp_min(1e-8)
    if y.shape[0]:
        with torch.no_grad():
            # Assign after ownership, maximizing number of >.5 matches first,
            # then their IoU. Auxiliary mask assignment remains independent.
            # A match is worth N + 1, more than any possible IoU sum, which makes the order lexicographic.
            reward = ((iou.detach() > iou_threshold).float() * (y.shape[0] + 1) +
                      iou.detach()) * kept[:, None]
            row, column = linear_sum_assignment(-reward.cpu().numpy())
            row = torch.as_tensor(row, device=logits.device)
            column = torch.as_tensor(column, device=logits.device)
        matched_iou = iou[row, column]
        quality_soft = ((matched_iou - iou_threshold) / iou_temperature).sigmoid()
        matched = straight_through(quality_soft, matched_iou.detach() > iou_threshold) * detected[row]
        numerator = (matched * matched_iou).sum()
        mean_iou = matched_iou.detach().mean()
    else:
        numerator = owned.sum() * 0
        mean_iou = numerator.detach()
    denominator = .5 * (detected.sum() + y.shape[0])
    return numerator, denominator, {
        'sampled_hard_pq': numerator.detach() / denominator.detach().clamp_min(1e-8),
        'sampled_kept_queries': kept.float().sum(),
        'sampled_matched_owned_iou': mean_iou,
    }
