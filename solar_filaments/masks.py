"""One definition of native-resolution masks for supervision and inference.

Instance logits = mask embedding . stride-2 basis (bilinearly upsampled) + phase embedding . the four
sub-pixel phase maps (pixel shuffle), so a one-pixel boundary is resolved at native 2048 px.
"""
import torch
from torch.nn import functional as F


def foreground_probability_mean(logits, weights=None, threshold=.5):
    """Mean on the same detached foreground support used at inference.

    A sigmoid gate in probability space has a nonzero background floor. On a
    2048-square disk that floor can outweigh an entire thin filament. Detaching
    the support keeps gradients through foreground probabilities without that
    image-area-dependent bias. Empty masks use their peak to retain a gradient.
    Optional inverse-proposal weights correct nonuniform native pixel sampling.
    """
    probability = logits.float().sigmoid().flatten(1)
    support = (probability.detach() >= threshold).to(probability.dtype)
    if weights is not None:
        support = support * weights.flatten()[None]
    area = support.sum(-1)
    mean = (probability * support).sum(-1) / area.clamp_min(1e-8)
    return torch.where(area > 0, mean, probability.amax(-1))


def sample_features(features, points, mode="bilinear"):
    # points: P x 2, normalized full-image pixel centers.
    b, c, h, w = features.shape
    if len(points) > h * w // 8:
        return F.grid_sample(features.float(), points[None, None].expand(b, -1, -1, -1) * 2 - 1,
                             mode=mode, padding_mode="border", align_corners=False).squeeze(2)
    # Sparse gathers cast sampled values only. Casting a 64x1024x1024 basis to
    # FP32 for 8192 points wastes bandwidth and a 256-MiB temporary each call.
    xy = points * points.new_tensor([w, h]) - .5
    if mode == "nearest":
        ij = xy.round().long()
        idx = ij[:, 1].clamp(0, h - 1) * w + ij[:, 0].clamp(0, w - 1)
        return features.flatten(2).gather(2, idx[None, None].expand(b, c, -1)).float()
    lower = xy.floor().long()
    delta = xy - lower
    dx = lower.new_tensor([0, 1, 0, 1])
    dy = lower.new_tensor([0, 0, 1, 1])
    x = (lower[:, 0:1] + dx).clamp(0, w - 1)
    y = (lower[:, 1:2] + dy).clamp(0, h - 1)
    indices = (y * w + x).flatten()
    sampled = features.flatten(2).gather(2, indices[None, None].expand(b, c, -1)).float().reshape(b, c, -1, 4)
    wx = torch.where(dx.bool(), delta[:, 0:1], 1 - delta[:, 0:1])
    wy = torch.where(dy.bool(), delta[:, 1:2], 1 - delta[:, 1:2])
    return (sampled * (wx * wy)).sum(-1)


def phase_samples(features, points):
    """Sample pixel-shuffled features without materializing a native feature map.

    Uses bilinear interpolation in the *native* pixel coordinate system, including
    the four different subpixel phases. Identical to sampling pixel_shuffle(...).
    """
    b, channels, h, w = features.shape
    c = channels // 4
    xy = points * points.new_tensor([2 * w, 2 * h]) - 0.5
    origin = xy.floor().long()
    delta = xy - origin
    result = features.new_zeros((b, c, len(points)), dtype=torch.float32)
    flat = features.reshape(b, c, 4, h * w)
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        x = (origin[:, 0] + dx).clamp(0, 2 * w - 1)
        y = (origin[:, 1] + dy).clamp(0, 2 * h - 1)
        phase = (y % 2) * 2 + x % 2
        cell = (y // 2) * w + x // 2
        values = flat[:, :, phase, cell].float()
        weight = (delta[:, 0] if dx else 1 - delta[:, 0]) * (delta[:, 1] if dy else 1 - delta[:, 1])
        result = result + values * weight
    return result


def fine_mask_points(output, index, queries, points):
    basis = sample_features(output["mask_features2"][index:index + 1], points)[0]
    result = output["mask_embeddings"][index, queries].float() @ basis
    if "phase_features2" in output:
        phases = phase_samples(output["phase_features2"][index:index + 1], points)[0]
        result = result + output["phase_embeddings"][index, queries].float() @ phases
    return result


def fine_mask_logits(output, index, queries):
    basis = output["mask_features2"][index]
    logits = torch.einsum("qc,chw->qhw", output["mask_embeddings"][index, queries].to(basis.dtype), basis).float()
    logits = F.interpolate(logits[None], size=output["image_size"], mode="bilinear", align_corners=False)[0]
    if "phase_features2" in output:
        phase = output["phase_features2"][index].reshape(-1, 4, *basis.shape[-2:])
        residual = torch.einsum("qc,cphw->qphw", output["phase_embeddings"][index, queries].to(phase.dtype), phase).float()
        logits = logits + F.pixel_shuffle(residual, 2)[:, 0]
    return logits
