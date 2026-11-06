"""One definition of native-resolution masks for supervision and inference.

Instance logits = mask embedding . stride-2 basis (bilinearly upsampled) + phase embedding . the four
sub-pixel phase maps (pixel shuffle), so a one-pixel boundary is resolved at native 2048 px.
fine_mask_logits renders whole native masks (inference, evaluation); fine_mask_points evaluates the same
function at sampled native pixel centers (training losses) without building native-size tensors.
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
    logits: N x P (or N x H x W) native mask logits; weights: one per point; returns N means.
    """
    probability = logits.float().sigmoid().flatten(1)
    support = (probability.detach() >= threshold).to(probability.dtype)
    if weights is not None:
        support = support * weights.flatten()[None]
    area = support.sum(-1)
    mean = (probability * support).sum(-1) / area.clamp_min(1e-8)
    return torch.where(area > 0, mean, probability.amax(-1))


def sample_features(features, points, mode="bilinear"):
    """Sample B x C x h x w features at P points; returns B x C x P in float32.

    points: P x 2, normalized full-image pixel centers: (x, y) in [0, 1], a native pixel (i, j) being
    ((j + .5) / W, (i + .5) / H). Bilinear or nearest sampling as F.grid_sample with align_corners=False and
    border padding, so a stride-2 map sampled at native pixel centers equals its bilinear 2x upsampling.
    More than h * w / 8 points use grid_sample on the whole map; fewer gather only the needed neighbors.
    """
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
    features: B x 4C x h x w, channel 4c + phase with phase = 2 * (row % 2) + (column % 2) of the native pixel;
    points as in sample_features. Returns B x C x P in float32.
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
    """Native mask logits of the selected queries of image `index` at P normalized points: Q x P in float32.

    The same function as fine_mask_logits at those pixel centers (tests/test_native_pq.py checks this), but
    only the needed basis and phase values are sampled. queries indexes the query axis (slice, list, tensor).
    """
    basis = sample_features(output["mask_features2"][index:index + 1], points)[0]
    result = output["mask_embeddings"][index, queries].float() @ basis
    if "phase_features2" in output:
        phases = phase_samples(output["phase_features2"][index:index + 1], points)[0]
        result = result + output["phase_embeddings"][index, queries].float() @ phases
    return result


def fine_mask_logits(output, index, queries):
    """Native-resolution mask logits of the selected queries of image `index`: Q x H x W in float32.

    The embedding . basis product is formed at stride 2 and bilinearly upsampled to output['image_size'];
    the phase residual (four sub-pixel values per stride-2 cell) is pixel-shuffled to native resolution and
    added. A 2048 x 2048 float32 mask takes 16 MiB, so callers render queries in small chunks.
    """
    basis = output["mask_features2"][index]
    logits = torch.einsum("qc,chw->qhw", output["mask_embeddings"][index, queries].to(basis.dtype), basis).float()
    logits = F.interpolate(logits[None], size=output["image_size"], mode="bilinear", align_corners=False)[0]
    if "phase_features2" in output:
        phase = output["phase_features2"][index].reshape(-1, 4, *basis.shape[-2:])
        residual = torch.einsum("qc,cphw->qphw", output["phase_embeddings"][index, queries].to(phase.dtype), phase).float()
        logits = logits + F.pixel_shuffle(residual, 2)[:, 0]
    return logits
