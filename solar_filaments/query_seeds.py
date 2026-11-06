"""Image-conditioned query locations from native-detail candidate features.

The candidates are peaks of the decoder's seed score, a 1x1 head on the stride-4 detail features trained
against GT skeleton occupancy. Each query starts at a different strong, spatially separated candidate, so the
initial queries depend on the image instead of being fixed learned embeddings only.
"""
import torch
from torch.nn import functional as F


@torch.no_grad()
def query_seed_indices(logits, queries, candidate_count=1024, diversity=.5):
    """Keep strong peaks, then cover remaining locations with quality-weighted FPS.

    Coordinates and ranking are detached; gathered feature values still receive
    gradients. Nearby peaks along the same long filament otherwise consume a
    disproportionate number of initial queries. The spatial preference is soft:
    low-confidence distant background does not automatically outrank a filament.
    logits: B x 1 x h x w seed logits; returns B x queries indices into the flattened h x w grid. Candidates are
    the candidate_count best 7x7 local maxima. The first queries // 4 are the strongest; every further one
    maximizes sigmoid(logit)^2 * d^(2 * diversity), d being the distance (normalized coordinates) to the
    nearest location already chosen. diversity = 0 returns the strongest peaks.
    """
    b, _, h, w = logits.shape
    value = logits[:, 0].float()
    peaks = value == F.max_pool2d(value[:, None], 7, 1, 3)[:, 0]
    ranked = torch.where(peaks, value, value - 100).flatten(1)
    count = min(max(queries, candidate_count), h * w)
    indices = ranked.topk(count, dim=1).indices
    if diversity == 0:
        return indices[:, :queries]
    quality = value.flatten(1).gather(1, indices).sigmoid().square()
    xy = torch.stack(((indices % w + .5) / w, (indices // w + .5) / h), -1)
    # Preserve the most reliable quarter, then spend the other slots on distinct
    # high-quality locations. All arithmetic remains on the CUDA device.
    initial = max(1, queries // 4)
    chosen = [torch.full((b,), i, device=logits.device, dtype=torch.long) for i in range(initial)]
    distance = ((xy[:, :, None] - xy[:, None, :initial]).square().sum(-1)).amin(-1)
    batch = torch.arange(b, device=logits.device)
    for _ in range(initial, queries):
        priority = quality * distance.clamp_min(0).pow(diversity)
        priority = torch.where(distance > 0, priority, -torch.ones_like(priority))
        next_index = priority.argmax(-1)
        chosen.append(next_index)
        center = xy[batch, next_index]
        distance = torch.minimum(distance, (xy - center[:, None]).square().sum(-1))
    chosen = torch.stack(chosen, 1)
    return indices.gather(1, chosen)
