"""Training-only mask pilots derived exclusively from segmentation polygons.

MP-Former (Zhang et al., CVPR 2023): extra decoder queries receive noised ground-truth masks as attention
guides and are trained to reconstruct them, which stabilizes mask-guided attention. The branch shares the
decoder weights, is isolated from the normal queries and is absent at inference (decoder.piloted_decode).
"""
import torch
from torch.nn import functional as F


@torch.no_grad()
def prepare_mask_pilots(targets, slots):
    """Boolean attention guides B x slots x H/16 x W/16 for the pilot branch, or None when slots <= 0.

    Slot i holds GT instance i of an image: its stride-4 occupancy (targets[k]['occupancy4']) max-pooled by a
    further factor of 4 (at least 4 x 4 cells). Remaining slots stay empty and are trained as no object.
    """
    if slots <= 0:
        return None
    result = []
    for target in targets:
        occupancy = target['occupancy4']
        n, h, w = occupancy.shape
        if n > slots:
            raise ValueError(f'{n} GT instances exceed the {slots} pilot slots')
        # Max pooling preserves thin foreground. Independent area-proportional
        # point noise is added inside each decoder layer, following MP-Former.
        size = (max(4, h // 4), max(4, w // 4))
        guide = torch.zeros(slots, *size, dtype=torch.bool, device=occupancy.device)
        if n:
            guide[:n] = F.adaptive_max_pool2d((occupancy > 0).float()[None], size)[0] > 0
        result.append(guide)
    return torch.stack(result)


def noisy_pilot_support(support, ratio):
    """Flip points with E[count] <= ratio * foreground area per instance.

    The noise is local point corruption, not whole-mask translation/scaling;
    MP-Former's published ablation favored point noise over those alternatives.
    No noise is introduced into unused padding slots.
    support: B x slots x M boolean. Each slot draws a flip rate uniformly in [0, ratio x its foreground
    fraction] and flips every token (foreground or background) independently with that probability.
    """
    rate = torch.rand_like(support[..., :1].float()) * ratio * support.float().mean(-1, keepdim=True)
    flip = torch.rand_like(support.float()) < rate
    return support ^ flip
