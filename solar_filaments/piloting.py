"""Training-only mask pilots derived exclusively from segmentation polygons."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def prepare_mask_pilots(targets, slots):
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
    """
    rate = torch.rand_like(support[..., :1].float()) * ratio * support.float().mean(-1, keepdim=True)
    flip = torch.rand_like(support.float()) < rate
    return support ^ flip
