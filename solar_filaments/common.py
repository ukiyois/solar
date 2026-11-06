"""Small layers shared by the detail path, the curve blocks and the context pyramid.

DropPath and trunc_normal_ are also imported by the vendored FlashInternImage (vendor/flash_intern_image.py),
which therefore does not need timm.
"""
import torch
from torch import nn
from torch.nn.init import trunc_normal_


class DropPath(nn.Module):
    """Stochastic depth for residual branches (backbone drop_path_rate).

    In training, each sample's branch output is zeroed with probability drop_prob and the kept ones are
    scaled by 1 / (1 - drop_prob), so the expected output is unchanged; identity in evaluation.
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class LayerNorm2d(nn.Module):
    """Channel-only normalization: independent of tile extent and batch size.

    LayerNorm over the C channels of every pixel of a B x C x H x W map (eps 1e-6). No spatial or batch
    statistics enter, so the tiled detail path computes the same values as the full-image path.
    """
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x):
        # LayerNorm returns contiguous NHWC. Its NCHW view is already channels-
        # last; forcing conventional contiguous storage adds a full-map copy.
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNormAct(nn.Sequential):
    """Conv2d with 'same' padding (kernel // 2), then LayerNorm2d and GELU."""

    def __init__(self, cin, cout, kernel=3, stride=1, groups=1):
        super().__init__(nn.Conv2d(cin, cout, kernel, stride, kernel // 2, groups=groups),
                         LayerNorm2d(cout), nn.GELU())


class SeparableBlock(nn.Module):
    """Residual depthwise-separable block: x + 1x1 conv(GELU(LayerNorm2d(depthwise 3x3 conv(x))))."""

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(ConvNormAct(channels, channels, groups=channels),
                                 nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        return x + self.net(x)
