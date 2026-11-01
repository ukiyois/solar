import torch
from torch import nn
from torch.nn.init import trunc_normal_


class DropPath(nn.Module):
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
    """Channel-only normalization: independent of tile extent and batch size."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)

    def forward(self, x):
        # LayerNorm returns contiguous NHWC. Its NCHW view is already channels-
        # last; forcing conventional contiguous storage adds a full-map copy.
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNormAct(nn.Sequential):
    def __init__(self, cin, cout, kernel=3, stride=1, groups=1):
        super().__init__(nn.Conv2d(cin, cout, kernel, stride, kernel // 2, groups=groups),
                         LayerNorm2d(cout), nn.GELU())


class SeparableBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(ConvNormAct(channels, channels, groups=channels),
                                 nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        return x + self.net(x)
