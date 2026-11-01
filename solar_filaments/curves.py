"""Continuous orthogonal curve sampling, inspired by ICCV 2023 DSConv.

This is a new portable implementation, not a copy of DSCNet. It constrains
successive perpendicular increments and retains a conventional local path.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .common import LayerNorm2d, SeparableBlock
from .ops.dcnv4 import native_function
from .ops.native import dcnv4_cuda


class CurveAggregation(nn.Module):
    def __init__(self, channels, points=7, step=0.75, backend="auto"):
        super().__init__()
        self.points, self.step = points, step
        self.backend = backend
        self.groups = max(1, channels // 16)
        if channels % self.groups:
            raise ValueError("Curve channels must split into equal groups")
        self.offsets = nn.Conv2d(channels, 2 * points, 3, padding=1)
        nn.init.zeros_(self.offsets.weight)
        nn.init.zeros_(self.offsets.bias)
        # Shared spatial aggregation within a group; surrounding learned 1x1
        # projections retain channel mixing. This maps to the fused CUDA op.
        self.weights = nn.Parameter(torch.full((2, self.groups, points), 1 / points))
        self.mix = nn.Sequential(nn.Conv2d(2 * channels, channels, 1), LayerNorm2d(channels), nn.GELU())

    def bends(self, x):
        b, c, h, w = x.shape
        k, middle = self.points, self.points // 2
        offsets = self.offsets(x).reshape(b, 2, k, h, w).tanh() * self.step
        # The center stays fixed. Each side accumulates outward from the center.
        negative = -offsets[:, :, :middle].flip(2).cumsum(2).flip(2)
        positive = offsets[:, :, middle + 1:].cumsum(2)
        return torch.cat((negative, offsets[:, :, middle:middle + 1] * 0, positive), 2)

    def forward_native(self, x, bends):
        """Pack the continuous 7-point curves into a 9-sample DCNv4 kernel.

        Two unused samples receive zero weight. Offsets cancel the native 3x3
        base grid and replace it with the constrained curve. No KxCxHxW sample
        volume is materialized. This is still curve-constrained aggregation.
        """
        b, c, h, w = x.shape
        k, groups = self.points, self.groups
        value = x.permute(0, 2, 3, 1).contiguous()
        base = bends.new_tensor([(i - 1, j - 1) for i in range(3) for j in range(3)])
        axis = torch.arange(-(k // 2), k // 2 + 1, device=x.device, dtype=bends.dtype).view(1, 1, 1, k)
        outputs = []
        for direction in range(2):
            bend = bends[:, direction].permute(0, 2, 3, 1)
            dx = (axis.expand_as(bend) if direction == 0 else bend) - base[:k, 0]
            dy = (bend if direction == 0 else axis.expand_as(bend)) - base[:k, 1]
            xy = F.pad(torch.stack((dx, dy), -1), (0, 0, 0, 9 - k))
            xy = xy.reshape(b, h, w, 1, 18).expand(-1, -1, -1, groups, -1)
            weights = F.pad(self.weights[direction], (0, 9 - k)).to(x.dtype)
            weights = weights[None, None, None].expand(b, h, w, -1, -1)
            packed = torch.cat((xy, weights), -1).flatten(3)
            # CUDA autocast may promote pad to FP32. The extension requires
            # packed offsets and values to have the same storage dtype.
            packed = F.pad(packed, (0, (-packed.shape[-1]) % 8)).to(value.dtype).contiguous()
            sampled = dcnv4_cuda(value, packed, groups, 1., 3, 0)
            outputs.append(sampled.permute(0, 3, 1, 2))
        return self.mix(torch.cat(outputs, 1))

    def forward_reference(self, x, bends):
        b, c, h, w = x.shape
        k, middle = self.points, self.points // 2
        yy, xx = torch.meshgrid(torch.arange(h, device=x.device, dtype=torch.float32),
                                torch.arange(w, device=x.device, dtype=torch.float32), indexing="ij")
        axis = torch.arange(-middle, middle + 1, device=x.device).view(1, k, 1, 1)
        outputs = []
        for direction in range(2):
            gx = xx + (axis if direction == 0 else bends[:, direction].float())
            gy = yy + (bends[:, direction].float() if direction == 0 else axis)
            gx, gy = torch.broadcast_tensors(gx, gy)
            grid = torch.stack((2 * (gx + 0.5) / w - 1, 2 * (gy + 0.5) / h - 1), -1)
            grid = grid.permute(0, 2, 3, 1, 4).reshape(b, h, w * k, 2)
            with torch.autocast(x.device.type, enabled=False):
                sampled = F.grid_sample(x.float(), grid.float(), align_corners=False, padding_mode="zeros")
                sampled = sampled.reshape(b, c, h, w, k)
                weight = self.weights[direction].repeat_interleave(c // self.groups, 0)
                output = (sampled * weight.float()[None, :, None, None, :]).sum(-1)
            outputs.append(output.to(x.dtype))
        return self.mix(torch.cat(outputs, 1))

    def forward(self, x):
        bends = self.bends(x)
        supported = x.is_cuda and x.shape[1] // self.groups % 16 == 0 and self.points <= 9
        use_native = supported and (self.backend == "native" or
                                   self.backend == "auto" and native_function() is not None)
        return self.forward_native(x, bends) if use_native else self.forward_reference(x, bends)


class CurveResidualBlock(nn.Module):
    def __init__(self, channels, points=7, step=0.75, enabled=True, curve_channels=16, backend="auto"):
        super().__init__()
        self.local = SeparableBlock(channels)
        width = min(channels, curve_channels)
        if width >= 16:
            width = (width // 16) * 16
        self.reduce = nn.Conv2d(channels, width, 1) if enabled else None
        self.curve = CurveAggregation(width, points, step, backend) if enabled else None
        self.expand = nn.Conv2d(width, channels, 1) if enabled else None
        self.gamma = nn.Parameter(torch.full((1, channels, 1, 1), 1e-4)) if enabled else None

    def forward(self, x):
        local = self.local(x)
        return local if self.curve is None else local + self.gamma.to(local.dtype) * self.expand(self.curve(self.reduce(x)))
