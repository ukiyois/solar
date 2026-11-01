import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .common import ConvNormAct, SeparableBlock
from .curves import CurveResidualBlock


def haar_split(x):
    """Orthogonal, invertible rearrangement of all four 2x2 subpixels."""
    b, c, h, w = x.shape
    p = F.pixel_unshuffle(x, 2).reshape(b, c, 4, h // 2, w // 2)
    a, b_, c_, d = p.unbind(2)
    return torch.stack((a + b_ + c_ + d, a - b_ + c_ - d,
                        a + b_ - c_ - d, a - b_ - c_ + d), 2).flatten(1, 2) * 0.5


def haar_inverse(x):
    b, c4, h, w = x.shape
    ll, lh, hl, hh = x.reshape(b, c4 // 4, 4, h, w).unbind(2)
    p = torch.stack((ll + lh + hl + hh, ll - lh + hl - hh,
                     ll + lh - hl - hh, ll - lh - hl + hh), 2) * 0.5
    return F.pixel_shuffle(p.flatten(1, 2), 2)


class DetailDownsample(nn.Module):
    def __init__(self, cin, cout, method):
        super().__init__()
        self.method = method
        self.project = ConvNormAct(4 * cin, cout, kernel=1) if method == "haar" else ConvNormAct(cin, cout, stride=2)

    def forward(self, x):
        return self.project(haar_split(x) if self.method == "haar" else x)


def sample_context(context, bounds, full_size, output_size):
    """Native pixel centers -> the FlashInternImage stride-4 stem anchors.

    Two 3x3/stride-2/pad-1 convolutions keep the first anchor at view pixel .5,
    rather than at the stride-4 cell center 2. This is a .375-feature-cell shift.
    The original-image resize affine is included via full_size/context.shape.
    """
    y0, x0, y1, x1 = bounds
    height, width = full_size
    oh, ow = output_size
    y = y0 + (torch.arange(oh, device=context.device, dtype=torch.float32) + 0.5) * (y1 - y0) / oh
    x = x0 + (torch.arange(ow, device=context.device, dtype=torch.float32) + 0.5) * (x1 - x0) / ow
    yy, xx = torch.meshgrid(2 * y / height - 1 + .75 / context.shape[-2],
                            2 * x / width - 1 + .75 / context.shape[-1], indexing="ij")
    grid = torch.stack((xx, yy), -1).unsqueeze(0).expand(context.shape[0], -1, -1, -1)
    with torch.autocast(context.device.type, enabled=False):
        result = F.grid_sample(context.float(), grid, align_corners=False, padding_mode="border")
    return result.to(context.dtype)


def upsample_stem_aligned(source, size):
    """All padded stride-2 backbone stages share their first spatial anchor.

    Target index j maps to source j/2, not j/2-.25 as in an ordinary 2x resize.
    This preserves the same anchor through the top-down context pyramid.
    """
    h, w = source.shape[-2:]
    oh, ow = size
    y = torch.arange(oh, device=source.device, dtype=torch.float32) * h / oh
    x = torch.arange(ow, device=source.device, dtype=torch.float32) * w / ow
    yy, xx = torch.meshgrid(2 * (y + .5) / h - 1, 2 * (x + .5) / w - 1, indexing="ij")
    grid = torch.stack((xx, yy), -1)[None].expand(source.shape[0], -1, -1, -1)
    with torch.autocast(source.device.type, enabled=False):
        result = F.grid_sample(source.float(), grid, align_corners=False, padding_mode="border")
    return result.to(source.dtype)


class ContextPyramid(nn.Module):
    def __init__(self, channels, dim):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, dim, 1) for c in channels])
        self.refine = nn.ModuleList([SeparableBlock(dim) for _ in channels])

    def forward(self, features):
        outputs = [None] * len(features)
        previous = None
        for i in range(len(features) - 1, -1, -1):
            current = self.lateral[i](features[i])
            if previous is not None:
                current = current + upsample_stem_aligned(previous, current.shape[-2:])
            previous = self.refine[i](current)
            outputs[i] = previous
        return outputs


class ContextGate(nn.Module):
    def __init__(self, dim, channels):
        super().__init__()
        self.project = nn.Conv2d(dim, channels, 1)
        self.gate = nn.Conv2d(channels * 2, channels, 1)
        self.strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, local, context, bounds, full_size, projected=False):
        # A linear 1x1 projection commutes with bilinear sampling (border pad).
        # Project first: 32x1024² instead of 192x1024² for the finest gate.
        context = context if projected else self.project(context)
        context = sample_context(context, bounds, full_size, local.shape[-2:])
        gate = self.gate(torch.cat((local, context), 1)).sigmoid()
        return local + self.strength * gate * context


class NativeDetailPath(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.detail_channels
        self.stem = DetailDownsample(3 if config.detail_flattening else 2, c, config.detail_downsample)
        self.gate2 = ContextGate(config.feature_dim, c)
        self.curve2 = CurveResidualBlock(c, config.curve_points, config.curve_step, config.curve_enabled, config.curve_channels, config.dcn_backend)
        self.down = DetailDownsample(c, 2 * c, config.detail_downsample)
        self.gate4 = ContextGate(config.feature_dim, 2 * c)
        self.curve4 = CurveResidualBlock(2 * c, config.curve_points, config.curve_step, config.curve_enabled, config.curve_channels, config.dcn_backend)
        self.output4 = nn.Conv2d(2 * c, config.feature_dim, 1)
        self.context4 = ContextGate(config.feature_dim, config.feature_dim)
        self.context4.strength.data.fill_(1.0)
        self.mask4 = nn.Conv2d(2 * c, config.mask_dim, 1)
        self.mask2 = nn.Conv2d(c, config.mask_dim, 1)
        self.output2 = nn.Sequential(ConvNormAct(config.mask_dim, config.mask_dim, groups=config.mask_dim),
                                     nn.Conv2d(config.mask_dim, config.mask_dim, 1))
        self.mask_dim, self.feature_dim = config.mask_dim, config.feature_dim
        self.checkpointing = config.detail_checkpointing

    def apply_curve(self, block, features):
        if self.training and self.checkpointing and features.requires_grad:
            return checkpoint(block, features, use_reentrant=False)
        return block(features)

    def project_context(self, context):
        return self.gate2.project(context), self.gate4.project(context), self.context4.project(context)

    def forward(self, image, context, bounds, full_size, projected_context=None):
        p2, p4, semantic = self.project_context(context) if projected_context is None else projected_context
        x2 = self.apply_curve(self.curve2, self.gate2(self.stem(image), p2, bounds, full_size, projected=True))
        x4 = self.apply_curve(self.curve4, self.gate4(self.down(x2), p4, bounds, full_size, projected=True))
        mask4 = self.mask4(x4)
        mask2 = self.output2(self.mask2(x2) + F.interpolate(mask4, size=x2.shape[-2:], mode="bilinear", align_corners=False))
        pixels4 = self.context4(self.output4(x4), semantic, bounds, full_size, projected=True)
        return pixels4, mask2

    def tiled(self, image, context, tile, halo):
        if self.training and torch.is_grad_enabled():
            raise RuntimeError("Tiled detail assembly is inference-only; training uses the complete differentiable path")
        b, _, h, w = image.shape
        if h % 4 or w % 4:
            raise ValueError("Pad images to multiples of four before detail tiling")
        output4 = image.new_zeros(b, self.feature_dim, h // 4, w // 4, dtype=context.dtype)
        output2 = image.new_zeros(b, self.mask_dim, h // 2, w // 2, dtype=context.dtype)
        projected = self.project_context(context)
        # Read overlapping halo, keep only disjoint cores. IDs are never generated per tile.
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
                bounds = (max(0, y0 - halo), max(0, x0 - halo), min(h, y1 + halo), min(w, x1 + halo))
                by, bx, ey, ex = bounds
                f4, f2 = self(image[..., by:ey, bx:ex], context, bounds, (h, w), projected)
                for stride, source, destination in ((4, f4, output4), (2, f2, output2)):
                    sy, sx = (y0 - by) // stride, (x0 - bx) // stride
                    hh, ww = (y1 - y0) // stride, (x1 - x0) // stride
                    destination[..., y0 // stride:y1 // stride, x0 // stride:x1 // stride] = source[..., sy:sy + hh, sx:sx + ww]
        return output4, output2
