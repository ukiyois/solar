"""Checkpoint-compatible DCNv4 with an explicit, differentiable reference backend.

The reference follows OpenGVLab CUDA's group-major [2*K offsets, K weights]
packing, x-major kernel ordering, scaled base dilation and unnormalized weights.
It is for portability/debugging; it is not the accelerated DCNv4 kernel.
"""
import importlib
import math
import warnings
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
from .native import dcnv4_cuda

_warned = False


@lru_cache(maxsize=1)
def native_function():
    try:
        module = importlib.import_module("DCNv4.functions.dcnv4_func")
        # Upstream's uncached heuristic can request 1280 backward threads for
        # L's 80 groups, exceeding CUDA's block limit. Change launch geometry,
        # not the operator: vectorize channels until a <=256-thread block fits.
        module.find_spec_bwd = safe_backward_spec
        return module.DCNv4Function
    except (ImportError, OSError):
        return None


@lru_cache(maxsize=256)
def safe_backward_spec(batch, height, width, groups, group_channels):
    stride = 2 if group_channels >= 64 else 1
    while groups * group_channels // stride > 256:
        stride *= 2
    if stride > 8 or group_channels % stride:
        raise ValueError("Unsupported native DCNv4 backward group geometry")
    sequence = batch * height * width
    multiplier = max(m for m in range(1, min(64, sequence) + 1)
                     if sequence % m == 0 and m * groups * group_channels // stride <= 256)
    return stride, multiplier * groups * group_channels // stride


def dcnv4_reference(value, offset_mask, group, offset_scale=1.0, kernel=3,
                    remove_center=False, group_chunk=4):
    b, h, w, channels = value.shape
    k = kernel * kernel - int(remove_center)
    dim = channels // group
    packed = offset_mask[..., :group * k * 3].reshape(b, h, w, group, k * 3)
    xy = packed[..., :2 * k].reshape(b, h, w, group, k, 2)
    weights = packed[..., 2 * k:]
    radius = kernel // 2
    base = [(i - radius, j - radius) for i in range(kernel) for j in range(kernel)
            if not (remove_center and i == radius and j == radius)]
    # grid_sample in float32 keeps the CPU path valid and avoids half coordinate loss.
    with torch.autocast(value.device.type, enabled=False):
        yy, xx = torch.meshgrid(torch.arange(h, device=value.device, dtype=torch.float32),
                                torch.arange(w, device=value.device, dtype=torch.float32), indexing="ij")
        origin = torch.stack((xx, yy), -1).view(1, h, w, 1, 1, 2)
        base = value.new_tensor(base, dtype=torch.float32).view(1, 1, 1, 1, k, 2)
        outputs = []
        for start in range(0, group, group_chunk):
            end = min(start + group_chunk, group)
            g = end - start
            grid = origin + (base + xy[..., start:end, :, :].float()) * offset_scale
            grid = grid * grid.new_tensor((2 / w, 2 / h)) + grid.new_tensor((1 / w - 1, 1 / h - 1))
            grid = grid.permute(0, 3, 1, 2, 4, 5).reshape(b * g, h, w * k, 2)
            chunk = value[..., start * dim:end * dim].reshape(b, h, w, g, dim)
            chunk = chunk.permute(0, 3, 4, 1, 2).reshape(b * g, dim, h, w).float()
            sampled = F.grid_sample(chunk, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            sampled = sampled.reshape(b, g, dim, h, w, k)
            mask = weights[..., start:end, :].permute(0, 3, 1, 2, 4).unsqueeze(2).float()
            output = (sampled * mask).sum(-1).permute(0, 3, 4, 1, 2).reshape(b, h, w, g * dim)
            outputs.append(output.to(value.dtype))
        return torch.cat(outputs, -1)


class DCNv4(nn.Module):
    def __init__(self, channels=64, kernel_size=3, stride=1, pad=1, dilation=1,
                 group=4, offset_scale=1.0, dw_kernel_size=None,
                 center_feature_scale=False, remove_center=False,
                 output_bias=True, without_pointwise=False, **kwargs):
        super().__init__()
        if channels % group or channels // group % 16:
            raise ValueError("Official DCNv4 requires group_channels divisible by 16")
        if (stride, pad, dilation) != (1, kernel_size // 2, 1):
            raise ValueError("This wrapper supports the official FlashInternImage spatial configuration")
        self.channels, self.group, self.group_channels = channels, group, channels // group
        self.kernel_size, self.offset_scale = kernel_size, offset_scale
        self.remove_center = int(remove_center)
        self.without_pointwise, self.dw_kernel_size = without_pointwise, dw_kernel_size
        self.center_feature_scale = center_feature_scale
        self.backend = "auto"
        self.K = group * (kernel_size * kernel_size - self.remove_center)
        if dw_kernel_size is not None:
            self.offset_mask_dw = nn.Conv2d(channels, channels, dw_kernel_size,
                                           padding=dw_kernel_size // 2, groups=channels)
        self.offset_mask = nn.Linear(channels, math.ceil(self.K * 3 / 8) * 8)
        if not without_pointwise:
            self.value_proj = nn.Linear(channels, channels)
            self.output_proj = nn.Linear(channels, channels, bias=output_bias)
        if center_feature_scale:
            self.center_feature_scale_proj_weight = nn.Parameter(torch.zeros(group, channels))
            self.center_feature_scale_proj_bias = nn.Parameter(torch.zeros(group))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)
        if not self.without_pointwise:
            for layer in (self.value_proj, self.output_proj):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, input, shape=None):
        global _warned
        b, length, c = input.shape
        h, w = shape if shape is not None else (math.isqrt(length), math.isqrt(length))
        if h * w != length:
            raise ValueError("DCNv4 spatial shape mismatch")
        value = input if self.without_pointwise else self.value_proj(input)
        value = value.reshape(b, h, w, c)
        features = input
        if self.dw_kernel_size is not None:
            features = self.offset_mask_dw(input.reshape(b, h, w, c).permute(0, 3, 1, 2))
            features = features.flatten(2).transpose(1, 2)
        packed = self.offset_mask(features).reshape(b, h, w, -1).contiguous()
        use_native = input.is_cuda and (self.backend == "native" or
                                      (self.backend == "auto" and native_function() is not None))
        if self.backend == "native" and not use_native:
            raise RuntimeError("Native CUDA DCNv4 is unavailable; build it or explicitly select reference")
        if use_native:
            result = dcnv4_cuda(value.contiguous(), packed, self.group, float(self.offset_scale),
                                self.kernel_size, self.remove_center)
        else:
            if self.backend == "auto" and not _warned:
                warnings.warn("DCNv4 uses the slower PyTorch reference backend; accelerated training needs the CUDA extension")
                _warned = True
            result = dcnv4_reference(value, packed, self.group, self.offset_scale,
                                     self.kernel_size, self.remove_center)
        if self.center_feature_scale:
            scale = F.linear(result, self.center_feature_scale_proj_weight,
                             self.center_feature_scale_proj_bias).sigmoid().repeat_interleave(self.group_channels, -1)
            result = result * (1 - scale) + value * scale
        result = result.reshape(b, length, c)
        return result if self.without_pointwise else self.output_proj(result)
