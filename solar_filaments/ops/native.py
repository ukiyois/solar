"""torch.library contract around the pinned CUDA extension.

Opaque forward/backward kernels, fake implementations, explicit autograd. AOT
Autograd can compile surrounding PyTorch operations without tracing pybind.
The extension still performs exactly the author's DCNv4 arithmetic.

The extension is the official OpenGVLab DCNv4 operator at a pinned commit (third_party/DCNv4_op, built by
scripts/build_dcnv4.py). Tensors are channels-last: value B x H x W x C, packed offsets and weights
B x H x W x (multiple of 8) in the layout described in solar_filaments.ops.dcnv4.
"""
from functools import lru_cache
import importlib
import torch


@lru_cache(maxsize=512)
def launch_spec(batch, height, width, groups, channels, backward):
    """CUDA launch geometry (d_stride, threads per block) for the DCNv4 forward or backward kernel.

    channels is the per-group width. d_stride (channels per thread) starts at upstream's value (forward 8;
    backward 1, or 2 for wide groups) and doubles while one position would need more threads than the limit
    (forward 512, backward 256); a d_stride above 8 or one that does not divide channels raises ValueError.
    A block covers the largest number of positions (a divisor of batch * height * width, at most 64) that
    fits. This is upstream's findspec/find_spec_bwd heuristic without its tuned lookup table, plus the thread
    cap; it changes the launch, not the arithmetic.
    """
    stride = (2 if channels >= 64 else 1) if backward else 8
    limit = 256 if backward else 512
    while groups * channels // stride > limit:
        stride *= 2
    if stride > 8 or channels % stride:
        raise ValueError("Unsupported DCNv4 launch geometry")
    sequence = batch * height * width
    multiplier = max(m for m in range(1, min(64, sequence) + 1)
                     if sequence % m == 0 and m * groups * channels // stride <= limit)
    return stride, multiplier * groups * channels // stride


@torch.library.custom_op("solar::dcnv4", mutates_args=(), device_types="cuda")
def dcnv4_cuda(value: torch.Tensor, packed: torch.Tensor, groups: int,
               scale: float, kernel: int, remove: int) -> torch.Tensor:
    """DCNv4 forward on CUDA: B x H x W x C values and packed offsets/weights -> B x H x W x C.

    groups: channel groups; scale: offset scale; kernel: kernel size (stride 1, same padding, dilation 1);
    remove: 1 drops the center sample. No softmax on the weights. Differentiable via register_autograd below.
    """
    if value.dtype != packed.dtype or value.device != packed.device:
        raise ValueError("DCNv4 values and offsets must share dtype and device")
    # Keep the layout contract inside the opaque boundary. A compiler may
    # otherwise change surrounding layouts before the raw-pointer CUDA call.
    value, packed = value.contiguous(), packed.contiguous()
    ext = importlib.import_module("DCNv4.ext")
    channels = value.shape[-1] // groups
    stride, threads = launch_spec(*value.shape[:3], groups, channels, False)
    # Arguments: kernel h/w, stride h/w, pad h/w, dilation h/w, groups, group channels, offset scale,
    # im2col_step (images per chunk), remove_center, d_stride, threads per block, softmax (off in DCNv4).
    return ext.dcnv4_forward(value, packed, kernel, kernel, 1, 1, kernel // 2, kernel // 2,
                             1, 1, groups, channels, scale, 256, remove, stride, threads, False)


@dcnv4_cuda.register_fake
def _fake(value, packed, groups, scale, kernel, remove):
    """Output metadata (shape, dtype, device) for tracing and torch.compile; no computation."""
    return torch.empty(value.shape, dtype=value.dtype, device=value.device)


@torch.library.custom_op("solar::dcnv4_backward", mutates_args=(), device_types="cuda")
def dcnv4_backward(value: torch.Tensor, packed: torch.Tensor, grad: torch.Tensor,
                   groups: int, scale: float, kernel: int, remove: int) -> tuple[torch.Tensor, torch.Tensor]:
    """DCNv4 backward on CUDA: gradients with respect to value and packed, given the output gradient grad."""
    value, packed = value.contiguous(), packed.contiguous()
    ext = importlib.import_module("DCNv4.ext")
    channels = value.shape[-1] // groups
    stride, threads = launch_spec(*value.shape[:3], groups, channels, True)
    # Same arguments as the forward call, with the output gradient inserted before remove_center.
    a, b = ext.dcnv4_backward(value, packed, kernel, kernel, 1, 1, kernel // 2, kernel // 2,
                             1, 1, groups, channels, scale, 256, grad.contiguous(), remove, stride, threads, False)
    return a, b


@dcnv4_backward.register_fake
def _fake_backward(value, packed, grad, groups, scale, kernel, remove):
    """Gradient metadata for tracing and torch.compile; no computation."""
    return (torch.empty(value.shape, dtype=value.dtype, device=value.device),
            torch.empty(packed.shape, dtype=packed.dtype, device=packed.device))


def _setup_context(ctx, inputs, output):
    """Save the forward inputs and settings that the backward kernel needs."""
    value, packed, groups, scale, kernel, remove = inputs
    ctx.save_for_backward(value, packed)
    ctx.settings = (groups, scale, kernel, remove)


def _backward(ctx, grad):
    """Gradients for (value, packed); the integer and float settings get None."""
    a, b = dcnv4_backward(*ctx.saved_tensors, grad, *ctx.settings)
    return a, b, None, None, None, None


dcnv4_cuda.register_autograd(_backward, setup_context=_setup_context)
