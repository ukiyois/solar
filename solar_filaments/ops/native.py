"""torch.library contract around the pinned CUDA extension.

Opaque forward/backward kernels, fake implementations, explicit autograd. AOT
Autograd can compile surrounding PyTorch operations without tracing pybind.
The extension still performs exactly the author's DCNv4 arithmetic.
"""
from functools import lru_cache
import importlib
import torch


@lru_cache(maxsize=512)
def launch_spec(batch, height, width, groups, channels, backward):
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
    if value.dtype != packed.dtype or value.device != packed.device:
        raise ValueError("DCNv4 values and offsets must share dtype and device")
    # Keep the layout contract inside the opaque boundary. A compiler may
    # otherwise change surrounding layouts before the raw-pointer CUDA call.
    value, packed = value.contiguous(), packed.contiguous()
    ext = importlib.import_module("DCNv4.ext")
    channels = value.shape[-1] // groups
    stride, threads = launch_spec(*value.shape[:3], groups, channels, False)
    return ext.dcnv4_forward(value, packed, kernel, kernel, 1, 1, kernel // 2, kernel // 2,
                             1, 1, groups, channels, scale, 256, remove, stride, threads, False)


@dcnv4_cuda.register_fake
def _fake(value, packed, groups, scale, kernel, remove):
    return torch.empty(value.shape, dtype=value.dtype, device=value.device)


@torch.library.custom_op("solar::dcnv4_backward", mutates_args=(), device_types="cuda")
def dcnv4_backward(value: torch.Tensor, packed: torch.Tensor, grad: torch.Tensor,
                   groups: int, scale: float, kernel: int, remove: int) -> tuple[torch.Tensor, torch.Tensor]:
    value, packed = value.contiguous(), packed.contiguous()
    ext = importlib.import_module("DCNv4.ext")
    channels = value.shape[-1] // groups
    stride, threads = launch_spec(*value.shape[:3], groups, channels, True)
    a, b = ext.dcnv4_backward(value, packed, kernel, kernel, 1, 1, kernel // 2, kernel // 2,
                             1, 1, groups, channels, scale, 256, grad.contiguous(), remove, stride, threads, False)
    return a, b


@dcnv4_backward.register_fake
def _fake_backward(value, packed, grad, groups, scale, kernel, remove):
    return (torch.empty(value.shape, dtype=value.dtype, device=value.device),
            torch.empty(packed.shape, dtype=packed.dtype, device=packed.device))


def _setup_context(ctx, inputs, output):
    value, packed, groups, scale, kernel, remove = inputs
    ctx.save_for_backward(value, packed)
    ctx.settings = (groups, scale, kernel, remove)


def _backward(ctx, grad):
    a, b = dcnv4_backward(*ctx.saved_tensors, grad, *ctx.settings)
    return a, b, None, None, None, None


dcnv4_cuda.register_autograd(_backward, setup_context=_setup_context)
