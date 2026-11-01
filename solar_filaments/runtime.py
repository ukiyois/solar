"""Training throughput utilities; matching and file I/O remain outside compile."""
from contextlib import contextmanager
import os
import torch


def configure_cuda():
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark_limit = 4


def move_batch(batch, device):
    images, targets = batch
    images = images.to(device, non_blocking=True)
    if images.dtype == torch.uint8:
        images = images.float().div_(255)
    targets = [{k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in target.items()} for target in targets]
    return images, targets


class DevicePrefetcher:
    def __init__(self, loader, device, enabled=True):
        self.loader, self.device = loader, device
        self.enabled = enabled and device.type == "cuda"

    def __iter__(self):
        if not self.enabled:
            for batch in self.loader:
                yield move_batch(batch, self.device)
            return
        stream = torch.cuda.Stream(device=self.device)
        iterator = iter(self.loader)
        def preload():
            try:
                batch = next(iterator)
            except StopIteration:
                return None
            with torch.cuda.stream(stream):
                return move_batch(batch, self.device)
        pending = preload()
        while pending is not None:
            current_stream = torch.cuda.current_stream(self.device)
            current_stream.wait_stream(stream)
            images, targets = pending
            images.record_stream(current_stream)
            for target in targets:
                for value in target.values():
                    if isinstance(value, torch.Tensor):
                        value.record_stream(current_stream)
            pending = preload()
            yield images, targets


def photometric(images, settings):
    if not settings.get("photometric", True):
        return images
    b = images.shape[0]
    random = torch.rand(b, 4, 1, 1, device=images.device)
    brightness = 1 + (random[:, 0:1] - .5) * .2
    contrast = 1 + (random[:, 1:2] - .5) * .3
    gamma = 1 + (random[:, 2:3] - .5) * .2
    mean = images.mean((-2, -1), keepdim=True)
    images = ((images - mean) * contrast + mean).clamp_min(0) * brightness
    images = images.clamp(0, 1).pow(gamma)
    return (images + torch.randn_like(images) * random[:, 3:4] * .005).clamp(0, 1)


class ModelEMA:
    def __init__(self, model, decay=.999, warmup=100):
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.decay, self.warmup, self.updates = decay, warmup, 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (self.warmup + self.updates))
        old, new = [], []
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                old.append(self.shadow[name])
                new.append(value.detach())
            else:
                self.shadow[name].copy_(value)
        torch._foreach_lerp_(old, new, 1 - decay)

    @contextmanager
    def apply(self, model):
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
        try:
            yield
        finally:
            model.load_state_dict(backup, strict=True)

    def state_dict(self):
        return {"shadow": self.shadow, "updates": self.updates}

    def load_state_dict(self, state):
        for key, value in state["shadow"].items():
            self.shadow[key].copy_(value)
        self.updates = state["updates"]


def compile_training(model, settings):
    requested = settings.get("compile", "off")
    if requested in (False, "off", None):
        return {"enabled": False}
    if requested == "auto" and os.name == "nt":
        return {"enabled": False, "reason": "auto selects eager on Windows; explicit compile is available for local audits"}
    backend = settings.get("compile_backend", "inductor")
    mode = settings.get("compile_mode", "default")
    # Compile only tensor computation. CPU Hungarian, preprocessing, optimizer,
    # validation, checkpointing and reporting intentionally stay eager.
    forward = torch.compile(model.forward, backend=backend, mode=mode if backend == "inductor" else None,
                            dynamic=False, fullgraph=settings.get("compile_fullgraph", False))
    return {"enabled": True, "backend": backend, "mode": mode, "forward": forward}
