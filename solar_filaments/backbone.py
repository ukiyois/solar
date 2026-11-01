"""FlashInternImage-L context encoder on the downsampled full-disk view (pretrained, DCNv4 operators)."""
import contextlib
import io
from pathlib import Path
import torch
from torch import nn
from .ops.dcnv4 import DCNv4
from .vendor.flash_intern_image import FlashInternImage


class ContextBackbone(nn.Module):
    """Four-level feature pyramid (strides 4..32 of the view) of the official FlashInternImage encoder;
    the classification head is removed and `load_pretrained` loads the ImageNet weights strictly."""

    def __init__(self, variant="large", backend="auto", checkpointing=True, drop_path_rate=0.1):
        super().__init__()
        settings = {
            "large": (160, [5, 5, 22, 5], [10, 20, 40, 80], True, 1e-5, 2.0),
            # The tiny structure is ONLY for cheap software checks, not a competition backbone.
            "diagnostic": (16, [1, 1, 1, 1], [1, 2, 4, 8], True, 1e-5, 2.0),
        }
        if variant not in settings:
            raise ValueError(f"Unknown backbone {variant}")
        channels, depths, groups, post_norm, layer_scale, offset_scale = settings[variant]
        with contextlib.redirect_stdout(io.StringIO()):
            self.encoder = FlashInternImage(channels=channels, depths=depths, groups=groups,
                post_norm=post_norm, layer_scale=layer_scale, offset_scale=offset_scale,
                dw_kernel_size=3, dcn_output_bias=True, mlp_fc2_bias=True,
                drop_path_rate=drop_path_rate, with_cp=checkpointing, num_classes=0)
        self.encoder.conv_head = nn.Identity()
        self.encoder.head = nn.Identity()
        self.channels = [channels * 2 ** i for i in range(4)]
        self.depths = list(depths)
        self.variant = variant
        for module in self.modules():
            if isinstance(module, DCNv4):
                module.backend = backend

    def forward(self, x):
        return self.encoder.forward_features_seq_out(x)

    def load_pretrained(self, path):
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True, mmap=True)
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        clean = {}
        for key, value in state.items():
            key = key.removeprefix("module.").removeprefix("backbone.")
            if not key.startswith(("head.", "conv_head.")):
                clean[key] = value
        missing, unexpected = self.encoder.load_state_dict(clean, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Backbone checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        return {"loaded_tensors": len(clean), "loaded_parameters": sum(v.numel() for v in clean.values()),
                "source": str(path), "excluded": "classification head and conv_head only"}

