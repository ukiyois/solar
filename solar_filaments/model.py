"""Full model: full-disk context view -> FlashInternImage-L pyramid; native 2048 px detail path; query decoder.

Inputs are B x 1 x H x W H-alpha images in [0, 1] (H, W divisible by 4). The backbone sees a downsampled
view (context_size on the longer side); the detail path works at native resolution, in overlapping tiles
at inference. Each decoder query yields a class logit, a quality score and a mask embedding; native masks
are rendered by solar_filaments.masks from the stride-2 basis plus the sub-pixel phase residual.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .backbone import ContextBackbone
from .config import ModelConfig
from .detail import ContextPyramid, NativeDetailPath
from .decoder import InstanceDecoder


def limb_flattened(images, factor=8, sigma=4.0):
    """Large-scale (center-to-limb) normalization restricted to the solar disk.

    Normalized convolution at 1/8 resolution: blur(I*disk)/blur(disk), sigma
    ~32 native pixels, so off-disk darkness never enters the background and
    filaments (a few to tens of pixels wide) are not absorbed by it. Output is
    the relative darkness I/B - 1, scaled like the local-contrast channel,
    and zero off the disk. Same role as the radial flattening / Zernike
    center-to-limb corrections of EdgeAttNet (2025) and Diercke et al. (2024).
    """
    with torch.autocast(images.device.type, enabled=False):
        x = images.float()
        small = F.avg_pool2d(x, factor)
        level = torch.quantile(small.flatten(1), .99, dim=1).view(-1, 1, 1, 1)
        disk = (small > .25 * level).float()
        radius = int(3 * sigma)
        axis = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float32)
        kernel = torch.exp(-.5 * (axis / sigma) ** 2)
        kernel = kernel / kernel.sum()
        def blur(t):
            t = F.conv2d(F.pad(t, (radius, radius, 0, 0), mode='replicate'), kernel.view(1, 1, 1, -1))
            return F.conv2d(F.pad(t, (0, 0, radius, radius), mode='replicate'), kernel.view(1, 1, -1, 1))
        background = blur(small * disk) / blur(disk).clamp_min(1e-3)
        background = F.interpolate(background, size=x.shape[-2:], mode='bilinear', align_corners=False)
        # Exact limb at native resolution; off-disk pixels are exactly zero.
        disk = (x > .25 * level).float()
        return ((x / background.clamp_min(1e-3) - 1) * 4 * disk).clamp(-4, 4).to(images.dtype)


class SolarFilamentModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.backbone = ContextBackbone(config.backbone, config.dcn_backend, config.backbone_checkpointing,
                                        config.drop_path_rate)
        self.context = ContextPyramid(self.backbone.channels, config.feature_dim)
        self.detail = NativeDetailPath(config)
        self.decoder = InstanceDecoder(config)
        self.semantic_head = nn.Conv2d(config.mask_dim, 1, 1)
        self.centerline_head = nn.Conv2d(config.mask_dim, 1, 1) if config.auxiliary_heads else None
        self.boundary_head = nn.Conv2d(config.mask_dim, 1, 1) if config.auxiliary_heads else None
        inputs = 3 if config.detail_flattening else 2
        self.phase_head = nn.Conv2d(config.mask_dim + 4 * inputs, 4 * config.phase_dim, 1) if config.phase_dim else None
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def context_view(self, images, context_size=None):
        """Antialiased full-disk view (longer side context_size, multiple of 32), 3 channels, ImageNet-normalized."""
        h, w = images.shape[-2:]
        ratio = (context_size or self.config.context_size) / max(h, w)
        size = (max(32, round(h * ratio / 32) * 32), max(32, round(w * ratio / 32) * 32))
        view = F.interpolate(images, size=size, mode="bilinear", align_corners=False, antialias=True)
        return (view.expand(-1, 3, -1, -1) - self.mean) / self.std, size

    def forward(self, images, tiled_detail=False, denoising_masks=None, context_size=None):
        """tiled_detail: native path in overlapping tiles (inference memory); denoising_masks: training-only
        mask pilots; context_size overrides the backbone view size (multi-scale training of the E6 teacher)."""
        if denoising_masks is not None and not self.training:
            raise ValueError('Mask pilots are training-only')
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("Expected Bx1xHxW H-alpha images in [0,1]")
        h, w = images.shape[-2:]
        if h % 4 or w % 4:
            raise ValueError("Input dimensions must be divisible by four")
        view, size = self.context_view(images, context_size)
        # Compute contrast over the full image so overlapping tiles receive identical inputs.
        contrast = (images - F.avg_pool2d(images, 15, 1, 7)) * 4.0
        detail_input = torch.cat((images, contrast), 1)
        if self.config.detail_flattening:
            detail_input = torch.cat((detail_input, limb_flattened(images)), 1)
        pyramid = self.context(self.backbone(view))
        if tiled_detail:
            pixels4, basis2 = self.detail.tiled(detail_input, pyramid[0], self.config.detail_tile, self.config.detail_halo)
        else:
            pixels4, basis2 = self.detail(detail_input, pyramid[0], (0, 0, h, w), (h, w))
        output = self.decoder(pixels4, pyramid, denoising_masks=denoising_masks)
        output.update(mask_features2=basis2, semantic_logits=self.semantic_head(basis2),
                      image_size=(h, w), context_size=size)
        if self.centerline_head is not None:
            output.update(centerline_logits=self.centerline_head(basis2), boundary_logits=self.boundary_head(basis2))
        if self.phase_head is not None:
            # Raw subpixel phases survive even when a one-pixel edge lies between
            # stride-2 cells. This branch participates in extraction and training.
            output["phase_features2"] = self.phase_head(torch.cat((basis2, F.pixel_unshuffle(detail_input, 2)), 1))
        if 'denoising' in output:
            output['denoising'].update(mask_features2=basis2, image_size=(h, w))
            if 'phase_features2' in output:
                output['denoising']['phase_features2'] = output['phase_features2']
        return output


def build_model(config, load_backbone=True):
    model = SolarFilamentModel(config)
    report = None
    if load_backbone and config.pretrained:
        if not Path(config.pretrained).exists():
            raise FileNotFoundError(f"Missing backbone weights: {config.pretrained}; run scripts/download_weights.py")
        report = model.backbone.load_pretrained(config.pretrained)
    return model, report
