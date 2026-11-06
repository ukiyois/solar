"""Full model: full-disk context view -> FlashInternImage-L pyramid; native 2048 px detail path; query decoder.

Inputs are B x 1 x H x W H-alpha images in [0, 1] (H, W divisible by 4). The competition images are GONG
full-disk frames of 2048 x 2048 px, about 0.74 Mm per pixel. Filaments are hundreds of pixels long but narrow
(median maximum width 14 px, about 10 Mm; one in ten is at most 8 px wide), so the model reads two views.
The backbone sees a downsampled view of the whole disk (context_size on the longer side, 1024 px), whose
context separates filaments from sunspots, plage and the limb; the detail path works at native resolution, in
overlapping tiles at inference. Each decoder query yields a class logit, a quality score and a mask
embedding; native masks are rendered by solar_filaments.masks from the stride-2 basis plus the sub-pixel
phase residual.
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

    images: B x 1 x H x W in [0, 1]. Returns the relative darkness 4 * (I/B - 1) with the same shape and dtype,
    scaled like the local-contrast channel, clamped to [-4, 4] and exactly 0 off the disk (the disk is
    I > 0.25 x the 99th percentile of the 1/factor image). B is a normalized convolution at 1/factor resolution,
    blur(I*disk)/blur(disk) with sigma x factor native pixels (default 1/8 and ~32 px, about 24 Mm): a
    Gaussian-weighted mean over on-disk pixels only, so off-disk darkness never enters the background and
    filaments (a few to tens of pixels wide) are not absorbed by it. B is the smooth, mostly quiet-Sun
    background and follows limb darkening, so the ratio removes the center-to-limb fall-off. For an absorbing
    filament (Beckers' cloud model: source function S, optical depth tau, background I0 = B),
    I = I0 exp(-tau) + S (1 - exp(-tau)), hence I/B - 1 = (S/I0 - 1)(1 - exp(-tau)): the cloud-model contrast
    C, negative for a filament. Same role as the radial flattening / Zernike center-to-limb corrections of
    EdgeAttNet (2025) and Diercke et al. (2024). Computed in float32.
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
        # Separable Gaussian (radius 3 sigma) with replicate padding at the image border.
        def blur(t):
            t = F.conv2d(F.pad(t, (radius, radius, 0, 0), mode='replicate'), kernel.view(1, 1, 1, -1))
            return F.conv2d(F.pad(t, (0, 0, radius, radius), mode='replicate'), kernel.view(1, 1, -1, 1))
        background = blur(small * disk) / blur(disk).clamp_min(1e-3)
        background = F.interpolate(background, size=x.shape[-2:], mode='bilinear', align_corners=False)
        # Exact limb at native resolution; off-disk pixels are exactly zero.
        disk = (x > .25 * level).float()
        return ((x / background.clamp_min(1e-3) - 1) * 4 * disk).clamp(-4, 4).to(images.dtype)


class SolarFilamentModel(nn.Module):
    """Filament instance segmentation: context backbone, native detail path and query decoder.

    backbone + context: FlashInternImage-L on the full-disk view and its 4-level pyramid (feature_dim
    channels, strides 4 to 32 of the view). detail: the native path at strides 2 and 4 of the image, fused
    with the finest pyramid level. decoder: the instance query decoder. The 1x1 heads on the stride-2 mask basis
    give semantic (filament vs. background) and optional centerline/boundary logits, auxiliary outputs with
    zero loss weight in the final recipe. phase_head maps the basis and the four sub-pixel phases of the raw
    inputs to 4 x phase_dim channels per stride-2 cell, the native phase residual of every mask.
    """

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
        # Native input channels: intensity, local contrast and, optionally, the limb-flattened darkness.
        inputs = 3 if config.detail_flattening else 2
        self.phase_head = nn.Conv2d(config.mask_dim + 4 * inputs, 4 * config.phase_dim, 1) if config.phase_dim else None
        # ImageNet statistics of the pretrained backbone, applied to the gray view replicated to 3 channels.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def context_view(self, images, context_size=None):
        """Antialiased full-disk view (longer side context_size, multiple of 32), 3 channels, ImageNet-normalized.

        context_size defaults to config.context_size (1024 px, half the native 2048 px). Each side is rounded to a
        multiple of 32, the backbone's total stride. Returns (view B x 3 x h x w, (h, w)).
        """
        h, w = images.shape[-2:]
        ratio = (context_size or self.config.context_size) / max(h, w)
        size = (max(32, round(h * ratio / 32) * 32), max(32, round(w * ratio / 32) * 32))
        view = F.interpolate(images, size=size, mode="bilinear", align_corners=False, antialias=True)
        return (view.expand(-1, 3, -1, -1) - self.mean) / self.std, size

    def forward(self, images, tiled_detail=False, denoising_masks=None, context_size=None):
        """Instance queries and native mask features for B x 1 x H x W H-alpha images in [0, 1].

        tiled_detail: native path in overlapping tiles (inference memory; same result as the full image).
        denoising_masks: training-only mask pilots (solar_filaments.piloting). context_size overrides the
        backbone view size; the multi-scale-view teacher configuration (configs/v9_e6.yaml) trains with
        1024, 1152 and 1280 px views. Returns the decoder outputs (pred_logits, pred_quality, mask_embeddings,
        stride-4 pred_masks, aux_outputs, seed_logits, ...) plus mask_features2 (the B x mask_dim x H/2 x W/2
        mask basis), phase_features2 (B x 4*phase_dim x H/2 x W/2), semantic_logits, the optional
        centerline/boundary logits (all stride 2), image_size (H, W) and context_size (view h, w).
        """
        if denoising_masks is not None and not self.training:
            raise ValueError('Mask pilots are training-only')
        if images.ndim != 4 or images.shape[1] != 1:
            raise ValueError("Expected Bx1xHxW H-alpha images in [0,1]")
        h, w = images.shape[-2:]
        if h % 4 or w % 4:
            raise ValueError("Input dimensions must be divisible by four")
        view, size = self.context_view(images, context_size)
        # Local contrast 4 * (I - 15 x 15 box mean) removes slow brightness variation.
        # Compute contrast over the full image so overlapping tiles receive identical inputs.
        contrast = (images - F.avg_pool2d(images, 15, 1, 7)) * 4.0
        detail_input = torch.cat((images, contrast), 1)
        if self.config.detail_flattening:
            detail_input = torch.cat((detail_input, limb_flattened(images)), 1)
        # Level 0 (stride 4 of the view) conditions the detail path; the decoder memory adds levels 1 and 2.
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
            # pixel_unshuffle hands the head the raw inputs at all four native sub-pixels of every stride-2 cell,
            # so an edge that falls inside a cell is still placed to one native pixel. The phase features are
            # used both for instance extraction at inference and in training.
            output["phase_features2"] = self.phase_head(torch.cat((basis2, F.pixel_unshuffle(detail_input, 2)), 1))
        # The pilot branch renders its native masks from the same basis and phase features.
        if 'denoising' in output:
            output['denoising'].update(mask_features2=basis2, image_size=(h, w))
            if 'phase_features2' in output:
                output['denoising']['phase_features2'] = output['phase_features2']
        return output


def build_model(config, load_backbone=True):
    """Build SolarFilamentModel(config); returns (model, report).

    With load_backbone and config.pretrained set, the ImageNet FlashInternImage-L weights are loaded into the
    backbone and report is the summary from ContextBackbone.load_pretrained; otherwise report is None. Use
    load_backbone=False when a complete checkpoint is loaded afterwards (inference, warm starts, tests).
    """
    model = SolarFilamentModel(config)
    report = None
    if load_backbone and config.pretrained:
        if not Path(config.pretrained).exists():
            raise FileNotFoundError(f"Missing backbone weights: {config.pretrained}; run scripts/download_weights.py")
        report = model.backbone.load_pretrained(config.pretrained)
    return model, report
