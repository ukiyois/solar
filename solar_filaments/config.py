"""Model configuration and YAML experiment files.

A YAML file holds `model`, `train`, `loss` and `inference` sections and may name a `base:` file whose
sections it updates key by key, so every stage of the training history is a small, reviewable delta
(configs/flash_large.yaml -> ... -> configs/v9_ffinal_a.yaml).
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import yaml


@dataclass
class ModelConfig:
    """Architecture of the filament instance segmentation model (values used: configs/v9_ffinal_a.yaml)."""
    backbone: str = "large"                # FlashInternImage-L; 'diagnostic' = tiny stand-in for software tests
    context_size: int = 1024               # longer side of the downsampled full-disk view seen by the backbone
    feature_dim: int = 192                 # width of the context pyramid, native detail features and decoder
    mask_dim: int = 64                     # instance mask embedding / stride-2 mask basis channels
    detail_channels: int = 32              # native detail path width at stride 2 (twice that at stride 4)
    queries: int = 64                      # instance queries; one query is at most one filament
    decoder_layers: int = 6
    heads: int = 6
    memory_grid: int = 48                  # pooled grid of the finest decoder memory level
    seed_tokens: int = 256                 # unpooled stride-4 tokens at seed-score peaks, appended to the memory
    curve_points: int = 7                  # continuous curve sampling (DSConv-style) in the detail path
    curve_step: float = 0.75               # max shift across the curve axis between neighboring points (cells)
    detail_tile: int = 512                 # native-resolution inference in tiles: core size and halo (pixels)
    detail_halo: int = 48
    dcn_backend: str = "auto"              # DCNv4: 'native' CUDA kernel, 'reference' PyTorch, 'auto'
    backbone_checkpointing: bool = True    # activation checkpointing (training memory)
    detail_checkpointing: bool = True
    curve_enabled: bool = True             # curve branch of the detail blocks; False keeps the local blocks only
    curve_channels: int = 16               # curve branch width (rounded down to a multiple of 16)
    detail_downsample: str = "haar"        # 'haar': invertible 2x2 Haar split before learned mixing; 'stride'
    phase_dim: int = 8                     # native sub-pixel phase residual of every instance mask
    global_attention_every: int = 2        # every n-th decoder layer attends globally, the others mask-guided
    mask_feedback: bool = True             # mask-weighted memory pooling fed back into each query
    query_detail_points: int = 0           # per-query attention to native stride-4 foreground/edge points
    denoising_queries: int = 0             # training-only mask-pilot slots (MP-Former); 0 = off
    pilot_point_noise: float = .2          # guide point noise: expected flips <= ratio x guide area
    pilot_label_noise: float = .2          # probability of flipping a pilot's filament/no-object label
    image_query_init: bool = False         # initialize queries from spatially diverse seed-score peaks
    query_init_candidates: int = 1024      # seed-score peaks considered for the initial queries
    query_init_diversity: float = .5       # 0 = strongest peaks only; larger values spread the queries apart
    # ImageNet weights of the backbone (scripts/download_weights.py); read only when no checkpoint is loaded
    pretrained: str = "weights/flash_intern_image_l_backbone.pt"
    auxiliary_heads: bool = True           # centerline/boundary heads (zero loss weight in the final recipe)
    detail_flattening: bool = False        # extra limb-darkening-normalized input channel (model.limb_flattened)
    drop_path_rate: float = 0.1            # backbone stochastic depth

    def __post_init__(self):
        """Reject inconsistent settings when the configuration is created, before any model is built."""
        if self.feature_dim % self.heads or self.feature_dim % 4:
            raise ValueError("feature_dim must be divisible by heads and 4")
        if not 0 <= self.drop_path_rate < 1:
            raise ValueError('drop_path_rate must be in [0, 1)')
        if self.queries < 26:
            raise ValueError("At least 26 queries are needed for the observed training counts")
        if self.context_size % 32 or self.detail_tile % 4 or self.detail_halo % 4:
            raise ValueError("context_size must align to 32; detail tile/halo to 4")
        if self.curve_points % 2 != 1 or self.curve_points < 3:
            raise ValueError("curve_points must be odd and >=3")
        if self.detail_downsample not in ("haar", "stride") or self.phase_dim < 0:
            raise ValueError("Invalid detail downsampling / phase dimension")
        if self.curve_channels < 1 or self.global_attention_every < 0 or self.decoder_layers < 1:
            raise ValueError("Invalid curve width / decoder depth / global attention interval")
        if self.query_detail_points < 0 or self.query_detail_points % 2:
            raise ValueError('query_detail_points must be a nonnegative even integer')
        if self.denoising_queries < 0 or self.denoising_queries > self.queries:
            raise ValueError('denoising_queries must fit inside the learned query table')
        if not 0 <= self.pilot_point_noise <= 1 or not 0 <= self.pilot_label_noise <= 1:
            raise ValueError('Pilot noise ratios must lie in [0,1]')
        if self.query_init_candidates < self.queries or not 0 <= self.query_init_diversity <= 1:
            raise ValueError('Invalid image-query candidate count or spatial diversity')
        # Tiled inference must reproduce the full-image result, so the halo must cover the detail path's
        # receptive field in native pixels: the block reach of curve_radius cells enters at stride 2 and at
        # stride 4 (6 * curve_radius px), plus a fixed 9 px allowance for the other layers (3x3 convolutions,
        # upsampling); the sum is rounded up to the 4 px tiling alignment.
        curve_radius = math.ceil(max(self.curve_points // 2, (self.curve_points // 2) * self.curve_step, 1)) if self.curve_enabled else 1
        required_halo = math.ceil((9 + 6 * curve_radius) / 4) * 4
        if self.detail_halo < required_halo:
            raise ValueError(f"detail_halo must be >= {required_halo} to cover local receptive fields")


# Options of experiments that were not retained (see docs/TECHNICAL_REPORT.md, negative results).
# Checkpoints written while they existed store them; such a checkpoint loads only if the option was
# off (the value below), so no removed code path is ever needed. None = value irrelevant when off.
RETIRED_OPTIONS = {"native_input_adapter": False, "decoder_mode": "global", "memory_mode": "pooled",
                   "attention_mask_pool": "bilinear", "efficient_attention": False, "flattening_mode": "nc",
                   "dynamic_mask": False, "annotators": (), "hypotheses": 0, "curve_attention_points": None,
                   "curve_attention_anchors": None, "dynamic_mask_channels": None}


def checkpoint_config(saved, **overrides):
    """ModelConfig of a checkpoint's `model_config` dict; `overrides` replace operational fields
    (context size, DCN backend, tiling, checkpointing) that do not change the parameters."""
    values = dict(saved)
    for key, neutral in RETIRED_OPTIONS.items():
        if key not in values:
            continue
        value = values.pop(key)
        if neutral is not None and (tuple(value) if isinstance(value, list) else value) != neutral:
            raise ValueError(f"Checkpoint uses the retired option {key}={value!r}")
    values.update(overrides)
    return ModelConfig(**values)


def read_config(path):
    """Read a YAML experiment file as a dict, resolving `base:` inheritance recursively.

    Sections present as dicts in both files are merged one level deep: each key of the child's section
    replaces the parent's value, nested dicts (such as loss.weights) included. Other top-level keys replace
    the parent's. `base:` paths are relative to the working directory (the repository root).
    """
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "base" in document:
        parent = read_config(document.pop("base"))
        for key, value in document.items():
            if isinstance(value, dict) and isinstance(parent.get(key), dict):
                parent[key].update(value)
            else:
                parent[key] = value
        return parent
    return document


def model_config(document):
    """ModelConfig from the `model` section of a configuration document (or from a dict of model fields)."""
    return ModelConfig(**document.get("model", document))


def config_dict(config):
    """Plain dict of a ModelConfig, the form stored in checkpoints under 'model_config'."""
    return asdict(config)
