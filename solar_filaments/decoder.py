"""Bounded-memory, mask-guided instance queries over full-image features.

Memory = context features average-pooled to small grids (finest level `memory_grid`), plus `seed_tokens`
unpooled stride-4 tokens taken at peaks of a learned seed score, so thin filaments are not averaged away.
Queries are initialized from spatially diverse seed peaks (`image_query_init`). Each layer runs
self-attention, cross-attention (masked by the previous layer's masks, Mask2Former; every
`global_attention_every`-th layer attends globally so a missed branch can be recovered), an optional
mask-weighted memory feedback, and an attention read of native foreground/edge points of its own mask
(`query_detail_points`). Training adds an isolated mask-pilot branch (MP-Former) that shares weights.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .piloting import noisy_pilot_support
from .query_seeds import query_seed_indices


def position_encoding(height, width, dim, device, dtype):
    """encode_xy of the cell centers of a height x width grid: 1 x (height * width) x dim, row-major."""
    y, x = torch.meshgrid((torch.arange(height, device=device) + 0.5) / height,
                          (torch.arange(width, device=device) + 0.5) / width, indexing="ij")
    return encode_xy(torch.stack((x.flatten(), y.flatten()), -1), dim).to(dtype).unsqueeze(0)


def encode_xy(xy, dim):
    """Sine/cosine encoding of normalized image coordinates: ... x 2 (x, y in [0, 1]) -> ... x dim.

    dim // 4 wavelengths per axis, from one image extent up to nearly 10^4 extents (geometric, as in DETR);
    positions on grids of different resolution and the seed tokens share one coordinate system.
    """
    frequency = 10000 ** (torch.arange(dim // 4, device=xy.device).float() / (dim // 4))
    x, y = xy[..., 0:1] * (2 * math.pi) / frequency, xy[..., 1:2] * (2 * math.pi) / frequency
    return torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), -1)


class QueryLayer(nn.Module):
    """Pre-norm decoder layer: query self-attention, cross-attention to the memory, feed-forward network."""

    def __init__(self, dim, heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, query, query_pos, memory, memory_pos, mask):
        """query B x Q x D, memory B x M x D; positions are added to attention queries and keys only.

        mask: boolean (B * heads) x Q x M, True = memory token not attended (masked attention); None = global.
        """
        q = self.norms[0](query)
        query = query + self.self_attn(q + query_pos, q + query_pos, q, need_weights=False)[0]
        q = self.norms[1](query)
        query = query + self.cross_attn(q + query_pos, memory + memory_pos, memory, attn_mask=mask, need_weights=False)[0]
        return query + self.ffn(self.norms[2](query))


class InstanceDecoder(nn.Module):
    """Mask2Former-style decoder of config.queries instance queries over a bounded scene memory (module doc).

    Heads per query: class logits (index 0 = filament, 1 = no object), a quality logit trained to predict the
    native IoU of the query's mask, a mask embedding (mask_dim) and, with phase_dim > 0, a phase embedding.
    Stride-4 masks are embedding . basis4, a 1x1 projection of the detail features; native masks are rendered
    from the stride-2 basis by solar_filaments.masks. seed_score is a 1x1 head on the stride-4 features,
    trained against GT skeleton occupancy; its peaks give the seed tokens and the query initialization.
    The learned gains of the optional branches start small (seed query .1, mask feedback 1e-2, native points
    1e-3, phase 1e-3), so a branch added to a trained model initially changes its outputs only slightly.
    """

    def __init__(self, config):
        super().__init__()
        dim = config.feature_dim
        self.grid, self.heads = config.memory_grid, config.heads
        self.seed_tokens = config.seed_tokens
        self.global_every = config.global_attention_every
        self.seed_score = nn.Conv2d(dim, 1, 1)
        self.query = nn.Embedding(config.queries, dim)
        self.query_pos = nn.Embedding(config.queries, dim)
        self.level_pos = nn.Embedding(4, dim)
        self.layers = nn.ModuleList([QueryLayer(dim, config.heads) for _ in range(config.decoder_layers)])
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, 2)  # foreground filament / no object
        self.quality = nn.Linear(dim, 1)
        self.mask_embed = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, config.mask_dim))
        self.mask_features4 = nn.Conv2d(dim, config.mask_dim, 1)
        self.phase_embed = nn.Linear(dim, config.phase_dim) if config.phase_dim else None
        self.phase_gain = nn.Parameter(torch.tensor(1e-3)) if config.phase_dim else None
        self.feedback = nn.Linear(dim, dim) if config.mask_feedback else None
        self.feedback_gain = nn.Parameter(torch.tensor(1e-2)) if config.mask_feedback else None
        self.detail_points = config.query_detail_points
        self.pilot_point_noise, self.pilot_label_noise = config.pilot_point_noise, config.pilot_label_noise
        self.pilot_labels = nn.Embedding(2, dim) if config.denoising_queries else None
        self.image_query_init = config.image_query_init
        self.query_init_candidates = config.query_init_candidates
        self.query_init_diversity = config.query_init_diversity
        if self.image_query_init:
            self.seed_query_projection = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim))
            nn.init.eye_(self.seed_query_projection[0].weight)
            nn.init.zeros_(self.seed_query_projection[0].bias)
            self.seed_query_gain = nn.Parameter(torch.tensor(.1))
            self.seed_position_gain = nn.Parameter(torch.tensor(1.))
        if self.detail_points:
            self.detail_attention = nn.MultiheadAttention(dim, config.heads, batch_first=True)
            self.detail_norm = nn.LayerNorm(dim)
            self.detail_gain = nn.Parameter(torch.tensor(1e-3))

    def attend_native_details(self, query, pixels, masks, seeds):
        """Per-instance foreground/edge points from the native stride-4 path.

        Global memory pooling can average an 8-pixel-wide filament out of a
        42-pixel cell. These tokens bypass that pooling. The point selection is
        detached, as in mask-guided attention; features and attention train.
        Each query takes query_detail_points / 2 body points (3x3 local maxima of its mask probability,
        favoring high seed scores) and as many edge points (probability near .5) on the stride-4 grid, and
        attends to their unpooled features. query B x Q x D, pixels B x D x h x w, masks B x Q x h x w and
        seeds B x 1 x h x w (logits); the update is added with a learned gain (initially 1e-3).
        """
        b, d, h, w = pixels.shape
        q = query.shape[1]
        half = min(self.detail_points // 2, h * w)
        with torch.no_grad():
            p = masks.detach().sigmoid()
            saliency = seeds.detach().sigmoid()
            body = p * (.5 + .5 * saliency)
            peaks = body == F.max_pool2d(body, 3, 1, 1)
            body = torch.where(peaks, body, body * .01)
            edge = 4 * p * (1 - p) * (.5 + .5 * saliency)
            chosen = torch.cat((body.flatten(2).topk(half, -1).indices,
                                edge.flatten(2).topk(half, -1).indices), -1)
            xy = torch.stack(((chosen % w + .5) / w, (chosen // w + .5) / h), -1)
        features = pixels.flatten(2).transpose(1, 2)
        batch = torch.arange(b, device=pixels.device)[:, None, None]
        local = features[batch, chosen].reshape(b * q, -1, d)
        position = encode_xy(xy, d).to(local.dtype).reshape(b * q, -1, d)
        normalized = self.detail_norm(query).reshape(b * q, 1, d)
        update = self.detail_attention(normalized, local + position, local, need_weights=False)[0]
        return query + self.detail_gain * update.reshape(b, q, d)

    def predict(self, query, basis4):
        """Heads on the normalized queries: class and quality logits, embeddings, stride-4 masks B x Q x h x w."""
        query = self.norm(query)
        embed = self.mask_embed(query)
        result = {"pred_logits": self.classifier(query), "pred_quality": self.quality(query).squeeze(-1),
                  "mask_embeddings": embed, "pred_masks": torch.einsum("bqc,bchw->bqhw", embed, basis4)}
        if self.phase_embed is not None:
            result["phase_embeddings"] = self.phase_gain * self.phase_embed(query)
        return result

    def piloted_decode(self, pixels4, basis4, memory, memory_pos, shapes, indices, seeds, guides):
        """An isolated training branch shares decoder weights with normal queries.

        Keeping separate self-attention prevents GT-guided tokens from leaking
        into the normal predictions. The branch is absent at inference.
        guides: B x slots x h x w boolean GT masks (prepare_mask_pilots). Each pilot query starts from a label
        embedding (filament for a GT slot, no object for an empty one; flipped with probability
        pilot_label_noise) and, in every layer, attends only to the memory tokens covered by its guide after
        fresh point noise (noisy_pilot_support). The losses ask it to reconstruct its GT mask (MP-Former).
        """
        b, slots = guides.shape[:2]
        valid = guides.flatten(2).any(-1)
        labels = (~valid).long()
        labels = labels ^ (torch.rand_like(labels.float()) < self.pilot_label_noise).long()
        query = self.pilot_labels(labels) + .01 * self.query.weight[:slots][None]
        query_pos = self.query_pos.weight[:slots][None]
        # Guide support per memory token: any guide pixel inside a pooled cell, and the guide at each seed token.
        supports = [F.adaptive_max_pool2d(guides.float(), size).flatten(2) > 0 for size in shapes]
        if indices is not None:
            guide4 = F.interpolate(guides.float(), size=pixels4.shape[-2:], mode='nearest').flatten(2)
            supports.append(guide4.gather(2, indices[:, None].expand(-1, slots, -1)) > 0)
        support = torch.cat(supports, -1)
        predictions = []
        for layer in self.layers:
            noisy = noisy_pilot_support(support, self.pilot_point_noise)
            if self.feedback is not None:
                weights = noisy.float()
                pooled = weights @ memory.float() / weights.sum(-1, keepdim=True).clamp_min(1)
                query = query + self.feedback_gain * self.feedback(pooled.to(query.dtype))
            blocked = ~noisy
            blocked = blocked & ~blocked.all(-1, keepdim=True)
            mask = blocked[:, None].expand(-1, self.heads, -1, -1).reshape(b * self.heads, slots, memory.shape[1])
            query = layer(query, query_pos, memory, memory_pos, mask)
            prediction = self.predict(query, basis4)
            if self.detail_points:
                # Native points follow the mask predicted after this layer; predict again to include them.
                query = self.attend_native_details(query, pixels4, prediction['pred_masks'], seeds)
                prediction = self.predict(query, basis4)
            predictions.append(prediction)
        prediction['aux_outputs'] = predictions[:-1]
        return prediction

    def attention_values(self, masks, shapes, indices):
        """Mask logits on every memory token: bilinearly resized grids, then the exact seed positions (B x Q x M)."""
        values = [F.interpolate(masks, size=size, mode="bilinear", align_corners=False).flatten(2) for size in shapes]
        if indices is not None:
            values.append(masks.flatten(2).gather(2, indices[:, None].expand(-1, masks.shape[1], -1)))
        return torch.cat(values, -1)

    def forward(self, pixels4, context_pyramid, denoising_masks=None):
        """Decode the instance queries of a batch.

        pixels4: B x D x H/4 x W/4 native detail features; context_pyramid: the four context levels (D channels,
        strides 4 to 32 of the view); denoising_masks: training-only pilot guides. Returns the last layer's
        prediction (pred_logits B x Q x 2, pred_quality B x Q, mask_embeddings B x Q x mask_dim, pred_masks
        B x Q x H/4 x W/4, phase_embeddings) with aux_outputs (the earlier layers' predictions), memory_tokens,
        seed_logits (B x 1 x H/4 x W/4) and, with pilots, the outputs of the pilot branch under 'denoising'.
        """
        # Scene memory: stride-4 detail features and two context levels, pooled to bounded grids.
        # By default 48, 24 and 12 cells on the longer side; a 48-cell grid over 2048 px is ~43 px per cell.
        sources = [pixels4, context_pyramid[1], context_pyramid[2]]
        tokens, positions, shapes = [], [], []
        for i, source in enumerate(sources):
            limit = max(4, self.grid // (2 ** i))
            h, w = source.shape[-2:]
            scale = min(1, limit / max(h, w))
            size = (max(1, round(h * scale)), max(1, round(w * scale)))
            feature = F.adaptive_avg_pool2d(source, size)
            tokens.append(feature.flatten(2).transpose(1, 2))
            positions.append(position_encoding(*size, feature.shape[1], feature.device, feature.dtype) + self.level_pos.weight[i][None, None])
            shapes.append(size)
        memory, memory_pos = torch.cat(tokens, 1), torch.cat(positions, 1)
        seed_logits = self.seed_score(pixels4)
        indices = None
        # Seed tokens: unpooled stride-4 features at the strongest 7x7 local maxima of the detached seed score
        # (non-peaks are pushed 100 logits down), appended to the memory with their exact positions.
        if self.seed_tokens:
            b, d, h, w = pixels4.shape
            scores = seed_logits.detach()
            peaks = scores == F.max_pool2d(scores, 7, 1, 3)
            scores = torch.where(peaks, scores, scores - 100)
            indices = scores.flatten(1).topk(min(self.seed_tokens, h * w), dim=-1).indices
            detail_tokens = pixels4.flatten(2).transpose(1, 2).gather(1, indices[..., None].expand(-1, -1, d))
            xy = torch.stack(((indices % w + 0.5) / w, (indices // w + 0.5) / h), -1)
            detail_pos = encode_xy(xy, d).to(detail_tokens.dtype) + self.level_pos.weight[3][None, None]
            memory, memory_pos = torch.cat((memory, detail_tokens), 1), torch.cat((memory_pos.expand(b, -1, -1), detail_pos), 1)
        query = self.query.weight.unsqueeze(0).expand(pixels4.shape[0], -1, -1)
        query_pos = self.query_pos.weight.unsqueeze(0)
        # Image-dependent queries: each learned query receives the projected feature and the position of one
        # spatially diverse seed peak (query_seeds.py).
        if self.image_query_init:
            locations = query_seed_indices(seed_logits, query.shape[1], self.query_init_candidates,
                                           self.query_init_diversity)
            b, d, h, w = pixels4.shape
            seed_features = pixels4.flatten(2).transpose(1, 2).gather(
                1, locations[..., None].expand(-1, -1, d))
            seed_xy = torch.stack(((locations % w + .5) / w, (locations // w + .5) / h), -1)
            query = query + self.seed_query_gain * self.seed_query_projection(seed_features)
            query_pos = query_pos + self.seed_position_gain * encode_xy(seed_xy, d).to(query.dtype)
        basis4 = self.mask_features4(pixels4)
        # The masks of the initial queries guide the first layer's attention.
        prediction = self.predict(query, basis4)
        predictions = []
        for layer_index, layer in enumerate(self.layers):
            # Masked attention (Mask2Former): a query sees the tokens where its previous mask logit is >= 0, or
            # every token if its mask is empty. Mask feedback adds the mask-weighted mean of the memory.
            values = self.attention_values(prediction["pred_masks"].detach(), shapes, indices)
            if self.feedback is not None:
                weights = values.float().sigmoid()
                pooled = weights @ memory.float() / weights.sum(-1, keepdim=True).clamp_min(1e-6)
                query = query + self.feedback_gain * self.feedback(pooled.to(query.dtype))
            blocked = values < 0
            blocked = blocked & ~blocked.all(-1, keepdim=True)
            mask = blocked[:, None].expand(-1, self.heads, -1, -1).reshape(-1, query.shape[1], memory.shape[1])
            # Periodic global attention can recover distant branches initially
            # excluded by a bad mask. Other layers retain local mask guidance.
            use_global = self.global_every > 0 and layer_index % self.global_every == 0
            query = layer(query, query_pos, memory, memory_pos, None if use_global else mask)
            if self.detail_points:
                query = self.attend_native_details(query, pixels4, prediction['pred_masks'], seed_logits)
            prediction = self.predict(query, basis4)
            predictions.append(prediction)
        prediction["aux_outputs"] = predictions[:-1]
        prediction["memory_tokens"] = memory.shape[1]
        prediction["seed_logits"] = seed_logits
        if denoising_masks is not None:
            if self.pilot_labels is None or not self.training:
                raise ValueError('Mask pilots require a configured training-only branch')
            prediction['denoising'] = self.piloted_decode(pixels4, basis4, memory, memory_pos, shapes,
                                                          indices, seed_logits, denoising_masks)
        return prediction
