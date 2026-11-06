"""Training objective: set matching, a differentiable Panoptic Quality loss and auxiliary terms.

Every annotator record (one annotator's drawing of one observation) is one sample. Per record, the queries
are matched one-to-one to the GT instances (match_instances). The primary loss is -log of a soft Panoptic
Quality computed from the native-resolution (2048 px) mask logits at importance-sampled pixels
(native_area_points), with a soft pixel ownership that mirrors the inference-time pixel competition
(inference.instance_predictions). Its numerator and denominator are summed over the batch and all GPUs
before dividing, as the competition pools TP/FP/FN over records. The auxiliary terms (classification,
point-sampled mask BCE + Dice, quality, intermediate decoder layers and mask pilots, seed map) are scaled
by `auxiliary_scale`. The remaining terms (area IoU, clDice topology, boundary distance, skeleton coverage
and recall, pixel competition, instance discrimination, dense heads) are implemented but have weight 0 in
the final recipe (configs/pq_recovery_v7.yaml onward).

The topology prior is applied to each annotator's matched instance separately.
No loss joins disconnected annotations or assumes that all foreground connects.
"""
import torch
import math
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from .geometry import derive_geometry, target_at_points, unpack_masks
from .masks import sample_features, phase_samples, fine_mask_points, fine_mask_logits, foreground_probability_mean
from .panoptic_surrogate import sampled_panoptic_terms

# Alias of masks.sample_features: B x C x h x w maps -> B x C x P values at P normalized (x, y) points.
sample_points = sample_features


def dice_loss(logits, target, weights=None):
    """Soft Dice loss along the last axis, averaged over all other axes.

    logits, target (0/1) and the optional weights share the shape ... x P (P sample points). The +1 in
    numerator and denominator keeps the loss finite and makes an empty prediction on an empty target cost 0.
    """
    p = logits.float().sigmoid()
    weights = torch.ones_like(target) if weights is None else weights
    return (1 - (2 * (p * target * weights).sum(-1) + 1) /
            ((p * weights).sum(-1) + (target * weights).sum(-1) + 1)).mean()


def soft_skeleton(probability, iterations=8):
    """Differentiable soft skeleton of clDice (Shit et al., CVPR 2021).

    probability: B x 1 x H x W in [0, 1]. Soft erosion is a min-pool over the 3 x 3 cross (a pixel and its
    4 neighbors) and opening is an erosion followed by a 3 x 3 max-pool. The thin ridges that opening removes
    are collected, as a soft union, over `iterations` successive erosions. As clDice notes, `iterations` must
    be at least the largest structure radius in pixels, otherwise thick parts get no skeleton. Returns
    B x 1 x H x W in [0, 1].
    """
    def erode(x):
        return torch.minimum(-F.max_pool2d(-x, (3, 1), 1, (1, 0)),
                             -F.max_pool2d(-x, (1, 3), 1, (0, 1)))
    def opened(x):
        return F.max_pool2d(erode(x), 3, 1, 1)
    skeleton = F.relu(probability - opened(probability))
    for _ in range(iterations):
        probability = erode(probability)
        delta = F.relu(probability - opened(probability))
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def cldice_loss(logits, truth, skeleton, iterations=8):
    """clDice loss (Shit et al., CVPR 2021) of matched instances, each in its own ROI.

    logits: N x R x R ROI mask logits (fine_rois); truth, skeleton: N x R x R GT mask and precomputed GT
    skeleton of the matched instance. Topology precision is the share of the soft predicted skeleton inside
    the GT mask; topology sensitivity is the share of the GT skeleton covered by the predicted mask. The loss
    is one minus their harmonic mean (+1 smoothing), averaged over instances.
    """
    p = logits.float().sigmoid()[:, None]
    predicted_skeleton = soft_skeleton(p, iterations)[:, 0]
    precision = ((predicted_skeleton * truth).sum((-2, -1)) + 1) / (predicted_skeleton.sum((-2, -1)) + 1)
    recall = ((p[:, 0] * skeleton).sum((-2, -1)) + 1) / (skeleton.sum((-2, -1)) + 1)
    return (1 - 2 * precision * recall / (precision + recall).clamp_min(1e-6)).mean()


def balanced_binary(logits, truth):
    """Class-balanced BCE plus Dice for a dense binary map (seed map and semantic/centerline/boundary heads).

    logits, truth: C x H x W. Positive pixels are up-weighted by sqrt(#negative / #positive), bounded to
    [1, 20]. Skeleton and edge pixels cover a small fraction of the disk, and the bound keeps the background
    constraining false alarms.
    """
    positive = truth.sum().clamp_min(1)
    pos_weight = ((truth.numel() - positive) / positive).sqrt().clamp(1, 20)
    bce = ((1 - truth) * F.softplus(logits) + pos_weight * truth * F.softplus(-logits)).mean()
    return bce + dice_loss(logits.flatten(1), truth.flatten(1))


def training_points(target, count, coarse):
    """Mask-loss points: 1/4 uniform, 1/4 per-instance foreground, 1/4 edge, 1/4 uncertain.

    target: geometry dict of one record (geometry.derive_geometry); count: number of points (8192);
    coarse: Q x H/4 x W/4 stride-4 mask logits of the same image. Foreground and edge points cycle over the
    instances, so the smallest filament gets as many points as the largest; edge anchors get a Gaussian
    jitter of 2 native pixels. As in PointRend (Kirillov et al., CVPR 2020), uncertain points are those
    3x-oversampled uniform candidates whose highest query probability is closest to 0.5. Returns count x 2
    normalized (x, y) coordinates snapped to native pixel centers.

    Cached fixed-size anchor lists avoid dynamic CUDA nonzero and synchronize
    only once for the small Hungarian assignment, not for pixel sampling.
    """
    device = coarse.device
    size = target["auxiliary"].shape[-2:]
    h, w = size[0] * 2, size[1] * 2
    n = target["positive_points"].shape[0]
    quarter = count // 4
    uniform = torch.rand(quarter, 2, device=device)
    if n:
        # Cycling instances guarantees coverage even for the smallest filament.
        owners = torch.arange(quarter, device=device) % n
        choices = torch.randint(target["positive_points"].shape[1], (quarter,), device=device)
        positive = target["positive_points"][owners, choices]
        edges = target["edge_points"][owners, choices]
        edges = edges + torch.randn_like(edges) * edges.new_tensor([2 / w, 2 / h])
    else:
        positive, edges = torch.rand_like(uniform), torch.rand_like(uniform)
    candidates = torch.rand(max(1, (count - 3 * quarter) * 3), 2, device=device)
    with torch.no_grad():
        # Uncertainty near the most likely foreground mask, rather than every
        # irrelevant no-object query's zero crossing.
        confidence = coarse.detach().sigmoid().amax(0, keepdim=True)[None]
        values = sample_features(confidence, candidates)[0, 0]
        indices = (values - 0.5).abs().topk(count - 3 * quarter, largest=False).indices
    points = torch.cat((uniform, positive, edges, candidates[indices]), 0).clamp(0, 1 - 1e-7)
    # Native binary labels and the phase branch refer to identical pixel centers.
    scale = points.new_tensor([w, h])
    return ((points * scale).floor() + 0.5) / scale


@torch.no_grad()
def match_instances(logits, mask_logits, target_samples, dense_masks=None, occupancy=None, sample_weights=None,
                    mode='legacy'):
    """Hungarian matching of one record's GT instances to queries (Mask2Former-style set prediction).

    logits: Q x 2 class logits (index 0 = filament); mask_logits: Q x P query mask logits at P points;
    target_samples: N x P GT labels (0/1) at the same points; sample_weights: optional P inverse-proposal
    weights of importance-sampled points (native_area_points). Point BCE and soft Dice are computed for all
    Q x N pairs as matrix products. Cost 'legacy': -2 P(filament) + 2 BCE + 5 Dice; cost 'pq':
    -P(filament) * (1 - Dice) + 0.001 BCE. dense_masks (Q x h x w logits) together with occupancy
    (N x H/4 x W/4 GT pixels per 4 x 4 cell) add 2 * (1 - soft IoU). scipy's linear_sum_assignment solves
    the assignment on the CPU. Returns (query indices, GT indices), N of each, matched pairwise; raises
    ValueError if N > Q.
    """
    if target_samples.shape[0] == 0:
        empty = torch.empty(0, dtype=torch.long, device=logits.device)
        return empty, empty
    if target_samples.shape[0] > logits.shape[0]:
        raise ValueError("More annotated instances than queries")
    p, y = mask_logits.float().sigmoid(), target_samples.float()
    # BCE(x, y) = softplus(x) - x * y for y in {0, 1}, so the pairwise mean BCE is one matrix product.
    if sample_weights is None:
        bce = F.softplus(mask_logits.float()).mean(-1, keepdim=True) - mask_logits.float() @ y.T / y.shape[-1]
        dice = 1 - (2 * p @ y.T + 1) / (p.sum(-1, keepdim=True) + y.sum(-1)[None] + 1)
    else:
        weight = sample_weights.float()[None]
        bce = ((F.softplus(mask_logits.float()) * weight).sum(-1, keepdim=True) -
               (mask_logits.float() * weight) @ y.T) / weight.sum().clamp_min(1e-8)
        intersection = (p * weight) @ y.T
        dice = 1 - (2 * intersection + 1e-6) / (
            (p * weight).sum(-1, keepdim=True) + (y * weight).sum(-1)[None] + 1e-6)
    probability = logits.float().softmax(-1)[:, :1]
    if mode == 'pq':
        cost = -probability * (1 - dice) + 1e-3 * bce
    elif mode == 'legacy':
        cost = -2 * probability + 2 * bce + 5 * dice
    else:
        raise ValueError('Unknown instance matching mode')
    if occupancy is not None:
        dense = dense_masks.float().sigmoid().flatten(1)
        gt = F.interpolate(occupancy[None].float() / 16, size=dense_masks.shape[-2:], mode="area")[0].flatten(1)
        intersection = dense @ gt.T
        area_iou = intersection / (dense.sum(-1, keepdim=True) + gt.sum(-1)[None] - intersection).clamp_min(1)
        cost = cost + 2 * (1 - area_iou)
    rows, columns = linear_sum_assignment(cost.cpu().numpy())
    return torch.as_tensor(rows, device=logits.device), torch.as_tensor(columns, device=logits.device)


@torch.no_grad()
def native_area_points(target, coarse_logits, count, uniform_fraction=.25, mode='union'):
    """Native-pixel samples with a known proposal, not biased foreground IoU.

    Sample cells from a uniform/active mixture, then a uniform native pixel in
    each 4x4 cell. The active set contains *every* GT cell plus predicted support.
    Inverse proposal weights make foreground and background area estimates
    unbiased while providing many observations of the smallest filaments.

    target['occupancy4']: N x H/4 x W/4 GT pixels per cell; coarse_logits: Q x H/4 x W/4 stride-4 mask
    logits (predicted support: any query logit > -2, i.e. probability > 0.12). 'union': uniform_fraction
    of the mass is uniform over all cells, the rest uniform over the union of GT and predicted cells.
    'balanced_mixture' (final recipe): uniform_fraction uniform; half of the rest split equally between the
    GT instances (uniform within each), half uniform over predicted cells. Returns points (count x 2
    normalized (x, y) native pixel centers) and weights (count) = 1 / (number of cells * proposal(cell)):
    the mean of weight * f over the samples is an unbiased estimate of the mean of f over all native pixels.
    """
    occupancy = target['occupancy4']
    h4, w4 = occupancy.shape[-2:]
    predicted = F.interpolate(coarse_logits[None].detach().float(), size=(h4, w4),
                              mode='bilinear', align_corners=False)[0].amax(0) > -2
    total = h4 * w4
    if mode == 'balanced_mixture':
        # Keep GT-instance coverage independent of the size of predicted support.
        # A broad low-confidence query must not turn the entire proposal uniform.
        per_instance = (occupancy > 0).float().flatten(1)
        gt_proposal = (per_instance / per_instance.sum(-1, keepdim=True).clamp_min(1)).sum(0)
        gt_proposal = gt_proposal / max(1, len(occupancy))
        pred_proposal = predicted.flatten().float()
        pred_proposal = pred_proposal / pred_proposal.sum().clamp_min(1)
        proposal = (uniform_fraction / total + .5 * (1 - uniform_fraction) *
                    (gt_proposal + pred_proposal))
    elif mode == 'union':
        active = ((occupancy.sum(0) > 0) | predicted).flatten().float()
        proposal = uniform_fraction / total + (1 - uniform_fraction) * active / active.sum().clamp_min(1)
    else:
        raise ValueError('Unknown native area proposal: ' + mode)
    proposal = proposal / proposal.sum()
    cells = torch.multinomial(proposal, count, replacement=True)
    phase = torch.randint(4, (count, 2), device=cells.device)
    xy = torch.stack(((cells % w4) * 4 + phase[:, 0], (cells // w4) * 4 + phase[:, 1]), -1)
    points = (xy.float() + .5) / xy.new_tensor([w4 * 4, h4 * 4])
    weights = 1 / (total * proposal[cells])
    return points, weights


@torch.no_grad()
def sampled_query_quality(native_logits, native_truth, weights, mask_threshold=.5):
    """Best hard native-mask IoU, independently of Hungarian object assignment.

    The inverse-proposal weights are essential: foreground-biased samples alone
    would overestimate IoU. The ratio is a finite-sample estimate of native IoU,
    not an exact full-image label. No objectness probability enters this target.

    native_logits: Q x P, native_truth: N x P, weights: P (native_area_points). Masks are binarized at
    mask_threshold. Returns Q values in [0, 1]: each query's best IoU with any GT instance (0 without GT).
    """
    if native_truth.shape[0] == 0:
        return native_logits.new_zeros(native_logits.shape[0], dtype=torch.float32)
    binary = (native_logits.detach() >= math.log(mask_threshold / (1 - mask_threshold))).float()
    truth, weight = native_truth.float(), weights.float()[None]
    intersection = (binary * weight) @ truth.T
    union = (binary * weight).sum(-1, keepdim=True) + (truth * weight).sum(-1)[None] - intersection
    return (intersection / union.clamp_min(1e-8)).amax(-1).clamp(0, 1)


def fine_rois(output, index, rows, boxes, resolution):
    """Native refined mask logits of matched queries, each on an R x R grid over its own GT instance's ROI.

    rows: N matched query indices; boxes: N x 4 normalized (x0, y0, x1, y1) ROIs of the matched GT
    instances (geometry 'rois'); resolution: R. Grid points are snapped to native pixel centers and use the
    mask definition of masks.fine_mask_points (stride-2 basis plus sub-pixel phase residual), so they align
    with the cached GT 'roi_masks'. Points outside the image get logit -20. Returns N x R x R logits.
    """
    axis = (torch.arange(resolution, device=boxes.device) + 0.5) / resolution
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    unit = torch.stack((xx, yy), -1)[None]
    grid = boxes[:, None, None, :2] + unit * (boxes[:, None, None, 2:] - boxes[:, None, None, :2])
    points = grid.reshape(-1, 2)
    native_size = points.new_tensor([output["image_size"][1], output["image_size"][0]])
    points = ((points * native_size).floor() + .5) / native_size
    basis = sample_features(output["mask_features2"][index:index + 1], points)[0]
    basis = basis.reshape(basis.shape[0], len(rows), -1)
    result = torch.einsum("nc,cnp->np", output["mask_embeddings"][index, rows].float(), basis)
    if "phase_features2" in output:
        phases = phase_samples(output["phase_features2"][index:index + 1], points)[0]
        phases = phases.reshape(phases.shape[0], len(rows), -1)
        result = result + torch.einsum("nc,cnp->np", output["phase_embeddings"][index, rows].float(), phases)
    valid = ((grid >= 0) & (grid <= 1)).all(-1)
    return result.reshape(-1, resolution, resolution).masked_fill(~valid, -20)


@torch.no_grad()
def native_iou_targets(output, index, rows, columns, target, chunk=2, mask_threshold=.5):
    """Exact native-area IoU of the raw refined masks, before winner assignment.

    Geometry-biased point samples must never be used as IoU quality labels.
    Streaming bounds peak memory independently of the number of instances.

    Returns one IoU per matched pair (rows[i], columns[i]): the query's full-resolution mask binarized at
    mask_threshold against the unpacked GT mask, rendered `chunk` queries at a time. This is the quality-head
    target for quality_target='assigned_exact'.
    """
    values = []
    for start in range(0, len(rows), chunk):
        selected = rows[start:start + chunk]
        truth = unpack_masks(target["packed_masks"][columns[start:start + chunk]])
        binary = fine_mask_logits(output, index, selected) >= math.log(mask_threshold / (1 - mask_threshold))
        intersection = (binary & truth).sum((-2, -1)).float()
        union = (binary | truth).sum((-2, -1)).float().clamp_min(1)
        values.append(intersection / union)
    return torch.cat(values) if values else output["pred_quality"].new_empty(0)


class FilamentCriterion(nn.Module):
    """All training losses of a batch; forward() returns {term: weighted loss}, and their sum is minimized.

    point_count: points per record for the mask losses (training_points). weights: per-term weights (see
    self.weights; unknown names raise). The structural terms (topology, center_coverage, boundary_distance,
    competition) ramp in linearly over the first `structural_warmup` epochs. Every term except 'pq' is
    averaged over the records of the batch and multiplied by auxiliary_scale.

    PQ surrogate. Each query has the inference score c = P(filament) * sigmoid(quality) * mean probability
    on its foreground. pq_surrogate='soft' uses a soft detection sigmoid((c - pq_score_threshold) / T_score),
    a soft pixel ownership softmax over queries of (log sigmoid(logit) + log c + log detection) / T_owner
    (c and detection detached), owned masks sigmoid((logit - logit(pq_mask_threshold)) / T_mask) * ownership,
    and a soft true positive sigmoid((IoU - pq_iou_threshold) / T_iou) of each matched owned mask. The
    temperatures T anneal linearly from their initial to their *_final values over pq_temperature_epochs; the
    final recipe uses score threshold 0.15 with T_score 0.05, T_owner 0.15, T_mask 0.5 and T_iou 0.1.
    pq_surrogate='straight_through' uses the discrete inference decisions in the forward pass instead
    (panoptic_surrogate.py, with the inference filters pq_min_area and pq_min_retained_fraction) and needs
    pq_objective='linear'. pq_objective 'log' minimizes -log PQ, 'linear' 1 - PQ. pq_point_count native
    pixels per record are drawn with the pq_sampling proposal (native_area_points).

    quality_target: 'assigned_exact' = exact native IoU of each matched query's mask (0 for the others);
    'best_native_sampled' (final recipe) = every query's best sampled IoU with any GT instance.
    quality_hurdle replaces the target by IoU * 1{IoU > pq_iou_threshold} (off in the final recipe; it lowered
    fold-0 PQ). matching_mode selects the Hungarian cost (match_instances). instance_temperature and
    skeleton_recall_points configure the optional instance-discrimination and skeleton-recall terms, and
    pilot_weight scales the mask-pilot reconstruction inside 'decoder_aux'.
    """

    def __init__(self, point_count=8192, weights=None, topology_iterations=8, structural_warmup=5,
                 pq_iou_threshold=.5, pq_iou_temperature=.15,
                 pq_iou_temperature_final=.10, pq_score_threshold=.35,
                 pq_score_temperature=.2, pq_score_temperature_final=.05,
                 pq_temperature_epochs=20,
                 auxiliary_scale=.05, pq_point_count=16384, pq_objective='log',
                 pq_mask_temperature=1., pq_mask_temperature_final=.5,
                 pq_owner_temperature=.5, pq_owner_temperature_final=.15, pilot_weight=1.,
                 pq_sampling='union', quality_target='assigned_exact', pq_surrogate='soft',
                 pq_min_area=32, pq_min_retained_fraction=.5, pq_mask_threshold=.5, matching_mode='legacy',
                 instance_temperature=.3, skeleton_recall_points=64, quality_hurdle=False):
        super().__init__()
        self.point_count, self.iterations = point_count, topology_iterations
        self.structural_warmup, self.progress = structural_warmup, float(structural_warmup)
        # instance_discrimination / skeleton_recall default to 0 so older configs are unchanged.
        self.weights = {"class": 2., "mask": 1., "fine_mask": 1., "area_iou": 1., "quality": .5,
                        "topology": .5, "center_coverage": .25, "boundary_distance": .2,
                        "competition": .5, "decoder_aux": .25, "seed": .2,
                        "semantic": .3, "centerline": .2, "boundary": .1, "pq": 1.,
                        "instance_discrimination": 0., "skeleton_recall": 0.}
        if instance_temperature <= 0 or skeleton_recall_points < 1:
            raise ValueError("Invalid instance-discrimination temperature or skeleton sample count")
        self.instance_temperature, self.skeleton_recall_points = float(instance_temperature), int(skeleton_recall_points)
        unknown = set(weights or {}) - set(self.weights)
        if unknown:
            raise ValueError(f"Unknown loss weights: {sorted(unknown)}")
        self.weights.update(weights or {})
        # Cross-entropy weights of (filament, no object); the down-weighted no-object class follows DETR.
        self.register_buffer("class_weights", torch.tensor([1.0, 0.1]))
        if min(pq_iou_temperature, pq_iou_temperature_final, pq_score_temperature,
               pq_score_temperature_final, pq_temperature_epochs) <= 0:
            raise ValueError("PQ surrogate temperatures must be positive")
        if auxiliary_scale < 0:
            raise ValueError("Auxiliary loss scale must be non-negative")
        self.pq_iou_threshold = float(pq_iou_threshold)
        self.pq_iou_temperature = float(pq_iou_temperature)
        self.pq_iou_temperature_final = float(pq_iou_temperature_final)
        self.pq_score_threshold = float(pq_score_threshold)
        self.pq_score_temperature = float(pq_score_temperature)
        self.pq_score_temperature_final = float(pq_score_temperature_final)
        self.pq_temperature_epochs = float(pq_temperature_epochs)
        self.auxiliary_scale = float(auxiliary_scale)
        if pq_objective not in ('log', 'linear') or pq_point_count < 256:
            raise ValueError('Invalid PQ objective or native sampling budget')
        if min(pq_mask_temperature, pq_mask_temperature_final, pq_owner_temperature,
               pq_owner_temperature_final) <= 0:
            raise ValueError('Mask and ownership temperatures must be positive')
        self.pq_point_count, self.pq_objective = int(pq_point_count), pq_objective
        self.pq_mask_temperature, self.pq_mask_temperature_final = pq_mask_temperature, pq_mask_temperature_final
        self.pq_owner_temperature, self.pq_owner_temperature_final = pq_owner_temperature, pq_owner_temperature_final
        self.last_stats = {}
        if pilot_weight < 0:
            raise ValueError('pilot_weight must be nonnegative')
        self.pilot_weight = float(pilot_weight)
        if pq_sampling not in ('union', 'balanced_mixture'):
            raise ValueError('Unknown PQ sampling mode')
        if quality_target not in ('assigned_exact', 'best_native_sampled'):
            raise ValueError('Unknown query quality target')
        self.pq_sampling, self.quality_target = pq_sampling, quality_target
        # Hurdle quality target IoU * 1{IoU > pq_iou_threshold}: averaged over a physical image's
        # annotator records it equals u = P(match) * E[IoU | match], the per-candidate statistic whose
        # global threshold tau = PQ*/2 is the optimal selection rule for pooled PQ (Dinkelbach).
        # A raw-IoU target rewards near misses (IoU < .5) that count as FP + FN in PQ.
        self.quality_hurdle = bool(quality_hurdle)
        if pq_surrogate not in ('soft', 'straight_through'):
            raise ValueError('Unknown PQ surrogate')
        if pq_surrogate == 'straight_through' and pq_objective != 'linear':
            raise ValueError('Discrete-forward PQ needs the stable linear objective, including zero-TP batches')
        if pq_min_area < 1 or not 0 <= pq_min_retained_fraction <= 1:
            raise ValueError('Invalid inference-aligned PQ filters')
        self.pq_surrogate = pq_surrogate
        self.pq_min_area, self.pq_min_retained_fraction = pq_min_area, pq_min_retained_fraction
        if not 0 < pq_mask_threshold < 1 or matching_mode not in ('legacy', 'pq'):
            raise ValueError('Invalid mask operating point or matching mode')
        self.pq_mask_threshold, self.matching_mode = pq_mask_threshold, matching_mode

    def set_progress(self, epoch):
        """Set the training progress in fractional epochs (temperature annealing and structural warm-up)."""
        self.progress = float(epoch)

    def forward(self, outputs, targets):
        """Weighted loss terms of one batch.

        outputs: model output dict (pred_logits B x Q x 2, pred_quality B x Q, pred_masks B x Q x H/4 x W/4,
        the mask/phase embeddings and stride-2 bases rendered by masks.py, aux_outputs, seed_logits, the
        optional dense heads and the training-only mask-pilot branch 'denoising'). targets: one geometry dict
        per annotator record (data.FilamentDataset). Returns {term: weighted loss}; diagnostics such as the
        batch's soft PQ are stored in self.last_stats.
        """
        device = outputs["pred_logits"].device
        zero = outputs["pred_logits"].float().sum() * 0
        losses = {key: zero for key in self.weights}
        pq_numerator, pq_denominator, pq_gt_count = zero, zero, zero.detach().clone()
        matched_ious, matched_scores, pilot_losses = [], [], []
        quality_labels, quality_predictions = [], []
        panoptic_stats = []
        for index, original in enumerate(targets):
            # Legacy full masks are accepted for diagnostic scripts only.
            target = original
            if "packed_masks" not in target:
                target = {**target, **{k: torch.from_numpy(v) for k, v in
                          derive_geometry(original["masks"].cpu().numpy().astype(bool)).items()}}
            target = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in target.items()}
            # Two point sets: `points` (training_points) for the mask losses, and importance-sampled native
            # pixels with inverse-proposal weights (native_area_points) for matching, quality and PQ.
            points = training_points(target, self.point_count, outputs["pred_masks"][index])
            truth = target_at_points(target, points)
            coarse = sample_features(outputs["pred_masks"][index:index + 1], points)[0]
            fine = fine_mask_points(outputs, index, slice(None), points)
            area_points, area_weights = native_area_points(target, outputs['pred_masks'][index], self.pq_point_count,
                                                           mode=self.pq_sampling)
            area_truth = target_at_points(target, area_points)
            native_logits = fine_mask_points(outputs, index, slice(None), area_points)
            rows, columns = match_instances(outputs['pred_logits'][index], native_logits, area_truth,
                                             sample_weights=area_weights, mode=self.matching_mode)
            # Class targets: 0 (filament) for matched queries, 1 (no object) for all others.
            classes = torch.ones(outputs["pred_logits"].shape[1], dtype=torch.long, device=device)
            classes[rows] = 0
            def classification(prediction):
                return F.cross_entropy(prediction["pred_logits"][index].float(), classes, weight=self.class_weights)
            losses["class"] = losses["class"] + classification(outputs)
            # Zero-valued terms keep every output in the autograd graph, as DDP expects for all parameters.
            losses["mask"] = losses["mask"] + coarse.sum() * 0
            quality_truth = torch.zeros_like(outputs["pred_quality"][index], dtype=torch.float32)
            if self.quality_target == 'best_native_sampled':
                quality_truth = sampled_query_quality(native_logits, area_truth, area_weights, self.pq_mask_threshold)
            quality_raw = quality_truth.clone()
            matched_iou = outputs["pred_quality"].new_empty((0,), dtype=torch.float32)
            # Includes zero gradients on all head parameters for empty DDP batches.
            losses["fine_mask"] = losses["fine_mask"] + fine.sum() * 0
            if len(rows):
                gt = truth[columns]
                # Retain supervision in fuzzy edge bands, but reduce its weight.
                edge = sample_features(target["auxiliary"][None, 2:3], points, "nearest")[0, 0]
                reliability = 1 - .25 * edge
                def mask_loss(pred):
                    """2 x point BCE (edge-band points weighted 0.75) + 5 x Dice of matched masks vs. their GT."""
                    bce = F.binary_cross_entropy_with_logits(pred.float(), gt, reduction="none")
                    return 2 * (bce * reliability).mean() + 5 * dice_loss(pred, gt)
                losses["mask"] = losses["mask"] + mask_loss(coarse[rows])
                losses["fine_mask"] = losses["fine_mask"] + mask_loss(fine[rows])
                # area_iou: soft IoU of the stride-4 masks with each GT instance's pixel fraction per 4 x 4 cell.
                occupancy = F.interpolate(target["occupancy4"][None, columns].float() / 16,
                                          size=outputs["pred_masks"].shape[-2:], mode="area")[0]
                p = outputs["pred_masks"][index, rows].float().sigmoid()
                intersection = (p * occupancy).sum((-2, -1))
                iou = intersection / (p.sum((-2, -1)) + occupancy.sum((-2, -1)) - intersection).clamp_min(1)
                matched_iou = iou
                losses["area_iou"] = losses["area_iou"] + (1 - iou).mean()
                if self.quality_target == 'assigned_exact':
                    quality_truth[rows] = native_iou_targets(outputs, index, rows, columns, target,
                                                            mask_threshold=self.pq_mask_threshold)
                    quality_raw = quality_truth.clone()
                quality_labels.append(quality_truth[rows].mean().detach())
                quality_predictions.append(outputs['pred_quality'][index, rows].float().sigmoid().mean().detach())
                if self.weights["topology"] or self.weights["boundary_distance"]:
                    roi_gt, roi_sk = target["roi_masks"][columns].float(), target["roi_skeletons"][columns].float()
                    roi = fine_rois(outputs, index, rows, target["rois"][columns], roi_gt.shape[-1])
                    if self.weights["topology"]:
                        losses["topology"] = losses["topology"] + cldice_loss(roi, roi_gt, roi_sk, self.iterations)
                    if self.weights["boundary_distance"]:
                        # Boundary loss (Kervadec et al., MIDL 2019) with the signed distance of the GT ROI
                        # mask: probability outside the GT costs its distance, missing interior its depth.
                        distance = target["roi_distance"][columns].float()
                        losses["boundary_distance"] = losses["boundary_distance"] + ((roi.sigmoid() - roi_gt) * distance).mean()
                if self.weights["center_coverage"]:
                    # BCE towards 1 of each matched query's native logits on 128 skeleton anchors of its own
                    # GT instance (the diagonal of the query x instance table).
                    skeleton = target["skeleton_points"][columns, :128].reshape(-1, 2)
                    coverage = fine_mask_points(outputs, index, rows, skeleton).reshape(len(rows), len(rows), -1)
                    diagonal = coverage[torch.arange(len(rows), device=device), torch.arange(len(rows), device=device)]
                    losses["center_coverage"] = losses["center_coverage"] + F.softplus(-diagonal).mean()
                if self.weights["skeleton_recall"]:
                    # Skeleton Recall (Kirchhoff et al., ECCV 2024): soft recall of each
                    # matched query's native probability on its own GT skeleton.
                    k = self.skeleton_recall_points
                    pick = torch.randint(target["skeleton_points"].shape[1], (k,), device=device)
                    skeleton = target["skeleton_points"][columns][:, pick].reshape(-1, 2)
                    values = fine_mask_points(outputs, index, rows, skeleton).reshape(len(rows), len(rows), k)
                    diagonal = torch.arange(len(rows), device=device)
                    recall = values[diagonal, diagonal].float().sigmoid().mean(-1)
                    losses["skeleton_recall"] = losses["skeleton_recall"] + (1 - recall).mean()
            if self.weights["instance_discrimination"] and truth.shape[0] > 1:
                # MaX-DeepLab (CVPR 2021) per-pixel instance discrimination on the
                # stride-2 mask basis: each pixel must identify its own GT instance
                # embedding among the K instances of this annotation record.
                single = truth.sum(0) == 1
                if bool(single.any()):
                    membership = truth[:, single]
                    features = F.normalize(sample_features(outputs["mask_features2"][index:index + 1],
                                                           points[single])[0].float(), dim=0)
                    centers = F.normalize(features @ membership.T, dim=0)
                    logits = (centers.T @ features) / self.instance_temperature
                    losses["instance_discrimination"] = (losses["instance_discrimination"] +
                                                         F.cross_entropy(logits.T, membership.argmax(0)))
            elif self.weights["instance_discrimination"]:
                losses["instance_discrimination"] = (losses["instance_discrimination"] +
                                                     outputs["mask_features2"][index].float().sum() * 0)
            # PQ is computed from the native refined mask definition, including
            # soft exclusive ownership. Area weights correct the point proposal.
            # Use hard foreground support for confidence, matching inference and
            # avoiding the old sigmoid gate's catastrophic background mass floor.
            mean_foreground_probability = foreground_probability_mean(native_logits, area_weights,
                                                                       self.pq_mask_threshold)
            confidence = (outputs["pred_logits"][index].float().softmax(-1)[:, 0] *
                          outputs["pred_quality"][index].float().sigmoid() *
                          mean_foreground_probability)
            # Temperatures anneal linearly from their initial to their final values over pq_temperature_epochs.
            temperature_progress = min(1., max(0., self.progress / self.pq_temperature_epochs))
            iou_temperature = self.pq_iou_temperature + temperature_progress * (
                self.pq_iou_temperature_final - self.pq_iou_temperature)
            score_temperature = self.pq_score_temperature + temperature_progress * (
                self.pq_score_temperature_final - self.pq_score_temperature)
            mask_temperature = self.pq_mask_temperature + temperature_progress * (
                self.pq_mask_temperature_final - self.pq_mask_temperature)
            owner_temperature = self.pq_owner_temperature + temperature_progress * (
                self.pq_owner_temperature_final - self.pq_owner_temperature)
            pq_gt_count = pq_gt_count + len(columns)
            if self.pq_surrogate == 'straight_through':
                hard_numerator, hard_denominator, hard_stats = sampled_panoptic_terms(
                    native_logits, confidence, area_truth, area_weights,
                    outputs['image_size'][0] * outputs['image_size'][1],
                    self.pq_score_threshold, self.pq_iou_threshold,
                    score_temperature, iou_temperature, mask_temperature, owner_temperature,
                    self.pq_min_area, self.pq_min_retained_fraction, self.pq_mask_threshold)
                pq_numerator = pq_numerator + hard_numerator
                pq_denominator = pq_denominator + hard_denominator
                panoptic_stats.append(hard_stats)
                if len(rows):
                    matched_ious.append(hard_stats['sampled_matched_owned_iou'].reshape(1))
                    matched_scores.append(confidence[rows].detach())
            else:
                # Soft inference rule: detection gate, softmax pixel ownership over queries and a soft mask
                # threshold; soft IoU and soft TP of each matched owned mask from the weighted native samples.
                detection_probability = torch.sigmoid(
                    (confidence - self.pq_score_threshold) / score_temperature)
                owner_logits = (F.logsigmoid(native_logits.float()) +
                                confidence.detach().clamp_min(1e-8).log()[:, None] +
                                detection_probability.detach().clamp_min(1e-8).log()[:, None]) / owner_temperature
                ownership = owner_logits.softmax(0)
                centered = native_logits.float() - math.log(self.pq_mask_threshold / (1 - self.pq_mask_threshold))
                owned_probability = (centered / mask_temperature).sigmoid() * ownership
                matched = torch.zeros_like(classes, dtype=torch.bool)
                matched[rows] = True
                if len(rows):
                    y = area_truth[columns]
                    p_native = owned_probability[rows]
                    intersection = (p_native * y * area_weights).sum(-1)
                    union = ((p_native + y - p_native * y) * area_weights).sum(-1)
                    matched_iou = intersection / union.clamp_min(1e-8)
                    matched_ious.append(matched_iou.detach())
                    matched_scores.append(confidence[rows].detach())
                    iou_probability = torch.sigmoid(
                        (matched_iou - self.pq_iou_threshold) / iou_temperature)
                    match_probability = detection_probability[rows] * iou_probability
                    pq_numerator = pq_numerator + (match_probability * matched_iou).sum()
                # TP + FP/2 + FN/2 = (#predictions + #GT) / 2, with the expected number of predictions
                # equal to the sum of the detection probabilities.
                pq_denominator = pq_denominator + .5 * (detection_probability.sum() + len(columns))
            if self.quality_hurdle:
                quality_truth = quality_truth * (quality_truth > self.pq_iou_threshold)
            # Weight by the raw overlap so near misses are pushed firmly towards a zero hurdle target.
            # Weight 1 for queries with overlap (raw IoU > .1, or matched for 'assigned_exact'), 0.1 otherwise.
            quality_foreground = quality_raw > .1 if self.quality_target == 'best_native_sampled' else classes == 0
            qweight = torch.where(quality_foreground, 1., .1)
            losses["quality"] = losses["quality"] + (F.binary_cross_entropy_with_logits(
                outputs["pred_quality"][index].float(), quality_truth, reduction="none") * qweight).sum() / qweight.sum()
            # One owner per unambiguous native pixel. Include every query and a
            # background class; duplicate/split and merged masks compete here.
            if self.weights['competition']:
                count = truth.sum(0)
                owner = torch.full((len(points),), len(classes), device=device, dtype=torch.long)
                if len(rows):
                    gt_owner = truth.argmax(0)
                    inverse = torch.empty(len(rows), device=device, dtype=torch.long)
                    inverse[columns] = rows
                    owner = torch.where(count == 1, inverse[gt_owner], owner)
                competition_logits = torch.cat((fine, fine.new_zeros(1, fine.shape[-1])), 0)[None]
                ce = F.cross_entropy(competition_logits, owner[None], reduction="none")[0]
                losses["competition"] = losses["competition"] + (ce * (count <= 1)).sum() / (count <= 1).sum().clamp_min(1)
            # Deep supervision: intermediate decoder layers get the class and stride-4 mask losses of the final
            # layer's matching.
            for prediction in outputs["aux_outputs"]:
                value = classification(prediction) + prediction["pred_masks"][index].float().sum() * 0
                if len(rows):
                    sampled = sample_features(prediction["pred_masks"][index:index + 1], points)[0][rows]
                    value = value + 2 * F.binary_cross_entropy_with_logits(sampled, truth[columns]) + 5 * dice_loss(sampled, truth[columns])
                losses["decoder_aux"] = losses["decoder_aux"] + value / max(1, len(outputs["aux_outputs"]))
            if 'denoising' in outputs:
                # Mask pilots (MP-Former): slot i was guided by GT instance i and must reconstruct it; unused slots
                # learn 'no object'. Refined native masks of the last layer plus class and stride-4 mask losses
                # averaged over all layers.
                pilot = outputs['denoising']
                n = truth.shape[0]
                classes_pilot = torch.ones(pilot['pred_logits'].shape[1], dtype=torch.long, device=device)
                classes_pilot[:n] = 0
                value = F.cross_entropy(pilot['pred_logits'][index].float(), classes_pilot, weight=self.class_weights)
                refined = fine_mask_points(pilot, index, slice(None), points)
                value = value + refined.sum() * 0
                if n:
                    value = value + 2 * F.binary_cross_entropy_with_logits(refined[:n], truth) + 5 * dice_loss(refined[:n], truth)
                for layer_pilot in [*pilot['aux_outputs'], pilot]:
                    auxiliary = F.cross_entropy(layer_pilot['pred_logits'][index].float(), classes_pilot,
                                                weight=self.class_weights)
                    sampled = sample_features(layer_pilot['pred_masks'][index:index + 1], points)[0]
                    auxiliary = auxiliary + sampled.sum() * 0
                    if n:
                        auxiliary = auxiliary + 2 * F.binary_cross_entropy_with_logits(sampled[:n], truth) + 5 * dice_loss(sampled[:n], truth)
                    value = value + auxiliary / (1 + len(pilot['aux_outputs']))
                losses['decoder_aux'] = losses['decoder_aux'] + self.pilot_weight * value
                pilot_losses.append(value.detach())
            # Seed map (stride 4) against the GT skeleton channel, max-pooled from stride 2. Dense heads
            # (stride 2) against auxiliary channel 0 = union of masks, 1 = skeletons, 2 = edge bands.
            aux = target["auxiliary"].float()
            seed_gt = F.adaptive_max_pool2d(aux[None, 1:2], outputs["seed_logits"].shape[-2:])[0]
            losses["seed"] = losses["seed"] + balanced_binary(outputs["seed_logits"][index].float(), seed_gt)
            for i, name in enumerate(("semantic", "centerline", "boundary")):
                if f"{name}_logits" not in outputs:  # head cut from the model configuration
                    continue
                pred = outputs[f"{name}_logits"][index].float()
                if self.weights[name]:
                    gt = F.interpolate(aux[None, i:i + 1], size=pred.shape[-2:], mode="nearest")[0]
                    losses[name] = losses[name] + balanced_binary(pred, gt)
                else:
                    losses[name] = losses[name] + pred.sum() * 0
        ramp = min(1., max(0., self.progress / max(1e-6, self.structural_warmup)))
        structural = {"topology", "center_coverage", "boundary_distance", "competition"}
        # The official score aggregates TP/FP/FN across the validation set.
        # Match that aggregation over the global DDP batch while preserving
        # gradients through both expected-IoU numerator and denominator.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            from torch.distributed.nn.functional import all_reduce
            pq_numerator = all_reduce(pq_numerator, op=torch.distributed.ReduceOp.SUM)
            pq_denominator = all_reduce(pq_denominator, op=torch.distributed.ReduceOp.SUM)
            pq_gt_count = all_reduce(pq_gt_count, op=torch.distributed.ReduceOp.SUM)
        soft_pq = pq_numerator / pq_denominator.clamp_min(1e-8)
        # -log(PQ) has the same optimum and ordering as 1-PQ, but rescales a
        # nearly-zero PQ's gradient instead of leaving the primary signal tiny.
        primary = ((pq_denominator + 1e-8).log() - (pq_numerator + 1e-8).log()
                   if self.pq_objective == 'log' else 1 - soft_pq)
        # Without any GT instance in the global batch PQ is undefined; the loss is then the denominator,
        # half the expected number of detections, which only suppresses false positives.
        losses['pq'] = torch.where(pq_gt_count > 0, primary, pq_denominator)
        self.last_stats = {('soft_pq' if self.pq_surrogate == 'soft' else 'sampled_global_pq'): soft_pq.detach(),
                           'matched_native_iou': torch.cat(matched_ious).mean() if matched_ious else zero.detach(),
                           'matched_score': torch.cat(matched_scores).mean() if matched_scores else zero.detach()}
        if pilot_losses:
            self.last_stats['pilot_reconstruction'] = torch.stack(pilot_losses).mean()
        if quality_labels:
            self.last_stats['matched_quality_target'] = torch.stack(quality_labels).mean()
            self.last_stats['matched_quality_prediction'] = torch.stack(quality_predictions).mean()
        if panoptic_stats:
            self.last_stats.update({key: torch.stack([row[key] for row in panoptic_stats]).mean()
                                    for key in panoptic_stats[0]})
        # 'pq' is already pooled over the batch; every other term becomes a per-record mean scaled by
        # auxiliary_scale.
        return {key: value * self.weights[key] * (ramp if key in structural else 1.) *
                (1 if key == "pq" else self.auxiliary_scale) /
                (1 if key == "pq" else len(targets)) for key, value in losses.items()}
