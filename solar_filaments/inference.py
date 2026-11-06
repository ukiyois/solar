"""Single-view instance selection: query scores, exclusive pixel ownership and COCO RLE encoding.

Each query yields at most one instance, scored P(filament) * sigmoid(quality). Queries claim the pixels of
their binarized native mask in a score-ordered competition, and an instance survives only if enough of its
area is retained, so the output masks never overlap. fast_eval.owned_masks reproduces this rule on the GPU
for threshold sweeps, and the training PQ surrogate (losses.py) mirrors it with a soft ownership.
"""
import numpy as np
from pycocotools import mask as mask_utils
import torch
from torch.nn import functional as F
from .masks import fine_mask_logits


def encode_mask(mask):
    """COCO RLE of an H x W boolean mask, with 'counts' as an ASCII string for JSON and CSV output."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


@torch.no_grad()
def instance_predictions(output, index=0, score_threshold=0.35, mask_threshold=0.5,
                         min_area=32, query_chunk=4, min_retained_fraction=0.5,
                         original_size=None, mask_cache=None):
    """Stream masks onto one full-image canvas; one global query is one instance.

    Queries with score = P(filament) * sigmoid(quality) >= score_threshold are rendered at native
    resolution in descending score order, query_chunk at a time. A query is skipped if its mask
    (probability >= mask_threshold) has fewer than min_area pixels, or if score * its mean probability on
    that mask (the final score) is below score_threshold. Each pixel goes to the query with the highest
    probability * final score among the masks that contain it. Afterwards an instance is dropped if fewer
    than min_area pixels or less than min_retained_fraction of its mask remain; its pixels stay background.
    original_size (H, W) crops away the padding of pad_image. Returns (instances, labels): a list of dicts
    with instance_id (1..K), final score, retained area and COCO RLE segmentation, and an H x W uint16 map
    of instance ids (0 = background).

    Audit callers may supply {query_id: FP32 probability on this device} to
    reuse native masks across score/binarization sweeps without rerendering.
    """
    logits = output["pred_logits"][index].float()
    scores = logits.softmax(-1)[:, 0] * output["pred_quality"][index].float().sigmoid()
    selected = torch.where(scores >= score_threshold)[0]
    h, w = output["image_size"]
    oh, ow = original_size or (h, w)
    canvas = torch.zeros((oh, ow), dtype=torch.int32, device=logits.device)
    winning = torch.zeros((oh, ow), dtype=torch.float32, device=logits.device)
    area_before, score_map = {}, {}
    selected = selected[scores[selected].argsort(descending=True)]
    for start in range(0, len(selected), query_chunk):
        ids = selected[start:start + query_chunk]
        masks = (fine_mask_logits(output, index, ids)[:, :oh, :ow].sigmoid() if mask_cache is None else
                 torch.stack([mask_cache[q] for q in ids.tolist()])[:, :oh, :ow])
        for query_id, probability in zip(ids.tolist(), masks):
            foreground = probability >= mask_threshold
            area = int(foreground.sum())
            if area < min_area:
                continue
            score = float(scores[query_id] * probability[foreground].mean())
            if score < score_threshold:
                continue
            # Pixel competition: a pixel moves to this query if probability * final score beats its owner.
            confidence = probability * score
            replace = foreground & (confidence > winning)
            winning[replace] = confidence[replace]
            canvas[replace] = query_id + 1
            area_before[query_id + 1], score_map[query_id + 1] = area, score
    labels = canvas.cpu().numpy()
    result, compact = [], np.zeros_like(labels, dtype=np.uint16)
    # Retained-area filter, then consecutive instance ids; pixels of dropped instances are not reassigned.
    for temporary_id, area in area_before.items():
        mask = labels == temporary_id
        retained = int(mask.sum())
        if retained < min_area or retained / area < min_retained_fraction:
            continue
        instance_id = len(result) + 1
        compact[mask] = instance_id
        result.append({"instance_id": instance_id, "score": score_map[temporary_id],
                       "area": retained, "segmentation": encode_mask(mask)})
    return result, compact


def pad_image(image):
    """Replicate-pad B x C x H x W images at the bottom and right to multiples of 4 (the model requires it).

    Returns (padded image, original (H, W)).
    """
    h, w = image.shape[-2:]
    return F.pad(image, (0, (-w) % 4, 0, (-h) % 4), mode="replicate"), (h, w)
