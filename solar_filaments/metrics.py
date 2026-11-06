"""Competition metric: Panoptic Quality (Kirillov et al., CVPR 2019) per annotator record, pooled over records.

A GT/prediction pair with IoU > 0.5 is a true positive, and PQ = sum of TP IoUs / (TP + FP/2 + FN/2) = SQ * RQ.
A prediction is scored separately against every annotator record of its observation.
"""
import numpy as np
from pycocotools import mask as mask_utils


def official_pq(predictions, ground_truth):
    """Organizer self-evaluation convention: every pair with IoU strictly > .5.

    predictions, ground_truth: lists of COCO RLEs of one annotator record. With non-overlapping masks an
    IoU > 0.5 match is unique (Kirillov et al.), so no assignment step is needed. Returns pq, sq, rq and the
    raw counts tp, fp, fn, iou_sum that aggregate_pq pools.
    """
    if predictions and ground_truth:
        ious = np.asarray(mask_utils.iou(predictions, ground_truth, [0] * len(ground_truth)))
        matches = ious > 0.5
        tp = int(matches.sum())
        fp = int((~matches.any(axis=1)).sum())
        fn = int((~matches.any(axis=0)).sum())
        iou_sum = float(ious[matches].sum())
    else:
        tp, fp, fn, iou_sum = 0, len(predictions), len(ground_truth), 0.0
    denominator = tp + 0.5 * fp + 0.5 * fn
    return {"pq": iou_sum / denominator if denominator else 0.0,
            "sq": iou_sum / tp if tp else 0.0,
            "rq": tp / denominator if denominator else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "iou_sum": iou_sum}


def aggregate_pq(scores):
    """Pool official_pq rows: TP, FP, FN and matched IoU are summed over all records before dividing.

    This is the competition aggregation, under which records with many filaments weigh more; the plain mean
    of the per-record PQ is reported separately as mean_record_pq.
    """
    tp, fp, fn = (sum(row[key] for row in scores) for key in ("tp", "fp", "fn"))
    denominator = tp + 0.5 * fp + 0.5 * fn
    iou_sum = sum(row['iou_sum'] for row in scores)
    return {"pq": iou_sum / denominator if denominator else 0.0,
            'sq': iou_sum / tp if tp else 0., 'rq': tp / denominator if denominator else 0.,
            'precision': tp / (tp + fp) if tp + fp else 0.,
            'recall': tp / (tp + fn) if tp + fn else 0., 'iou_sum': iou_sum,
            "mean_record_pq": float(np.mean([row["pq"] for row in scores])) if scores else 0.0,
            "tp": tp, "fp": fp, "fn": fn}
