"""Inter-annotator agreement on the training set: PQ of one annotator's record scored against another's.

For every observation with several annotator records, each pair of records (prediction = one record,
ground truth = the other) is scored with the official PQ (IoU > 0.5); PQ is symmetric in the two roles, so
each pair is scored once. The aggregate pools TP/FP/FN over all pairs exactly like the competition metric.
research/label_visibility.py scores a model the same way against each record.
Output: research/interannotator_agreement.json (pairwise PQ distribution, pooled PQ and counts, and the
pairs with the largest instance-count difference and with the lowest PQ).

usage (repository root): python research/interannotator_agreement.py
"""
import itertools, json, statistics
from collections import defaultdict
from pathlib import Path
import numpy as np
from pycocotools import mask as mask_utils

REPOSITORY = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY / 'data/raw/MAGFiLO_1.0_Kaggle_2026'
ANNOTATIONS = ROOT/'train'/'MAGFiLO_1.0_Annotations_kaggle2026_train.json'
OUT = REPOSITORY / 'research/interannotator_agreement.json'
SIZE = 2048

data = json.loads(ANNOTATIONS.read_text(encoding='utf-8'))
image_records = {str(x['id']): x for x in data['images']}
annotations_by_record = defaultdict(list)
for ann in data['annotations']:
    annotations_by_record[str(ann['image_id'])].append(ann)
records_by_file = defaultdict(list)
for record_id, image in image_records.items():
    records_by_file[image['file_name']].append(record_id)

def encode_record(record_id):
    """pycocotools RLEs (2048 x 2048) of the instances of one annotator record, first polygon of each."""
    encoded = []
    for ann in annotations_by_record[record_id]:
        polys = ann['segmentation']
        rles = mask_utils.frPyObjects(polys, SIZE, SIZE)
        if isinstance(rles, dict):
            encoded.append(rles)
        else:
            # Competition JSON says each instance is a single polygon.
            encoded.append(rles[0])
    return encoded

pair_scores=[]
pair_rows=[]
all_tp_iou=0.0
all_tp=all_fp=all_fn=0
for filename, record_ids in records_by_file.items():
    if len(record_ids)<2:
        continue
    record_ids=sorted(record_ids)
    encoded={rid:encode_record(rid) for rid in record_ids}
    for rid_a,rid_b in itertools.combinations(record_ids,2):
        a,b=encoded[rid_a],encoded[rid_b]
        # pycocotools.iou expects detections first and crowd flags for GT.
        iou=np.asarray(mask_utils.iou(a,b,[0]*len(b)),dtype=np.float64)
        hit=iou>0.5
        tp_iou=float(iou[hit].sum())
        tp=int(hit.sum())
        fp=int((hit.sum(axis=0)==0).sum()) if tp or len(a) else len(a)
        fn=int((hit.sum(axis=1)==0).sum()) if tp or len(b) else len(b)
        # Correct empty-side handling explicitly.
        if not a: fp=0; fn=len(b); tp=0; tp_iou=0.0
        if not b: fp=len(a); fn=0; tp=0; tp_iou=0.0
        denom=tp+0.5*fp+0.5*fn
        pq=tp_iou/denom if denom else 0.0
        pair_scores.append(pq)
        pair_rows.append({'file':filename,'records':[rid_a,rid_b],'counts':[len(a),len(b)],'pq':pq,'tp':tp,'fp':fp,'fn':fn})
        all_tp_iou+=tp_iou; all_tp+=tp; all_fp+=fp; all_fn+=fn

sorted_scores=sorted(pair_scores)
def quantile(q):
    """Linearly interpolated q-quantile of the pairwise PQ values, 4 decimals (None without pairs)."""
    if not sorted_scores: return None
    pos=(len(sorted_scores)-1)*q
    lo=int(pos); hi=min(lo+1,len(sorted_scores)-1); frac=pos-lo
    return round(sorted_scores[lo]*(1-frac)+sorted_scores[hi]*frac,4)
summary={
 'groups_with_multiple_annotators':sum(len(v)>1 for v in records_by_file.values()),
 'annotator_pairs':len(pair_scores),
 'pairwise_pq_mean':round(statistics.mean(pair_scores),4) if pair_scores else None,
 'pairwise_pq_median':round(statistics.median(pair_scores),4) if pair_scores else None,
 'pairwise_pq_quantiles':{'p10':quantile(.1),'p25':quantile(.25),'p75':quantile(.75),'p90':quantile(.9)},
 'pairwise_pq_at_least_0_5':sum(x>=.5 for x in pair_scores),
 'pairwise_pq_below_0_2':sum(x<.2 for x in pair_scores),
 'aggregate_pq_over_all_annotator_pairs':round(all_tp_iou/(all_tp+.5*all_fp+.5*all_fn),4) if all_tp+.5*all_fp+.5*all_fn else 0.0,
 'aggregate_tp_fp_fn':{'tp':all_tp,'fp':all_fp,'fn':all_fn},
 'largest_instance_count_disagreements':sorted(pair_rows,key=lambda x:abs(x['counts'][0]-x['counts'][1]),reverse=True)[:10],
 'lowest_pq_pairs':sorted(pair_rows,key=lambda x:x['pq'])[:10],
 'metric':'Organizer self-evaluation PQ: pair counted as TP when IoU > 0.5; FP/FN from unmatched columns/rows.'
}
OUT.write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
