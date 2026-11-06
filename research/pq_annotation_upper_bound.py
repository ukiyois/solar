"""A conservative geometry/count upper bound, not an achievable model score.

For disjoint predictions and disjoint GT within each annotator, IoU > .5
matches are one-to-one within each record. The pooled PQ denominator is
(sum annotator_count(image) * prediction_count(image) + total_GT) / 2.

A prediction that matches GT masks A_i must have x_i = IoU(P, A_i) > .5.
Jaccard distance is a metric, hence x_i+x_j <= 1+IoU(A_i,A_j). A group of
two targets therefore contributes <= 1+IoU(A_i,A_j). For three targets let
y_i=x_i-.5 in [0,.5]. Maximizing sum(y_i) subject to y_i+y_j<=IoU(A_i,A_j)
gives the minimum of 1.5, .5+min(edge), .5*sum(edge), and min(two-edge sums).

Enumerate groups with at most one GT per annotator. Every physically feasible
prediction maps to one such group. Relax spatial feasibility and binary group
selection to fractional set packing. The resulting optimum is an upper bound.
This script reads only the competition TRAIN annotations in the selected fold.

The bound is found by bisection on the target PQ (certificate()); --target is also tested on its own.

usage (repository root): python research/pq_annotation_upper_bound.py [--fold 0] [--target .7]
"""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as mu
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from solar_filaments.data import FilamentDataset


def construct(ds):
    """One fractional set-packing problem per physical image of the dataset split.

    The GT instances of all annotator records of an image (at most three) are rasterized; an overlap inside
    one record raises, since the one-to-one argument needs disjoint GT. Groups take at most one GT per record:
    single GT instances (weight 1), cross-record pairs with IoU > 0 (weight 1 + IoU) and, with three records,
    triples whose three pairwise IoUs are > 0 (weight 1.5 + y, y as in the module docstring). A weight bounds
    the IoU sum one prediction can earn on its group. Returns dicts with file name, annotator count, GT
    count, group weights and the sparse GT x group incidence matrix.
    """
    by_file = defaultdict(list)
    for r in ds.records:
        by_file[r['file_name']].append(r)
    problems = []
    for name, records in sorted(by_file.items()):
        if len(records) > 3:
            raise ValueError('This bound implementation covers at most three annotators')
        truth, ranges = [], []
        for r in records:
            start = len(truth)
            for ann in ds.annotations[str(r['id'])]:
                encoded = mu.frPyObjects(ann['segmentation'], r['height'], r['width'])
                truth.append(mu.merge(encoded) if isinstance(encoded, list) else encoded)
            ranges.append(list(range(start, len(truth))))
        n, m = len(truth), len(records)
        overlaps = np.asarray(mu.iou(truth, truth, [0] * n))
        for indices in ranges:
            block = overlaps[np.ix_(indices, indices)]
            if np.triu(block, 1).max(initial=0) > 0:
                raise ValueError('Within-annotator overlap invalidates the one-to-one proof: ' + name)
        groups = [(tuple([i]), 1.) for i in range(n)]
        for a, b in itertools.combinations(ranges, 2):
            for i, j in itertools.product(a, b):
                if overlaps[i, j] > 0:
                    groups.append(((i, j), 1. + float(overlaps[i, j])))
        if m == 3:
            for i, j, k in itertools.product(*ranges):
                edges = [float(overlaps[i, j]), float(overlaps[i, k]), float(overlaps[j, k])]
                if min(edges) <= 0:
                    continue
                y = min(1.5, .5 + min(edges), .5 * sum(edges), sum(edges) - max(edges))
                groups.append(((i, j, k), 1.5 + y))
        row, col = [], []
        for c, (indices, _) in enumerate(groups):
            for i in indices:
                row.append(i); col.append(c)
        incidence = coo_matrix((np.ones(len(row)), (row, col)), shape=(n, len(groups))).tocsr()
        problems.append({'file_name': name, 'annotators': m, 'GT': n,
                         'weights': np.array([g[1] for g in groups]), 'incidence': incidence})
    return problems


def certificate(problems, score):
    """Test the target PQ `score`; a negative first return value proves it unreachable.

    PQ >= score requires 2 sum(IoU) - score (sum m P + total GT) >= 0 (pooled denominator of the module
    docstring; m = annotator count, P = prediction count of an image). A prediction on a group of weight w adds
    at most 2 w - score m to the left side, so each image contributes at most the optimum of its relaxed
    packing (each GT in at most one group, 0 <= x <= 1, solved with HiGHS). Returns
    (sum of the image optima minus score x total GT, per-image terms).
    """
    terms = []
    total_gt = sum(p['GT'] for p in problems)
    for p in problems:
        profit = 2 * p['weights'] - score * p['annotators']
        solved = linprog(-profit, A_ub=p['incidence'], b_ub=np.ones(p['GT']),
                         bounds=(0, 1), method='highs')
        if not solved.success:
            raise RuntimeError(solved.message)
        terms.append({'file_name': p['file_name'], 'annotators': p['annotators'],
                      'GT': p['GT'], 'clusters': len(p['weights']),
                      'maximum_relaxed_profit': float(-solved.fun)})
    return sum(t['maximum_relaxed_profit'] for t in terms) - score * total_gt, terms


def main():
    """Build the problems of the fold, test --target and bisect the largest PQ the bound does not rule out."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/raw/MAGFiLO_1.0_Kaggle_2026')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--target', type=float, default=.7)
    parser.add_argument('--output', default='research/pq_annotation_upper_bound.json')
    args = parser.parse_args()
    ds = FilamentDataset(args.data_root, 'val', fold=args.fold, geometry=False, decode_masks=False)
    problems = construct(ds)
    gap, terms = certificate(problems, args.target)
    # bisection: largest target whose certificate stays positive (20 halvings of [0, 1])
    lo, hi = 0., 1.
    for _ in range(20):
        middle = (lo + hi) / 2
        difference, _ = certificate(problems, middle)
        if difference > 0:
            lo = middle
        else:
            hi = middle
    # Round upward and retain a conservative numerical tolerance.
    upper = min(1., hi + 1e-6)
    result = {'fold': args.fold, 'physical_images': len(problems), 'annotator_records': len(ds),
              'GT_instances': sum(p['GT'] for p in problems),
              'within_annotator_GT_overlap_pairs': 0,
              'fractional_Jaccard_upper_bound': upper, 'target': args.target,
              'target_certificate_gap': gap,
              'target_ruled_out_by_this_bound': bool(gap < -1e-5),
              'meaning': 'Upper bound on any disjoint single-image prediction shared across all annotator versions. Positive target gap does NOT prove target achievable.',
              'relaxations': ['Prediction shapes need not be spatially realizable.',
                              'Selected prediction groups may overlap spatially.',
                              'Group selection is fractional, not binary.',
                              'IoU>.5 lower bounds relaxed to >=.5.'],
              'per_image_target_certificate': terms}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_image_target_certificate'}, indent=2))


if __name__ == '__main__':
    main()
