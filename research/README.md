# research/

These analyses back the statements in `docs/TECHNICAL_REPORT.md` and `paper/`. Run each script from the
repository root. Its JSON output is stored next to it. "Fold-0 dump" means the output of
`python -m scripts.dump_tta_candidates --checkpoint <fold-0 model> --split val --output <dump>`.

| Script | Output | What it measures | Key result |
|---|---|---|---|
| `interannotator_agreement.py` | `interannotator_agreement.json` | Official PQ of one annotator record scored against another, over 598 annotator pairs on the 296 observations with several records | PQ 0.340 (SQ 0.634, RQ 0.536) |
| `label_visibility.py --folder <fold-0 dump>` | `label_visibility.json`, `../docs/figures/visibility.{png,pdf}` | Annotator agreement and the model's matches and misses against the cloud-model contrast C = I/B − 1, area and disk position. It also scores the model and the annotators the same way on the same observations | Agreement falls from 0.79 to 0.25 as contrast decreases, and model misses rise from 0 % to 65 %. In units of the quiet-Sun texture σ (median 0.048), agreement is 0.26 below 1σ and 0.70 beyond 3σ, and 58 % of all filaments lie within 2σ. On the 47 multi-record fold-0 observations, the model scores PQ 0.430 against the annotators, and the annotators score 0.327 against each other: paired bootstrap difference +0.103, 95 % CI 0.076–0.129 |
| `magnetogram_pil.py --folder <fold-0 dump>` | `magnetogram_pil.json` | Distance of annotated filaments, fold-0 predictions and missed filaments to magnetic polarity inversion lines. The line-of-sight magnetograms are listed in `magnetogram_pairs.csv`. They are used for this analysis only, never for training or inference | 23.3 % of filaments lie within 20 Mm of a PIL, against 4.9 % of random disk points. Matched predictions (24 %), unsupported predictions (22 %) and misses (23 %) are not distinguishable. From disk centre to limb the share falls from 30 % to 12 %, but random points fall alike, so the enrichment stays at about 4: the line-of-sight magnetograms lose PILs toward the limb |
| `filament_geometry.py` | `filament_geometry.json` | Area, width, length and elongation of the 8,199 annotated filaments, measured from the polygons only | Median area 1,228 px. Median width along the skeleton 8.2 px (P10 5.7, P90 15.8). Median length (oriented box) 105 px, P99 548 px |
| `pq_annotation_upper_bound.py` | `pq_annotation_upper_bound.json` | A conservative upper bound on the PQ that a single prediction set can reach against every annotator record of fold 0. It relaxes the problem to fractional set packing, so the bound is not an achievable score | 0.83 |

`magnetogram_pairs.csv` pairs each of the 707 training observations with one magnetogram:
- 612 GONG QuickReduce `bqa` 10-minute averages from the same site. 610 of them are within 15 min of the H-alpha frame and 2 are within 16–18 min.
- 95 SDO/HMI `hmi.M_720s` maps. 92 of them are within 12 min and 3 are within 19–60 min.

The HMI rows carry the WCS keywords needed to map the disk. The files are expected in `data/gong_bqa/fits` and
`data/hmi_720s/fits`; other locations can be passed with `--gong-dir` and `--hmi-dir`.
