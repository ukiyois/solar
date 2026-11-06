# Solar filament instance segmentation: Filament Segmentation Challenge 2026

This repository segments every filament in a GONG H-alpha full-disk image (2048 × 2048) as a separate instance. The submitted model, **final-A**, is a single network:
- a FlashInternImage-L context encoder on a 1024 px view of the disk;
- a native-resolution 2048 px detail path with a limb-darkening-normalized contrast channel;
- 64 image-seeded instance queries that render native-resolution masks;
- training on a differentiable Panoptic Quality surrogate, with Noisy-Student self-training on unlabelled GONG archive frames.

At test time, 8 rotations/flips of the image are fused.

| | |
|---|---|
| Public leaderboard | **PQ 0.40** (single model + 8-view TTA) |
| Fold 0 (held-out), fold-0 twin of the final recipe | PQ 0.446 · mean pair Dice 0.728 · torchmetrics DiceScore 0.656 per record |
| Human vs human (training set) | PQ 0.340 |

**End to end:** [`notebooks/end_to_end.ipynb`](notebooks/end_to_end.ipynb) runs the whole pipeline. It covers installation, data and weight checks, the unit tests, test inference with 8-view TTA, fusion, the validated submission file and morphology figures. It also shows the judging metrics and analyses (recomputed with a fold-0 model on request) and contains the complete training chain behind a flag.

**Full description:** [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md). It covers preprocessing, architecture, objective, training stages, self-training and compliance, inference, the judging metrics (Dice/IoU distributions, one-to-many and many-to-one relations), morphology, the label-visibility analysis, negative results and reproduction. The 4-page report, written in the organizers' Overleaf template, is in [`paper/`](paper/).

## Repository layout

```text
solar_filaments/   the package: model (backbone, detail path, curve blocks, query decoder, mask rendering),
                   losses (PQ surrogate), data, training loop, inference, official PQ metric
configs/           YAML stages with `base:` inheritance: flash_large -> ... -> v9_ffinal_a (final model);
                   v9_e6 (pseudo-label teacher), local (inference settings), diagnostic (tiny test model)
scripts/           setup, archive download + pseudo-labelling, TTA dump + fusion + submission checks,
                   judging metrics, morphology and error analyses, packaging
notebooks/         end_to_end.ipynb: the whole pipeline in one notebook
tests/             33 unit tests (CPU; 2 need the DCNv4 CUDA extension)
docs/              technical report, figures, analysis outputs
paper/             4-page report in the organizers' Overleaf template (ACM acmart sigconf) and its figures
research/          inter-annotator agreement, label visibility, magnetogram / polarity-inversion-line check,
                   dataset geometry, PQ upper bound (see research/README.md)
third_party/       pinned official DCNv4 CUDA operator source (built by scripts/build_dcnv4.py)
weights/           final_a.pt (submitted model, released separately), ImageNet backbone (downloaded)
```

## Setup

Use Linux with an NVIDIA GPU and a CUDA 12.8 toolkit (nvcc), which the official DCNv4 kernel needs. The project was verified with Python 3.10 and PyTorch 2.10.0 + cu128.

```bash
python -m pip install -r requirements.txt      # every package, pinned (PyTorch from the cu128 wheel index)
python -m pip install -e . --no-build-isolation
python scripts/build_dcnv4.py          # official DCNv4 CUDA operator (pinned commit 4b848f7)
python scripts/download_weights.py     # ImageNet FlashInternImage-L backbone (training only)
```

Data layout (competition files, not distributed):

```text
data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images/*.jpeg
data/raw/MAGFiLO_1.0_Kaggle_2026/test/test_images/*.jpeg
```

## Reproduce the submission from the final weights

`weights/final_a.pt` (894.7 MB, SHA-256 `6784bcc33d38f6495aea322863d83ccd062f7297686c0b9d5edcbc89f48933cb`) is attached to the [v1.0.0 release](https://github.com/ukiyois/solar/releases/tag/v1.0.0), together with the scored submission file and the 4-page report.

```bash
curl -L -o weights/final_a.pt https://github.com/ukiyois/solar/releases/download/v1.0.0/final_a.pt
sha256sum weights/final_a.pt
# 1. 8 dihedral views per test image -> per-query probability crops (GPU, ~3.7 s / image on an RTX 3090)
python -m scripts.dump_tta_candidates --checkpoint weights/final_a.pt --split test --output runs/tta/final_a/test
# 2. fuse views, select (votes >= 1/8, score >= .10, mask .40), pixel ownership -> CSV (CPU)
python -m scripts.fuse_tta --folders runs/tta/final_a/test --votes 0.125 --scores 0.10 --masks 0.40 \
    --test --workers 4 --output submission.csv
# 3. mechanical checks (header, ids, RLE decoding, no overlaps)
python scripts/validate_submission.py submission.csv
```

The submitted file is `submission_final_a.csv` in the release (SHA-256 `d5896cfb…c20bf1e`, 1,072 instances on 179 of 180 images, public PQ 0.40). For a quick single-view prediction (PNG label maps + CSV), use `python -m solar_filaments.predict --checkpoint weights/final_a.pt --input <images>`. Its thresholds come from `--config` (default `configs/local.yaml`). It is a quick look, not the submission setting: one view instead of eight, and without the TTA fusion.

## Reproduce training

Each stage warm-starts from the previous one (`--init-from`). The pseudo-label pools are directories under `data/`, set by `extra_data_roots` in `configs/v9_e9.yaml`, `v9_f1.yaml` and `v9_ffinal_a.yaml`. The submitted chain used these pools:
- `data/gong_pseudo`: round 1, used by E9;
- `data/gong_pseudo_nt` and `data/gong_pseudo_b2`: two round-1 pools without test-day frames, used by F1;
- `data/gong_pseudo_r2`: round 2, used by final-A.

To reproduce with a single round-1 pool, point both E9 and F1 at it. A two-line YAML with `base: configs/v9_e9.yaml` and `train: {extra_data_roots: [...]}` does this; section 9 of the notebook writes these files for you.

```bash
python scripts/prepare_data.py --config configs/v9_ffinal_a.yaml      # geometry cache, CPU, once
T="python -m torch.distributed.run --standalone --nproc_per_node=2 -m solar_filaments.train"
$T --config configs/dual_3090.yaml                                       # S1 from the ImageNet backbone
$T --config configs/pq_aligned_v1.yaml  --init-from runs/dual_3090/best.pt
$T --config configs/pq_seeded_v3.yaml   --init-from runs/pq_aligned_v1/best.pt
$T --config configs/pq_recovery_v7.yaml --init-from runs/pq_seeded_v3/best.pt   # = v7
# fold-0 teachers (fold 0 held out)
$T --config configs/v9_e3.yaml --init-from runs/pq_recovery_v7/best.pt
$T --config configs/v9_e6.yaml --init-from runs/pq_recovery_v7/best.pt
# unlabelled GONG archive frames; frames from test-image days are never labelled
python scripts/download_gong.py --output data/gong_extra
mkdir -p data/gong_pseudo data/gong_pseudo_r2          # exclude_fold.txt: skip fold-0 validation days as well
echo 0 > data/gong_pseudo/exclude_fold.txt; echo 0 > data/gong_pseudo_r2/exclude_fold.txt
L="python -m scripts.pseudo_label --images data/gong_extra/images --views d0 f2 --votes 1 --score .10 --mask .70"
$L --checkpoints runs/v9_e3/best.pt runs/v9_e6/best.pt --output data/gong_pseudo           # round 1
python -m scripts.pseudo_label assemble data/gong_pseudo data/gong_extra/images
$T --config configs/v9_e9.yaml --init-from runs/pq_recovery_v7/best.pt   # fold-0 twin, round-2 teacher
# final track: all labels, no held-out data, EMA weights saved per epoch as final.pt
$T --config configs/v9_f1.yaml --init-from runs/pq_recovery_v7/best.pt
$L --checkpoints runs/v9_e9/best.pt runs/v9_e6/best.pt --output data/gong_pseudo_r2        # round 2
python -m scripts.pseudo_label assemble data/gong_pseudo_r2 data/gong_extra/images
$T --config configs/v9_ffinal_a.yaml --init-from runs/v9_f1/final.pt      # = final-A
```

A single GPU works as well (`python -m solar_filaments.train ...`). The batch is per GPU and gradient accumulation is disabled.

## Judging metrics and figures

```bash
python -m scripts.dump_tta_candidates --checkpoint <fold-0 model> --split val --output runs/tta/<name>/val
V="--folders runs/tta/<name>/val --folder-views d0,d1,d2,d3,f0,f1,f2,f3 --votes 1 --score .10 --mask .40"
python -m scripts.rubric_metrics $V --output-dir docs/figures/rubric_fold0 \
    --compare docs/figures/rubric_fold0_v7/per_record.json                   # PQ, Dice/IoU, 1:n / n:1, bootstrap CIs
python -m scripts.morphology_figures $V --per-record docs/figures/rubric_fold0/per_record.json --output-dir docs/figures/morphology
python -m scripts.morphology_figures --csv submission.csv --output-dir docs/figures/morphology_test
python -m scripts.shape_statistics $V --csv submission.csv --output-dir docs/figures/shape
python -m scripts.morphology_details $V --output docs/figures/morphology_details.png
python -m scripts.error_modes $V --output docs/analysis/error_modes_fold0.json
python research/interannotator_agreement.py
python research/label_visibility.py --folder runs/tta/<name>/val       # agreement / misses vs contrast, model vs human
python research/magnetogram_pil.py --folder runs/tta/<name>/val        # distance to polarity inversion lines
```

`rubric_metrics` reproduces the organizers' self-evaluation notebook and adds the `torchmetrics` DiceScore of the binary filament mask. It reports 95 % bootstrap intervals over observations; resampling whole observations keeps all annotator records of an image together. With `--compare`, it also reports paired differences against another run's `per_record.json`.

`magnetogram_pil` reads the line-of-sight magnetograms listed in `research/magnetogram_pairs.csv`: GONG QuickReduce `bqa` FITS in `data/gong_bqa/fits` and SDO/HMI `hmi.M_720s` FITS in `data/hmi_720s/fits`. The magnetograms are used for this analysis only, never for training or inference.

The report in `paper/` is written in the organizers' Overleaf template: ACM `acmart` sigconf, with `preamble.tex` unchanged and citations in `main.bib`. It builds with `latexmk -pdf main.tex`. Its figures come from three sources:
- the architecture diagram is TikZ (`paper/fig_architecture.tex`) with thumbnails from `python paper/make_thumbnails.py --csv <submission.csv>`;
- the visibility curves come from `research/label_visibility.py` (`docs/figures/visibility.pdf`, copied to `paper/figures/`);
- the close-ups come from `scripts.morphology_details ... --paper-output paper/figures/morphology_details.png`.

## Tests

```bash
python -m unittest discover -s tests -v
```

## Compliance

- Inference uses only the provided test image.
- Training uses the competition segmentation polygons plus our pseudo-labels of unlabelled GONG archive frames. No annotator codes, bounding boxes, spines, chirality or other ground-truth metadata are used.
- No test image, and no archive frame from a test-image day, is in the training data of the submitted model.

Details are in report §11.

## License

MIT for this project's code (`LICENSE`). Third-party components and the pretrained weights are described in `THIRD_PARTY_NOTICES.md`. Competition data remain under their own terms.
