"""Solar filament instance segmentation of 2048 x 2048 GONG H-alpha full-disk images (MAGFiLO data).

Entry points: train, predict and evaluate (python -m solar_filaments.<name>). Model: config, model,
backbone (vendor/: FlashInternImage), detail, curves, common, decoder, query_seeds, masks, ops (DCNv4).
Data and targets: data, geometry, piloting. Objective: losses, panoptic_surrogate. Instance selection
and metric: inference, fast_eval, metrics. Training runtime: runtime.
"""
