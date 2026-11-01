"""Losses, warm starts, physics preprocessing, data mixing, fast evaluation and checkpoint handling."""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch import nn
from torch.nn import functional as F

from solar_filaments.config import checkpoint_config, config_dict, model_config, read_config
from solar_filaments.data import rotate_record
from solar_filaments.fast_eval import owned_masks, pq_counts, query_probabilities
from solar_filaments.inference import instance_predictions
from solar_filaments.losses import FilamentCriterion
from solar_filaments.metrics import official_pq
from solar_filaments.model import build_model, limb_flattened
from solar_filaments.train import load_warmstart, save_atomic

torch.set_num_threads(2)


def diagnostic(**changes):
    return replace(model_config(read_config("configs/diagnostic.yaml")), **changes)


def two_filaments(h=96, w=128):
    masks = torch.zeros(2, h, w, dtype=torch.bool)
    masks[0, 20:24, 10:115], masks[1, 60:65, 30:90] = True, True
    auxiliary = F.max_pool2d(masks.any(0).float()[None, None].repeat(1, 3, 1, 1), 2)[0]
    return masks, auxiliary


class V9Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)

    def test_optional_losses_and_gradients_without_feedback_or_auxiliary_heads(self):
        config = diagnostic(mask_feedback=False, auxiliary_heads=False, global_attention_every=0,
                            detail_flattening=True, query_detail_points=8)
        model, _ = build_model(config, load_backbone=False)
        masks, auxiliary = two_filaments()
        output = model(torch.rand(1, 1, 96, 128))
        self.assertNotIn("centerline_logits", output)
        weights = {"competition": .3, "instance_discrimination": 1., "skeleton_recall": 1., "semantic": .5}
        criterion = FilamentCriterion(256, weights=weights, auxiliary_scale=1.)
        losses = criterion(output, [{"masks": masks, "auxiliary": auxiliary}])
        for key in ("instance_discrimination", "skeleton_recall", "competition"):
            self.assertGreater(float(losses[key]), 0, key)
        sum(losses.values()).backward()
        for name in ("decoder.query.weight", "decoder.level_pos.weight", "decoder.layers.0.cross_attn.in_proj_weight",
                     "detail.stem.project.0.weight", "detail.mask2.weight"):
            grad = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(float(grad.abs().sum()), 0, name)

    def test_hurdle_quality_target_zeroes_near_misses(self):
        torch.manual_seed(0)
        model, _ = build_model(diagnostic(detail_flattening=True), load_backbone=False)
        masks, auxiliary = two_filaments()
        output = model(torch.rand(1, 1, 96, 128))
        targets = [{"masks": masks, "auxiliary": auxiliary}]
        plain = FilamentCriterion(256, auxiliary_scale=1., quality_target="best_native_sampled")
        hurdle = FilamentCriterion(256, auxiliary_scale=1., quality_target="best_native_sampled", quality_hurdle=True)
        a, b = plain(output, targets), hurdle(output, targets)
        self.assertTrue(torch.isfinite(b["quality"]))
        # an untrained model has no query above IoU .5, so every hurdle target is 0 while raw IoUs are not
        self.assertNotEqual(float(a["quality"]), float(b["quality"]))
        sum(b.values()).backward()
        grad = dict(model.named_parameters())["decoder.quality.weight"].grad if "decoder.quality.weight" in dict(model.named_parameters()) else None
        if grad is not None:
            self.assertTrue(torch.isfinite(grad).all())

    def test_warmstart_cuts_dead_modules_and_zero_widens_inputs(self):
        source, _ = build_model(diagnostic(), load_backbone=False)
        target, _ = build_model(diagnostic(mask_feedback=False, auxiliary_heads=False, detail_flattening=True),
                                load_backbone=False)
        missing = load_warmstart(target, {"model": source.state_dict()})
        self.assertEqual(missing, [])
        stem = target.state_dict()["detail.stem.project.0.weight"]
        self.assertTrue(torch.equal(stem[:, :8], source.state_dict()["detail.stem.project.0.weight"]))
        self.assertEqual(float(stem[:, 8:].abs().sum()), 0.)
        # With zero weights on the new channel the detail path computes the same function.
        source.eval(), target.eval()
        image = torch.rand(1, 1, 64, 64)
        contrast = (image - F.avg_pool2d(image, 15, 1, 7)) * 4
        context = torch.randn(1, 48, 8, 8)
        old = source.detail(torch.cat((image, contrast), 1), context, (0, 0, 64, 64), (64, 64))
        new = target.detail(torch.cat((image, contrast, limb_flattened(image)), 1), context, (0, 0, 64, 64), (64, 64))
        for a, b in zip(old, new):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_mixed_dataset_epoch_sampling(self):
        from solar_filaments.data import MixedDataset

        class Toy(torch.utils.data.Dataset):
            def __init__(self, tag, n):
                self.tag, self.n, self.records = tag, n, list(range(n))
            def __len__(self):
                return self.n
            def set_epoch(self, epoch):
                pass
            def __getitem__(self, i):
                return (self.tag, i)

        mixed = MixedDataset(Toy("L", 5), [Toy("A", 4), Toy("B", 6)], 7, seed=1)
        self.assertEqual(len(mixed), 12)
        first = [mixed[i] for i in range(12)]
        self.assertEqual(first[:5], [("L", i) for i in range(5)])
        extra = first[5:]
        self.assertEqual(len(set(extra)), 7)  # sampled without replacement within an epoch
        self.assertTrue(all(t in ("A", "B") for t, _ in extra))
        mixed.set_epoch(1)
        self.assertNotEqual([mixed[i] for i in range(5, 12)], extra)  # new subset next epoch
        mixed.set_epoch(0)
        self.assertEqual([mixed[i] for i in range(5, 12)], extra)    # deterministic per epoch

    def test_limb_flattening_removes_center_to_limb_variation(self):
        # Disk radius / smoothing scale comparable to GONG (R ~ 950 px, sigma 32 px).
        size, radius = 1024, 450
        yy, xx = torch.meshgrid(torch.arange(size, dtype=torch.float32), torch.arange(size, dtype=torch.float32),
                                indexing="ij")
        r = ((yy - (size - 1) / 2) ** 2 + (xx - (size - 1) / 2) ** 2).sqrt() / radius
        disk = (r < 1).float()
        mu = (1 - r.clamp(max=1) ** 2).sqrt()
        image = .8 * (1 - .6 * (1 - mu)) * disk + .02 * (1 - disk)  # limb darkening, faint sky
        image[400:406, 300:700] *= .7  # a dark filament, 30% absorption
        flat = limb_flattened(image[None, None])[0, 0]
        inner = (r < .9) & ~((yy >= 380) & (yy < 426))
        self.assertLess(float(flat[inner].abs().median()), .05)
        self.assertLess(float(flat[400:406, 350:650].mean()), -.8)
        self.assertEqual(float(flat[r > 1.01].abs().max()), 0.)

    def test_rotate_record_quarter_turn_matches_rot90(self):
        h = w = 64
        image = np.random.default_rng(0).integers(0, 255, (h, w), dtype=np.uint8)
        polygon = [[10.0, 12.0, 40.0, 12.0, 40.0, 20.0, 10.0, 20.0]]
        rotated, masks = rotate_record(image, [{"segmentation": polygon}], 90., 1., False)
        reference = mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(polygon, h, w))).astype(bool)
        self.assertTrue(np.array_equal(rotated, np.rot90(image)))
        self.assertTrue(np.array_equal(masks[0], np.rot90(reference)))
        flipped, flipped_masks = rotate_record(image, [{"segmentation": polygon}], 0., 1., True)
        self.assertTrue(np.array_equal(flipped, image[:, ::-1]))
        self.assertTrue(np.array_equal(flipped_masks[0], reference[:, ::-1]))

    def test_fast_eval_equals_reference_inference_and_official_pq(self):
        model, _ = build_model(diagnostic(), load_backbone=False)
        model.eval()
        with torch.no_grad():
            output = model(torch.rand(1, 1, 96, 128))
        masks, _ = two_filaments()
        truth = [mask_utils.encode(np.asfortranarray(m.numpy().astype(np.uint8))) for m in masks]
        scores, probabilities = query_probabilities(output, 0, 0., dtype=torch.float32)
        for score_threshold, mask_threshold in ((0., .5), (.3, .6), (.5, .4)):
            reference, _ = instance_predictions(output, score_threshold=score_threshold, mask_threshold=mask_threshold,
                                                min_area=4)
            fast, _ = owned_masks(scores, probabilities, score_threshold, mask_threshold, min_area=4)
            self.assertEqual(len(reference), len(fast))
            expected = official_pq([r["segmentation"] for r in reference], truth)
            counts = pq_counts(fast, masks)
            for key in ("tp", "fp", "fn"):
                self.assertEqual(expected[key], counts[key])
            self.assertAlmostEqual(expected["iou_sum"], counts["iou_sum"], places=4)


    def test_checkpoint_config_accepts_only_neutral_retired_options(self):
        saved = config_dict(diagnostic())
        neutral = {**saved, "memory_mode": "pooled", "decoder_mode": "global", "dynamic_mask": False,
                   "annotators": [], "hypotheses": 0, "curve_attention_points": 7}
        self.assertEqual(checkpoint_config(neutral, context_size=96).context_size, 96)
        for key, value in (("memory_mode", "multiscale"), ("dynamic_mask", True), ("annotators", ["010101"])):
            with self.assertRaises(ValueError):
                checkpoint_config({**saved, key: value})

    def test_failed_atomic_save_retains_previous_checkpoint_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            save_atomic({"value": torch.tensor(3)}, path)

            def fail(value, temporary):
                Path(temporary).write_bytes(b"incomplete")
                raise OSError("simulated disk full")
            with patch("torch.save", side_effect=fail), self.assertRaises(OSError):
                save_atomic({"value": torch.tensor(9)}, path)
            self.assertEqual(int(torch.load(path, weights_only=True)["value"]), 3)
            self.assertFalse(path.with_suffix(".tmp").exists())


if __name__ == "__main__":
    unittest.main()
