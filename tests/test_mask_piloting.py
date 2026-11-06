"""Training-only mask pilots: no GT path into ordinary inference queries.

Mask pilots (MP-Former) are extra decoder queries that receive noised GT masks as attention guides during
training only (solar_filaments.piloting). These tests run the diagnostic model on CPU.
"""
import unittest
import numpy as np
import torch
from solar_filaments.config import model_config, read_config
from solar_filaments.geometry import derive_geometry
from solar_filaments.losses import FilamentCriterion
from solar_filaments.model import build_model
from solar_filaments.piloting import prepare_mask_pilots, noisy_pilot_support

torch.set_num_threads(2)


def target(empty=False):
    """Geometry targets of two thin horizontal filaments on a 64 x 64 image (no instance if empty)."""
    masks = np.zeros((0 if empty else 2, 64, 64), bool)
    if not empty:
        masks[0, 12:15, 5:54] = True
        masks[1, 45:48, 10:58] = True
    return {k: torch.from_numpy(v) for k, v in derive_geometry(masks).items()}


def config(pilots):
    """Diagnostic model configuration with `pilots` pilot slots (0 = no pilot branch)."""
    result = model_config(read_config('configs/diagnostic.yaml'))
    result.denoising_queries = pilots
    result.query_detail_points = 16
    return result


class MaskPilotingTests(unittest.TestCase):
    """Protects the pilot guides and their noise, and that the pilot branch never changes inference outputs
    or the ordinary queries, while its losses still back-propagate finite gradients (also with empty targets)."""

    def test_thin_support_padding_and_capacity(self):
        """Thin instances keep a non-empty guide, unused slots stay empty, 0 slots disable the pilots and more
        GT instances than slots raise."""
        t = target()
        guides = prepare_mask_pilots([t], 4)
        self.assertEqual(guides.dtype, torch.bool)
        self.assertFalse(guides.requires_grad)
        self.assertTrue(guides[0, :2].flatten(1).any(-1).all())
        self.assertFalse(guides[0, 2:].any())
        self.assertIsNone(prepare_mask_pilots([t], 0))
        with self.assertRaises(ValueError):
            prepare_mask_pilots([t], 1)

    def test_noise_is_area_proportional_and_ignores_padding(self):
        torch.manual_seed(17)
        support = torch.zeros(8192, 256, dtype=torch.bool)
        support[:4096, :16] = True
        noisy = noisy_pilot_support(support, .2)
        self.assertFalse(noisy[4096:].any())
        actual = float((noisy[:4096] ^ support[:4096]).float().sum(-1).mean())
        self.assertAlmostEqual(actual, 1.6, delta=.12)
        self.assertTrue(torch.equal(noisy_pilot_support(support, 0), support))

    def test_inference_unchanged_by_pilot_capability(self):
        plain, _ = build_model(config(0), load_backbone=False)
        piloted, _ = build_model(config(4), load_backbone=False)
        missing, unexpected = piloted.load_state_dict(plain.state_dict(), strict=False)
        self.assertEqual(missing, ['decoder.pilot_labels.weight'])
        self.assertEqual(unexpected, [])
        plain.eval(); piloted.eval()
        image = torch.rand(1, 1, 64, 64)
        with torch.no_grad():
            a, b = plain(image), piloted(image)
        for key in ('pred_logits', 'pred_masks', 'pred_quality', 'mask_features2'):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        self.assertNotIn('denoising', b)
        with self.assertRaises(ValueError):
            piloted(image, denoising_masks=prepare_mask_pilots([target()], 4))

    def test_normal_queries_independent_of_pilots_and_shared_gradients_finite(self):
        model, _ = build_model(config(4), load_backbone=False)
        image = torch.rand(1, 1, 64, 64)
        t = target()
        torch.manual_seed(91)
        ordinary = model(image)
        torch.manual_seed(91)
        piloted = model(image, denoising_masks=prepare_mask_pilots([t], 4))
        torch.testing.assert_close(ordinary['pred_masks'], piloted['pred_masks'], rtol=0, atol=0)
        torch.testing.assert_close(ordinary['pred_logits'], piloted['pred_logits'], rtol=0, atol=0)
        self.assertEqual(piloted['denoising']['pred_masks'].shape[:2], (1, 4))
        criterion = FilamentCriterion(256, pq_point_count=1024)
        loss = sum(criterion(piloted, [t]).values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIn('pilot_reconstruction', criterion.last_stats)
        for p in (model.decoder.pilot_labels.weight, model.decoder.layers[0].cross_attn.in_proj_weight,
                  model.backbone.encoder.patch_embed.conv1.weight):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()), 0)

    def test_empty_targets_have_finite_pilot_backward(self):
        model, _ = build_model(config(4), load_backbone=False)
        t = target(empty=True)
        output = model(torch.rand(1, 1, 64, 64), denoising_masks=prepare_mask_pilots([t], 4))
        loss = sum(FilamentCriterion(256, pq_point_count=512)(output, [t]).values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.decoder.pilot_labels.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
