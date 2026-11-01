"""Regressions for sparse foreground confidence and native PQ gradients."""
import unittest
import numpy as np
import torch
from torch.nn import functional as F
from solar_filaments.geometry import derive_geometry, target_at_points
from solar_filaments.losses import FilamentCriterion, native_area_points
from solar_filaments.masks import foreground_probability_mean, fine_mask_logits, fine_mask_points
from solar_filaments.config import model_config, read_config
from solar_filaments.model import build_model
from solar_filaments.inference import instance_predictions

torch.set_num_threads(2)


def fixture(shift=0, foreground=.5):
    masks = np.zeros((2, 128, 128), bool)
    masks[0, 24:30, 12:112] = True
    masks[1, 84:90, 20:104] = True
    target = {k: torch.from_numpy(v) for k, v in derive_geometry(masks).items()}
    logits = torch.from_numpy(np.roll(masks, shift, axis=1)).float() * (foreground + 7) - 7
    basis = F.interpolate(logits[None], size=(64, 64), mode='nearest').requires_grad_()
    output = {'pred_logits': torch.tensor([[[3., -3.], [3., -3.]]], requires_grad=True),
              'pred_quality': torch.full((1, 2), 2., requires_grad=True),
              'pred_masks': torch.full((1, 2, 32, 32), -5., requires_grad=True),
              'mask_embeddings': torch.eye(2)[None].requires_grad_(),
              'mask_features2': basis, 'image_size': (128, 128), 'aux_outputs': [],
              'seed_logits': torch.zeros(1, 1, 32, 32, requires_grad=True)}
    for name in ('semantic', 'centerline', 'boundary'):
        output[name + '_logits'] = torch.zeros(1, 1, 64, 64, requires_grad=True)
    return output, target


class NativePQTests(unittest.TestCase):
    def test_cached_inference_preserves_masks_filters_and_scores(self):
        basis = torch.full((1, 3, 16, 16), -5.)
        basis[0, 0, 2:10, 2:10] = 3
        basis[0, 1, 7:13, 6:14] = 2
        basis[0, 2, 1:15, 7:8] = 4
        output = {'mask_features2': basis, 'mask_embeddings': torch.eye(3)[None],
                  'pred_logits': torch.tensor([[[4., -4.], [2., -2.], [3., -3.]]]),
                  'pred_quality': torch.tensor([[2., 1., 2.]]), 'image_size': (32, 32)}
        cache = {q: p for q, p in enumerate(fine_mask_logits(output, 0, slice(None)).sigmoid())}
        for score in (.1, .35, .6):
            for mask in (.4, .5, .6):
                for retention in (.3, .8):
                    settings = dict(score_threshold=score, mask_threshold=mask, min_area=2,
                                    min_retained_fraction=retention)
                    a, a_labels = instance_predictions(output, **settings)
                    b, b_labels = instance_predictions(output, mask_cache=cache, **settings)
                    np.testing.assert_array_equal(a_labels, b_labels)
                    self.assertEqual([p['segmentation'] for p in a], [p['segmentation'] for p in b])
                    self.assertEqual([p['area'] for p in a], [p['area'] for p in b])
                    np.testing.assert_allclose([p['score'] for p in a], [p['score'] for p in b], atol=1e-6)

    def test_native_point_definition_matches_render_with_subpixel_phases(self):
        torch.manual_seed(3)
        output = {'mask_features2': torch.randn(1, 4, 32, 40),
                  'mask_embeddings': torch.randn(1, 3, 4),
                  'phase_features2': torch.randn(1, 12, 32, 40),
                  'phase_embeddings': torch.randn(1, 3, 3), 'image_size': (64, 80)}
        pixels = torch.cat((torch.tensor([0, 79, 80 * 63, 80 * 64 - 1]),
                            torch.randint(64 * 80, (1000,))))
        points = torch.stack(((pixels % 80 + .5) / 80, (pixels // 80 + .5) / 64), -1)
        expected = fine_mask_logits(output, 0, slice(None)).flatten(1)[:, pixels]
        actual = fine_mask_points(output, 0, slice(None), points)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)

    def test_sparse_confidence_has_no_background_area_floor(self):
        means = []
        for side in (64, 512):
            logits = torch.full((1, side, side), -8.)
            logits[0, 20:24, 16:40] = 4.
            fixed = foreground_probability_mean(logits)
            means.append(fixed)
            torch.testing.assert_close(fixed, torch.sigmoid(torch.tensor([4.])))
            if side == 512:
                p = logits.sigmoid()
                gate = ((p - .5) / .1).sigmoid()
                old = (p * gate).sum() / gate.sum()
                self.assertLess(float(old), .1)
        torch.testing.assert_close(means[0], means[1])

    def test_importance_samples_estimate_native_area(self):
        torch.manual_seed(2026)
        out, target = fixture()
        points, weights = native_area_points(target, out['pred_masks'][0], 65536)
        truth = target_at_points(target, points)
        estimate = (truth * weights).mean(-1)
        exact = target['occupancy4'].float().sum((-2, -1)) / (128 * 128)
        torch.testing.assert_close(estimate, exact, atol=.002, rtol=.02)

    def test_pq_uses_refined_masks_and_has_native_gradients(self):
        criterion = FilamentCriterion(512, weights={'pq': 1.}, pq_point_count=8192)
        torch.manual_seed(7)
        good, target = fixture(foreground=3.)
        good_loss = criterion(good, [target])['pq']
        good_loss.backward()
        grad = good['mask_features2'].grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0)
        torch.manual_seed(7)
        bad, target = fixture(shift=20, foreground=3.)
        bad_loss = criterion(bad, [target])['pq']
        self.assertLess(float(good_loss.detach()), float(bad_loss.detach()))
        # The coarse masks were identical, so their values cannot explain it.
        torch.testing.assert_close(good['pred_masks'], bad['pred_masks'])

    def test_primary_objective_can_improve_refined_mask(self):
        out, target = fixture(foreground=.1)
        optimizer = torch.optim.AdamW([out['mask_features2'], out['pred_logits'], out['pred_quality']], lr=.15)
        criterion = FilamentCriterion(256, weights={'pq': 1.}, pq_point_count=4096)
        values = []
        for step in range(16):
            torch.manual_seed(29)
            optimizer.zero_grad()
            loss = criterion(out, [target])['pq']
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            optimizer.step()
            values.append(float(loss.detach()))
        self.assertLess(values[-1], values[0] * .8)

    def test_native_query_attention_backward(self):
        config = model_config(read_config('configs/diagnostic.yaml'))
        config.query_detail_points = 16
        model, _ = build_model(config, load_backbone=False)
        output = model(torch.rand(1, 1, 64, 64))
        output['pred_masks'].square().mean().backward()
        for p in (model.decoder.detail_gain, model.decoder.detail_attention.in_proj_weight):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
