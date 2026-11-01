import unittest
from dataclasses import asdict
from pathlib import Path
import tempfile
import numpy as np
import torch
from torch.nn import functional as F
from solar_filaments.config import model_config, read_config
from solar_filaments.curves import CurveResidualBlock
from solar_filaments.detail import NativeDetailPath
from solar_filaments.inference import encode_mask, instance_predictions
from solar_filaments.losses import FilamentCriterion
from solar_filaments.metrics import official_pq, aggregate_pq
from solar_filaments.model import build_model
from solar_filaments.ops.dcnv4 import dcnv4_reference, native_function


torch.set_num_threads(2)


def diagnostic_config():
    return model_config(read_config("configs/diagnostic.yaml"))


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)

    def test_dcn_group_packing_axis_order_scale_and_no_softmax(self):
        y, x = torch.meshgrid(torch.arange(6), torch.arange(7), indexing="ij")
        plane = (3 * x + 2 * y).float()
        value = plane[None, :, :, None].repeat(1, 1, 1, 32)
        packed = torch.zeros(1, 6, 7, 56)
        # Group 0: center sample with (dx=.25, dy=.125), weight 2.
        packed[..., 8], packed[..., 9], packed[..., 22] = 0.25, 0.125, 2.0
        # Group 1: x-major kernel index 7 = (+1,0), weight -1.
        packed[..., 27 + 18 + 7] = -1.0
        result = dcnv4_reference(value, packed, group=2, offset_scale=2)
        expected0 = 2 * (plane[1:5, 1:4] + 3 * 0.5 + 2 * 0.25)
        expected1 = -(plane[1:5, 1:4] + 3 * 2)
        torch.testing.assert_close(result[0, 1:5, 1:4, 0], expected0)
        torch.testing.assert_close(result[0, 1:5, 1:4, 16], expected1)

    def test_dcn_offset_gradient_finite_difference(self):
        value = torch.randn(1, 4, 5, 16)
        packed = (torch.randn(1, 4, 5, 32) * 0.15).requires_grad_()
        weighting = torch.randn_like(value)
        loss = (dcnv4_reference(value, packed, 1, 1.5) * weighting).sum()
        gradient = torch.autograd.grad(loss, packed)[0]
        index = (0, 2, 2, 8)
        plus, minus = packed.detach().clone(), packed.detach().clone()
        plus[index] += 1e-3
        minus[index] -= 1e-3
        difference = ((dcnv4_reference(value, plus, 1, 1.5) - dcnv4_reference(value, minus, 1, 1.5)) * weighting).sum() / 0.002
        torch.testing.assert_close(gradient[index], difference, atol=2e-3, rtol=2e-3)

    @unittest.skipUnless(torch.cuda.is_available() and native_function() is not None, "Official CUDA extension not installed")
    def test_native_reference_outputs_and_gradients(self):
        value = torch.randn(1, 8, 8, 32, device="cuda", requires_grad=True)
        packed = (torch.randn(1, 8, 8, 56, device="cuda") * 0.2).requires_grad_()
        function = native_function()
        native = function.apply(value, packed, 3, 3, 1, 1, 1, 1, 1, 1, 2, 16, 2.0, 256, 0)
        reference = dcnv4_reference(value, packed, 2, 2.0)
        torch.testing.assert_close(native, reference, atol=3e-5, rtol=3e-5)
        weighting = torch.randn_like(native)
        actual_grad = torch.autograd.grad((native * weighting).sum(), (value, packed))
        expected_grad = torch.autograd.grad((reference * weighting).sum(), (value, packed))
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    @unittest.skipUnless(torch.cuda.is_available() and native_function() is not None, "Official CUDA extension not installed")
    def test_native_half_backward_for_all_large_backbone_group_counts(self):
        for groups in (10, 20, 40, 80):
            with self.subTest(groups=groups):
                value = torch.randn(1, 4, 4, groups * 16, device="cuda", dtype=torch.float16, requires_grad=True)
                packed_dim = ((groups * 27 + 7) // 8) * 8
                packed = (torch.randn(1, 4, 4, packed_dim, device="cuda", dtype=torch.float16) * 0.1).requires_grad_()
                native = native_function().apply(value, packed, 3, 3, 1, 1, 1, 1, 1, 1, groups, 16, 2.0, 256, 0)
                reference = dcnv4_reference(value, packed, groups, 2.0)
                torch.testing.assert_close(native, reference, atol=2e-3, rtol=2e-3)
                weighting = torch.randn_like(native)
                actual_grad = torch.autograd.grad((native * weighting).sum(), (value, packed))
                expected_grad = torch.autograd.grad((reference * weighting).sum(), (value, packed))
                for actual, expected in zip(actual_grad, expected_grad):
                    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=1e-2)

    def test_halo_tiling_matches_complete_detail_path_with_curves_active(self):
        config = diagnostic_config()
        path = NativeDetailPath(config).eval()
        for module in path.modules():
            if isinstance(module, CurveResidualBlock) and module.curve is not None:
                module.gamma.data.fill_(0.3)
                module.curve.offsets.weight.data.normal_(0, 0.02)
        image, context = torch.randn(1, 2, 128, 192), torch.randn(1, config.feature_dim, 16, 24)
        with torch.no_grad():
            complete = path(image, context, (0, 0, 128, 192), (128, 192))
            tiled = path.tiled(image, context, tile=64, halo=32)
        for actual, expected in zip(tiled, complete):
            torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)

    def test_full_model_matching_loss_and_curve_gradients(self):
        model, _ = build_model(diagnostic_config(), load_backbone=False)
        image = torch.rand(1, 1, 96, 128)
        masks = torch.zeros(2, 96, 128, dtype=torch.bool)
        masks[0, 20:24, 10:115], masks[1, 60:65, 30:90] = True, True
        foreground = masks.any(0).float()
        auxiliary = F.max_pool2d(foreground[None, None].repeat(1, 3, 1, 1), 2)[0]
        output = model(image)
        losses = FilamentCriterion(256)(output, [{"masks": masks, "auxiliary": auxiliary}])
        loss = sum(losses.values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in (model.backbone.encoder.patch_embed.conv1.weight,
                          model.detail.curve2.curve.offsets.weight,
                          model.detail.curve4.curve.offsets.weight,
                          model.decoder.query.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertEqual(output["pred_masks"].shape, (1, 32, 24, 32))
        self.assertEqual(output["mask_features2"].shape, (1, 16, 48, 64))

    def test_empty_target_has_valid_loss_and_backward(self):
        model, _ = build_model(diagnostic_config(), load_backbone=False)
        output = model(torch.rand(1, 1, 64, 64))
        losses = FilamentCriterion(64)(output, [{"masks": torch.zeros(0, 64, 64, dtype=torch.bool),
                                               "auxiliary": torch.zeros(3, 32, 32)}])
        loss = sum(losses.values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_mixed_empty_and_nonempty_batch_and_checkpoint_roundtrip(self):
        config = diagnostic_config()
        model, _ = build_model(config, load_backbone=False)
        model.eval()
        images = torch.rand(2, 1, 64, 64)
        masks = torch.zeros(1, 64, 64, dtype=torch.bool)
        masks[:, 20:24, 8:55] = True
        targets = [{"masks": masks, "auxiliary": torch.zeros(3, 32, 32)},
                   {"masks": masks[:0], "auxiliary": torch.zeros(3, 32, 32)}]
        outputs = model(images)
        loss = sum(FilamentCriterion(64)(outputs, targets).values())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"model": model.state_dict(), "model_config": asdict(config), "task_trained": False}, path)
            restored, _ = build_model(config, load_backbone=False)
            state = torch.load(path, weights_only=True)
            restored.load_state_dict(state["model"], strict=True)
            restored.eval()
            with torch.no_grad():
                reloaded = restored(images)
            torch.testing.assert_close(reloaded["pred_masks"], outputs["pred_masks"].detach())

    def test_pq_strict_threshold_and_split_merge_penalty(self):
        a, b = np.zeros((8, 12), bool), np.zeros((8, 12), bool)
        a[2:6, 1:4], b[2:6, 7:10] = True, True
        truth = [encode_mask(a), encode_mask(b)]
        self.assertEqual(official_pq(truth, truth)["pq"], 1)
        merged = official_pq([encode_mask(a | b)], truth)
        self.assertEqual((merged["pq"], merged["fp"], merged["fn"]), (0, 1, 2))
        self.assertEqual(official_pq([], truth)["fn"], 2)

    def test_pq_aggregates_counts_instead_of_averaging_records(self):
        scores = [{"pq": 1.0, "iou_sum": 1.0, "tp": 1, "fp": 0, "fn": 0},
                  {"pq": 0.0, "iou_sum": 0.0, "tp": 0, "fp": 0, "fn": 4}]
        result = aggregate_pq(scores)
        self.assertAlmostEqual(result["pq"], 1 / 3)
        self.assertEqual(result["mean_record_pq"], 0.5)

    def test_one_query_keeps_disconnected_parts_as_one_instance(self):
        basis = torch.full((1, 2, 4, 8), -12.0)
        basis[0, 0, 0:2, 0:2], basis[0, 0, 2:4, 6:8] = 12, 12
        basis[0, 1, 1:3, 3:5] = 12
        output = {"pred_logits": torch.tensor([[[12.0, -12.0], [12.0, -12.0]]]),
                  "pred_quality": torch.full((1, 2), 12.0), "mask_embeddings": torch.eye(2)[None],
                  "mask_features2": basis, "image_size": (8, 16)}
        predictions, labels = instance_predictions(output, min_area=2, score_threshold=0.1)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(np.unique(labels[labels > 0])), 2)
        from scipy.ndimage import label
        self.assertEqual(label(labels == 1)[1], 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU not available")
    def test_2048_streamed_masks_and_rle_roundtrip(self):
        basis = torch.full((1, 2, 1024, 1024), -12.0, device="cuda", dtype=torch.float16)
        basis[0, 0, 200:204, 10:1000] = 12
        basis[0, 1, 30:950, 500:504] = 12
        output = {"pred_logits": torch.tensor([[[12.0, -12.0], [12.0, -12.0]]], device="cuda"),
                  "pred_quality": torch.full((1, 2), 12.0, device="cuda"),
                  "mask_embeddings": torch.eye(2, device="cuda")[None],
                  "mask_features2": basis, "image_size": (2048, 2048)}
        predictions, labels = instance_predictions(output, query_chunk=1, min_area=32)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(labels.shape, (2048, 2048))
        from pycocotools import mask as mask_utils
        for prediction in predictions:
            decoded = mask_utils.decode(prediction["segmentation"]).astype(bool)
            np.testing.assert_array_equal(decoded, labels == prediction["instance_id"])


if __name__ == "__main__":
    unittest.main()
