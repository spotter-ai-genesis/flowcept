"""Tests for the timm (Vision Transformer attention) adapter.

Real model, real forward passes, no mocks -- per this repo's testing
conventions. Requires torch plus a timm build exposing set_attn_capture/
get_patch_attention_scores (the timm-instrumentation fork); skipped entirely
otherwise, since that's a local/unpublished dependency, not a hard
requirement of the base flowcept package.
"""

import unittest

import pytest

from flowcept import Flowcept

try:
    import torch
    import timm

    from flowcept.flowceptor.adapters.timm.timm_interceptor import TimmInterceptor

    def _build_native_vit(img_size=224):
        m = timm.create_model(
            "vit_small_patch16_224",
            pretrained=False,
            features_only=True,
            out_indices=(-1,),
            img_size=img_size,
        )
        return m.model

    _HAS_TIMM_INSTRUMENTATION = hasattr(_build_native_vit(), "set_attn_capture")
except Exception:
    _HAS_TIMM_INSTRUMENTATION = False

pytestmark = pytest.mark.skipif(
    not _HAS_TIMM_INSTRUMENTATION,
    reason="Requires torch and a timm build with set_attn_capture/get_patch_attention_scores "
    "(the timm-instrumentation fork) -- not a hard dependency of base flowcept.",
)


class TestTimmInterceptor(unittest.TestCase):
    """Real (no-mock) tests for TimmInterceptor's transparent attention capture."""

    def setUp(self):
        TimmInterceptor.get_instance().detach_all()

    def tearDown(self):
        TimmInterceptor.get_instance().detach_all()

    def test_attach_rejects_module_without_required_api(self):
        class Dummy(torch.nn.Module):
            def forward(self, x):
                return x

        with self.assertRaises(AttributeError):
            TimmInterceptor.get_instance().attach(Dummy())

    def test_attach_enables_capture_on_last_block_only_by_default(self):
        native = _build_native_vit()
        TimmInterceptor.get_instance().attach(native)
        self.assertTrue(native.blocks[-1].attn.store_attn_map)
        self.assertFalse(native.blocks[0].attn.store_attn_map)

    def test_transparent_capture_on_plain_forward_call(self):
        """The core claim: a normal model(x) call, with zero Flowcept API at
        the call site, produces a correctly-shaped captured task."""
        native = _build_native_vit()
        TimmInterceptor.get_instance().attach(native, activity_id="vit_forward")

        with Flowcept("timm", workflow_name="test_timm_transparent_capture") as fc:
            wf_id = Flowcept.current_workflow_id
            with torch.no_grad():
                native(torch.randn(2, 3, 224, 224))  # <-- no Flowcept API touched here
            buf = fc.get_buffer()

        tasks = [r for r in buf if r.get("activity_id") == "vit_forward"]
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task["workflow_id"], wf_id)
        self.assertEqual(task["status"], "FINISHED")
        self.assertEqual(task["used"]["input_shape"], [2, 3, 224, 224])
        scores = task["generated"]["patch_attention"]
        self.assertIsInstance(scores, list)
        self.assertEqual(len(scores), 196)  # patch16 @ tile 224 -> 14x14 patches
        self.assertTrue(all(isinstance(v, float) for v in scores))

    def test_multiple_forward_calls_produce_multiple_tasks(self):
        native = _build_native_vit()
        TimmInterceptor.get_instance().attach(native, activity_id="vit_forward_multi")

        with Flowcept("timm", workflow_name="test_timm_multi_call") as fc:
            with torch.no_grad():
                native(torch.randn(1, 3, 224, 224))
                native(torch.randn(3, 3, 224, 224))
            buf = fc.get_buffer()

        tasks = [r for r in buf if r.get("activity_id") == "vit_forward_multi"]
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["used"]["input_shape"], [1, 3, 224, 224])
        self.assertEqual(tasks[1]["used"]["input_shape"], [3, 3, 224, 224])

    def test_detach_all_stops_capture(self):
        native = _build_native_vit()
        TimmInterceptor.get_instance().attach(native, activity_id="vit_forward_detach")
        TimmInterceptor.get_instance().detach_all()

        with Flowcept("timm", workflow_name="test_timm_detach") as fc:
            with torch.no_grad():
                native(torch.randn(1, 3, 224, 224))
            buf = fc.get_buffer()

        tasks = [r for r in buf if r.get("activity_id") == "vit_forward_detach"]
        self.assertEqual(len(tasks), 0)

    def test_custom_reduction_and_activity_id(self):
        native = _build_native_vit()
        TimmInterceptor.get_instance().attach(native, activity_id="vit_cls_reduction", reduction="cls")

        with Flowcept("timm", workflow_name="test_timm_reduction") as fc:
            with torch.no_grad():
                native(torch.randn(1, 3, 224, 224))
            buf = fc.get_buffer()

        tasks = [r for r in buf if r.get("activity_id") == "vit_cls_reduction"]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(len(tasks[0]["generated"]["patch_attention"]), 196)

    def test_different_tile_size_and_patch_count(self):
        """patch16 @ tile 448 -> 28x28 = 784 patches, sanity-checking the
        adapter isn't hardcoded to any one tile size."""
        native = _build_native_vit(img_size=448)
        TimmInterceptor.get_instance().attach(native, activity_id="vit_forward_448")

        with Flowcept("timm", workflow_name="test_timm_448") as fc:
            with torch.no_grad():
                native(torch.randn(1, 3, 448, 448))
            buf = fc.get_buffer()

        tasks = [r for r in buf if r.get("activity_id") == "vit_forward_448"]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(len(tasks[0]["generated"]["patch_attention"]), 784)


if __name__ == "__main__":
    unittest.main()
