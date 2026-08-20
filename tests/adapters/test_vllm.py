"""Tests for the vLLM (KV cache token importance) adapter.

Real interceptor, real Flowcept buffer, no mocks -- per this repo's testing
conventions. The adapter deliberately takes plain Python data rather than vLLM
objects, so these tests need neither vLLM nor a GPU: the boundary being tested
is "score data in -> correct Flowcept task/workflow out".

End-to-end capture against a live vLLM engine is covered separately, in the
experiments repo (`experiments/vllm-kvnorm/`), since that does require a GPU.
"""

import unittest

from flowcept import Flowcept
from flowcept.flowceptor.adapters.vllm.vllm_interceptor import VLLMInterceptor

MODEL_CONF = {
    "model": "facebook/opt-125m",
    "tokenizer": "facebook/opt-125m",
    "tokenizer_mode": "auto",
    "dtype": "torch.float16",
    "max_model_len": 512,
    "architectures": ["OPTForCausalLM"],
}


class TestVLLMInterceptor(unittest.TestCase):
    """Real (no-mock) tests for VLLMInterceptor's request capture."""

    def test_capture_request_emits_task(self):
        """The core claim: one finished request produces one correctly shaped task."""
        with Flowcept("vllm", workflow_name="test_vllm_capture") as fc:
            wf_id = Flowcept.current_workflow_id
            VLLMInterceptor.get_instance().capture_request(
                workflow_id=wf_id,
                request_id="0-abc123",
                scores=[0.13, 0.16, 0.14, 0.15],
                prompt_token_ids=[2, 133, 812],
                metadata={"num_layers": 12, "num_kv_heads": 12},
            )
            buf = fc.get_buffer()

        tasks = [r for r in buf if r.get("task_id") == "0-abc123"]
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task["workflow_id"], wf_id)
        self.assertEqual(task["status"], "FINISHED")
        self.assertEqual(task["activity_id"], "kv_token_importance")
        self.assertEqual(task["generated"]["score"], [0.13, 0.16, 0.14, 0.15])
        self.assertEqual(task["used"]["prompt_token_ids"], [2, 133, 812])
        self.assertEqual(task["used"]["num_prompt_tokens"], 3)
        self.assertEqual(task["used"]["num_computed_tokens"], 4)
        self.assertEqual(task["custom_metadata"]["num_layers"], 12)

    def test_one_score_per_token(self):
        """num_computed_tokens must track the score vector, not the prompt."""
        with Flowcept("vllm", workflow_name="test_vllm_lengths") as fc:
            VLLMInterceptor.get_instance().capture_request(
                workflow_id=Flowcept.current_workflow_id,
                request_id="0-lengths",
                scores=[0.1] * 29,
                prompt_token_ids=[1] * 6,
                metadata={},
            )
            buf = fc.get_buffer()

        task = next(r for r in buf if r.get("task_id") == "0-lengths")
        self.assertEqual(len(task["generated"]["score"]), 29)
        self.assertEqual(task["used"]["num_computed_tokens"], 29)
        self.assertEqual(task["used"]["num_prompt_tokens"], 6)

    def test_multiple_requests_produce_multiple_tasks(self):
        with Flowcept("vllm", workflow_name="test_vllm_multi") as fc:
            wf_id = Flowcept.current_workflow_id
            for i in range(3):
                VLLMInterceptor.get_instance().capture_request(
                    workflow_id=wf_id,
                    request_id=f"{i}-multi",
                    scores=[0.1 * (i + 1)],
                    prompt_token_ids=[i],
                    metadata={},
                )
            buf = fc.get_buffer()

        tasks = sorted(
            (r for r in buf if str(r.get("task_id", "")).endswith("-multi")),
            key=lambda r: r["task_id"],
        )
        self.assertEqual(len(tasks), 3)
        self.assertEqual([t["used"]["prompt_token_ids"] for t in tasks], [[0], [1], [2]])

    def test_model_workflow_carries_tokenizer_identity(self):
        """Without the tokenizer on the workflow, prompt_token_ids is undecodable.

        Uses a workflow id the controller has not already claimed, which is the
        real deployment shape: vLLM runs its scheduler in a separate EngineCore
        process, so the connector there owns its own workflow id (handed to it
        via ``kv_connector_extra_config``) and registers the model itself.
        """
        vllm_wf_id = "test-vllm-engine-core-workflow"

        with Flowcept("vllm", workflow_name="test_vllm_workflow") as fc:
            returned = VLLMInterceptor.get_instance().send_model_workflow(
                vllm_wf_id, MODEL_CONF, parent_workflow_id=Flowcept.current_workflow_id
            )
            parent_id = Flowcept.current_workflow_id
            VLLMInterceptor.get_instance().capture_request(
                workflow_id=vllm_wf_id,
                request_id="0-wf",
                scores=[0.1, 0.2],
                prompt_token_ids=[2, 133],
                metadata={},
            )
            buf = fc.get_buffer()

        self.assertEqual(returned, vllm_wf_id)

        workflows = [
            r for r in buf if r.get("type") == "workflow" and r.get("workflow_id") == vllm_wf_id
        ]
        self.assertEqual(len(workflows), 1)
        # Nested under the caller's workflow, not reusing its id -- reusing it
        # would collide with the already-registered workflow and drop the conf.
        self.assertEqual(workflows[0]["parent_workflow_id"], parent_id)
        self.assertNotEqual(workflows[0]["workflow_id"], parent_id)

        conf = workflows[0]["conf"]
        self.assertEqual(conf["tokenizer"], "facebook/opt-125m")
        self.assertEqual(conf["model"], "facebook/opt-125m")
        self.assertEqual(conf["max_model_len"], 512)

        # The task must be joinable back to that workflow.
        task = next(r for r in buf if r.get("task_id") == "0-wf")
        self.assertEqual(task["workflow_id"], vllm_wf_id)

    def test_model_workflow_is_registered_only_once(self):
        """A run registers its model workflow once; repeat calls are no-ops."""
        wf_id = "test-vllm-dedupe-workflow"
        interceptor = VLLMInterceptor.get_instance()

        with Flowcept("vllm", workflow_name="test_vllm_dedupe") as fc:
            first = interceptor.send_model_workflow(wf_id, MODEL_CONF)
            second = interceptor.send_model_workflow(wf_id, MODEL_CONF)
            buf = fc.get_buffer()

        self.assertEqual(first, wf_id)
        self.assertIsNone(second)
        emitted = [
            r for r in buf if r.get("type") == "workflow" and r.get("workflow_id") == wf_id
        ]
        self.assertEqual(len(emitted), 1)

    def test_build_returns_singleton(self):
        """BaseInterceptor.build('vllm') must resolve to this adapter."""
        from flowcept.flowceptor.adapters.base_interceptor import BaseInterceptor

        built = BaseInterceptor.build("vllm")
        self.assertIsInstance(built, VLLMInterceptor)
        self.assertIs(built, VLLMInterceptor.get_instance())

    def test_observe_is_not_supported(self):
        """This adapter is event-driven; observe() must fail loudly."""
        with self.assertRaises(NotImplementedError):
            VLLMInterceptor.get_instance().observe()


if __name__ == "__main__":
    unittest.main()
