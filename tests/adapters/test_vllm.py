"""Tests for the vLLM adapter.

Real interceptor, real Flowcept buffer, no mocks -- per this repo's testing
conventions. The adapter deliberately takes plain Python data rather than vLLM
objects, so these tests need neither vLLM nor a GPU: the boundary being tested
is "statistics reference in -> correct Flowcept task/workflow out".

End-to-end capture against a live vLLM engine is covered by the connector's own
`tests/e2e_smoke.py`, which does require a GPU.
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

ATTENTION_CONFIG = {
    "connector": "vllm-attn-connector",
    "metric": "decode_attention",
    "top_pct": 10.0,
    "chunk_size": 32,
    "num_query_heads": 12,
    "num_layers_scored": 12,
}


def stats(uri="file:///prov/wf/0-abc123_g0.safetensors", **over):
    """A statistics-file descriptor, as the connector writes one."""
    d = {
        "uri": uri,
        "format": "safetensors",
        "bytes": 4096,
        "sha256": "0" * 64,
        "kv_cache_group_id": 0,
        "written_by_tp_rank": 0,
        "segment_mode": "fixed",
        "decode_steps_dropped": 0,
        "decode_steps_nonfinite": 0,
        "restarts": 0,
        "tensors": {
            "attn_sum": {"shape": [3], "dtype": "float32"},
            "val_all_max": {"shape": [4, 2], "dtype": "float32"},
        },
    }
    d.update(over)
    return d


class TestVLLMInterceptor(unittest.TestCase):
    """Real (no-mock) tests for VLLMInterceptor's request capture."""

    def test_capture_request_emits_task(self):
        """The core claim: one finished request produces one correctly shaped task."""
        with Flowcept("vllm", workflow_name="test_vllm_capture") as fc:
            wf_id = Flowcept.current_workflow_id
            VLLMInterceptor.get_instance().capture_request(
                workflow_id=wf_id,
                request_id="0-abc123",
                attention_stats=stats(),
                num_prompt_tokens=3,
                num_decode_tokens=4,
                activity="decode_attention",
            )
            buf = fc.get_buffer()

        tasks = [r for r in buf if r.get("task_id") == "0-abc123"]
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task["workflow_id"], wf_id)
        self.assertEqual(task["status"], "FINISHED")
        self.assertEqual(task["activity_id"], "decode_attention")
        self.assertEqual(task["used"]["num_prompt_tokens"], 3)
        self.assertEqual(task["used"]["num_decode_tokens"], 4)
        self.assertEqual(
            task["attention_stats"]["uri"], "file:///prov/wf/0-abc123_g0.safetensors"
        )
        self.assertEqual(sorted(task["attention_stats"]["tensors"]), ["attn_sum", "val_all_max"])

    def test_record_size_does_not_grow_with_the_prompt(self):
        """Why the statistics live in a file: the record must stay small.

        A 128k-token prompt used to serialise its arrays inline, past what a
        message queue accepts. The descriptor is the same size either way.
        """
        import json

        with Flowcept("vllm", workflow_name="test_vllm_size") as fc:
            for name, n_prompt in (("0-small", 8), ("0-large", 131072)):
                VLLMInterceptor.get_instance().capture_request(
                    workflow_id=Flowcept.current_workflow_id,
                    request_id=name,
                    attention_stats=stats(uri=f"file:///prov/wf/{name}_g0.safetensors"),
                    num_prompt_tokens=n_prompt,
                    num_decode_tokens=16,
                )
            buf = fc.get_buffer()

        by_id = {r["task_id"]: r for r in buf if r.get("task_id") in ("0-small", "0-large")}
        small = len(json.dumps(by_id["0-small"]))
        large = len(json.dumps(by_id["0-large"]))
        # only the recorded token count differs, a handful of characters
        self.assertLess(abs(large - small), 32)

    def test_group_suffix_is_stripped_from_the_request_id(self):
        """Multi-group models emit one task per group, all one request."""
        with Flowcept("vllm", workflow_name="test_vllm_groups") as fc:
            for gid in (0, 1):
                VLLMInterceptor.get_instance().capture_request(
                    workflow_id=Flowcept.current_workflow_id,
                    request_id=f"0-grouped:g{gid}",
                    attention_stats=stats(kv_cache_group_id=gid),
                    num_prompt_tokens=5,
                    num_decode_tokens=2,
                )
            buf = fc.get_buffer()

        tasks = sorted(
            (r for r in buf if str(r.get("task_id", "")).startswith("0-grouped")),
            key=lambda r: r["task_id"],
        )
        self.assertEqual(len(tasks), 2)
        self.assertEqual([t["task_id"] for t in tasks], ["0-grouped:g0", "0-grouped:g1"])
        # the task id keeps the group, `used.request_id` does not
        self.assertEqual({t["used"]["request_id"] for t in tasks}, {"0-grouped"})

    def test_multiple_requests_produce_multiple_tasks(self):
        with Flowcept("vllm", workflow_name="test_vllm_multi") as fc:
            wf_id = Flowcept.current_workflow_id
            for i in range(3):
                VLLMInterceptor.get_instance().capture_request(
                    workflow_id=wf_id,
                    request_id=f"{i}-multi",
                    attention_stats=stats(),
                    num_prompt_tokens=i + 1,
                    num_decode_tokens=1,
                )
            buf = fc.get_buffer()

        tasks = sorted(
            (r for r in buf if str(r.get("task_id", "")).endswith("-multi")),
            key=lambda r: r["task_id"],
        )
        self.assertEqual(len(tasks), 3)
        self.assertEqual([t["used"]["num_prompt_tokens"] for t in tasks], [1, 2, 3])

    def test_model_workflow_carries_tokenizer_and_attention_config(self):
        """Without the tokenizer on the workflow, prompt_token_ids is undecodable.

        `attention_config` rides here for the same reason: it is constant for
        the run, so repeating it on every task would be duplication that can
        drift.

        Uses a workflow id the controller has not already claimed, which is the
        real deployment shape: vLLM runs its scheduler in a separate EngineCore
        process, so the connector there owns its own workflow id (handed to it
        via ``kv_connector_extra_config``) and registers the model itself.
        """
        vllm_wf_id = "test-vllm-engine-core-workflow"

        with Flowcept("vllm", workflow_name="test_vllm_workflow") as fc:
            returned = VLLMInterceptor.get_instance().send_model_workflow(
                vllm_wf_id,
                MODEL_CONF,
                parent_workflow_id=Flowcept.current_workflow_id,
                attention_config=ATTENTION_CONFIG,
            )
            parent_id = Flowcept.current_workflow_id
            VLLMInterceptor.get_instance().capture_request(
                workflow_id=vllm_wf_id,
                request_id="0-wf",
                attention_stats=stats(),
                num_prompt_tokens=2,
                num_decode_tokens=2,
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

        ac = workflows[0]["attention_config"]
        self.assertEqual(ac["metric"], "decode_attention")
        self.assertEqual(ac["top_pct"], 10.0)
        # needed to decode topk_head, so it must be a field of its own
        self.assertEqual(ac["num_query_heads"], 12)

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
