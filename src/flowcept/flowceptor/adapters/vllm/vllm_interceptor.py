"""Flowcept adapter for vLLM KV connectors.

Records one task per served request. The attention statistics themselves are
written by the connector to a SafeTensors file; this adapter records the
reference, so a record stays a fixed ~1.6 KB whatever the prompt length.
Carrying them inline did not scale -- a 128k-context request serialises to
hundreds of megabytes of JSON, past what a message queue will accept.

Connector settings that are constant for a run go on the workflow
(`attention_config`), not on every task.

    interceptor.send_model_workflow(
        "my-workflow-id", {"model": "facebook/opt-125m"},
        attention_config={"metric": "decode_attention", "top_pct": 10},
    )
    interceptor.capture_request(
        workflow_id="my-workflow-id",
        request_id="cmpl-abc:g0",
        attention_stats={"uri": "file:///prov/cmpl-abc_g0.safetensors",
                         "format": "safetensors", "bytes": 23688,
                         "tensors": {"attn_sum": {"shape": [953],
                                                  "dtype": "float32"}}},
        num_prompt_tokens=953,
        num_decode_tokens=8,
        activity="decode_attention",
    )
"""

from time import time
from typing import Any, Dict, Optional

from flowcept.commons.flowcept_dataclasses.task_object import Status, TaskObject
from flowcept.commons.flowcept_dataclasses.workflow_object import WorkflowObject
from flowcept.flowceptor.adapters.base_interceptor import BaseInterceptor


class VLLMInterceptor(BaseInterceptor):
    """Interceptor that captures per-token series emitted by a vLLM KVConnector.

    The adapter imposes no meaning on the series. A connector passes whatever
    it measured -- one list, or several named lists of equal length -- together
    with a metadata dict describing the metric. `activity` lets the connector
    label its own records so several connectors can coexist in one store.

    Examples
    --------
    >>> from flowcept.flowceptor.adapters.vllm.vllm_interceptor import VLLMInterceptor
    >>> interceptor = VLLMInterceptor.get_instance()
    >>> interceptor.start(bundle_exec_id="my-run")
    >>> interceptor.send_model_workflow("my-workflow-id", {"model": "facebook/opt-125m"})
    >>> interceptor.capture_request(
    ...     workflow_id="my-workflow-id",
    ...     request_id="0-abc:g0",
    ...     series={"k_avg": [12.4, 30.0], "v_avg": [1.05, 10.8]},
    ...     prompt_token_ids=[2, 133],
    ...     metadata={"metric": "kv_l2_norms", "num_layers": 12},
    ...     activity="kv_token_importance",
    ... )
    """

    _instance: Optional["VLLMInterceptor"] = None

    def __init__(self, plugin_key=None, kind="vllm"):
        """Initialize the interceptor.

        `plugin_key=None` (matching `InstrumentationInterceptor`'s own pattern)
        means this works with zero settings.yaml configuration; pass
        `plugin_key="vllm"` explicitly if you've added a `vllm:` entry under
        `adapters:` in your settings.yaml and want it honored.
        """
        super().__init__(plugin_key=plugin_key, kind=kind)

    @classmethod
    def get_instance(cls) -> "VLLMInterceptor":
        """Get the singleton instance for this interceptor."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def send_model_workflow(
        self,
        workflow_id: str,
        conf: Dict[str, Any],
        parent_workflow_id: Optional[str] = None,
        attention_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Register the served model as a workflow, once per run.

        Everything needed to interpret later tasks lives here -- in particular
        the tokenizer identity, without which `prompt_token_ids` cannot be
        decoded. It is constant for the run, so it belongs on the workflow
        rather than being repeated on every task.

        Parameters
        ----------
        workflow_id : str
            Workflow id the emitted tasks will reference. Must be distinct from
            any workflow the caller has already registered: Flowcept records a
            given workflow once, so reusing the caller's id would silently drop
            this configuration.
        conf : dict
            Model/tokenizer configuration (model, tokenizer, tokenizer_mode,
            dtype, max_model_len, ...).
        parent_workflow_id : str, optional
            The caller's workflow, if the vLLM run is nested inside one.
        attention_config : dict, optional
            Connector settings constant for the run. Recorded here for the same
            reason as `conf`: repeating it on every task is duplication that can
            drift.

        Returns
        -------
        The workflow id, as recorded.
        """
        workflow_obj = WorkflowObject()
        workflow_obj.workflow_id = workflow_id
        workflow_obj.parent_workflow_id = parent_workflow_id
        workflow_obj.name = conf.get("model", "vllm")
        workflow_obj.conf = conf
        if attention_config:
            workflow_obj.attention_config = attention_config
        return self.send_workflow_message(workflow_obj)

    def prepare_task_msg(
        self,
        workflow_id: str,
        request_id: str,
        attention_stats: Dict[str, Any],
        num_prompt_tokens: int,
        num_decode_tokens: int,
        started_at: Optional[float] = None,
        activity: Optional[str] = None,
    ) -> TaskObject:
        """Build a TaskObject for one finished vLLM request.

        Overrides `BaseInterceptor.prepare_task_msg` (which otherwise raises
        `NotImplementedError`) with the params `callback`/`capture_request`
        actually provide, rather than the generic `*args, **kwargs`.

        `num_prompt_tokens`/`num_decode_tokens` are passed rather than derived:
        they used to be inferred from the length of an inline series, and the
        series now lives in a file.
        """
        # Precedence: what the connector asked for, then settings.yaml, then a
        # neutral default. Distinct activities let several connectors write into
        # one store without their records being conflated.
        activity_id = (
            activity
            or getattr(self.settings, "activity_id", None)
            or "vllm_token_series"
        )

        task_msg = TaskObject()
        task_msg.task_id = request_id
        task_msg.activity_id = activity_id
        task_msg.subtype = activity_id
        task_msg.workflow_id = workflow_id
        task_msg.status = Status.FINISHED
        task_msg.started_at = started_at if started_at is not None else time()
        task_msg.ended_at = time()

        # prompt_token_ids is the model input; decode it with the tokenizer
        # recorded on the workflow to recover text.
        # One per-token series or several named ones. A bare list is kept under
        # `score` so older consumers are unaffected; a dict is passed through
        # under its own keys, whatever the connector chose to call them.
        task_msg.used = {
            "request_id": request_id.rsplit(":g", 1)[0],
            "num_prompt_tokens": num_prompt_tokens,
            "num_decode_tokens": num_decode_tokens,
        }
        # Top level rather than under `generated`: the statistics themselves are
        # in the file this points at, so the record carries a reference, not the
        # data. That keeps a record flat in size -- a long-context request would
        # otherwise serialise to hundreds of megabytes, past what a message
        # queue will accept.
        task_msg.attention_stats = attention_stats

        if self.telemetry_capture is not None:
            task_msg.telemetry_at_end = self.telemetry_capture.capture()

        return task_msg

    def capture_request(
        self,
        workflow_id: str,
        request_id: str,
        attention_stats: Dict[str, Any],
        num_prompt_tokens: int,
        num_decode_tokens: int,
        started_at: Optional[float] = None,
        activity: Optional[str] = None,
    ) -> None:
        """Capture one finished request, by reference to its statistics file.

        `attention_stats` is the descriptor written by the connector: uri,
        format, byte size, checksum, and a manifest of the tensors inside.
        """
        self.callback(
            workflow_id, request_id, attention_stats, num_prompt_tokens,
            num_decode_tokens, started_at, activity
        )

    def callback(
        self,
        workflow_id: str,
        request_id: str,
        attention_stats: Dict[str, Any],
        num_prompt_tokens: int,
        num_decode_tokens: int,
        started_at: Optional[float] = None,
        activity: Optional[str] = None,
    ) -> None:
        """Decide what to do when a request finishes.

        Always "interesting" here (every finished request gets a task) -- there
        is no filtering analogous to Dask's task-state-transition branching,
        since a completed request is unconditionally worth recording.
        """
        task_msg = self.prepare_task_msg(
            workflow_id, request_id, attention_stats, num_prompt_tokens,
            num_decode_tokens, started_at, activity
        )
        self.intercept(task_msg.to_dict())

    def observe(self, *args, **kwargs):
        """Not applicable: this adapter is event-driven, not a polling observer."""
        raise NotImplementedError(
            "VLLMInterceptor captures via vLLM's KVConnector request-finished hook, "
            "not a continuous observe() loop."
        )
