"""vLLM (KV cache token importance) interceptor module.

Captures a per-token importance score for every request served by vLLM, taken
from the KV cache at the moment the request finishes -- as produced by the
out-of-tree `vllm_kvnorm` KV connector. No application-level task/workflow code
is required at the call site: once vLLM is started with that connector
configured, every completed request transparently emits a Flowcept task.

This mirrors the Dask adapter's architecture rather than the timm one. The timm
adapter attaches a forward hook inside the caller's own process; vLLM instead
runs its scheduler and workers in a separate EngineCore process, so this
interceptor is constructed *there*, buffers into its own MQ connection, and is
handed the workflow id explicitly -- exactly as Dask worker plugins are. The
extension point used is vLLM's own `KVConnector` interface, not something
Flowcept invents.

This adapter has no import-time dependency on vLLM or on `vllm_kvnorm`: it
receives plain Python data structures, so it is importable and testable in a
base flowcept install.
"""

from time import time
from typing import Any, Dict, List, Optional

from flowcept.commons.flowcept_dataclasses.task_object import TaskObject
from flowcept.commons.flowcept_dataclasses.workflow_object import WorkflowObject
from flowcept.commons.vocabulary import Status
from flowcept.flowceptor.adapters.base_interceptor import BaseInterceptor


class VLLMInterceptor(BaseInterceptor):
    """Interceptor that captures per-token KV importance scores from vLLM.

    The scores use the PagedEviction proxy (Chitty-Venkata et al., Findings of
    EACL 2026, arXiv:2509.04377): ``mean(||V_i||_2 / ||K_i||_2)`` over layers
    and KV heads. High score means the token is important -- a key's L2 norm is
    inversely proportional to its cumulative attention (Devoto et al. 2024,
    arXiv:2406.11430). The proxy exists so that no attention weights, and hence
    no FlashAttention kernel changes, are needed.

    Examples
    --------
    >>> from flowcept.flowceptor.adapters.vllm.vllm_interceptor import VLLMInterceptor
    >>> interceptor = VLLMInterceptor.get_instance()
    >>> interceptor.start(bundle_exec_id="my-run")
    >>> interceptor.send_model_workflow("my-workflow-id", {"model": "facebook/opt-125m"})
    >>> interceptor.capture_request(
    ...     workflow_id="my-workflow-id",
    ...     request_id="0-abc",
    ...     scores=[0.13, 0.16, 0.14],
    ...     prompt_token_ids=[2, 133, 812],
    ...     metadata={"num_layers": 12},
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
        self, workflow_id: str, conf: Dict[str, Any], parent_workflow_id: Optional[str] = None
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

        Returns
        -------
        The workflow id, as recorded.
        """
        workflow_obj = WorkflowObject()
        workflow_obj.workflow_id = workflow_id
        workflow_obj.parent_workflow_id = parent_workflow_id
        workflow_obj.name = conf.get("model", "vllm")
        workflow_obj.conf = conf
        return self.send_workflow_message(workflow_obj)

    def prepare_task_msg(
        self,
        workflow_id: str,
        request_id: str,
        scores: List[float],
        prompt_token_ids: List[int],
        metadata: Dict[str, Any],
        started_at: Optional[float] = None,
    ) -> TaskObject:
        """Build a TaskObject for one finished vLLM request.

        Overrides `BaseInterceptor.prepare_task_msg` (which otherwise raises
        `NotImplementedError`) with the params `callback`/`capture_request`
        actually provide, rather than the generic `*args, **kwargs`.
        """
        activity_id = getattr(self.settings, "activity_id", None) or "kv_token_importance"

        task_msg = TaskObject()
        task_msg.task_id = request_id
        task_msg.activity_id = activity_id
        task_msg.subtype = "kv_token_importance"
        task_msg.workflow_id = workflow_id
        task_msg.status = Status.FINISHED
        task_msg.started_at = started_at if started_at is not None else time()
        task_msg.ended_at = time()

        # prompt_token_ids is the model input; decode it with the tokenizer
        # recorded on the workflow to recover text.
        task_msg.used = {
            "request_id": request_id,
            "prompt_token_ids": prompt_token_ids,
            "num_prompt_tokens": len(prompt_token_ids),
            "num_computed_tokens": len(scores),
        }
        task_msg.generated = {"score": scores}
        task_msg.custom_metadata = metadata

        if self.telemetry_capture is not None:
            task_msg.telemetry_at_end = self.telemetry_capture.capture()

        return task_msg

    def capture_request(
        self,
        workflow_id: str,
        request_id: str,
        scores: List[float],
        prompt_token_ids: List[int],
        metadata: Dict[str, Any],
        started_at: Optional[float] = None,
    ) -> None:
        """Capture one finished request. Convenience wrapper over `callback`."""
        self.callback(workflow_id, request_id, scores, prompt_token_ids, metadata, started_at)

    def callback(
        self,
        workflow_id: str,
        request_id: str,
        scores: List[float],
        prompt_token_ids: List[int],
        metadata: Dict[str, Any],
        started_at: Optional[float] = None,
    ) -> None:
        """Decide what to do when a request finishes.

        Always "interesting" here (every finished request gets a task) -- there
        is no filtering analogous to Dask's task-state-transition branching,
        since a completed request is unconditionally worth recording.
        """
        task_msg = self.prepare_task_msg(
            workflow_id, request_id, scores, prompt_token_ids, metadata, started_at
        )
        self.intercept(task_msg.to_dict())

    def observe(self, *args, **kwargs):
        """Not applicable: this adapter is event-driven, not a polling observer."""
        raise NotImplementedError(
            "VLLMInterceptor captures via vLLM's KVConnector request-finished hook, "
            "not a continuous observe() loop."
        )
