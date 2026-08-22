"""Timm (Vision Transformer attention) interceptor module.

Transparently captures ViT patch-attention scores from a timm
VisionTransformer -- as instrumented by the timm-instrumentation fork's
`set_attn_capture`/`get_attn_maps`/`get_patch_attention_scores` API -- via a
native PyTorch forward hook. No application-level task/workflow code is
required at the call site: once attached, every normal `model(x)` call
transparently emits a Flowcept task.

This mirrors the Dask adapter's architecture: hook into the host framework's
own callback mechanism rather than requiring the caller to instrument every
call by hand. `torch.nn.Module.register_forward_hook` plays the role here
that `distributed.WorkerPlugin` plays for the Dask adapter -- both are the
*target framework's own* extension points, not something Flowcept invents.

This adapter has no import-time dependency on timm or on any specific fork:
`attach()` duck-types the module at runtime (checks for `set_attn_capture`/
`get_patch_attention_scores`) and raises a clear error if they're absent,
rather than pinning a hard dependency that can't be expressed for a local,
unpublished fork.
"""

from time import time
from typing import Optional

from flowcept.commons.flowcept_dataclasses.task_object import TaskObject
from flowcept.commons.utils import replace_non_serializable
from flowcept.commons.vocabulary import Status
from flowcept.configs import REPLACE_NON_JSON_SERIALIZABLE
from flowcept.flowceptor.adapters.base_interceptor import BaseInterceptor


class TimmInterceptor(BaseInterceptor):
    """Interceptor that transparently captures ViT attention via a forward hook.

    Unlike the Dask adapter (which observes an external distributed
    cluster), this adapter observes a single, local PyTorch module: attach
    it once to a timm VisionTransformer instance, and every subsequent
    `model(x)` call transparently emits a Flowcept task carrying the
    captured patch-attention scores -- no per-call instrumentation needed
    at the application level.

    Examples
    --------
    >>> from flowcept import Flowcept
    >>> from flowcept.flowceptor.adapters.timm.timm_interceptor import TimmInterceptor
    >>> TimmInterceptor.get_instance().attach(native_vision_transformer)
    >>> with Flowcept("timm", workflow_name="my_workflow"):
    ...     output = native_vision_transformer(x)  # captured automatically
    >>> TimmInterceptor.get_instance().detach_all()
    """

    _instance: Optional["TimmInterceptor"] = None

    def __init__(self, plugin_key=None, kind="timm"):
        """Initialize the interceptor.

        `plugin_key=None` (matching `InstrumentationInterceptor`'s own
        pattern) means this works with zero settings.yaml configuration;
        pass `plugin_key="timm"` explicitly if you've added a `timm:` entry
        under `adapters:` in your settings.yaml and want it honored.
        """
        super().__init__(plugin_key=plugin_key, kind=kind)
        self._hook_handles = []

    @classmethod
    def get_instance(cls) -> "TimmInterceptor":
        """Get the singleton instance for this interceptor."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def attach(self, vit_module, activity_id: Optional[str] = None, reduction: Optional[str] = None,
               indices=1):
        """Attach this interceptor to a native timm VisionTransformer instance.

        Parameters
        ----------
        vit_module : torch.nn.Module
            A timm VisionTransformer (or API-compatible module) exposing
            `set_attn_capture`/`get_patch_attention_scores`. Pass the
            *native* module -- unwrap `torch.compile` and any
            `features_only`/`FeatureGetterNet` wrapping yourself first; this
            adapter is intentionally agnostic to any specific wrapping
            scheme so it stays reusable across projects.
        activity_id : str, optional
            Name recorded for captured tasks. Defaults to the module's class
            name (e.g. "VisionTransformer").
        reduction : str, optional
            Passed straight through to `get_patch_attention_scores` on every
            forward call ("incoming_mean" or "cls"). Defaults to
            `self.settings.reduction` if configured via settings.yaml,
            otherwise "incoming_mean".
        indices : int or list[int], optional
            Which transformer blocks to capture, same semantics as
            `set_attn_capture`/`get_patch_attention_scores` (None -> all
            blocks, int -> last n, list -> specific indices). Defaults to
            `1` (last block only) -- capturing every block by default would
            force the non-fused attention path everywhere, which is
            unnecessary overhead for a single summary signal per forward
            pass.

        Returns
        -------
        A forward-hook handle. Call `.remove()` on it, or use
        `detach_all()`, to stop capturing.

        Raises
        ------
        AttributeError
            If `vit_module` doesn't expose the required timm-instrumentation
            API. Checked here at attach-time, not at import time, so this
            adapter has no hard dependency on any specific timm build.
        """
        if not hasattr(vit_module, "set_attn_capture") or not hasattr(vit_module, "get_patch_attention_scores"):
            raise AttributeError(
                "TimmInterceptor.attach() requires a module exposing set_attn_capture()/"
                "get_patch_attention_scores() (the timm-instrumentation fork's "
                "VisionTransformer API) -- got a plain/unmodified module instead."
            )
        if reduction is None:
            reduction = getattr(self.settings, "reduction", None) or "incoming_mean"

        vit_module.set_attn_capture(True, indices=indices)
        resolved_activity_id = activity_id or type(vit_module).__name__
        handle = vit_module.register_forward_hook(
            self._make_hook(vit_module, resolved_activity_id, reduction, indices)
        )
        self._hook_handles.append(handle)
        return handle

    def detach_all(self):
        """Remove every hook registered via `attach()`.

        Does not disable capture on the underlying module (call
        `vit_module.set_attn_capture(False)` yourself if you also want that).
        """
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _make_hook(self, vit_module, activity_id, reduction, indices):
        def hook(module, args, output):
            self.callback(vit_module, args, output, activity_id, reduction, indices)

        return hook

    def prepare_task_msg(self, module, args, output, activity_id, reduction, indices) -> TaskObject:
        """Build a TaskObject from one completed `forward()` call.

        Overrides `BaseInterceptor.prepare_task_msg` (which otherwise raises
        `NotImplementedError`) with the params `callback`/`_make_hook`
        actually provide, rather than the generic `*args, **kwargs`.
        """
        from flowcept.flowcept_api.flowcept_controller import Flowcept
        from flowcept.instrumentation.flowcept_task import (
            get_current_context_agent_id,
            get_current_context_campaign_id,
            get_current_context_task_id,
            get_current_context_workflow_id,
        )

        started_at = time()
        task_msg = TaskObject()
        task_msg.task_id = str(started_at)
        task_msg.activity_id = activity_id
        # Prefer the enclosing task's workflow/campaign over Flowcept's class attributes: those
        # track whichever controller started last, which is the wrong workflow when one long-lived
        # controller drives several (e.g. a per-plan workflow under a persistent agent workflow).
        task_msg.workflow_id = get_current_context_workflow_id() or Flowcept.current_workflow_id
        task_msg.campaign_id = get_current_context_campaign_id() or Flowcept.campaign_id
        # Nest under whatever task the forward pass ran inside, when there is one, so captured
        # attention is a child of the enclosing task rather than a bare sibling under the workflow.
        task_msg.parent_task_id = get_current_context_task_id()
        # Same rationale: attribute the capture to whichever agent is driving the enclosing task.
        task_msg.agent_id = get_current_context_agent_id()
        task_msg.started_at = started_at
        task_msg.status = Status.FINISHED

        used = {}
        if args and hasattr(args[0], "shape"):
            used["input_shape"] = list(args[0].shape)
        task_msg.used = used

        scores = module.get_patch_attention_scores(indices=indices, reduction=reduction)
        # Flatten the common case (a single captured block) to a plain list of
        # floats, matching the shape produced by the other two examples in
        # examples/vit/ (extract_patch_attention()) for direct comparability.
        generated = {"patch_attention": scores[0] if len(scores) == 1 else scores}
        if REPLACE_NON_JSON_SERIALIZABLE:
            generated = replace_non_serializable(generated)
        task_msg.generated = generated

        if self.telemetry_capture is not None:
            task_msg.telemetry_at_end = self.telemetry_capture.capture()
        task_msg.ended_at = time()

        return task_msg

    def callback(self, module, args, output, activity_id, reduction, indices):
        """Decide what to do when a forward() call completes.

        Always "interesting" here (every forward pass gets a task) -- there's
        no filtering analogous to Dask's task-state-transition branching,
        since a completed forward pass is unconditionally worth recording.
        """
        task_msg = self.prepare_task_msg(module, args, output, activity_id, reduction, indices)
        self.intercept(task_msg.to_dict())

    def observe(self, *args, **kwargs):
        """Not applicable: this adapter is event-driven (forward hooks), not a polling observer."""
        raise NotImplementedError(
            "TimmInterceptor captures via PyTorch forward hooks registered in attach(), "
            "not a continuous observe() loop."
        )
