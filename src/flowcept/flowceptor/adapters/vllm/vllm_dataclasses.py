"""Dataclasses module."""

from dataclasses import dataclass

from flowcept.commons.flowcept_dataclasses.base_settings_dataclasses import (
    BaseSettings,
)


@dataclass
class VLLMSettings(BaseSettings):
    """Settings for the vLLM (KV cache token importance) adapter.

    Note: like the timm adapter, VLLMInterceptor does not require an entry
    here to function -- it defaults to plugin_key=None (same pattern as
    InstrumentationInterceptor) so it works with zero settings.yaml
    configuration. This dataclass exists so `activity_id` (and future
    options) *can* be configured via settings.yaml, following this project's
    "prefer settings.yaml over hardcoded behavior" convention.
    """

    key: str = "vllm"
    kind: str = "vllm"
    activity_id: str = "kv_token_importance"

    def __post_init__(self):
        """Set attributes after init."""
        self.observer_type = "outsourced"
        self.observer_subtype = None
