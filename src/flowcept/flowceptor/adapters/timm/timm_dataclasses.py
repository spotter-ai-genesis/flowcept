"""Dataclasses module."""

from dataclasses import dataclass

from flowcept.commons.flowcept_dataclasses.base_settings_dataclasses import (
    BaseSettings,
)


@dataclass
class TimmSettings(BaseSettings):
    """Settings for the timm (Vision Transformer attention) adapter.

    Note: unlike the Dask/MLflow/Tensorboard adapters, TimmInterceptor does
    not require an entry here to function -- it defaults to plugin_key=None
    (same pattern as InstrumentationInterceptor) so it works with zero
    settings.yaml configuration. This dataclass exists so `reduction` (and
    future options) *can* be configured via settings.yaml for callers who
    want that, following this project's "prefer settings.yaml over hardcoded
    behavior" convention.
    """

    key: str = "timm"
    kind: str = "timm"
    reduction: str = "incoming_mean"

    def __post_init__(self):
        """Set attributes after init."""
        self.observer_type = "outsourced"
        self.observer_subtype = None
