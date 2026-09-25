"""Lightweight scene-conditioned VLA decision adapter."""

from .src.decision_adapter import (
    ACTION_LABELS,
    AdapterOutput,
    LightweightDecisionAdapter,
    decode_proposal,
)
def __getattr__(name):
    # Model-only checks must not import the external control integration.
    if name == "LightweightVLAPipeline":
        from .src.pipeline import LightweightVLAPipeline
        return LightweightVLAPipeline
    if name in {"advance_vla_control_plan", "gate_vla_proposal"}:
        from .src import safety_bridge
        return getattr(safety_bridge, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ACTION_LABELS",
    "AdapterOutput",
    "LightweightDecisionAdapter",
    "LightweightVLAPipeline",
    "advance_vla_control_plan",
    "decode_proposal",
    "gate_vla_proposal",
]
