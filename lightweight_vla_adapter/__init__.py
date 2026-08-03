"""Lightweight scene-conditioned VLA decision adapter.

JSON contracts and the VLA-first coordinator can be imported on deployment
hosts that do not install PyTorch.  Neural inference symbols become available
when the optional ML runtime is installed.
"""

from .src.contracts import ACTION_LABELS
from .src.decision_coordinator import (
    CoordinatorConfig,
    VLAFirstDecisionCoordinator,
    infer_input_health,
    validate_input_health,
)
from .src.safety_bridge import advance_vla_control_plan, gate_vla_proposal

try:
    from .src.decision_adapter import (
        AdapterOutput,
        LightweightDecisionAdapter,
        decode_proposal,
    )
    from .src.pipeline import LightweightVLAPipeline
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    AdapterOutput = None  # type: ignore[assignment]
    LightweightDecisionAdapter = None  # type: ignore[assignment]
    LightweightVLAPipeline = None  # type: ignore[assignment]
    decode_proposal = None  # type: ignore[assignment]

__all__ = [
    "ACTION_LABELS",
    "AdapterOutput",
    "LightweightDecisionAdapter",
    "LightweightVLAPipeline",
    "CoordinatorConfig",
    "VLAFirstDecisionCoordinator",
    "infer_input_health",
    "validate_input_health",
    "advance_vla_control_plan",
    "decode_proposal",
    "gate_vla_proposal",
]
