"""Stable JSON-file interface for deterministic WorldState risk assessment."""

from __future__ import annotations

from typing import Any

from scene_understanding.core.risk_assessment import assess_world_state, validate_risk_assessment


def assess_scene_risk(world_state: dict[str, Any]) -> dict[str, Any]:
    """Return the validated risk contract consumed by decision modules."""

    result = assess_world_state(world_state)
    errors = validate_risk_assessment(result)
    if errors:
        raise ValueError("invalid risk-assessment output: " + "; ".join(errors))
    return result
