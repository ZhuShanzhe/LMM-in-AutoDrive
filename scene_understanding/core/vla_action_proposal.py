"""Validate action proposals emitted by a multimodal VLA model.

A VlaActionProposal is an untrusted model recommendation.  It must
be validated and passed through a deterministic safety gate before
it can become a ControlDecision or reach the CARLA controller.
"""

from __future__ import annotations

import math
from typing import Any


VLA_ACTION_PROPOSAL_SCHEMA_VERSION = "1.0.0"

VLA_PROPOSAL_STATUSES = {
    "VALID",
    "LOW_CONFIDENCE",
    "INVALID",
    "INCOMPLETE_INPUT",
}

VLA_ACTIONS = {
    "keep_lane",
    "accelerate",
    "decelerate",
    "stop",
    "emergency_brake",
    "lane_change_left",
    "lane_change_right",
    "turn_left",
    "turn_right",
}

EVIDENCE_MODALITIES = {
    "instruction",
    "front_rgb",
    "left_rgb",
    "right_rgb",
    "rear_rgb",
    "lidar",
    "world_state",
}

EXPECTED_FIELDS = {
    "schema_version",
    "proposal_id",
    "request_id",
    "bundle_id",
    "frame_id",
    "simulation_frame",
    "status",
    "action",
    "target_speed_kmh",
    "target_lane",
    "target_location",
    "confidence",
    "model_name",
    "inference_latency_ms",
    "matched_entity_id",
    "evidence_modalities",
    "reason_codes",
}


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _validate_nonempty_string(
    value: Any,
    path: str,
    errors: list[str],
) -> None:
    if not isinstance(value, str) or not value:
        errors.append(
            f"{path}: expected a non-empty string"
        )


def _validate_string_array(
    value: Any,
    path: str,
    errors: list[str],
    *,
    allowed: set[str] | None = None,
) -> None:
    if not isinstance(value, list):
        errors.append(
            f"{path}: expected an array"
        )
        return

    if any(
        not isinstance(item, str) or not item
        for item in value
    ):
        errors.append(
            f"{path}: expected non-empty strings"
        )
        return

    if len(value) != len(set(value)):
        errors.append(
            f"{path}: entries must be unique"
        )

    if allowed is not None:
        unsupported = sorted(
            set(value) - allowed
        )
        if unsupported:
            errors.append(
                f"{path}: unsupported entries "
                + ", ".join(unsupported)
            )


def _validate_target_location(
    value: Any,
    errors: list[str],
) -> None:
    if value is None:
        return

    if (
        not isinstance(value, dict)
        or set(value) != {"x", "y", "z"}
        or any(
            not _is_number(value.get(key))
            for key in ("x", "y", "z")
        )
    ):
        errors.append(
            "target_location: expected null or "
            "a finite x/y/z object"
        )


def _validate_action_invariants(
    data: dict[str, Any],
    errors: list[str],
) -> None:
    action = data.get("action")
    target_lane = data.get("target_lane")
    target_speed = data.get(
        "target_speed_kmh"
    )
    target_location = data.get(
        "target_location"
    )

    if action == "lane_change_left":
        if target_lane != "left":
            errors.append(
                "target_lane: must be 'left' "
                "for lane_change_left"
            )
    elif action == "lane_change_right":
        if target_lane != "right":
            errors.append(
                "target_lane: must be 'right' "
                "for lane_change_right"
            )
    elif target_lane is not None:
        errors.append(
            "target_lane: must be null for "
            "non-lane-change actions"
        )

    if (
        action in {"stop", "emergency_brake"}
        and _is_number(target_speed)
        and float(target_speed) != 0.0
    ):
        errors.append(
            "target_speed_kmh: must be 0 "
            "for stop or emergency_brake"
        )

    if action in {"turn_left", "turn_right"}:
        if target_location is None:
            errors.append(
                "target_location: required for "
                "turn actions"
            )
    elif target_location is not None:
        errors.append(
            "target_location: must be null for "
            "non-turn actions"
        )


def validate_vla_action_proposal(
    data: Any,
) -> list[str]:
    """Return deterministic validation errors for a VLA proposal."""

    if not isinstance(data, dict):
        return ["root: expected an object"]

    errors: list[str] = []

    missing = sorted(
        EXPECTED_FIELDS - data.keys()
    )
    unexpected = sorted(
        data.keys() - EXPECTED_FIELDS
    )

    if missing:
        errors.append(
            "root: missing fields: "
            + ", ".join(missing)
        )
    if unexpected:
        errors.append(
            "root: unexpected fields: "
            + ", ".join(unexpected)
        )

    if (
        data.get("schema_version")
        != VLA_ACTION_PROPOSAL_SCHEMA_VERSION
    ):
        errors.append(
            "schema_version: expected "
            f"{VLA_ACTION_PROPOSAL_SCHEMA_VERSION!r}"
        )

    for key in (
        "proposal_id",
        "request_id",
        "bundle_id",
        "frame_id",
        "model_name",
    ):
        _validate_nonempty_string(
            data.get(key),
            key,
            errors,
        )

    simulation_frame = data.get(
        "simulation_frame"
    )
    if (
        isinstance(simulation_frame, bool)
        or not isinstance(
            simulation_frame,
            int,
        )
        or simulation_frame < 0
    ):
        errors.append(
            "simulation_frame: expected a "
            "non-negative integer"
        )

    if (
        data.get("status")
        not in VLA_PROPOSAL_STATUSES
    ):
        errors.append(
            "status: invalid value"
        )

    if data.get("action") not in VLA_ACTIONS:
        errors.append(
            "action: invalid value"
        )

    target_speed = data.get(
        "target_speed_kmh"
    )
    if (
        not _is_number(target_speed)
        or not 0 <= float(target_speed) <= 100
    ):
        errors.append(
            "target_speed_kmh: expected a finite "
            "number between 0 and 100"
        )

    target_lane = data.get("target_lane")
    if target_lane not in {
        None,
        "left",
        "right",
    }:
        errors.append(
            "target_lane: expected null, "
            "'left', or 'right'"
        )

    _validate_target_location(
        data.get("target_location"),
        errors,
    )

    confidence = data.get("confidence")
    if (
        not _is_number(confidence)
        or not 0 <= float(confidence) <= 1
    ):
        errors.append(
            "confidence: expected a finite number "
            "between 0 and 1"
        )

    latency = data.get(
        "inference_latency_ms"
    )
    if (
        not _is_number(latency)
        or float(latency) < 0
    ):
        errors.append(
            "inference_latency_ms: expected a "
            "finite non-negative number"
        )

    matched_entity_id = data.get(
        "matched_entity_id"
    )
    if (
        matched_entity_id is not None
        and (
            not isinstance(
                matched_entity_id,
                str,
            )
            or not matched_entity_id
        )
    ):
        errors.append(
            "matched_entity_id: expected null "
            "or a non-empty string"
        )

    _validate_string_array(
        data.get("evidence_modalities"),
        "evidence_modalities",
        errors,
        allowed=EVIDENCE_MODALITIES,
    )
    _validate_string_array(
        data.get("reason_codes"),
        "reason_codes",
        errors,
    )

    _validate_action_invariants(
        data,
        errors,
    )

    return errors


def ensure_valid_vla_action_proposal(
    data: Any,
) -> dict[str, Any]:
    """Return the proposal or raise ValueError when it is invalid."""

    errors = validate_vla_action_proposal(
        data
    )
    if errors:
        raise ValueError(
            "invalid VlaActionProposal: "
            + "; ".join(errors)
        )
    return data
