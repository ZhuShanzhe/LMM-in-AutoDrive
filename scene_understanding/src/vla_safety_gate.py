"""Convert an untrusted VLA proposal into a safety-gated decision."""

from __future__ import annotations

import math
from typing import Any, Mapping

from scene_understanding.core.multimodal_frame_bundle import (
    validate_multimodal_frame_bundle,
)
from scene_understanding.core.vla_action_proposal import (
    ensure_valid_vla_action_proposal,
)
from scene_understanding.src.control_decision import (
    CONTROL_DECISION_SCHEMA_VERSION,
    validate_control_decision,
)


RISK_LEVELS = {
    "none",
    "low",
    "medium",
    "high",
}

RISK_ACTIONS = {
    "maintain_speed",
    "monitor",
    "decelerate",
    "emergency_brake",
}


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _current_speed_kmh(
    world_state: Mapping[str, Any],
) -> float:
    ego = world_state.get("ego")
    if not isinstance(ego, Mapping):
        raise ValueError(
            "WorldState ego must be an object"
        )

    speed_mps = ego.get("speed_mps")
    if (
        not _is_number(speed_mps)
        or float(speed_mps) < 0
    ):
        raise ValueError(
            "WorldState ego.speed_mps must be "
            "a finite non-negative number"
        )

    return round(
        min(float(speed_mps) * 3.6, 100.0),
        6,
    )


def _validate_risk_assessment(
    risk: Any,
) -> None:
    if not isinstance(risk, dict):
        raise ValueError(
            "RiskAssessment must be an object"
        )

    if risk.get("risk_level") not in RISK_LEVELS:
        raise ValueError(
            "RiskAssessment risk_level is invalid"
        )

    if (
        risk.get("recommended_action")
        not in RISK_ACTIONS
    ):
        raise ValueError(
            "RiskAssessment recommended_action "
            "is invalid"
        )

    reason_codes = risk.get("reason_codes")
    if (
        not isinstance(reason_codes, list)
        or any(
            not isinstance(item, str)
            or not item
            for item in reason_codes
        )
    ):
        raise ValueError(
            "RiskAssessment reason_codes must "
            "contain non-empty strings"
        )

    lane_change = risk.get("lane_change")
    if not isinstance(lane_change, dict):
        raise ValueError(
            "RiskAssessment lane_change must "
            "be an object"
        )

    for direction in ("left", "right"):
        judgment = lane_change.get(direction)
        if (
            not isinstance(judgment, dict)
            or not isinstance(
                judgment.get("is_safe"),
                bool,
            )
        ):
            raise ValueError(
                "RiskAssessment lane_change."
                f"{direction}.is_safe must be "
                "a boolean"
            )

        lane_reasons = judgment.get(
            "reason_codes"
        )
        if (
            not isinstance(lane_reasons, list)
            or any(
                not isinstance(item, str)
                or not item
                for item in lane_reasons
            )
        ):
            raise ValueError(
                "RiskAssessment lane_change."
                f"{direction}.reason_codes must "
                "contain non-empty strings"
            )


def _make_decision(
    *,
    proposal: Mapping[str, Any],
    bundle: Mapping[str, Any],
    risk: Mapping[str, Any],
    status: str,
    action: str,
    target_speed_kmh: float,
    target_lane: str | None,
    target_location: (
        dict[str, float] | None
    ),
    reason: str,
    blocked_reason_codes: list[str],
    parse_status: str = "VALID",
    matched_entity_id: str | None = None,
) -> dict[str, Any]:
    decision = {
        "schema_version": (
            CONTROL_DECISION_SCHEMA_VERSION
        ),
        "request_id": bundle["request_id"],
        "frame_id": bundle["frame_id"],
        "decision_status": status,
        "action": action,
        "target_speed_kmh": round(
            float(target_speed_kmh),
            6,
        ),
        "target_lane": target_lane,
        "target_location": target_location,
        "emergency": (
            action == "emergency_brake"
        ),
        "reason": reason,
        "parse_status": parse_status,
        "parse_confidence": round(
            float(proposal["confidence"]),
            6,
        ),
        "source_step_id": proposal[
            "proposal_id"
        ],
        "source_step_action": proposal[
            "action"
        ],
        "source_step_count": 1,
        "matched_entity_id": (
            matched_entity_id
        ),
        "risk_level": risk["risk_level"],
        "risk_reason_codes": list(
            risk["reason_codes"]
        ),
        "blocked_reason_codes": list(
            dict.fromkeys(
                blocked_reason_codes
            )
        ),
    }

    errors = validate_control_decision(
        decision
    )
    if errors:
        raise ValueError(
            "invalid safety-gated "
            "ControlDecision: "
            + "; ".join(errors)
        )

    return decision


def _safe_stop(
    *,
    proposal: Mapping[str, Any],
    bundle: Mapping[str, Any],
    risk: Mapping[str, Any],
    reason: str,
    parse_status: str = "INVALID",
    blocked_reason_codes: (
        list[str] | None
    ) = None,
    decision_status: str = "SAFE_FALLBACK",
) -> dict[str, Any]:
    return _make_decision(
        proposal=proposal,
        bundle=bundle,
        risk=risk,
        status=decision_status,
        action="stop",
        target_speed_kmh=0.0,
        target_lane=None,
        target_location=None,
        reason=reason,
        blocked_reason_codes=(
            [reason]
            if blocked_reason_codes is None
            else blocked_reason_codes
        ),
        parse_status=parse_status,
        matched_entity_id=None,
    )


def _available_modalities(
    bundle: Mapping[str, Any],
) -> set[str]:
    available = {
        "instruction",
        "world_state",
    }

    for camera in bundle.get(
        "cameras",
        [],
    ):
        if isinstance(camera, Mapping):
            name = camera.get("sensor_name")
            if isinstance(name, str) and name:
                available.add(name)

    if bundle.get("lidar") is not None:
        available.add("lidar")

    return available


def _identity_matches(
    proposal: Mapping[str, Any],
    bundle: Mapping[str, Any],
    world_state: Mapping[str, Any],
    risk: Mapping[str, Any],
) -> bool:
    return (
        proposal.get("request_id")
        == bundle.get("request_id")
        and proposal.get("bundle_id")
        == bundle.get("bundle_id")
        and proposal.get("frame_id")
        == bundle.get("frame_id")
        and proposal.get("simulation_frame")
        == bundle.get("simulation_frame")
        and world_state.get("frame_id")
        == bundle.get("frame_id")
        and world_state.get(
            "simulation_frame"
        )
        == bundle.get("simulation_frame")
        and risk.get("frame_id")
        == bundle.get("frame_id")
    )


def _matched_entity_exists(
    proposal: Mapping[str, Any],
    world_state: Mapping[str, Any],
) -> bool:
    matched_entity_id = proposal.get(
        "matched_entity_id"
    )
    if matched_entity_id is None:
        return True

    objects = world_state.get("objects")
    if not isinstance(objects, list):
        return False

    return any(
        isinstance(item, Mapping)
        and item.get("object_id")
        == matched_entity_id
        for item in objects
    )


def gate_vla_action_proposal(
    proposal: dict[str, Any],
    multimodal_bundle: dict[str, Any],
    world_state: dict[str, Any],
    risk_assessment: dict[str, Any],
    *,
    min_confidence: float = 0.70,
) -> dict[str, Any]:
    """Return a validated ControlDecision after deterministic gating."""

    ensure_valid_vla_action_proposal(
        proposal
    )

    bundle_errors = (
        validate_multimodal_frame_bundle(
            multimodal_bundle
        )
    )
    if bundle_errors:
        raise ValueError(
            "invalid MultimodalFrameBundle: "
            + "; ".join(bundle_errors)
        )

    if not isinstance(world_state, dict):
        raise ValueError(
            "WorldState must be an object"
        )

    _validate_risk_assessment(
        risk_assessment
    )

    if (
        not _is_number(min_confidence)
        or not 0 <= float(min_confidence) <= 1
    ):
        raise ValueError(
            "min_confidence must be between "
            "0 and 1"
        )

    current_speed = _current_speed_kmh(
        world_state
    )

    if not _identity_matches(
        proposal,
        multimodal_bundle,
        world_state,
        risk_assessment,
    ):
        return _safe_stop(
            proposal=proposal,
            bundle=multimodal_bundle,
            risk=risk_assessment,
            reason=(
                "vla_proposal_identity_mismatch"
            ),
        )

    synchronization = multimodal_bundle[
        "synchronization"
    ]
    if (
        synchronization["status"]
        == "INCOMPLETE"
    ):
        return _safe_stop(
            proposal=proposal,
            bundle=multimodal_bundle,
            risk=risk_assessment,
            reason=(
                "multimodal_bundle_incomplete"
            ),
        )

    proposal_status = proposal["status"]
    if proposal_status != "VALID":
        reason = {
            "LOW_CONFIDENCE": (
                "vla_proposal_low_confidence"
            ),
            "INCOMPLETE_INPUT": (
                "vla_proposal_incomplete_input"
            ),
            "INVALID": (
                "vla_proposal_invalid"
            ),
        }[proposal_status]

        return _safe_stop(
            proposal=proposal,
            bundle=multimodal_bundle,
            risk=risk_assessment,
            reason=reason,
            parse_status=(
                "NEEDS_CLARIFICATION"
                if proposal_status
                == "LOW_CONFIDENCE"
                else "INVALID"
            ),
        )

    if (
        float(proposal["confidence"])
        < float(min_confidence)
    ):
        return _safe_stop(
            proposal=proposal,
            bundle=multimodal_bundle,
            risk=risk_assessment,
            reason=(
                "vla_proposal_low_confidence"
            ),
            parse_status="NEEDS_CLARIFICATION",
        )

    evidence = set(
        proposal["evidence_modalities"]
    )
    available = _available_modalities(
        multimodal_bundle
    )
    required = set(
        synchronization[
            "required_modalities"
        ]
    )

    if not evidence.issubset(available):
        return _safe_stop(
            proposal=proposal,
            bundle=multimodal_bundle,
            risk=risk_assessment,
            reason="vla_evidence_unavailable",
        )

    if not required.issubset(evidence):
        return _safe_stop(
            proposal=proposal,
            bundle=multimodal_bundle,
            risk=risk_assessment,
            reason=(
                "vla_required_evidence_missing"
            ),
        )

    if not _matched_entity_exists(
        proposal,
        world_state,
    ):
        return _safe_stop(
            proposal=proposal,
            bundle=multimodal_bundle,
            risk=risk_assessment,
            reason=(
                "matched_entity_not_in_world_state"
            ),
        )

    recommended = risk_assessment[
        "recommended_action"
    ]

    # Deterministic metric risk has higher priority
    # than every learned VLA proposal.
    if recommended == "emergency_brake":
        return _make_decision(
            proposal=proposal,
            bundle=multimodal_bundle,
            risk=risk_assessment,
            status="BLOCKED",
            action="emergency_brake",
            target_speed_kmh=0.0,
            target_lane=None,
            target_location=None,
            reason=(
                "risk_requires_emergency_brake"
            ),
            blocked_reason_codes=[
                "risk_requires_emergency_brake"
            ],
            matched_entity_id=proposal[
                "matched_entity_id"
            ],
        )

    action = proposal["action"]

    if (
        recommended == "decelerate"
        and action
        not in {
            "decelerate",
            "stop",
            "emergency_brake",
        }
    ):
        return _make_decision(
            proposal=proposal,
            bundle=multimodal_bundle,
            risk=risk_assessment,
            status="BLOCKED",
            action="decelerate",
            target_speed_kmh=current_speed,
            target_lane=None,
            target_location=None,
            reason=(
                "risk_requires_deceleration"
            ),
            blocked_reason_codes=[
                "risk_requires_deceleration"
            ],
            matched_entity_id=proposal[
                "matched_entity_id"
            ],
        )

    if action in {
        "lane_change_left",
        "lane_change_right",
    }:
        direction = action.removeprefix(
            "lane_change_"
        )
        judgment = risk_assessment[
            "lane_change"
        ][direction]

        if not judgment["is_safe"]:
            lane_reasons = list(
                judgment["reason_codes"]
            )
            return _safe_stop(
                proposal=proposal,
                bundle=multimodal_bundle,
                risk=risk_assessment,
                reason=(
                    f"lane_change_{direction}"
                    "_blocked"
                ),
                blocked_reason_codes=(
                    lane_reasons
                    or [
                        "lane_change_"
                        f"{direction}_unsafe"
                    ]
                ),
                decision_status="BLOCKED",
                parse_status="VALID",
            )

    return _make_decision(
        proposal=proposal,
        bundle=multimodal_bundle,
        risk=risk_assessment,
        status="READY",
        action=action,
        target_speed_kmh=proposal[
            "target_speed_kmh"
        ],
        target_lane=proposal[
            "target_lane"
        ],
        target_location=proposal[
            "target_location"
        ],
        reason="vla_proposal_accepted",
        blocked_reason_codes=[],
        matched_entity_id=proposal[
            "matched_entity_id"
        ],
    )
