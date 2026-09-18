from __future__ import annotations

from typing import Any


def integration_documents(
    *,
    parser_action: str = "CHANGE_LANE",
    direction: str = "LEFT",
    recommended_action: str = "maintain_speed",
    risk_level: str = "low",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    request_id = "vla-test-0001"
    frame_id = "carla_100"
    driving_intent = {
        "schema_version": "1.2.0",
        "request_id": request_id,
        "input": {
            "modality": "TEXT",
            "language": "en-US",
            "raw_text": "Move to the left lane when safe.",
            "normalized_text": "Move to the left lane when safe.",
        },
        "intent": {
            "category": "BASIC_CONTROL",
            "urgency": "NORMAL",
            "steps": [
                {
                    "step_id": "step_1",
                    "action": parser_action,
                    "parameters": {"direction": direction},
                    "trigger": {"type": "IMMEDIATE"},
                    "depends_on": [],
                    "preconditions": [],
                    "on_blocked": "WAIT_FOR_SAFE",
                }
            ],
            "constraints": {
                "safety_first": True,
                "obey_traffic_rules": True,
                "driving_style": "NORMAL",
            },
        },
        "parse_result": {
            "status": "VALID",
            "method": "HYBRID",
            "model": "modernbert-drive-command-base",
            "confidence": 0.95,
            "missing_slots": [],
            "warnings": [],
            "latency_ms": 10.0,
        },
    }
    world_state = {
        "frame_id": frame_id,
        "timestamp": 5.0,
        "source": "carla_ground_truth",
        "ego": {
            "speed_mps": 5.0,
            "acceleration_mps2": 0.0,
            "yaw_rate_rps": 0.0,
            "speed_limit_mps": 13.9,
            "control": {"steer": 0.0, "throttle": 0.2, "brake": 0.0},
        },
        "objects": [
            {
                "entity_id": "vehicle_front",
                "class": "vehicle",
                "relative_position_m": {"x": 15.0, "y": 0.5, "z": 0.0},
                "relative_velocity_mps": {"x": -1.0, "y": 0.0},
                "lane_relation": "same_lane",
                "confidence": 0.99,
            }
        ],
        "environment": {"at_junction": False},
    }
    semantic_alignment = {
        "schema_version": "1.0.0",
        "request_id": request_id,
        "world_state_frame_id": frame_id,
        "parse_status": "VALID",
        "step_alignments": [
            {
                "step_id": "step_1",
                "alignment_required": False,
                "alignment_success": True,
                "reason_code": "not_required",
                "matched_entity": None,
            }
        ],
    }
    risk_assessment = {
        "schema_version": "1.0.0",
        "frame_id": frame_id,
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "reason_codes": [],
        "lane_change": {
            "left": {"is_safe": True, "reason_codes": ["target_lane_clear"]},
            "right": {"is_safe": True, "reason_codes": ["target_lane_clear"]},
        },
    }
    return driving_intent, world_state, semantic_alignment, risk_assessment


def proposal(
    *,
    action: str = "lane_change_left",
    target_speed_kmh: float = 18.0,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "request_id": "vla-test-0001",
        "frame_id": "carla_100",
        "action": action,
        "target_speed_kmh": target_speed_kmh,
        "target_lane": (
            "left"
            if action == "lane_change_left"
            else "right"
            if action == "lane_change_right"
            else None
        ),
        "target_location": None,
        "target_entity_id": "vehicle_front",
        "confidence": 0.9,
        "model": "student-test",
        "latency_ms": 5.0,
    }
