from __future__ import annotations

from lightweight_vla_adapter.src.temporal_supervisor import (
    TemporalProposalSupervisor,
)


def proposal(frame: int) -> dict:
    return {
        "schema_version": "1.0.0",
        "request_id": "request-1",
        "frame_id": f"frame-{frame}",
        "action": "keep_lane",
        "target_speed_kmh": 25.0,
        "target_lane": None,
        "target_location": None,
        "target_entity_id": None,
        "confidence": 0.9,
        "model": "test-model",
        "latency_ms": 10.0,
    }


def world(frame: int) -> dict:
    return {
        "frame_id": f"frame-{frame}",
        "timestamp_s": frame * 0.05,
        "ego": {"speed_mps": 0.0},
        "objects": [],
    }


def risk(level: str) -> dict:
    return {
        "risk_level": level,
        "recommended_action": {
            "low": "keep_lane",
            "medium": "decelerate",
            "high": "emergency_brake",
        }[level],
        "lane_change": {
            "left": {"is_safe": level == "low"},
            "right": {"is_safe": level == "low"},
        },
    }


def test_stationary_medium_risk_uses_bounded_caution_crawl() -> None:
    supervisor = TemporalProposalSupervisor()
    result = supervisor.stabilize(
        proposal(1), world(1), risk("medium"), stream_id="stream"
    )
    assert result["action"] == "decelerate"
    assert result["target_speed_kmh"] == 10.0


def test_high_to_medium_transition_can_resume_at_caution_crawl() -> None:
    supervisor = TemporalProposalSupervisor()
    stopped = supervisor.stabilize(
        proposal(1), world(1), risk("high"), stream_id="stream"
    )
    resumed = supervisor.stabilize(
        proposal(2), world(2), risk("medium"), stream_id="stream"
    )
    assert stopped["action"] == "emergency_brake"
    assert stopped["target_speed_kmh"] == 0.0
    assert resumed["action"] == "decelerate"
    assert resumed["target_speed_kmh"] == 10.0


def test_high_risk_never_uses_caution_crawl() -> None:
    result = TemporalProposalSupervisor().stabilize(
        proposal(1), world(1), risk("high"), stream_id="stream"
    )
    assert result["action"] == "emergency_brake"
    assert result["target_speed_kmh"] == 0.0
