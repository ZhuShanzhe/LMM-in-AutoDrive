"""Translate runner sensor telemetry into the shared WorldState contract."""

from __future__ import annotations

from typing import Any, Mapping


def scene_sensor_events(monitor_events: Mapping[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    """Return only newly observed events in ``CarlaWorldStateCollector`` form."""

    events = monitor_events or {}
    collisions = events.get("new_collision_events", [])
    lane_invasions = events.get("new_lane_invasion_events", [])
    return {
        "collisions": list(collisions) if isinstance(collisions, list) else [],
        "lane_invasions": list(lane_invasions) if isinstance(lane_invasions, list) else [],
    }
