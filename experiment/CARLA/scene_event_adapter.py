"""Translate runner sensor telemetry into the shared WorldState contract."""

from __future__ import annotations

from typing import Any, Mapping


def scene_sensor_events(monitor_events: Mapping[str, Any] | None) -> dict[str, list[dict[str, Any]]]:
    """Return only newly observed events in ``CarlaWorldStateCollector`` form."""

    events = monitor_events or {}
    collisions = events.get("new_collision_events", [])
    lane_invasions = events.get("new_lane_invasion_events", [])
    return {
        "collisions": _collisions(collisions),
        "lane_invasions": _lane_invasions(lane_invasions),
    }


def _timestamp(event):
    try:
        value = float(event.get("timestamp_s", event.get("time_s", 0.0)))
    except (AttributeError, TypeError, ValueError):
        return 0.0
    return max(0.0, value)


def _frame(event):
    try:
        return int(event.get("frame"))
    except (AttributeError, TypeError, ValueError):
        return None


def _collisions(events):
    if not isinstance(events, list):
        return []
    result = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            continue
        frame = _frame(event)
        other_actor_id = event.get("other_actor_id")
        result.append({
            "event_id": "collision_{0}_{1}".format(frame if frame is not None else "unknown", index),
            "frame": frame,
            "timestamp_s": _timestamp(event),
            "other_actor_id": str(other_actor_id) if other_actor_id is not None else "unknown",
            "normal_impulse_ns": {"x": 0.0, "y": 0.0, "z": 0.0},
            "impulse_magnitude_ns": 0.0,
        })
    return result


def _lane_invasions(events):
    if not isinstance(events, list):
        return []
    result = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            continue
        frame = _frame(event)
        markings = event.get("crossed_lane_markings", event.get("markings", []))
        if not isinstance(markings, list):
            markings = []
        result.append({
            "event_id": "lane_invasion_{0}_{1}".format(frame if frame is not None else "unknown", index),
            "frame": frame,
            "timestamp_s": _timestamp(event),
            "crossed_lane_markings": [str(marking) for marking in markings if str(marking).strip()],
        })
    return result
