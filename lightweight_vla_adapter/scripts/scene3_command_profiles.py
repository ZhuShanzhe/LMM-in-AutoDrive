"""Offline Scene-3 command profiles used only for training-label conversion.

These profiles are NOT part of the online policy chain.  The online chain uses
the generic instruction FSM; this module exists so the offline data conversion
scripts can reproduce the same deterministic labels used during V6 training.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


COMMAND_PROFILES = {
    "scene3_cruise": {
        "text_en": "Keep the current lane.",
        "action": "keep_lane",
        "target_speed_kmh": 40.0,
    },
    "scene3_general_hazard": {
        "text_en": "Slow down and keep the current lane.",
        "action": "decelerate",
        "target_speed_kmh": 30.0,
    },
    "scene3_cut_in_decelerate": {
        "text_en": "Brake immediately to avoid the vehicle ahead.",
        "action": "decelerate",
        "target_speed_kmh": 18.0,
    },
    "scene3_work_zone_warning": {
        "text_en": "Slow down and keep the current lane.",
        "action": "decelerate",
        "target_speed_kmh": 30.0,
    },
    "scene3_right_lane_closure": {
        "text_en": "Move to the left lane when safe.",
        "action": "lane_change_left",
        "target_speed_kmh": 25.0,
    },
    "scene3_pass_work_zone": {
        "text_en": "Keep the current lane and slow down.",
        "action": "keep_lane",
        "target_speed_kmh": 25.0,
    },
    "scene3_worker_crossing": {
        "text_en": "Brake immediately to avoid the pedestrian ahead.",
        "action": "decelerate",
        "target_speed_kmh": 10.0,
    },
    "scene3_blocked_lane_change_left": {
        "text_en": "Move to the left lane when safe.",
        "action": "lane_change_left",
        "target_speed_kmh": 20.0,
    },
    "scene3_resume_normal_driving": {
        "text_en": "Accelerate to 40 kilometers per hour and keep the current lane.",
        "action": "accelerate",
        "target_speed_kmh": 40.0,
    },
}


def active_text_command(
    commands: Sequence[Mapping[str, Any]],
    progress_m: float,
) -> dict[str, Any]:
    """Return the newest route-triggered command whose window is active."""

    active: dict[str, Any] = {
        "id": "scene3_cruise",
        "trigger_progress_m": 0.0,
        "text": "Continue driving safely in the current lane.",
        "semantic_goal": ["keep_lane"],
    }
    for command in sorted(
        commands,
        key=lambda item: float(item.get("trigger_progress_m", 0.0)),
    ):
        if progress_m + 1e-6 < float(
            command.get("trigger_progress_m", 0.0)
        ):
            break
        end_progress_m = command.get("end_progress_m")
        if (
            end_progress_m is not None
            and progress_m >= float(end_progress_m) - 1e-6
        ):
            continue
        active = dict(command)
    return active
