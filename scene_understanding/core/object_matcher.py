"""Normalize instruction references and match them to WorldState entities.

The functions in this module are deterministic and independent of CARLA.  An
upstream command parser may pass either plain text (for example ``"前车"``) or
a small mapping containing fields such as ``target_object`` or ``intent``.
"""

from __future__ import annotations

from typing import Any, Mapping


TARGET_TYPES = {
    "pedestrian",
    "cyclist",
    "front_vehicle",
    "slow_vehicle",
    "vehicle",
    "obstacle",
    "traffic_cone",
    "road_hazard",
    "left_lane",
    "right_lane",
    "junction",
    "traffic_light",
    "traffic_sign",
    "unknown",
}

# More specific phrases must appear before their shorter forms.
REFERENCE_ALIASES: tuple[tuple[str, str], ...] = (
    ("change_lane_left", "left_lane"),
    ("向左变道", "left_lane"),
    ("左侧车道", "left_lane"),
    ("左车道", "left_lane"),
    ("left lane", "left_lane"),
    ("change_lane_right", "right_lane"),
    ("向右变道", "right_lane"),
    ("右侧车道", "right_lane"),
    ("右车道", "right_lane"),
    ("right lane", "right_lane"),
    ("intersection", "junction"),
    ("junction", "junction"),
    ("十字路口", "junction"),
    ("路口", "junction"),
    ("slow vehicle", "slow_vehicle"),
    ("slow car", "slow_vehicle"),
    ("低速车辆", "slow_vehicle"),
    ("慢车", "slow_vehicle"),
    ("front vehicle", "front_vehicle"),
    ("vehicle ahead", "front_vehicle"),
    ("前方车辆", "front_vehicle"),
    ("前车", "front_vehicle"),
    ("pedestrian", "pedestrian"),
    ("walker", "pedestrian"),
    ("行人", "pedestrian"),
    ("traffic light", "traffic_light"),
    ("红绿灯", "traffic_light"),
    ("信号灯", "traffic_light"),
    ("traffic sign", "traffic_sign"),
    ("交通标志", "traffic_sign"),
    ("标志牌", "traffic_sign"),
    ("vehicle", "vehicle"),
    ("车辆", "vehicle"),
)

REFERENCE_FIELDS = (
    "target_object",
    "object",
    "target",
    "entity",
    "reference_text",
    "raw_text",
    "text",
    "command",
    "intent",
)


def _reference_text(reference_input: str | Mapping[str, Any]) -> str:
    if isinstance(reference_input, str):
        return reference_input.strip()
    if not isinstance(reference_input, Mapping):
        raise TypeError("reference input must be text or a mapping")
    for key in REFERENCE_FIELDS:
        value = reference_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalize_instruction_reference(
    reference_input: str | Mapping[str, Any],
) -> dict[str, str]:
    """Return the team-plan target vocabulary plus position/lane hints."""

    raw_text = _reference_text(reference_input)
    normalized = raw_text.lower().replace("-", " ")
    target_type = "unknown"
    for alias, candidate in REFERENCE_ALIASES:
        if alias in normalized:
            target_type = candidate
            break

    if target_type == "left_lane":
        position_hint = "left"
        lane_hint = "left_adjacent_lane"
    elif target_type == "right_lane":
        position_hint = "right"
        lane_hint = "right_adjacent_lane"
    elif target_type == "front_vehicle":
        position_hint = "front"
        lane_hint = "ego_lane"
    else:
        if any(token in normalized for token in ("左侧", "左边", "left")):
            position_hint = "left"
        elif any(token in normalized for token in ("右侧", "右边", "right")):
            position_hint = "right"
        elif any(token in normalized for token in ("后方", "后面", "rear", "behind")):
            position_hint = "rear"
        elif any(token in normalized for token in ("前方", "前面", "front", "ahead")):
            position_hint = "front"
        else:
            position_hint = "unknown"
        lane_hint = "unknown"

    return {
        "raw_text": raw_text,
        "target_type": target_type,
        "position_hint": position_hint,
        "lane_hint": lane_hint,
    }


def relative_position_label(obj: Mapping[str, Any]) -> str:
    """Convert an ego-frame metric position to a compact direction label."""

    position = obj.get("relative_position_ego_m")
    if not isinstance(position, Mapping):
        return "unknown"
    longitudinal = position.get("longitudinal")
    lateral = position.get("lateral")
    if not isinstance(longitudinal, (int, float)) or isinstance(longitudinal, bool):
        return "unknown"
    if not isinstance(lateral, (int, float)) or isinstance(lateral, bool):
        return "unknown"

    # CARLA adapter convention: positive lateral is right.
    if abs(float(lateral)) <= 2.5:
        return "front" if longitudinal >= 0 else "rear"
    if longitudinal >= 0:
        return "front_right" if lateral > 0 else "front_left"
    return "rear_right" if lateral > 0 else "rear_left"


def _candidate_score(
    obj: Mapping[str, Any],
    reference: Mapping[str, str],
) -> tuple[float, float, str]:
    score = 0.0
    relation = obj.get("lane_relation")
    direction = relative_position_label(obj)
    lane_hint = reference["lane_hint"]
    position_hint = reference["position_hint"]
    if lane_hint != "unknown" and relation == lane_hint:
        score -= 100.0
    if position_hint in {
        "front_left",
        "front_right",
        "rear_left",
        "rear_right",
    } and direction == position_hint:
        score -= 50.0
    elif position_hint == "front" and direction.startswith("front"):
        score -= 50.0
    elif position_hint == "rear" and direction.startswith("rear"):
        score -= 50.0
    elif position_hint == "left" and "left" in direction:
        score -= 50.0
    elif position_hint == "right" and "right" in direction:
        score -= 50.0
    distance = obj.get("distance_m")
    numeric_distance = float(distance) if isinstance(distance, (int, float)) else float("inf")
    return score, numeric_distance, str(obj.get("object_id", ""))


def candidate_world_objects(
    reference: Mapping[str, str],
    world_state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return compatible actor candidates in deterministic best-first order."""

    target_type = reference["target_type"]
    categories = {
        "pedestrian": {"pedestrian"},
        "cyclist": {"cyclist"},
        "traffic_light": {"traffic_light"},
        "traffic_sign": {"traffic_sign"},
        "traffic_cone": {"traffic_cone"},
        "obstacle": {
            "road_barrier",
            "traffic_cone",
            "other",
        },
        "road_hazard": {
            "road_barrier",
            "traffic_cone",
            "other",
        },
        "front_vehicle": {"vehicle"},
        "slow_vehicle": {"vehicle"},
        "vehicle": {"vehicle"},
    }.get(target_type)
    if categories is None:
        return []

    candidates: list[dict[str, Any]] = []
    ego_speed = world_state.get("ego", {}).get("speed_mps")
    for obj in world_state.get("objects", []):
        if (
            not isinstance(obj, dict)
            or obj.get("category") not in categories
        ):
            continue

        position_hint = reference["position_hint"]
        if (
            position_hint
            in {
                "front_left",
                "front_right",
                "rear_left",
                "rear_right",
            }
            and relative_position_label(obj) != position_hint
        ):
            continue

        if target_type == "front_vehicle":
            position = obj.get("relative_position_ego_m")
            if not isinstance(position, Mapping) or position.get("longitudinal", 0) <= 0:
                continue
            if obj.get("lane_relation") not in {"ego_lane", "unknown"}:
                continue
            if obj.get("lane_relation") == "unknown" and abs(position.get("lateral", 99)) > 2.5:
                continue
        if target_type == "slow_vehicle":
            speed = obj.get("speed_mps")
            if not isinstance(speed, (int, float)):
                continue
            is_slower_than_ego = isinstance(ego_speed, (int, float)) and speed < ego_speed - 0.5
            if not is_slower_than_ego and speed * 3.6 > 30.0:
                continue
        candidates.append(obj)
    candidates.sort(key=lambda item: _candidate_score(item, reference))
    return candidates


def select_world_object(
    reference: Mapping[str, str],
    world_state: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, int]:
    candidates = candidate_world_objects(reference, world_state)
    return (candidates[0] if candidates else None), len(candidates)
