"""Collect a schema-valid WorldState from CARLA 0.9.16 actors.

The module intentionally avoids importing :mod:`carla` at import time.  CARLA
objects are accepted through their public Python API, while unit tests use small
fakes and can therefore run on a login node without the simulator installed.
"""

from __future__ import annotations

import fnmatch
from typing import Any, Iterable

from scene_understanding.core.world_state import (
    empty_world_state,
    relative_kinematics,
    validate_world_state,
    vector3,
    vector_speed_mps,
)


def _enum_name(value: Any) -> str:
    """Normalize a CARLA enum such as ``LaneType.Driving`` to ``driving``."""

    if value is None:
        return "unknown"
    text = str(value).strip().split(".")[-1].lower()
    return text or "unknown"


def _carla_vector(value: Any) -> dict[str, float]:
    return vector3(value.x, value.y, value.z)


def _carla_rotation(value: Any) -> dict[str, float]:
    return {
        "pitch": float(value.pitch),
        "yaw": float(value.yaw),
        "roll": float(value.roll),
    }


def _lane_type(value: Any) -> str:
    name = _enum_name(value)
    aliases = {
        "driving": "driving",
        "shoulder": "shoulder",
        "sidewalk": "sidewalk",
        "biking": "biking",
        "parking": "parking",
        "restricted": "restricted",
        "none": "unknown",
    }
    return aliases.get(name, "other" if name != "unknown" else "unknown")


def _lane_change(value: Any) -> str:
    name = _enum_name(value)
    return name if name in {"none", "left", "right", "both"} else "unknown"


def _traffic_light_state(actor: Any) -> str:
    getter = getattr(actor, "get_state", None)
    value = getter() if callable(getter) else getattr(actor, "state", None)
    name = _enum_name(value)
    return name if name in {"red", "yellow", "green", "off"} else "unknown"


def _get_waypoint(carla_map: Any, location: Any, *, project_to_road: bool = True) -> Any:
    try:
        return carla_map.get_waypoint(location, project_to_road=project_to_road)
    except (RuntimeError, TypeError):
        return None


def _waypoint_lane_identity(waypoint: Any) -> tuple[int, int] | None:
    if waypoint is None:
        return None
    road_id = getattr(waypoint, "road_id", None)
    lane_id = getattr(waypoint, "lane_id", None)
    if isinstance(road_id, int) and isinstance(lane_id, int):
        return road_id, lane_id
    return None


def _adjacent_lane_info(waypoint: Any, getter_name: str) -> dict[str, Any] | None:
    getter = getattr(waypoint, getter_name, None)
    lane = getter() if callable(getter) else None
    if lane is None:
        return None
    return {
        "road_id": getattr(lane, "road_id", None),
        "lane_id": getattr(lane, "lane_id", None),
        "lane_type": _lane_type(getattr(lane, "lane_type", None)),
        "is_junction": getattr(lane, "is_junction", None),
    }


def classify_lane_relation(ego_waypoint: Any, object_waypoint: Any) -> str:
    """Classify a target lane using CARLA waypoint identities.

    Only relations directly supported by map topology are assigned.  Ambiguous
    cases remain ``unknown`` for the later semantic-alignment module.
    """

    ego_identity = _waypoint_lane_identity(ego_waypoint)
    object_identity = _waypoint_lane_identity(object_waypoint)
    if ego_identity is None or object_identity is None:
        return "unknown"
    if object_identity == ego_identity:
        return "ego_lane"

    for getter_name, relation in (
        ("get_left_lane", "left_adjacent_lane"),
        ("get_right_lane", "right_adjacent_lane"),
    ):
        getter = getattr(ego_waypoint, getter_name, None)
        adjacent = getter() if callable(getter) else None
        if _waypoint_lane_identity(adjacent) == object_identity:
            return relation

    ego_road, ego_lane = ego_identity
    object_road, object_lane = object_identity
    if ego_road == object_road and ego_lane * object_lane < 0:
        return "oncoming_lane"
    return "unknown"


def _actor_groups(actors: Any) -> Iterable[tuple[Any, str]]:
    """Yield CARLA actors once together with their WorldState category."""

    patterns = (
        ("vehicle.*", "vehicle"),
        ("walker.*", "pedestrian"),
        ("static.prop.*", "other"),
        ("traffic.traffic_light*", "traffic_light"),
        ("traffic.*sign*", "traffic_sign"),
    )
    seen: set[int] = set()
    for pattern, category in patterns:
        if hasattr(actors, "filter"):
            matched = actors.filter(pattern)
        else:
            matched = [
                actor
                for actor in actors
                if fnmatch.fnmatch(getattr(actor, "type_id", ""), pattern)
            ]
        for actor in matched:
            actor_id = getattr(actor, "id", None)
            if not isinstance(actor_id, int) or actor_id in seen:
                continue
            seen.add(actor_id)
            yield actor, category


def _weather_values(weather: Any) -> dict[str, Any]:
    precipitation = float(getattr(weather, "precipitation", 0.0))
    fog_density = float(getattr(weather, "fog_density", 0.0))
    if precipitation > 1.0:
        weather_name = "rain"
    elif fog_density > 10.0:
        weather_name = "fog"
    else:
        weather_name = "clear"

    if fog_density >= 70.0:
        visibility = "poor"
    elif fog_density >= 20.0 or precipitation >= 40.0:
        visibility = "reduced"
    else:
        visibility = "good"

    return {
        "weather": weather_name,
        "visibility": visibility,
        "precipitation_percent": precipitation,
        "fog_density_percent": fog_density,
    }


class CarlaWorldStateCollector:
    """Read one synchronized metric snapshot from a CARLA 0.9.16 world."""

    def __init__(self, world: Any, ego_vehicle: Any, *, max_distance_m: float = 80.0):
        if max_distance_m <= 0:
            raise ValueError("max_distance_m must be positive")
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.max_distance_m = float(max_distance_m)

    def collect(self, *, sensor_events: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
        """Return a validated WorldState for the world's latest completed tick."""

        snapshot = self.world.get_snapshot()
        frame = int(snapshot.frame)
        timestamp_s = float(snapshot.timestamp.elapsed_seconds)

        ego_transform = self.ego_vehicle.get_transform()
        ego_position = _carla_vector(ego_transform.location)
        ego_velocity = _carla_vector(self.ego_vehicle.get_velocity())
        ego_acceleration = _carla_vector(self.ego_vehicle.get_acceleration())
        carla_map = self.world.get_map()
        ego_waypoint = _get_waypoint(carla_map, ego_transform.location)

        ego = {
            "actor_id": str(self.ego_vehicle.id),
            "position_world_m": ego_position,
            "rotation_world_deg": _carla_rotation(ego_transform.rotation),
            "velocity_world_mps": ego_velocity,
            "acceleration_world_mps2": ego_acceleration,
            "speed_mps": vector_speed_mps(ego_velocity),
            "road_id": getattr(ego_waypoint, "road_id", None),
            "section_id": getattr(ego_waypoint, "section_id", None),
            "lane_id": getattr(ego_waypoint, "lane_id", None),
            "lane_type": _lane_type(getattr(ego_waypoint, "lane_type", None)),
            "lane_change": _lane_change(getattr(ego_waypoint, "lane_change", None)),
            "is_junction": getattr(ego_waypoint, "is_junction", None),
            "adjacent_lanes": {
                "left": _adjacent_lane_info(ego_waypoint, "get_left_lane"),
                "right": _adjacent_lane_info(ego_waypoint, "get_right_lane"),
            },
        }
        state = empty_world_state(
            frame_id=f"carla_{frame:08d}",
            simulation_frame=frame,
            timestamp_s=timestamp_s,
            ego=ego,
        )

        for actor, category in _actor_groups(self.world.get_actors()):
            if actor.id == self.ego_vehicle.id or not getattr(actor, "is_alive", True):
                continue
            record = self._collect_actor(
                actor,
                category=category,
                ego_position=ego_position,
                ego_velocity=ego_velocity,
                ego_yaw_deg=ego["rotation_world_deg"]["yaw"],
                ego_waypoint=ego_waypoint,
                carla_map=carla_map,
            )
            if record["distance_m"] <= self.max_distance_m:
                state["objects"].append(record)

        state["objects"].sort(key=lambda item: (item["distance_m"], item["object_id"]))

        weather = _weather_values(self.world.get_weather())
        state["environment"].update(weather)
        state["environment"]["is_intersection"] = ego["is_junction"]
        if sensor_events is not None:
            state["sensor_events"] = {
                "collisions": list(sensor_events.get("collisions", [])),
                "lane_invasions": list(sensor_events.get("lane_invasions", [])),
            }

        errors = validate_world_state(state)
        if errors:
            raise ValueError("invalid collected WorldState: " + "; ".join(errors))
        return state

    def _collect_actor(
        self,
        actor: Any,
        *,
        category: str,
        ego_position: dict[str, float],
        ego_velocity: dict[str, float],
        ego_yaw_deg: float,
        ego_waypoint: Any,
        carla_map: Any,
    ) -> dict[str, Any]:
        transform = actor.get_transform()
        position = _carla_vector(transform.location)
        get_velocity = getattr(actor, "get_velocity", None)
        velocity = _carla_vector(get_velocity()) if callable(get_velocity) else vector3(0, 0, 0)
        kinematics = relative_kinematics(
            ego_position_world_m=ego_position,
            ego_velocity_world_mps=ego_velocity,
            ego_yaw_deg=ego_yaw_deg,
            object_position_world_m=position,
            object_velocity_world_mps=velocity,
        )

        object_waypoint = None
        if category in {"vehicle", "pedestrian"}:
            object_waypoint = _get_waypoint(
                carla_map,
                transform.location,
                project_to_road=category == "vehicle",
            )
        lane_relation = classify_lane_relation(ego_waypoint, object_waypoint)
        if category == "pedestrian" and lane_relation == "unknown":
            position_ego = kinematics["relative_position_ego_m"]
            velocity_ego = kinematics["relative_velocity_ego_mps"]
            lateral = position_ego["lateral"]
            if position_ego["longitudinal"] > 0 and (
                abs(lateral) <= 2.5
                or (abs(lateral) <= 6.0 and lateral * velocity_ego["lateral"] < 0)
            ):
                lane_relation = "crossing_ego_path"
            elif abs(lateral) > 2.5:
                lane_relation = "roadside"

        return {
            "object_id": f"carla_actor_{actor.id}",
            "source_object_id": str(actor.id),
            "category": category,
            "subtype": getattr(actor, "type_id", "unknown") or "unknown",
            "position_world_m": position,
            "velocity_world_mps": velocity,
            "speed_mps": vector_speed_mps(velocity),
            **kinematics,
            "road_id": getattr(object_waypoint, "road_id", None),
            "lane_id": getattr(object_waypoint, "lane_id", None),
            "lane_relation": lane_relation,
            "is_junction": getattr(object_waypoint, "is_junction", None),
            "traffic_light_state": _traffic_light_state(actor)
            if category == "traffic_light"
            else None,
            "semantic_matches": [],
        }
