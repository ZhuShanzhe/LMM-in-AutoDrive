"""WorldState data contract and geometry helpers for the scene-understanding pipeline.

This module deliberately has no CARLA dependency.  The CARLA adapter can import
these helpers later, while schema validation and geometry tests can run on a
login node with only the Python standard library.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "1.0"

TOP_LEVEL_KEYS = {
    "schema_version",
    "frame_id",
    "source",
    "simulation_frame",
    "timestamp_s",
    "coordinate_convention",
    "ego",
    "objects",
    "environment",
    "sensor_events",
    "provenance",
}

COORDINATE_KEYS = {
    "world_frame",
    "ego_frame",
    "distance_unit",
    "speed_unit",
    "acceleration_unit",
}

EGO_KEYS = {
    "actor_id",
    "position_world_m",
    "rotation_world_deg",
    "velocity_world_mps",
    "acceleration_world_mps2",
    "speed_mps",
    "road_id",
    "section_id",
    "lane_id",
    "lane_type",
    "lane_change",
    "is_junction",
    "adjacent_lanes",
}

ADJACENT_LANE_KEYS = {"road_id", "lane_id", "lane_type", "is_junction"}

OBJECT_KEYS = {
    "object_id",
    "source_object_id",
    "category",
    "subtype",
    "position_world_m",
    "velocity_world_mps",
    "speed_mps",
    "relative_position_ego_m",
    "relative_velocity_ego_mps",
    "distance_m",
    "relative_longitudinal_speed_mps",
    "closing_speed_mps",
    "road_id",
    "lane_id",
    "lane_relation",
    "is_junction",
    "traffic_light_state",
    "semantic_matches",
}

SEMANTIC_MATCH_KEYS = {
    "camera_name",
    "visual_object_id",
    "bbox_2d",
    "description",
    "confidence",
}

ENVIRONMENT_KEYS = {
    "weather",
    "visibility",
    "road_type",
    "is_intersection",
    "precipitation_percent",
    "fog_density_percent",
    "scene_summary",
}

SENSOR_EVENT_KEYS = {"collisions", "lane_invasions"}
COLLISION_KEYS = {
    "event_id",
    "frame",
    "timestamp_s",
    "other_actor_id",
    "normal_impulse_ns",
    "impulse_magnitude_ns",
}
LANE_INVASION_KEYS = {
    "event_id",
    "frame",
    "timestamp_s",
    "crossed_lane_markings",
}
PROVENANCE_KEYS = {
    "metric_source",
    "semantic_source",
    "camera_names",
}

SOURCES = {"carla", "nuscenes", "waymo", "other"}
WORLD_FRAMES = {"carla_world", "dataset_world", "unavailable"}
LANE_TYPES = {
    "driving",
    "shoulder",
    "sidewalk",
    "biking",
    "parking",
    "restricted",
    "other",
    "unknown",
}
LANE_CHANGES = {"none", "left", "right", "both", "unknown"}
CATEGORIES = {
    "vehicle",
    "pedestrian",
    "cyclist",
    "motorcycle",
    "traffic_light",
    "traffic_sign",
    "road_barrier",
    "traffic_cone",
    "animal",
    "other",
    "unknown",
}
LANE_RELATIONS = {
    "ego_lane",
    "left_adjacent_lane",
    "right_adjacent_lane",
    "oncoming_lane",
    "crossing_ego_path",
    "roadside",
    "unknown",
}
TRAFFIC_LIGHT_STATES = {"red", "yellow", "green", "off", "unknown"}
WEATHER_VALUES = {"clear", "rain", "fog", "snow", "unknown"}
VISIBILITY_VALUES = {"good", "reduced", "poor", "unknown"}
ROAD_TYPES = {"urban", "residential", "highway", "rural", "parking", "unknown"}
METRIC_SOURCES = {"carla_actor_api", "dataset_annotation", "unavailable"}


def _is_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _exact_keys(value: Any, required: set[str], path: str, errors: list[str]) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{path}: expected an object")
        return False
    missing = sorted(required - value.keys())
    extra = sorted(value.keys() - required)
    if missing:
        errors.append(f"{path}: missing fields: {', '.join(missing)}")
    if extra:
        errors.append(f"{path}: unexpected fields: {', '.join(extra)}")
    return not missing and not extra


def _nonempty_string(value: Any, path: str, errors: list[str], *, allow_null: bool = False) -> None:
    if allow_null and value is None:
        return
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path}: expected a non-empty string" + (" or null" if allow_null else ""))


def _number(
    value: Any,
    path: str,
    errors: list[str],
    *,
    allow_null: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    if allow_null and value is None:
        return
    if not _is_number(value):
        errors.append(f"{path}: expected a finite number" + (" or null" if allow_null else ""))
        return
    number = float(value)
    if minimum is not None and number < minimum:
        errors.append(f"{path}: must be at least {minimum}")
    if maximum is not None and number > maximum:
        errors.append(f"{path}: must be at most {maximum}")


def _integer_or_null(value: Any, path: str, errors: list[str]) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        errors.append(f"{path}: expected an integer or null")


def _enum(value: Any, allowed: set[str], path: str, errors: list[str], *, allow_null: bool = False) -> None:
    if allow_null and value is None:
        return
    if value not in allowed:
        errors.append(f"{path}: invalid value {value!r}; allowed: {', '.join(sorted(allowed))}")


def _vector3(value: Any, path: str, errors: list[str], *, allow_null: bool = False) -> None:
    if allow_null and value is None:
        return
    if not isinstance(value, dict) or set(value) != {"x", "y", "z"}:
        errors.append(f"{path}: expected exactly x, y and z" + (" or null" if allow_null else ""))
        return
    for axis in ("x", "y", "z"):
        _number(value[axis], f"{path}.{axis}", errors)


def _ego_vector3(value: Any, path: str, errors: list[str], *, allow_null: bool = False) -> None:
    if allow_null and value is None:
        return
    expected = {"longitudinal", "lateral", "vertical"}
    if not isinstance(value, dict) or set(value) != expected:
        errors.append(
            f"{path}: expected exactly longitudinal, lateral and vertical"
            + (" or null" if allow_null else "")
        )
        return
    for axis in ("longitudinal", "lateral", "vertical"):
        _number(value[axis], f"{path}.{axis}", errors)


def _rotation(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, dict) or set(value) != {"pitch", "yaw", "roll"}:
        errors.append(f"{path}: expected exactly pitch, yaw and roll")
        return
    for axis in ("pitch", "yaw", "roll"):
        _number(value[axis], f"{path}.{axis}", errors)


def _adjacent_lanes(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, dict) or set(value) != {"left", "right"}:
        errors.append(f"{path}: expected exactly left and right")
        return
    for direction in ("left", "right"):
        lane = value[direction]
        lane_path = f"{path}.{direction}"
        if lane is None:
            continue
        if not _exact_keys(lane, ADJACENT_LANE_KEYS, lane_path, errors):
            if not isinstance(lane, dict):
                continue
        _integer_or_null(lane.get("road_id"), f"{lane_path}.road_id", errors)
        _integer_or_null(lane.get("lane_id"), f"{lane_path}.lane_id", errors)
        _enum(lane.get("lane_type"), LANE_TYPES, f"{lane_path}.lane_type", errors)
        if lane.get("is_junction") is not None and not isinstance(
            lane.get("is_junction"), bool
        ):
            errors.append(f"{lane_path}.is_junction: expected true, false, or null")


def _bbox(value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(value, list) or len(value) != 4:
        errors.append(f"{path}: expected four normalized coordinates")
        return
    if any(not _is_number(item) for item in value):
        errors.append(f"{path}: every coordinate must be a finite number")
        return
    if any(item < 0 or item > 1 for item in value):
        errors.append(f"{path}: every coordinate must be between 0 and 1")
        return
    if value[0] >= value[2] or value[1] >= value[3]:
        errors.append(f"{path}: expected x_min < x_max and y_min < y_max")


def vector3(x: float, y: float, z: float) -> dict[str, float]:
    """Return a JSON-ready finite world-frame vector."""

    values = (float(x), float(y), float(z))
    if not all(math.isfinite(item) for item in values):
        raise ValueError("vector coordinates must be finite")
    return {"x": values[0], "y": values[1], "z": values[2]}


def vector_speed_mps(value: Mapping[str, float]) -> float:
    """Return Euclidean speed from an x/y/z velocity in metres per second."""

    return math.sqrt(sum(float(value[axis]) ** 2 for axis in ("x", "y", "z")))


def world_vector_to_ego(value: Mapping[str, float], ego_yaw_deg: float) -> dict[str, float]:
    """Rotate a CARLA world-frame vector into the ego frame.

    The output convention is longitudinal=forward, lateral=right and
    vertical=up.  CARLA yaw is in degrees in its Z-up left-handed frame.
    """

    yaw = math.radians(float(ego_yaw_deg))
    x = float(value["x"])
    y = float(value["y"])
    z = float(value["z"])
    return {
        "longitudinal": x * math.cos(yaw) + y * math.sin(yaw),
        "lateral": -x * math.sin(yaw) + y * math.cos(yaw),
        "vertical": z,
    }


def relative_kinematics(
    *,
    ego_position_world_m: Mapping[str, float],
    ego_velocity_world_mps: Mapping[str, float],
    ego_yaw_deg: float,
    object_position_world_m: Mapping[str, float],
    object_velocity_world_mps: Mapping[str, float],
) -> dict[str, Any]:
    """Compute geometry needed later by TTC and lane-safety rules.

    ``relative_longitudinal_speed_mps`` is object minus ego along the ego's
    forward axis.  A negative value means ego is catching an object ahead.
    ``closing_speed_mps`` is the radial approach speed and is positive only
    while the 3-D separation is shrinking.
    """

    relative_world_position = {
        axis: float(object_position_world_m[axis]) - float(ego_position_world_m[axis])
        for axis in ("x", "y", "z")
    }
    relative_world_velocity = {
        axis: float(object_velocity_world_mps[axis]) - float(ego_velocity_world_mps[axis])
        for axis in ("x", "y", "z")
    }
    relative_position = world_vector_to_ego(relative_world_position, ego_yaw_deg)
    relative_velocity = world_vector_to_ego(relative_world_velocity, ego_yaw_deg)
    distance = vector_speed_mps(relative_world_position)

    if distance <= 1e-9:
        closing_speed = 0.0
    else:
        separation_rate = sum(
            relative_world_position[axis] * relative_world_velocity[axis]
            for axis in ("x", "y", "z")
        ) / distance
        closing_speed = max(0.0, -separation_rate)

    return {
        "relative_position_ego_m": relative_position,
        "relative_velocity_ego_mps": relative_velocity,
        "distance_m": distance,
        "relative_longitudinal_speed_mps": relative_velocity["longitudinal"],
        "closing_speed_mps": closing_speed,
    }


def empty_world_state(
    *,
    frame_id: str,
    simulation_frame: int,
    timestamp_s: float,
    ego: dict[str, Any],
) -> dict[str, Any]:
    """Create the fixed CARLA WorldState envelope for one simulator tick."""

    return {
        "schema_version": SCHEMA_VERSION,
        "frame_id": frame_id,
        "source": "carla",
        "simulation_frame": simulation_frame,
        "timestamp_s": timestamp_s,
        "coordinate_convention": {
            "world_frame": "carla_world",
            "ego_frame": "x_forward_y_right_z_up",
            "distance_unit": "m",
            "speed_unit": "mps",
            "acceleration_unit": "mps2",
        },
        "ego": ego,
        "objects": [],
        "environment": {
            "weather": "unknown",
            "visibility": "unknown",
            "road_type": "unknown",
            "is_intersection": ego.get("is_junction"),
            "precipitation_percent": None,
            "fog_density_percent": None,
            "scene_summary": None,
        },
        "sensor_events": {"collisions": [], "lane_invasions": []},
        "provenance": {
            "metric_source": "carla_actor_api",
            "semantic_source": None,
            "camera_names": [],
        },
    }


def validate_world_state(data: Any) -> list[str]:
    """Return validation errors; an empty list means the WorldState is valid."""

    errors: list[str] = []
    if not _exact_keys(data, TOP_LEVEL_KEYS, "root", errors):
        if not isinstance(data, dict):
            return errors

    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version: expected {SCHEMA_VERSION!r}")
    _nonempty_string(data.get("frame_id"), "frame_id", errors)
    _enum(data.get("source"), SOURCES, "source", errors)
    _integer_or_null(data.get("simulation_frame"), "simulation_frame", errors)
    _number(data.get("timestamp_s"), "timestamp_s", errors, allow_null=True, minimum=0)

    coordinates = data.get("coordinate_convention")
    if _exact_keys(coordinates, COORDINATE_KEYS, "coordinate_convention", errors):
        _enum(coordinates["world_frame"], WORLD_FRAMES, "coordinate_convention.world_frame", errors)
        if coordinates["ego_frame"] != "x_forward_y_right_z_up":
            errors.append("coordinate_convention.ego_frame: expected 'x_forward_y_right_z_up'")
        for key, expected in (
            ("distance_unit", "m"),
            ("speed_unit", "mps"),
            ("acceleration_unit", "mps2"),
        ):
            if coordinates[key] != expected:
                errors.append(f"coordinate_convention.{key}: expected {expected!r}")
    if data.get("source") == "carla" and isinstance(coordinates, dict):
        if coordinates.get("world_frame") != "carla_world":
            errors.append("coordinate_convention.world_frame: CARLA source requires 'carla_world'")

    ego = data.get("ego")
    if _exact_keys(ego, EGO_KEYS, "ego", errors):
        _nonempty_string(ego["actor_id"], "ego.actor_id", errors)
        _vector3(ego["position_world_m"], "ego.position_world_m", errors)
        _rotation(ego["rotation_world_deg"], "ego.rotation_world_deg", errors)
        _vector3(ego["velocity_world_mps"], "ego.velocity_world_mps", errors)
        _vector3(ego["acceleration_world_mps2"], "ego.acceleration_world_mps2", errors)
        _number(ego["speed_mps"], "ego.speed_mps", errors, minimum=0)
        for key in ("road_id", "section_id", "lane_id"):
            _integer_or_null(ego[key], f"ego.{key}", errors)
        _enum(ego["lane_type"], LANE_TYPES, "ego.lane_type", errors)
        _enum(ego["lane_change"], LANE_CHANGES, "ego.lane_change", errors)
        if ego["is_junction"] is not None and not isinstance(ego["is_junction"], bool):
            errors.append("ego.is_junction: expected true, false, or null")
        _adjacent_lanes(ego["adjacent_lanes"], "ego.adjacent_lanes", errors)

    objects = data.get("objects")
    object_ids: set[str] = set()
    if not isinstance(objects, list):
        errors.append("objects: expected an array")
    else:
        for index, obj in enumerate(objects):
            path = f"objects[{index}]"
            if not _exact_keys(obj, OBJECT_KEYS, path, errors):
                if not isinstance(obj, dict):
                    continue
            object_id = obj.get("object_id")
            _nonempty_string(object_id, f"{path}.object_id", errors)
            if isinstance(object_id, str):
                if object_id in object_ids:
                    errors.append(f"{path}.object_id: duplicate ID {object_id}")
                object_ids.add(object_id)
            _nonempty_string(obj.get("source_object_id"), f"{path}.source_object_id", errors, allow_null=True)
            _enum(obj.get("category"), CATEGORIES, f"{path}.category", errors)
            _nonempty_string(obj.get("subtype"), f"{path}.subtype", errors)
            _vector3(obj.get("position_world_m"), f"{path}.position_world_m", errors, allow_null=True)
            _vector3(obj.get("velocity_world_mps"), f"{path}.velocity_world_mps", errors, allow_null=True)
            _number(obj.get("speed_mps"), f"{path}.speed_mps", errors, allow_null=True, minimum=0)
            _ego_vector3(
                obj.get("relative_position_ego_m"),
                f"{path}.relative_position_ego_m",
                errors,
                allow_null=True,
            )
            _ego_vector3(
                obj.get("relative_velocity_ego_mps"),
                f"{path}.relative_velocity_ego_mps",
                errors,
                allow_null=True,
            )
            _number(obj.get("distance_m"), f"{path}.distance_m", errors, allow_null=True, minimum=0)
            _number(
                obj.get("relative_longitudinal_speed_mps"),
                f"{path}.relative_longitudinal_speed_mps",
                errors,
                allow_null=True,
            )
            _number(
                obj.get("closing_speed_mps"),
                f"{path}.closing_speed_mps",
                errors,
                allow_null=True,
                minimum=0,
            )
            for key in ("road_id", "lane_id"):
                _integer_or_null(obj.get(key), f"{path}.{key}", errors)
            _enum(obj.get("lane_relation"), LANE_RELATIONS, f"{path}.lane_relation", errors)
            if obj.get("is_junction") is not None and not isinstance(obj.get("is_junction"), bool):
                errors.append(f"{path}.is_junction: expected true, false, or null")
            _enum(
                obj.get("traffic_light_state"),
                TRAFFIC_LIGHT_STATES,
                f"{path}.traffic_light_state",
                errors,
                allow_null=True,
            )
            if obj.get("category") != "traffic_light" and obj.get("traffic_light_state") is not None:
                errors.append(f"{path}.traffic_light_state: only traffic_light objects may have a state")

            matches = obj.get("semantic_matches")
            match_ids: set[tuple[str, str]] = set()
            if not isinstance(matches, list):
                errors.append(f"{path}.semantic_matches: expected an array")
            else:
                for match_index, match in enumerate(matches):
                    match_path = f"{path}.semantic_matches[{match_index}]"
                    if not _exact_keys(match, SEMANTIC_MATCH_KEYS, match_path, errors):
                        if not isinstance(match, dict):
                            continue
                    _nonempty_string(match.get("camera_name"), f"{match_path}.camera_name", errors)
                    _nonempty_string(
                        match.get("visual_object_id"), f"{match_path}.visual_object_id", errors
                    )
                    if isinstance(match.get("camera_name"), str) and isinstance(
                        match.get("visual_object_id"), str
                    ):
                        key = (match["camera_name"], match["visual_object_id"])
                        if key in match_ids:
                            errors.append(f"{match_path}: duplicate camera/object match {key}")
                        match_ids.add(key)
                    _bbox(match.get("bbox_2d"), f"{match_path}.bbox_2d", errors)
                    _nonempty_string(
                        match.get("description"), f"{match_path}.description", errors, allow_null=True
                    )
                    _number(
                        match.get("confidence"),
                        f"{match_path}.confidence",
                        errors,
                        allow_null=True,
                        minimum=0,
                        maximum=1,
                    )

    environment = data.get("environment")
    if _exact_keys(environment, ENVIRONMENT_KEYS, "environment", errors):
        _enum(environment["weather"], WEATHER_VALUES, "environment.weather", errors)
        _enum(environment["visibility"], VISIBILITY_VALUES, "environment.visibility", errors)
        _enum(environment["road_type"], ROAD_TYPES, "environment.road_type", errors)
        if environment["is_intersection"] is not None and not isinstance(
            environment["is_intersection"], bool
        ):
            errors.append("environment.is_intersection: expected true, false, or null")
        for key in ("precipitation_percent", "fog_density_percent"):
            _number(
                environment[key],
                f"environment.{key}",
                errors,
                allow_null=True,
                minimum=0,
                maximum=100,
            )
        _nonempty_string(
            environment["scene_summary"], "environment.scene_summary", errors, allow_null=True
        )

    sensor_events = data.get("sensor_events")
    if _exact_keys(sensor_events, SENSOR_EVENT_KEYS, "sensor_events", errors):
        collisions = sensor_events["collisions"]
        if not isinstance(collisions, list):
            errors.append("sensor_events.collisions: expected an array")
        else:
            for index, event in enumerate(collisions):
                path = f"sensor_events.collisions[{index}]"
                if not _exact_keys(event, COLLISION_KEYS, path, errors):
                    if not isinstance(event, dict):
                        continue
                _nonempty_string(event.get("event_id"), f"{path}.event_id", errors)
                _integer_or_null(event.get("frame"), f"{path}.frame", errors)
                _number(event.get("timestamp_s"), f"{path}.timestamp_s", errors, minimum=0)
                _nonempty_string(event.get("other_actor_id"), f"{path}.other_actor_id", errors)
                _vector3(event.get("normal_impulse_ns"), f"{path}.normal_impulse_ns", errors)
                _number(
                    event.get("impulse_magnitude_ns"),
                    f"{path}.impulse_magnitude_ns",
                    errors,
                    minimum=0,
                )

        invasions = sensor_events["lane_invasions"]
        if not isinstance(invasions, list):
            errors.append("sensor_events.lane_invasions: expected an array")
        else:
            for index, event in enumerate(invasions):
                path = f"sensor_events.lane_invasions[{index}]"
                if not _exact_keys(event, LANE_INVASION_KEYS, path, errors):
                    if not isinstance(event, dict):
                        continue
                _nonempty_string(event.get("event_id"), f"{path}.event_id", errors)
                _integer_or_null(event.get("frame"), f"{path}.frame", errors)
                _number(event.get("timestamp_s"), f"{path}.timestamp_s", errors, minimum=0)
                markings = event.get("crossed_lane_markings")
                if not isinstance(markings, list) or any(
                    not isinstance(item, str) or not item.strip() for item in markings
                ):
                    errors.append(f"{path}.crossed_lane_markings: expected an array of strings")

    provenance = data.get("provenance")
    if _exact_keys(provenance, PROVENANCE_KEYS, "provenance", errors):
        _enum(provenance["metric_source"], METRIC_SOURCES, "provenance.metric_source", errors)
        _nonempty_string(
            provenance["semantic_source"], "provenance.semantic_source", errors, allow_null=True
        )
        camera_names = provenance["camera_names"]
        if not isinstance(camera_names, list) or any(
            not isinstance(item, str) or not item.strip() for item in camera_names
        ):
            errors.append("provenance.camera_names: expected an array of strings")
        elif len(camera_names) != len(set(camera_names)):
            errors.append("provenance.camera_names: duplicate camera names")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="WorldState JSON file")
    args = parser.parse_args()
    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"INVALID JSON: {exc}", file=sys.stderr)
        return 1
    errors = validate_world_state(data)
    if errors:
        print("INVALID WORLD STATE", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("VALID WORLD STATE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
