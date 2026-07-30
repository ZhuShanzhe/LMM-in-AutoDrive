"""Inspect Scene 2 route topology and command/event placement in CARLA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


CARLA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CARLA_ROOT.parents[1]
for path in (CARLA_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from carla_bootstrap import setup_carla_api
from scenarios.complex.town05_scene2 import (
    build_repeated_route,
    crossing_endpoints,
    distance_2d,
    route_index_at,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CARLA_ROOT / "configs" / "scene_2_town05_runtime.json",
    )
    parser.add_argument(
        "--commands",
        type=Path,
        default=CARLA_ROOT / "configs" / "scene_2_command_suite.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CARLA_ROOT / "outputs" / "scene2_route_audit.json",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    return parser.parse_args()


def _location(waypoint: Any) -> dict[str, float]:
    location = waypoint.transform.location
    return {
        "x": round(float(location.x), 3),
        "y": round(float(location.y), 3),
        "z": round(float(location.z), 3),
    }


def _lane_access(waypoint: Any) -> dict[str, Any]:
    left = waypoint.get_left_lane()
    right = waypoint.get_right_lane()
    return {
        "lane_change": str(waypoint.lane_change).split(".")[-1],
        "left_lane_id": (
            int(left.lane_id) if left is not None else None
        ),
        "left_lane_type": (
            str(left.lane_type).split(".")[-1]
            if left is not None
            else None
        ),
        "right_lane_id": (
            int(right.lane_id) if right is not None else None
        ),
        "right_lane_type": (
            str(right.lane_type).split(".")[-1]
            if right is not None
            else None
        ),
    }


def _maneuvers_ahead(
    route: list[tuple[Any, Any]],
    distances: list[float],
    start_index: int,
    lookahead_m: float = 300.0,
) -> list[dict[str, Any]]:
    start_distance = distances[start_index]
    output = []
    previous = None
    for index in range(start_index, len(route)):
        if distances[index] - start_distance > lookahead_m:
            break
        value = route[index][1]
        option = str(getattr(value, "name", value)).split(".")[-1].upper()
        if option in {"LEFT", "RIGHT", "STRAIGHT"} and option != previous:
            output.append(
                {
                    "option": option,
                    "route_progress_m": round(distances[index], 3),
                    "offset_m": round(distances[index] - start_distance, 3),
                    "location": _location(route[index][0]),
                }
            )
        previous = option
    return output


def _right_lane_change_windows(
    route: list[tuple[Any, Any]],
    distances: list[float],
    start_m: float,
    end_m: float,
) -> list[dict[str, float]]:
    windows = []
    active_start = None
    last_distance = None
    for (waypoint, _), distance in zip(route, distances):
        if distance < start_m or distance > end_m:
            continue
        right = waypoint.get_right_lane()
        permission = str(waypoint.lane_change).split(".")[-1].lower()
        legal = (
            permission in {"right", "both"}
            and right is not None
            and "Driving" in str(right.lane_type)
            and int(right.lane_id) * int(waypoint.lane_id) > 0
        )
        if legal and active_start is None:
            active_start = distance
        if not legal and active_start is not None:
            windows.append(
                {
                    "start_m": round(active_start, 3),
                    "end_m": round(float(last_distance), 3),
                }
            )
            active_start = None
        last_distance = distance
    if active_start is not None and last_distance is not None:
        windows.append(
            {
                "start_m": round(active_start, 3),
                "end_m": round(float(last_distance), 3),
            }
        )
    return windows


def main() -> None:
    args = parse_args()
    setup_carla_api()
    import carla

    config = json.loads(args.config.read_text(encoding="utf-8"))
    suite = json.loads(args.commands.read_text(encoding="utf-8"))
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.load_world(str(config["map"]))
    route_config = config["route"]
    route, distances = build_repeated_route(
        world.get_map(),
        int(route_config["start_spawn_index"]),
        int(route_config["turnaround_spawn_index"]),
        float(route_config["target_length_m"]),
        float(route_config["route_sampling_m"]),
    )
    command_audit = []
    for command in suite["commands"]:
        progress = float(command["announce_at_m"])
        index = route_index_at(distances, progress)
        waypoint = route[index][0]
        command_audit.append(
            {
                "id": command["id"],
                "announce_at_m": progress,
                "route_progress_m": round(distances[index], 3),
                "road_id": int(waypoint.road_id),
                "lane_id": int(waypoint.lane_id),
                "is_junction": bool(waypoint.is_junction),
                "lane_access": _lane_access(waypoint),
                "location": _location(waypoint),
                "maneuvers_next_300m": _maneuvers_ahead(
                    route,
                    distances,
                    index,
                ),
            }
        )

    event_points = []
    for event in config["special_events"]:
        index = route_index_at(
            distances,
            float(event["anchor_progress_m"]),
        )
        event_points.append(
            {
                "id": event["id"],
                "kind": event["kind"],
                "anchor_progress_m": float(event["anchor_progress_m"]),
                "route_progress_m": round(distances[index], 3),
                "location": _location(route[index][0]),
                "_waypoint": route[index][0],
            }
        )
        if event["kind"] == "crossing_pedestrian":
            start, target = crossing_endpoints(route[index][0])
            event_points[-1]["crossing_start"] = {
                "x": round(float(start.x), 3),
                "y": round(float(start.y), 3),
                "z": round(float(start.z), 3),
            }
            event_points[-1]["crossing_target"] = {
                "x": round(float(target.x), 3),
                "y": round(float(target.y), 3),
                "z": round(float(target.z), 3),
            }
    overlaps = []
    for left_index, left in enumerate(event_points):
        for right in event_points[left_index + 1 :]:
            separation = distance_2d(
                left["_waypoint"].transform.location,
                right["_waypoint"].transform.location,
            )
            if separation < 20.0:
                overlaps.append(
                    {
                        "left": left["id"],
                        "right": right["id"],
                        "separation_m": round(separation, 3),
                    }
                )
    for item in event_points:
        item.pop("_waypoint")

    result = {
        "schema_version": "scene2_route_audit/v1",
        "map": world.get_map().name,
        "route_length_m": round(distances[-1], 3),
        "route_points": len(route),
        "route_option_values": sorted(
            {
                str(getattr(option, "name", option))
                for _, option in route
            }
        ),
        "legal_right_lane_change_windows_350_590m": (
            _right_lane_change_windows(route, distances, 350.0, 590.0)
        ),
        "commands": command_audit,
        "events": event_points,
        "event_spatial_overlaps_under_20m": overlaps,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
