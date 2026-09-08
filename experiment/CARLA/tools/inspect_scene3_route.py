"""Inspect the real Town05 Scene 3 route before placing event actors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


CARLA_DIR = Path(__file__).resolve().parents[1]
if str(CARLA_DIR) not in sys.path:
    sys.path.insert(0, str(CARLA_DIR))

from carla_bootstrap import setup_carla_api
from scene3_town05_route import build_town05_route_context


DEFAULT_CONFIG = CARLA_DIR / "configs" / "scene_3_emergency_6km_runtime.json"
LOGICAL_LANES = (-1, -2, -3, -4, -5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sample-step-m", type=float, default=25.0)
    parser.add_argument(
        "--details-at",
        type=float,
        nargs="*",
        default=(),
        metavar="METRES",
        help="print actual road/lane ids at selected route distances",
    )
    return parser


def lane_signature(adapter: Any, progress_m: float) -> tuple[int, ...]:
    return tuple(
        lane
        for lane in LOGICAL_LANES
        if adapter.logical_waypoint(lane, progress_m) is not None
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_step_m <= 0.0:
        raise ValueError("--sample-step-m must be positive")

    setup_carla_api()
    import carla

    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    if not world.get_map().name.endswith("Town05_Opt"):
        world = client.load_world("Town05_Opt")
    context = build_town05_route_context(world.get_map(), config["map"]["route"])
    spawn_points = world.get_map().get_spawn_points()
    for key in ("start_spawn_index", "turnaround_spawn_index"):
        spawn_index = int(config["map"]["route"][key])
        transform = spawn_points[spawn_index]
        waypoint = world.get_map().get_waypoint(transform.location)
        nearby = []
        for index, candidate in enumerate(spawn_points):
            distance_m = candidate.location.distance(transform.location)
            if distance_m <= 15.0:
                candidate_waypoint = world.get_map().get_waypoint(
                    candidate.location
                )
                nearby.append(
                    (
                        round(float(distance_m), 2),
                        index,
                        int(candidate_waypoint.road_id),
                        int(candidate_waypoint.lane_id),
                    )
                )
        print(
            f"{key}={spawn_index} actual="
            f"({int(waypoint.road_id)}, {int(waypoint.lane_id)}) "
            f"nearby={sorted(nearby)}"
        )

    ranges: list[tuple[float, float, tuple[int, ...]]] = []
    start = 0.0
    previous = lane_signature(context.adapter, 0.0)
    progress = args.sample_step_m
    while progress <= context.length_m:
        current = lane_signature(context.adapter, progress)
        if current != previous:
            ranges.append((start, progress - args.sample_step_m, previous))
            start = progress
            previous = current
        progress += args.sample_step_m
    ranges.append((start, context.length_m, previous))

    print(f"route_length_m={context.length_m:.1f}")
    for start_m, end_m, lanes in ranges:
        print(f"{start_m:7.1f}-{end_m:7.1f} m lanes={lanes}")
    for progress_m in args.details_at:
        reference = context.adapter.route_waypoint(progress_m)
        mappings = {}
        for logical_lane in LOGICAL_LANES:
            waypoint = context.adapter.logical_waypoint(
                logical_lane,
                progress_m,
            )
            mappings[logical_lane] = (
                None
                if waypoint is None
                else (
                    int(waypoint.road_id),
                    int(waypoint.lane_id),
                    round(float(waypoint.transform.location.x), 2),
                    round(float(waypoint.transform.location.y), 2),
                )
            )
        print(
            f"detail {progress_m:.1f} m "
            f"reference=({int(reference.road_id)}, "
            f"{int(reference.lane_id)}) mappings={mappings}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
