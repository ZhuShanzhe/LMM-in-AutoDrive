"""Run a closed-loop slow-vehicle overtake experiment in CARLA 0.9.16.

The experiment resumes an ACTIVE step_3 plan with the ego vehicle already in
the passing lane.  It completes only after the grounded slow vehicle moves
from ahead of ego to a measured rear clearance while ego remains centred and
aligned in the passing lane without a collision.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driving-intent", required=True, type=Path)
    parser.add_argument("--initial-state", required=True, type=Path)
    parser.add_argument(
        "--scenario-root",
        required=True,
        type=Path,
        help="directory containing the team scenarios package",
    )
    parser.add_argument(
        "--control-root",
        required=True,
        type=Path,
        help="directory containing the team control package",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--spawn-index", type=int, default=60)
    parser.add_argument("--initial-lead-distance-m", type=float, default=12.0)
    parser.add_argument("--rear-clearance-m", type=float, default=8.0)
    parser.add_argument("--ego-speed-kmh", type=float, default=25.0)
    parser.add_argument("--slow-vehicle-speed-kmh", type=float, default=12.0)
    parser.add_argument("--maximum-ego-speed-kmh", type=float, default=40.0)
    parser.add_argument("--fixed-delta-s", type=float, default=0.05)
    parser.add_argument("--maximum-lane-center-offset-m", type=float, default=0.9)
    parser.add_argument("--minimum-heading-alignment", type=float, default=0.95)
    parser.add_argument("--required-stable-frames", type=int, default=5)
    parser.add_argument("--maximum-ticks", type=int, default=300)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False) + "\n")


def _speed_mps(actor: Any) -> float:
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def _world_object(world_state: dict[str, Any], object_id: str) -> dict[str, Any] | None:
    return next(
        (item for item in world_state["objects"] if item["object_id"] == object_id),
        None,
    )


def _alignment_entity_id(
    semantic_alignment: dict[str, Any], step_id: str
) -> str | None:
    matches = [
        item
        for item in semantic_alignment["step_alignments"]
        if item.get("step_id") == step_id
    ]
    if len(matches) != 1:
        return None
    entity = matches[0].get("matched_entity")
    return entity.get("entity_id") if isinstance(entity, dict) else None


def overtake_step_completed(
    *,
    observed_slow_vehicle_ahead: bool,
    slow_vehicle_present: bool,
    slow_vehicle_longitudinal_m: float | None,
    rear_clearance_m: float,
    current_lane_id: int | None,
    passing_lane_id: int,
    lane_center_offset_m: float | None,
    maximum_lane_center_offset_m: float,
    heading_alignment: float | None,
    minimum_heading_alignment: float,
    stable_frames: int,
    required_stable_frames: int,
    collision_count: int,
) -> tuple[bool, list[str]]:
    """Evaluate measurable completion conditions for the final plan step."""

    if collision_count:
        return False, ["collision_detected"]
    if not observed_slow_vehicle_ahead:
        return False, ["slow_vehicle_was_not_observed_ahead"]
    if not slow_vehicle_present:
        return False, ["slow_vehicle_not_in_world_state"]

    reasons: list[str] = []
    if slow_vehicle_longitudinal_m is None:
        reasons.append("slow_vehicle_longitudinal_position_unavailable")
    elif slow_vehicle_longitudinal_m > -rear_clearance_m:
        reasons.append("rear_clearance_not_reached")
    if current_lane_id != passing_lane_id:
        reasons.append("ego_left_passing_lane")
    if lane_center_offset_m is None:
        reasons.append("lane_center_offset_unavailable")
    elif lane_center_offset_m > maximum_lane_center_offset_m:
        reasons.append("ego_not_centered_in_passing_lane")
    if heading_alignment is None:
        reasons.append("lane_heading_alignment_unavailable")
    elif heading_alignment < minimum_heading_alignment:
        reasons.append("ego_not_aligned_with_passing_lane")
    if stable_frames < required_stable_frames:
        reasons.append("overtake_clearance_not_stable")
    if reasons:
        return False, reasons
    return True, [
        "slow_vehicle_passed_with_rear_clearance",
        "ego_stable_in_passing_lane",
        "slow_vehicle_grounded",
        "collision_free",
    ]


def _step_feedback(
    *,
    request_id: str,
    frame_id: str,
    step_id: str,
    outcome: str,
    reason_codes: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "request_id": request_id,
        "frame_id": frame_id,
        "step_id": step_id,
        "outcome": outcome,
        "reason_codes": reason_codes,
    }


def _forward_speed(carla_module: Any, actor: Any, speed_kmh: float) -> None:
    speed_mps = speed_kmh / 3.6
    forward = actor.get_transform().get_forward_vector()
    actor.set_target_velocity(
        carla_module.Vector3D(
            x=forward.x * speed_mps,
            y=forward.y * speed_mps,
            z=0.0,
        )
    )


def _spawn_transform(carla_module: Any, waypoint: Any) -> Any:
    location = waypoint.transform.location
    return carla_module.Transform(
        carla_module.Location(
            x=location.x,
            y=location.y,
            z=location.z + 0.6,
        ),
        waypoint.transform.rotation,
    )


def _lane_metrics(carla_map: Any, ego: Any) -> dict[str, Any]:
    transform = ego.get_transform()
    waypoint = carla_map.get_waypoint(
        transform.location,
        project_to_road=True,
    )
    if waypoint is None:
        return {
            "road_id": None,
            "lane_id": None,
            "center_offset_m": None,
            "heading_alignment": None,
        }
    dx = transform.location.x - waypoint.transform.location.x
    dy = transform.location.y - waypoint.transform.location.y
    ego_forward = transform.get_forward_vector()
    lane_forward = waypoint.transform.get_forward_vector()
    return {
        "road_id": int(waypoint.road_id),
        "lane_id": int(waypoint.lane_id),
        "center_offset_m": float(math.hypot(dx, dy)),
        "heading_alignment": float(
            ego_forward.x * lane_forward.x
            + ego_forward.y * lane_forward.y
            + ego_forward.z * lane_forward.z
        ),
    }


def _speed_capped_decision(
    decision: dict[str, Any],
    *,
    current_speed_kmh: float,
    maximum_speed_kmh: float,
) -> dict[str, Any]:
    """Cap overtake acceleration without weakening a safety override."""

    result = copy.deepcopy(decision)
    if (
        result.get("action") == "accelerate"
        and current_speed_kmh >= maximum_speed_kmh
    ):
        result.update(
            {
                "decision_status": "READY",
                "action": "keep_lane",
                "target_speed_kmh": maximum_speed_kmh,
                "target_lane": None,
                "target_location": None,
                "emergency": False,
                "reason": "overtake_speed_cap",
                "blocked_reason_codes": [],
            }
        )
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fixed_delta_s <= 0:
        raise SystemExit("--fixed-delta-s must be positive")
    if args.initial_lead_distance_m <= 0:
        raise SystemExit("--initial-lead-distance-m must be positive")
    if args.rear_clearance_m <= 0:
        raise SystemExit("--rear-clearance-m must be positive")
    if args.ego_speed_kmh <= args.slow_vehicle_speed_kmh:
        raise SystemExit("--ego-speed-kmh must exceed --slow-vehicle-speed-kmh")
    if args.slow_vehicle_speed_kmh < 0:
        raise SystemExit("--slow-vehicle-speed-kmh must be non-negative")
    if args.maximum_ego_speed_kmh <= args.ego_speed_kmh:
        raise SystemExit("--maximum-ego-speed-kmh must exceed --ego-speed-kmh")
    if args.maximum_lane_center_offset_m <= 0:
        raise SystemExit("--maximum-lane-center-offset-m must be positive")
    if not 0 < args.minimum_heading_alignment <= 1:
        raise SystemExit("--minimum-heading-alignment must be in (0, 1]")
    if args.required_stable_frames <= 0:
        raise SystemExit("--required-stable-frames must be positive")
    if args.maximum_ticks <= 0:
        raise SystemExit("--maximum-ticks must be positive")

    scenario_root = args.scenario_root.resolve()
    control_root = args.control_root.resolve()
    if not (scenario_root / "scenarios").is_dir():
        raise SystemExit(f"scenario package not found under {scenario_root}")
    if not (control_root / "control").is_dir():
        raise SystemExit(f"control package not found under {control_root}")
    sys.path.insert(0, str(scenario_root))
    sys.path.insert(0, str(control_root))

    import carla
    from control.pid_controller import EgoPIDController
    from scenarios.utils.vehicle import VehicleSpawner

    from scene_understanding.core.carla_sensor_manager import CarlaSensorManager
    from scene_understanding.core.carla_world_state import CarlaWorldStateCollector
    from scene_understanding.src.control_plan_executor import advance_control_plan
    from scene_understanding.src.driving_intent_alignment import align_driving_intent
    from scene_understanding.src.risk_interface import assess_scene_risk

    driving_intent = _read_json(args.driving_intent)
    state = _read_json(args.initial_state)
    if state.get("plan_status") != "ACTIVE" or state.get("active_step_id") != "step_3":
        raise SystemExit("initial state must have ACTIVE step_3")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    timeline_path = output_dir / "timeline.jsonl"

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    original_settings = world.get_settings()
    ego = None
    slow_vehicle = None
    sensors = None
    latest_alignment = None
    latest_risk = None
    latest_decision = None
    feedback = None
    result = "FAILED"
    error_message = None

    dynamic_actors = list(world.get_actors().filter("vehicle.*"))
    dynamic_actors += list(world.get_actors().filter("walker.pedestrian.*"))
    dynamic_actors += list(world.get_actors().filter("sensor.*"))
    if dynamic_actors:
        raise SystemExit(
            "CARLA world is not clean; existing dynamic actor IDs: "
            + ", ".join(str(actor.id) for actor in dynamic_actors)
        )

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = args.fixed_delta_s
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()
        if not 0 <= args.spawn_index < len(spawn_points):
            raise ValueError(
                f"spawn index {args.spawn_index} is outside 0..{len(spawn_points) - 1}"
            )
        base_waypoint = carla_map.get_waypoint(
            spawn_points[args.spawn_index].location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if base_waypoint is None:
            raise RuntimeError("selected spawn point has no driving waypoint")
        passing_waypoint = base_waypoint.get_left_lane()
        if (
            passing_waypoint is None
            or passing_waypoint.lane_type != carla.LaneType.Driving
        ):
            raise RuntimeError("selected spawn point has no left passing lane")
        base_forward = base_waypoint.transform.get_forward_vector()
        passing_forward = passing_waypoint.transform.get_forward_vector()
        direction_dot = (
            base_forward.x * passing_forward.x
            + base_forward.y * passing_forward.y
            + base_forward.z * passing_forward.z
        )
        if direction_dot < 0.8:
            raise RuntimeError("selected passing lane is not in the ego direction")

        slow_candidates = base_waypoint.next(args.initial_lead_distance_m)
        if not slow_candidates:
            raise RuntimeError("selected road is too short for the slow vehicle")

        spawner = VehicleSpawner(world)
        ego = spawner.spawn_ego_vehicle(_spawn_transform(carla, passing_waypoint))
        slow_vehicle = spawner.spawn_npc_vehicle(
            _spawn_transform(carla, slow_candidates[0])
        )
        sensors = CarlaSensorManager(
            world,
            ego,
            output_dir=output_dir / "sensors",
            enable_camera=False,
        )
        sensors.setup()
        collector = CarlaWorldStateCollector(world, ego, max_distance_m=100.0)
        ego_controller = EgoPIDController(
            ego,
            carla_map,
            target_speed_kmh=args.ego_speed_kmh,
        )
        slow_controller = EgoPIDController(
            slow_vehicle,
            carla_map,
            target_speed_kmh=args.slow_vehicle_speed_kmh,
        )

        _forward_speed(carla, ego, args.ego_speed_kmh)
        _forward_speed(carla, slow_vehicle, args.slow_vehicle_speed_kmh)
        world.tick()

        passing_lane_id = int(passing_waypoint.lane_id)
        slow_vehicle_id = f"carla_actor_{slow_vehicle.id}"
        observed_slow_vehicle_ahead = False
        collision_count = 0
        lane_invasion_count = 0
        stable_frames = 0
        peak_ego_speed_mps = _speed_mps(ego)
        initial_longitudinal_m = None
        final_longitudinal_m = None

        slow_intent = {
            "action": "keep_lane",
            "target_speed_kmh": args.slow_vehicle_speed_kmh,
            "target_lane": None,
            "target_location": None,
            "emergency": False,
            "reason": "slow_vehicle_speed_hold",
        }

        for controlled_tick in range(1, args.maximum_ticks + 1):
            slow_control, _ = slow_controller.run_step(
                slow_intent, args.fixed_delta_s
            )
            slow_vehicle.apply_control(slow_control)
            frame = world.tick()
            events = sensors.drain_events_through(frame)
            collision_count += len(events["collisions"])
            lane_invasion_count += len(events["lane_invasions"])
            world_state = collector.collect(sensor_events=events)
            latest_alignment = align_driving_intent(driving_intent, world_state)
            latest_risk = assess_scene_risk(world_state)

            matched_entity_id = _alignment_entity_id(latest_alignment, "step_3")
            slow_object = _world_object(world_state, slow_vehicle_id)
            slow_longitudinal = None
            if slow_object is not None:
                position = slow_object.get("relative_position_ego_m")
                if isinstance(position, dict):
                    value = position.get("longitudinal")
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        slow_longitudinal = float(value)
            if initial_longitudinal_m is None:
                initial_longitudinal_m = slow_longitudinal
            if matched_entity_id == slow_vehicle_id and (
                slow_longitudinal is not None and slow_longitudinal > 0
            ):
                observed_slow_vehicle_ahead = True

            lane = _lane_metrics(carla_map, ego)
            passed_and_stable = (
                slow_longitudinal is not None
                and slow_longitudinal <= -args.rear_clearance_m
                and lane["lane_id"] == passing_lane_id
                and lane["center_offset_m"] is not None
                and lane["center_offset_m"] <= args.maximum_lane_center_offset_m
                and lane["heading_alignment"] is not None
                and lane["heading_alignment"] >= args.minimum_heading_alignment
            )
            stable_frames = stable_frames + 1 if passed_and_stable else 0
            complete, completion_reasons = overtake_step_completed(
                observed_slow_vehicle_ahead=observed_slow_vehicle_ahead,
                slow_vehicle_present=slow_object is not None,
                slow_vehicle_longitudinal_m=slow_longitudinal,
                rear_clearance_m=args.rear_clearance_m,
                current_lane_id=lane["lane_id"],
                passing_lane_id=passing_lane_id,
                lane_center_offset_m=lane["center_offset_m"],
                maximum_lane_center_offset_m=args.maximum_lane_center_offset_m,
                heading_alignment=lane["heading_alignment"],
                minimum_heading_alignment=args.minimum_heading_alignment,
                stable_frames=stable_frames,
                required_stable_frames=args.required_stable_frames,
                collision_count=collision_count,
            )

            if collision_count:
                feedback = _step_feedback(
                    request_id=driving_intent["request_id"],
                    frame_id=world_state["frame_id"],
                    step_id="step_3",
                    outcome="FAILED",
                    reason_codes=["collision_detected"],
                )
                state, latest_decision = advance_control_plan(
                    driving_intent,
                    world_state,
                    latest_alignment,
                    latest_risk,
                    prior_state=state,
                    feedback=feedback,
                )
                result = "FAILED"
            elif complete:
                feedback = _step_feedback(
                    request_id=driving_intent["request_id"],
                    frame_id=world_state["frame_id"],
                    step_id="step_3",
                    outcome="COMPLETED",
                    reason_codes=completion_reasons,
                )
                state, latest_decision = advance_control_plan(
                    driving_intent,
                    world_state,
                    latest_alignment,
                    latest_risk,
                    prior_state=state,
                    feedback=feedback,
                )
                result = "COMPLETED"
            else:
                state, latest_decision = advance_control_plan(
                    driving_intent,
                    world_state,
                    latest_alignment,
                    latest_risk,
                    prior_state=state,
                )

            terminal_before_completion = (
                not collision_count
                and not complete
                and state["plan_status"] != "ACTIVE"
            )
            if terminal_before_completion:
                feedback = _step_feedback(
                    request_id=driving_intent["request_id"],
                    frame_id=world_state["frame_id"],
                    step_id="step_3",
                    outcome="FAILED",
                    reason_codes=[
                        "plan_terminal_before_step_completion",
                        f"plan_status_{state['plan_status'].lower()}",
                    ],
                )
                result = "FAILED"

            current_speed_mps = float(world_state["ego"]["speed_mps"])
            peak_ego_speed_mps = max(peak_ego_speed_mps, current_speed_mps)
            applied_decision = _speed_capped_decision(
                latest_decision,
                current_speed_kmh=current_speed_mps * 3.6,
                maximum_speed_kmh=args.maximum_ego_speed_kmh,
            )
            ego_control, normalized = ego_controller.run_step(
                applied_decision, args.fixed_delta_s
            )
            ego.apply_control(ego_control)
            final_longitudinal_m = slow_longitudinal
            record = {
                "controlled_tick": controlled_tick,
                "frame_id": world_state["frame_id"],
                "ego_speed_mps": round(current_speed_mps, 6),
                "peak_ego_speed_mps": round(peak_ego_speed_mps, 6),
                "slow_vehicle_speed_mps": round(_speed_mps(slow_vehicle), 6),
                "slow_vehicle_id": slow_vehicle_id,
                "slow_vehicle_distance_m": (
                    None if slow_object is None else slow_object.get("distance_m")
                ),
                "slow_vehicle_longitudinal_m": slow_longitudinal,
                "slow_vehicle_lane_relation": (
                    None if slow_object is None else slow_object.get("lane_relation")
                ),
                "matched_entity_id": matched_entity_id,
                "observed_slow_vehicle_ahead": observed_slow_vehicle_ahead,
                "passing_lane_id": passing_lane_id,
                "current_road_id": lane["road_id"],
                "current_lane_id": lane["lane_id"],
                "lane_center_offset_m": lane["center_offset_m"],
                "heading_alignment": lane["heading_alignment"],
                "stable_frames": stable_frames,
                "collision_count": collision_count,
                "lane_invasion_count": lane_invasion_count,
                "risk_level": latest_risk["risk_level"],
                "plan_decision_status": latest_decision["decision_status"],
                "plan_action": latest_decision["action"],
                "applied_action": normalized["action"],
                "speed_cap_active": normalized["reason"] == "overtake_speed_cap",
                "throttle": float(ego_control.throttle),
                "brake": float(ego_control.brake),
                "steer": float(ego_control.steer),
                "completion_ready": complete,
                "completion_checks": completion_reasons,
            }
            _append_jsonl(timeline_path, record)

            if collision_count or complete or terminal_before_completion:
                break
        else:
            world_state = collector.collect(
                sensor_events=sensors.drain_events_through(world.get_snapshot().frame)
            )
            latest_alignment = align_driving_intent(driving_intent, world_state)
            latest_risk = assess_scene_risk(world_state)
            feedback = _step_feedback(
                request_id=driving_intent["request_id"],
                frame_id=world_state["frame_id"],
                step_id="step_3",
                outcome="FAILED",
                reason_codes=["overtake_completion_timeout"],
            )
            state, latest_decision = advance_control_plan(
                driving_intent,
                world_state,
                latest_alignment,
                latest_risk,
                prior_state=state,
                feedback=feedback,
            )
            result = "FAILED"

        _write_json(output_dir / "step_feedback.json", feedback)
        _write_json(output_dir / "control_plan_state.json", state)
        _write_json(output_dir / "control_decision.json", latest_decision)
        _write_json(output_dir / "semantic_alignment.json", latest_alignment)
        _write_json(output_dir / "risk_assessment.json", latest_risk)
        final_lane = _lane_metrics(carla_map, ego)
        summary = {
            "schema_version": "1.0.0",
            "result": result,
            "request_id": driving_intent["request_id"],
            "spawn_index": args.spawn_index,
            "ego_actor_id": str(ego.id),
            "slow_vehicle_actor_id": str(slow_vehicle.id),
            "passing_lane_id": passing_lane_id,
            "final_road_id": final_lane["road_id"],
            "final_lane_id": final_lane["lane_id"],
            "final_lane_center_offset_m": final_lane["center_offset_m"],
            "final_heading_alignment": final_lane["heading_alignment"],
            "initial_slow_vehicle_longitudinal_m": initial_longitudinal_m,
            "final_slow_vehicle_longitudinal_m": final_longitudinal_m,
            "required_rear_clearance_m": args.rear_clearance_m,
            "peak_ego_speed_mps": round(peak_ego_speed_mps, 6),
            "stable_frames": stable_frames,
            "observed_slow_vehicle_ahead": observed_slow_vehicle_ahead,
            "collision_count": collision_count,
            "lane_invasion_count": lane_invasion_count,
            "feedback_outcome": feedback["outcome"],
            "feedback_reason_codes": feedback["reason_codes"],
            "final_plan_status": state["plan_status"],
            "active_step_id": state["active_step_id"],
        }
        _write_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Evidence: {output_dir}")
        return 0 if result == "COMPLETED" else 1
    except Exception as error:
        error_message = f"{type(error).__name__}: {error}"
        print(f"ERROR: {error_message}", file=sys.stderr)
        return 1
    finally:
        if error_message is not None:
            _write_json(output_dir / "error.json", {"error": error_message})
        if sensors is not None:
            sensors.destroy()
        for actor in (slow_vehicle, ego):
            if actor is not None and actor.is_alive:
                actor.destroy()
        world.apply_settings(original_settings)


if __name__ == "__main__":
    raise SystemExit(main())
