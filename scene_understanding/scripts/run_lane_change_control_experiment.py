"""Run a closed-loop, slow-vehicle left-lane-change experiment in CARLA.

The experiment resumes an ACTIVE step_2 ControlPlanState, creates one slower
vehicle ahead on a map-legal two-lane road, applies the team PID controller,
and emits COMPLETED StepFeedback only after CARLA reports the ego vehicle
settled in the intended adjacent lane without a collision.
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
    parser.add_argument("--spawn-index", type=int, default=1)
    parser.add_argument("--front-distance-m", type=float, default=35.0)
    parser.add_argument("--ego-speed-kmh", type=float, default=30.0)
    parser.add_argument("--slow-vehicle-speed-kmh", type=float, default=12.0)
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


def lane_change_step_completed(
    *,
    observed_slow_vehicle: bool,
    slow_vehicle_present: bool,
    current_lane_id: int | None,
    target_lane_id: int,
    lane_center_offset_m: float | None,
    maximum_lane_center_offset_m: float,
    heading_alignment: float | None,
    minimum_heading_alignment: float,
    stable_frames: int,
    required_stable_frames: int,
    collision_count: int,
) -> tuple[bool, list[str]]:
    """Evaluate conservative physical completion conditions for step 2."""

    if collision_count:
        return False, ["collision_detected"]
    if not observed_slow_vehicle:
        return False, ["slow_vehicle_not_observed"]
    if not slow_vehicle_present:
        return False, ["slow_vehicle_not_in_world_state"]

    reasons: list[str] = []
    # OpenDRIVE road_id can legitimately change at a connected road segment
    # while the physical lane continues.  Lane ID, centre offset and heading
    # together identify successful entry without pinning one road ID.
    if current_lane_id != target_lane_id:
        reasons.append("target_lane_not_reached")
    if lane_center_offset_m is None:
        reasons.append("lane_center_offset_unavailable")
    elif lane_center_offset_m > maximum_lane_center_offset_m:
        reasons.append("ego_not_centered_in_target_lane")
    if heading_alignment is None:
        reasons.append("lane_heading_alignment_unavailable")
    elif heading_alignment < minimum_heading_alignment:
        reasons.append("ego_not_aligned_with_target_lane")
    if stable_frames < required_stable_frames:
        reasons.append("target_lane_not_stable")
    if reasons:
        return False, reasons
    return True, [
        "target_lane_reached",
        "target_lane_centered_and_stable",
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
            "hold_location": None,
        }
    dx = transform.location.x - waypoint.transform.location.x
    dy = transform.location.y - waypoint.transform.location.y
    center_offset = math.hypot(dx, dy)
    ego_forward = transform.get_forward_vector()
    lane_forward = waypoint.transform.get_forward_vector()
    alignment = (
        ego_forward.x * lane_forward.x
        + ego_forward.y * lane_forward.y
        + ego_forward.z * lane_forward.z
    )
    ahead = waypoint.next(12.0)
    hold_location = None
    if ahead:
        location = ahead[0].transform.location
        hold_location = {
            "x": float(location.x),
            "y": float(location.y),
            "z": float(location.z),
        }
    return {
        "road_id": int(waypoint.road_id),
        "lane_id": int(waypoint.lane_id),
        "center_offset_m": float(center_offset),
        "heading_alignment": float(alignment),
        "hold_location": hold_location,
    }


def _target_lane_hold_decision(
    decision: dict[str, Any],
    *,
    current_speed_kmh: float,
    hold_location: dict[str, float] | None,
) -> dict[str, Any]:
    result = copy.deepcopy(decision)
    result["target_lane"] = None
    result["target_location"] = hold_location
    if result.get("action") in {"lane_change_left", "lane_change_right"}:
        result.update(
            {
                "decision_status": "READY",
                "action": "keep_lane",
                "target_speed_kmh": current_speed_kmh,
                "emergency": False,
                "reason": "target_lane_stabilization",
                "blocked_reason_codes": [],
            }
        )
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fixed_delta_s <= 0:
        raise SystemExit("--fixed-delta-s must be positive")
    if args.front_distance_m <= 0:
        raise SystemExit("--front-distance-m must be positive")
    if args.ego_speed_kmh <= args.slow_vehicle_speed_kmh:
        raise SystemExit("--ego-speed-kmh must exceed --slow-vehicle-speed-kmh")
    if args.slow_vehicle_speed_kmh < 0:
        raise SystemExit("--slow-vehicle-speed-kmh must be non-negative")
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
    if state.get("plan_status") != "ACTIVE" or state.get("active_step_id") != "step_2":
        raise SystemExit("initial state must have ACTIVE step_2")

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
        ego_spawn = spawn_points[args.spawn_index]
        ego_waypoint = carla_map.get_waypoint(
            ego_spawn.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_waypoint is None:
            raise RuntimeError("selected spawn point has no driving waypoint")
        target_waypoint = ego_waypoint.get_left_lane()
        if (
            target_waypoint is None
            or target_waypoint.lane_type != carla.LaneType.Driving
        ):
            raise RuntimeError("selected spawn point has no left driving lane")
        current_forward = ego_waypoint.transform.get_forward_vector()
        target_forward = target_waypoint.transform.get_forward_vector()
        direction_dot = (
            current_forward.x * target_forward.x
            + current_forward.y * target_forward.y
            + current_forward.z * target_forward.z
        )
        if direction_dot < 0.8:
            raise RuntimeError("selected left lane is not in the ego direction")
        lane_change = str(ego_waypoint.lane_change).lower()
        if "left" not in lane_change and "both" not in lane_change:
            raise RuntimeError("selected spawn point does not permit a left lane change")

        front_candidates = ego_waypoint.next(args.front_distance_m)
        if not front_candidates:
            raise RuntimeError("selected road is too short for the slow vehicle")
        front_waypoint = front_candidates[0]
        front_location = front_waypoint.transform.location
        front_spawn = carla.Transform(
            carla.Location(
                x=front_location.x,
                y=front_location.y,
                z=front_location.z + 0.6,
            ),
            front_waypoint.transform.rotation,
        )

        spawner = VehicleSpawner(world)
        ego = spawner.spawn_ego_vehicle(ego_spawn)
        slow_vehicle = spawner.spawn_npc_vehicle(front_spawn)
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

        initial_road_id = int(ego_waypoint.road_id)
        initial_lane_id = int(ego_waypoint.lane_id)
        target_road_id = int(target_waypoint.road_id)
        target_lane_id = int(target_waypoint.lane_id)
        slow_vehicle_id = f"carla_actor_{slow_vehicle.id}"
        observed_slow_vehicle = False
        collision_count = 0
        lane_invasion_count = 0
        stable_frames = 0

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

            matched_entity_id = _alignment_entity_id(latest_alignment, "step_2")
            if matched_entity_id == slow_vehicle_id:
                observed_slow_vehicle = True
            slow_object = _world_object(world_state, slow_vehicle_id)
            lane = _lane_metrics(carla_map, ego)
            centered = (
                lane["lane_id"] == target_lane_id
                and lane["center_offset_m"] is not None
                and lane["center_offset_m"] <= args.maximum_lane_center_offset_m
                and lane["heading_alignment"] is not None
                and lane["heading_alignment"] >= args.minimum_heading_alignment
            )
            stable_frames = stable_frames + 1 if centered else 0
            complete, completion_reasons = lane_change_step_completed(
                observed_slow_vehicle=observed_slow_vehicle,
                slow_vehicle_present=slow_object is not None,
                current_lane_id=lane["lane_id"],
                target_lane_id=target_lane_id,
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
                    step_id="step_2",
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
                    step_id="step_2",
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
                    step_id="step_2",
                    outcome="FAILED",
                    reason_codes=[
                        "plan_terminal_before_step_completion",
                        f"plan_status_{state['plan_status'].lower()}",
                    ],
                )
                result = "FAILED"

            applied_decision = latest_decision
            holding_target_lane = (
                not complete
                and lane["lane_id"] == target_lane_id
            )
            if holding_target_lane:
                applied_decision = _target_lane_hold_decision(
                    latest_decision,
                    current_speed_kmh=float(world_state["ego"]["speed_mps"]) * 3.6,
                    hold_location=lane["hold_location"],
                )

            ego_control, normalized = ego_controller.run_step(
                applied_decision, args.fixed_delta_s
            )
            ego.apply_control(ego_control)
            record = {
                "controlled_tick": controlled_tick,
                "frame_id": world_state["frame_id"],
                "ego_speed_mps": round(float(world_state["ego"]["speed_mps"]), 6),
                "slow_vehicle_speed_mps": round(_speed_mps(slow_vehicle), 6),
                "slow_vehicle_id": slow_vehicle_id,
                "slow_vehicle_distance_m": (
                    None if slow_object is None else slow_object.get("distance_m")
                ),
                "slow_vehicle_lane_relation": (
                    None if slow_object is None else slow_object.get("lane_relation")
                ),
                "matched_entity_id": matched_entity_id,
                "observed_slow_vehicle": observed_slow_vehicle,
                "initial_road_id": initial_road_id,
                "initial_lane_id": initial_lane_id,
                "target_road_id": target_road_id,
                "target_lane_id": target_lane_id,
                "current_road_id": lane["road_id"],
                "current_lane_id": lane["lane_id"],
                "lane_center_offset_m": lane["center_offset_m"],
                "heading_alignment": lane["heading_alignment"],
                "stable_frames": stable_frames,
                "collision_count": collision_count,
                "lane_invasion_count": lane_invasion_count,
                "risk_level": latest_risk["risk_level"],
                "left_lane_safe": latest_risk["lane_change"]["left"]["is_safe"],
                "plan_decision_status": latest_decision["decision_status"],
                "plan_action": latest_decision["action"],
                "applied_action": normalized["action"],
                "holding_target_lane": holding_target_lane,
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
                step_id="step_2",
                outcome="FAILED",
                reason_codes=["lane_change_completion_timeout"],
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
            "initial_road_id": initial_road_id,
            "initial_lane_id": initial_lane_id,
            "target_road_id": target_road_id,
            "target_lane_id": target_lane_id,
            "final_road_id": final_lane["road_id"],
            "final_lane_id": final_lane["lane_id"],
            "final_lane_center_offset_m": final_lane["center_offset_m"],
            "final_heading_alignment": final_lane["heading_alignment"],
            "stable_frames": stable_frames,
            "observed_slow_vehicle": observed_slow_vehicle,
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
