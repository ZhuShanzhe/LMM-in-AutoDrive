"""Run a closed-loop pedestrian-yield experiment against CARLA 0.9.16.

The script composes, without modifying, the team pedestrian scenario, the
team PID controller, and the scene-understanding JSON pipeline.  A COMPLETED
StepFeedback is emitted only after a crossing pedestrian has cleared the ego
path, ego speed has measurably decreased, and no collision was observed.
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--fixed-delta-s", type=float, default=0.05)
    parser.add_argument("--initial-speed-kmh", type=float, default=30.0)
    parser.add_argument("--minimum-speed-reduction-mps", type=float, default=3.0)
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


def pedestrian_step_completed(
    *,
    observed_crossing: bool,
    pedestrian: dict[str, Any] | None,
    initial_speed_mps: float,
    current_speed_mps: float,
    minimum_speed_reduction_mps: float,
    collision_count: int,
) -> tuple[bool, list[str]]:
    """Evaluate conservative, metric completion conditions for step 1."""

    reasons: list[str] = []
    if collision_count:
        return False, ["collision_detected"]
    if not observed_crossing:
        return False, ["crossing_not_observed"]
    if pedestrian is None:
        return False, ["pedestrian_not_in_world_state"]
    if pedestrian.get("lane_relation") == "crossing_ego_path":
        reasons.append("pedestrian_still_crossing")
    lateral = pedestrian.get("relative_position_ego_m", {}).get("lateral")
    if not isinstance(lateral, (int, float)) or isinstance(lateral, bool):
        reasons.append("pedestrian_lateral_position_unavailable")
    elif float(lateral) > -2.5:
        reasons.append("pedestrian_has_not_cleared_far_side")
    speed_reduction = initial_speed_mps - current_speed_mps
    if speed_reduction < minimum_speed_reduction_mps:
        reasons.append("ego_speed_reduction_insufficient")
    if reasons:
        return False, reasons
    return True, [
        "pedestrian_crossing_cleared",
        "ego_speed_reduced",
        "collision_free",
    ]


def _step_feedback(
    *, request_id: str, frame_id: str, step_id: str,
    outcome: str, reason_codes: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "request_id": request_id,
        "frame_id": frame_id,
        "step_id": step_id,
        "outcome": outcome,
        "reason_codes": reason_codes,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fixed_delta_s <= 0:
        raise SystemExit("--fixed-delta-s must be positive")
    if args.initial_speed_kmh <= 0:
        raise SystemExit("--initial-speed-kmh must be positive")
    if args.minimum_speed_reduction_mps <= 0:
        raise SystemExit("--minimum-speed-reduction-mps must be positive")
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
    from scenarios.pedestrian.pedestrian_crossing import PedestrianCrossingScenario

    from scene_understanding.core.carla_sensor_manager import CarlaSensorManager
    from scene_understanding.core.carla_world_state import CarlaWorldStateCollector
    from scene_understanding.src.control_plan_executor import advance_control_plan
    from scene_understanding.src.driving_intent_alignment import align_driving_intent
    from scene_understanding.src.risk_interface import assess_scene_risk

    driving_intent = _read_json(args.driving_intent)
    state = _read_json(args.initial_state)
    if state.get("plan_status") != "ACTIVE" or state.get("active_step_id") != "step_1":
        raise SystemExit("initial state must have ACTIVE step_1")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    timeline_path = output_dir / "timeline.jsonl"

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    original_settings = world.get_settings()
    scenario = None
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

        scenario = PedestrianCrossingScenario(world, external_control=True)
        scenario.setup()
        ego = scenario.ego_vehicle
        walker = scenario.walker

        sensors = CarlaSensorManager(
            world,
            ego,
            output_dir=output_dir / "sensors",
            enable_camera=False,
        )
        sensors.setup()
        collector = CarlaWorldStateCollector(world, ego, max_distance_m=80.0)
        controller = EgoPIDController(
            ego,
            world.get_map(),
            target_speed_kmh=args.initial_speed_kmh,
        )

        initial_speed_mps = args.initial_speed_kmh / 3.6
        transform = ego.get_transform()
        forward = transform.get_forward_vector()
        ego.set_target_velocity(
            carla.Vector3D(
                x=forward.x * initial_speed_mps,
                y=forward.y * initial_speed_mps,
                z=0.0,
            )
        )
        world.tick()
        measured_initial_speed_mps = _speed_mps(ego)
        observed_crossing = False
        collision_count = 0

        for controlled_tick in range(1, args.maximum_ticks + 1):
            scenario.tick()
            frame = world.tick()
            events = sensors.drain_events_through(frame)
            collision_count += len(events["collisions"])
            world_state = collector.collect(sensor_events=events)
            latest_alignment = align_driving_intent(driving_intent, world_state)
            latest_risk = assess_scene_risk(world_state)
            pedestrian_id = f"carla_actor_{walker.id}"
            pedestrian = _world_object(world_state, pedestrian_id)
            if pedestrian is not None and pedestrian.get("lane_relation") == "crossing_ego_path":
                observed_crossing = True
            current_speed_mps = float(world_state["ego"]["speed_mps"])
            complete, completion_reasons = pedestrian_step_completed(
                observed_crossing=observed_crossing,
                pedestrian=pedestrian,
                initial_speed_mps=measured_initial_speed_mps,
                current_speed_mps=current_speed_mps,
                minimum_speed_reduction_mps=args.minimum_speed_reduction_mps,
                collision_count=collision_count,
            )

            # Completion must be evaluated before advancing without feedback.
            # On the first cleared frame the pedestrian correctly stops matching
            # AHEAD_CROSSING; evaluating the SAFE_STOP policy first would turn a
            # successful step into a terminal BLOCKED plan.
            if collision_count:
                feedback = _step_feedback(
                    request_id=driving_intent["request_id"],
                    frame_id=world_state["frame_id"],
                    step_id="step_1",
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
                    step_id="step_1",
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
                    step_id="step_1",
                    outcome="FAILED",
                    reason_codes=[
                        "plan_terminal_before_step_completion",
                        f"plan_status_{state['plan_status'].lower()}",
                    ],
                )
                result = "FAILED"

            control, normalized = controller.run_step(
                latest_decision, args.fixed_delta_s
            )
            ego.apply_control(control)

            record = {
                "controlled_tick": controlled_tick,
                "frame_id": world_state["frame_id"],
                "ego_speed_mps": round(current_speed_mps, 6),
                "pedestrian_id": pedestrian_id,
                "pedestrian_lane_relation": (
                    None if pedestrian is None else pedestrian.get("lane_relation")
                ),
                "pedestrian_lateral_m": (
                    None
                    if pedestrian is None
                    else pedestrian.get("relative_position_ego_m", {}).get("lateral")
                ),
                "observed_crossing": observed_crossing,
                "collision_count": collision_count,
                "risk_level": latest_risk["risk_level"],
                "decision_status": latest_decision["decision_status"],
                "action": normalized["action"],
                "throttle": float(control.throttle),
                "brake": float(control.brake),
                "steer": float(control.steer),
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
                step_id="step_1",
                outcome="FAILED",
                reason_codes=["pedestrian_clearance_timeout"],
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
        summary = {
            "schema_version": "1.0.0",
            "result": result,
            "request_id": driving_intent["request_id"],
            "initial_speed_mps": round(measured_initial_speed_mps, 6),
            "final_speed_mps": round(_speed_mps(ego), 6),
            "speed_reduction_mps": round(
                measured_initial_speed_mps - _speed_mps(ego), 6
            ),
            "observed_crossing": observed_crossing,
            "collision_count": collision_count,
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
        if scenario is not None:
            scenario.destroy()
        world.apply_settings(original_settings)


if __name__ == "__main__":
    raise SystemExit(main())
