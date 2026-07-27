"""Evaluate a recorded CARLA run against basic scenario-quality gates."""

import argparse
import json
from pathlib import Path


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _range(values):
    return {"min": min(values), "max": max(values)} if values else {"min": 0, "max": 0}


def _event_evidence(frames, event_ids):
    evidence = {
        event_id: {"active_states": [], "behavior_observed": False}
        for event_id in event_ids
    }
    for record in frames:
        route_progress_m = record.get("scenario_status", {}).get("route_progress_m")
        active = record.get("scenario_status", {}).get("scenario_events", {}).get("active_details", [])
        for details in active:
            event_id = details.get("id")
            if event_id not in evidence:
                continue
            item = evidence[event_id]
            state = details.get("status")
            if state and state not in item["active_states"]:
                item["active_states"].append(state)
            item.setdefault("first_active_route_progress_m", route_progress_m)
            behavior_started = (
                state in {"BRAKING", "MERGING", "CROSSING"}
                or details.get("brake_started_s") is not None
                or bool(details.get("lane_change_requested"))
                or details.get("crossing_observed_s") is not None
            )
            if behavior_started and not item["behavior_observed"]:
                item["behavior_observed"] = True
                item["behavior_route_progress_m"] = route_progress_m
    return evidence


def _strong_brake_windows(frames, threshold=0.8, horizon_frames=30):
    windows = []
    braking = False
    for index, record in enumerate(frames):
        brake = float(record.get("control", {}).get("brake", 0.0))
        if brake >= threshold and not braking:
            horizon = frames[index:min(len(frames), index + horizon_frames + 1)]
            start_speed = float(record.get("ego", {}).get("speed_kmh", 0.0))
            minimum_speed = min(
                (float(item.get("ego", {}).get("speed_kmh", start_speed)) for item in horizon),
                default=start_speed,
            )
            windows.append({
                "sim_time_s": record.get("sim_time_s"),
                "route_progress_m": record.get("scenario_status", {}).get("route_progress_m"),
                "start_speed_kmh": round(start_speed, 3),
                "min_speed_next_2s_kmh": round(minimum_speed, 3),
            })
        braking = brake >= threshold
    return windows


def evaluate_run(
    run_dir,
    required_event_ids=(),
    require_emergency_brake=False,
    require_fresh_event_evidence=False,
    require_event_behavior=False,
):
    """Return an auditable pass/fail report for one completed run directory."""
    run_dir = Path(run_dir)
    metrics = _read_json(run_dir / "metrics.json")
    frames = _read_jsonl(run_dir / "frames.jsonl")
    events = _read_jsonl(run_dir / "events.jsonl")

    scenario_status = metrics.get("scenario_status", {})
    event_status = scenario_status.get("scenario_events", {})
    completed = set(event_status.get("completed", []))
    failed = set(event_status.get("failed", []))
    required = set(required_event_ids)
    activated = {
        item.get("event_id")
        for item in events
        if item.get("type") == "scenario_event" and item.get("transition") == "activated"
    }
    event_checks = {
        event_id: {
            "activated": event_id in activated,
            "completed": event_id in completed,
            "failed": event_id in failed,
        }
        for event_id in sorted(required)
    }
    event_evidence = _event_evidence(frames, required)

    traffic_counts = [
        int(record.get("scenario_status", {}).get("traffic", {}).get("background_actor_count", 0))
        for record in frames
    ]
    pedestrian_counts = [
        int(record.get("scenario_status", {}).get("pedestrians", {}).get("walker_count", 0))
        for record in frames
    ]
    controls = [record.get("control", {}) for record in frames]
    simultaneous_control_frames = sum(
        1
        for control in controls
        if float(control.get("throttle", 0.0)) > 0.01 and float(control.get("brake", 0.0)) > 0.01
    )
    max_brake = max((float(control.get("brake", 0.0)) for control in controls), default=0.0)
    brake_windows = _strong_brake_windows(frames)

    checks = {
        "collision_free": bool(metrics.get("collision_free", False)),
        "illegal_lane_invasion_free": bool(metrics.get("illegal_lane_invasion_free", False)),
        "no_failed_special_event": not failed,
        "required_events_completed": all(item["completed"] for item in event_checks.values()),
        "controls_are_mutually_exclusive": simultaneous_control_frames == 0,
    }
    if require_fresh_event_evidence:
        checks["required_events_activated_this_run"] = all(
            item["activated"] for item in event_checks.values()
        )
    if require_event_behavior:
        checks["required_event_behavior_observed"] = all(
            item["behavior_observed"] for item in event_evidence.values()
        )
    if require_emergency_brake:
        checks["emergency_brake_observed"] = max_brake >= 0.8

    return {
        "run_dir": str(run_dir),
        "passed": all(checks.values()),
        "checks": checks,
        "events": event_checks,
        "event_behavior": event_evidence,
        "traffic_actor_count": _range(traffic_counts),
        "pedestrian_count": _range(pedestrian_counts),
        "control": {
            "frame_count": len(controls),
            "simultaneous_throttle_brake_frames": simultaneous_control_frames,
            "max_brake": round(max_brake, 4),
            "strong_brake_windows": brake_windows,
        },
        "safety": {
            "collision_events": int(metrics.get("collision_events", 0)),
            "illegal_lane_invasion_events": int(metrics.get("illegal_lane_invasion_events", 0)),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Directory containing metrics.json and frame/event logs")
    parser.add_argument("--require-event", action="append", default=[], help="Event ID that must activate and complete")
    parser.add_argument("--require-emergency-brake", action="store_true", help="Require an observed brake command of at least 0.8")
    parser.add_argument(
        "--require-fresh-event-evidence",
        action="store_true",
        help="Require an explicit activated transition in this run's event log",
    )
    parser.add_argument(
        "--require-event-behavior",
        action="store_true",
        help="Require event-specific active-state evidence, not only a completed final state",
    )
    args = parser.parse_args()
    report = evaluate_run(
        args.run_dir,
        args.require_event,
        args.require_emergency_brake,
        args.require_fresh_event_evidence,
        args.require_event_behavior,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
