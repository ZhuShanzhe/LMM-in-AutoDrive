"""Metrics calculated from per-frame CARLA experiment records."""

import math


def _percentile(values, percentile):
    if not values:
        return None
    values = sorted(values)
    index = int(math.ceil(percentile / 100.0 * len(values))) - 1
    return values[max(0, min(index, len(values) - 1))]

def _steering_metrics(records):
    samples = []
    for record in records:
        dynamics = record.get("steering_dynamics", {})
        if "normalized_steer" not in dynamics:
            continue
        samples.append({
            "steer": float(dynamics["normalized_steer"]),
            "rate": float(dynamics.get("steer_rate_per_s", 0.0)),
            "accel": float(dynamics.get("steer_accel_per_s2", 0.0)),
            "action": str(dynamics.get("action", record.get("intent", {}).get("action", ""))),
        })
    if not samples:
        return {}

    abs_rates = [abs(sample["rate"]) for sample in samples]
    abs_accels = [abs(sample["accel"]) for sample in samples]
    straight = [
        sample for sample in samples
        if sample["action"] in {"keep_lane", "accelerate", "decelerate"}
    ]
    reversals = 0
    previous_sign = 0
    for sample in samples:
        sign = 1 if sample["steer"] >= 0.03 else -1 if sample["steer"] <= -0.03 else 0
        if sign and previous_sign and sign != previous_sign:
            reversals += 1
        if sign:
            previous_sign = sign

    result = {
        "steer_rate_abs_p95_per_s": round(_percentile(abs_rates, 95), 4),
        "steer_rate_abs_max_per_s": round(max(abs_rates), 4),
        "steer_accel_abs_p95_per_s2": round(_percentile(abs_accels, 95), 4),
        "steer_accel_abs_max_per_s2": round(max(abs_accels), 4),
        "steer_direction_reversal_count": reversals,
    }
    if straight:
        result.update({
            "straight_steer_abs_p95": round(
                _percentile([abs(sample["steer"]) for sample in straight], 95), 4
            ),
            "straight_steer_rate_abs_p95_per_s": round(
                _percentile([abs(sample["rate"]) for sample in straight], 95), 4
            ),
        })
    return result


def summarize(records, scenario, goal_distance_m=None):
    if not records:
        raise ValueError("Cannot summarize an empty experiment")

    first = records[0]
    last = records[-1]
    elapsed_s = max(float(last["sim_time_s"]) - float(first["sim_time_s"]), 1e-6)
    distance_m = max(float(last["distance_m"]) - float(first["distance_m"]), 0.0)
    collisions = max(record["events"]["collision_count"] for record in records)
    lane_invasions = max(record["events"]["lane_invasion_count"] for record in records)
    illegal_lane_invasions = max(
        record["events"].get("illegal_lane_invasion_count", record["events"]["lane_invasion_count"])
        for record in records
    )
    speeds = [float(record["ego"]["speed_kmh"]) for record in records]
    response_latencies = [float(record["latency_ms"]["end_to_end"]) for record in records]
    control_latencies = [float(record["latency_ms"]["control"]) for record in records]
    slowing_actions = {
        "stop", "decelerate", "emergency_brake", "turn_left", "turn_right",
    }
    speeding_frames = 0
    for record in records:
        decision_action = (
            record.get("scene_decision", {})
            .get("control_decision", {})
            .get("action")
        )
        effective_action = decision_action or record["intent"]["action"]
        if (
            effective_action not in slowing_actions
            and float(record.get("control", {}).get("brake", 0.0)) <= 0.01
            and record["ego"]["speed_kmh"]
            > record["intent"]["target_speed_kmh"] + 5.0
        ):
            speeding_frames += 1

    goal_reached = False if goal_distance_m is None else distance_m >= float(goal_distance_m)
    violation_free = collisions == 0 and illegal_lane_invasions == 0 and speeding_frames == 0
    task_completed = bool(goal_reached and violation_free)
    summary = {
        "scenario": scenario,
        "frames": len(records),
        "duration_s": round(elapsed_s, 3),
        "distance_m": round(distance_m, 3),
        "average_speed_kmh": round(sum(speeds) / len(speeds), 3),
        "max_speed_kmh": round(max(speeds), 3),
        "task_completed": task_completed,
        "goal_distance_m": goal_distance_m,
        "goal_reached": goal_reached,
        "collision_events": collisions,
        "collision_free": collisions == 0,
        "lane_invasion_free": illegal_lane_invasions == 0,
        "collision_rate_per_1000_frames": round(collisions * 1000.0 / len(records), 4),
        "collision_rate_per_km": round(collisions / max(distance_m / 1000.0, 1e-6), 4),
        "lane_invasion_events": lane_invasions,
        "lane_invasion_rate_per_1000_frames": round(lane_invasions * 1000.0 / len(records), 4),
        "illegal_lane_invasion_events": illegal_lane_invasions,
        "illegal_lane_invasion_free": illegal_lane_invasions == 0,
        "speeding_frames": speeding_frames,
        "speeding_rate": round(speeding_frames / len(records), 5),
        "violation_free": violation_free,
        "response_latency_ms_mean": round(sum(response_latencies) / len(response_latencies), 3),
        "response_latency_ms_p95": round(_percentile(response_latencies, 95), 3),
        "control_latency_ms_mean": round(sum(control_latencies) / len(control_latencies), 3),
        "control_latency_ms_p95": round(_percentile(control_latencies, 95), 3),
    }
    summary.update(_steering_metrics(records))
    return summary
