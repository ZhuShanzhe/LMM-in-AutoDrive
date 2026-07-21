"""Metrics calculated from per-frame CARLA experiment records."""

import math


def _percentile(values, percentile):
    if not values:
        return None
    values = sorted(values)
    index = int(math.ceil(percentile / 100.0 * len(values))) - 1
    return values[max(0, min(index, len(values) - 1))]


def summarize(records, scenario, goal_distance_m=None):
    if not records:
        raise ValueError("Cannot summarize an empty experiment")

    first = records[0]
    last = records[-1]
    elapsed_s = max(float(last["sim_time_s"]) - float(first["sim_time_s"]), 1e-6)
    distance_m = max(float(last["distance_m"]) - float(first["distance_m"]), 0.0)
    collisions = max(record["events"]["collision_count"] for record in records)
    lane_invasions = max(record["events"]["lane_invasion_count"] for record in records)
    speeds = [float(record["ego"]["speed_kmh"]) for record in records]
    response_latencies = [float(record["latency_ms"]["end_to_end"]) for record in records]
    control_latencies = [float(record["latency_ms"]["control"]) for record in records]
    speeding_frames = sum(
        1
        for record in records
        if record["intent"]["action"] not in ("stop", "emergency_brake")
        and record["ego"]["speed_kmh"] > record["intent"]["target_speed_kmh"] + 5.0
    )

    goal_reached = True if goal_distance_m is None else distance_m >= float(goal_distance_m)
    violation_free = collisions == 0 and lane_invasions == 0 and speeding_frames == 0
    task_completed = bool(goal_reached and violation_free)
    return {
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
        "lane_invasion_free": lane_invasions == 0,
        "collision_rate_per_1000_frames": round(collisions * 1000.0 / len(records), 4),
        "collision_rate_per_km": round(collisions / max(distance_m / 1000.0, 1e-6), 4),
        "lane_invasion_events": lane_invasions,
        "lane_invasion_rate_per_1000_frames": round(lane_invasions * 1000.0 / len(records), 4),
        "speeding_frames": speeding_frames,
        "speeding_rate": round(speeding_frames / len(records), 5),
        "violation_free": violation_free,
        "response_latency_ms_mean": round(sum(response_latencies) / len(response_latencies), 3),
        "response_latency_ms_p95": round(_percentile(response_latencies, 95), 3),
        "control_latency_ms_mean": round(sum(control_latencies) / len(control_latencies), 3),
        "control_latency_ms_p95": round(_percentile(control_latencies, 95), 3),
    }
