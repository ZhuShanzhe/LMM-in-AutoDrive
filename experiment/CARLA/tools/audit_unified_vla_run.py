"""Audit one unified VLA decision log into the final decision-audit JSON."""

import json
import math
import sys
from collections import Counter
from pathlib import Path


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


path = Path(sys.argv[1])
actions = Counter()
risks = Counter()
target_lane_risks = Counter()
overrides = Counter()
latencies = []
sensor_latencies = []
count = 0
model_applied = 0
truth_access = 0
candidate_nonzero = 0
safety_candidate_nonzero = 0
fallbacks = 0
route_min = None
route_max = None

with path.open(encoding="utf-8") as source:
    for line in source:
        if not line.strip():
            continue
        row = json.loads(line)
        count += 1
        route = row.get("route_s_m")
        if isinstance(route, (int, float)):
            route_min = route if route_min is None else min(route_min, route)
            route_max = route if route_max is None else max(route_max, route)
        decision = row.get("control_decision") or {}
        actions[str(decision.get("action", "missing"))] += 1
        risk = row.get("risk_assessment") or {}
        risks[str(risk.get("risk_level", "missing"))] += 1
        target_lane_risk = row.get("target_lane_risk_assessment")
        if target_lane_risk:
            target_lane_risks[
                str(target_lane_risk.get("risk_level", "missing"))
            ] += 1
        override = row.get("liveness_override")
        if override:
            overrides[str(override)] += 1
        model_applied += int(bool(row.get("model_output_applied")))
        truth_access += int(bool(row.get("policy_truth_access")))
        candidate_nonzero += int((row.get("candidate_count") or 0) != 0)
        safety_candidate_nonzero += int(
            (row.get("safety_observation_candidate_count") or 0) != 0
        )
        fallbacks += int(decision.get("decision_status") not in {None, "READY"})
        latency = row.get("full_decision_latency_ms")
        if isinstance(latency, (int, float)):
            latencies.append(float(latency))
        sensor_latency = row.get("sensor_to_decision_response_ms")
        if isinstance(sensor_latency, (int, float)):
            sensor_latencies.append(float(sensor_latency))


def latency_summary(values):
    return {
        "count": len(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values) if values else None,
        "within_120ms_ratio": (
            sum(value <= 120.0 for value in values) / len(values) if values else None
        ),
    }


result = {
    "decision_count": count,
    "route_min_m": route_min,
    "route_max_m": route_max,
    "model_output_applied_count": model_applied,
    "model_output_applied_ratio": model_applied / count if count else None,
    "fallback_count": fallbacks,
    "policy_truth_access_count": truth_access,
    "candidate_nonzero_count": candidate_nonzero,
    "safety_candidate_nonzero_count": safety_candidate_nonzero,
    "actions": dict(actions),
    "risk_levels": dict(risks),
    "target_lane_risk_levels": dict(target_lane_risks),
    "liveness_overrides": dict(overrides),
    "full_decision_latency": latency_summary(latencies),
    "sensor_to_decision_latency": latency_summary(sensor_latencies),
}
print(json.dumps(result, ensure_ascii=False, indent=2))
