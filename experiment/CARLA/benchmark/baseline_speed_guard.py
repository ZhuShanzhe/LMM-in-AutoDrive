"""Optional speed-limit cap for the configured simulator test driver only."""
import math


def guard_speed(intent, speed_limit_kmh):
    result = dict(intent)
    requested = intent.get('target_speed_kmh')
    valid_limit = (isinstance(speed_limit_kmh, (int, float)) and
                   not isinstance(speed_limit_kmh, bool) and
                   math.isfinite(speed_limit_kmh) and speed_limit_kmh > 0)
    valid_target = (isinstance(requested, (int, float)) and
                    not isinstance(requested, bool) and math.isfinite(requested) and requested >= 0)
    cap = max(0., float(speed_limit_kmh) - 1.) if valid_limit else 0.
    applied = min(float(requested), cap) if valid_target else 0.
    result['target_speed_kmh'] = applied
    if not valid_limit or not valid_target:
        result.update(action='stop', emergency=True)
    evidence = dict(requested_target_kmh=requested, reported_limit_kmh=speed_limit_kmh,
                    applied_target_kmh=applied, capped=(applied != requested),
                    reason='reported_speed_limit_cap' if valid_limit and valid_target else 'invalid_speed_input',
                    scope='configured_simulator_baseline_not_model_decision')
    return result, evidence
