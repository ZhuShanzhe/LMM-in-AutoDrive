"""Mandatory execution holds must not consume active instruction time."""
import math


def traffic_wait_feedback(request_id, timestamp_s, decision, traffic):
    cap = traffic.get('speed_cap_kmh')
    held = (traffic.get('active') is True
            and traffic.get('status') == 'wait_for_observed_green'
            and isinstance(cap, (int, float)) and math.isfinite(cap) and cap <= 0.
            and decision.get('action') in ('stop', 'emergency_brake')
            and float(decision.get('target_speed_kmh', 1.)) <= 0.)
    return dict(request_id=request_id, observed_at_s=timestamp_s,
                safety_wait=held, reason='traffic_signal_hold' if held else None)


def matching_execution_wait(feedback, request_id, observed_at_s):
    if not feedback or feedback.get('request_id') != request_id:
        return False
    stamp = feedback.get('observed_at_s')
    return (isinstance(stamp, (int, float)) and math.isfinite(stamp)
            and abs(stamp - observed_at_s) <= 1e-6
            and feedback.get('safety_wait') is True)
