"""Shared, label-free observation preparation for longitudinal event memory."""

import math


OBSERVATION_VERSION = 'longitudinal_corridor_radar/1.0'


def prepare_event_radar(observation, *, half_width_m=1.6, maximum_speed_mps=80.):
    result = dict(observation or {})
    bins = result.get('azimuth_obstacle_bins')
    if bins is None:
        # Synthetic inputs explicitly represent a centered, single return.
        bins = [] if result.get('nearest_distance_m') is None else [dict(
            distance_m=result.get('nearest_distance_m'),
            relative_velocity_mps=result.get('nearest_relative_velocity_mps'),
            azimuth_deg=result.get('nearest_azimuth_deg', 0.) or 0.)]
    selected, invalid = [], 0
    for item in bins:
        try:
            distance = float(item['distance_m'])
            velocity = float(item['relative_velocity_mps'])
            azimuth = float(item['azimuth_deg'])
        except (KeyError, TypeError, ValueError):
            invalid += 1
            continue
        if not all(map(math.isfinite, (distance, velocity, azimuth))) or distance < 0 or abs(velocity) > maximum_speed_mps:
            invalid += 1
            continue
        angle = math.radians(azimuth)
        if distance * math.cos(angle) <= 0 or abs(distance * math.sin(angle)) > half_width_m:
            continue
        selected.append((distance, velocity, azimuth))
    nearest = min(selected, default=None)
    closing = min((x for x in selected if x[1] < 0), default=None)
    result.update(nearest_distance_m=nearest[0] if nearest else None,
        nearest_relative_velocity_mps=nearest[1] if nearest else None,
        nearest_azimuth_deg=nearest[2] if nearest else None,
        nearest_closing_distance_m=closing[0] if closing else None,
        nearest_closing_velocity_mps=-closing[1] if closing else None,
        event_observation_version=OBSERVATION_VERSION,
        event_corridor_half_width_m=half_width_m, event_invalid_returns=invalid)
    # A physically impossible packet is unknown, not evidence of an empty road.
    if bins and invalid == len(bins):
        result['sensor_frame'] = -1
    return result


def prepare_event_ego(ego, memory):
    result = ego.clone()
    result[:, 0] = result[:, 0].clamp(0, 70)
    result[:, 1] = (memory[:, -1, 41] * 12).abs()
    result[:, 2] = result[:, 2].clamp(-3, 3)
    result[:, 3:6] = 0
    result[:, 6] = result[:, 6].clamp(0, 70)
    return result
