"""Weather changes the legal operating speed cap, not collision labels."""
import math


def environment_speed_bound(road_limit_kmh, route_limit_kmh, *, precipitation, wetness, sun_altitude):
    values = (road_limit_kmh, route_limit_kmh, precipitation, wetness, sun_altitude)
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
        raise ValueError('Finite environment inputs required')
    if min(road_limit_kmh, route_limit_kmh) < 0:
        raise ValueError('Nonnegative speed limits required')
    base = min(road_limit_kmh, route_limit_kmh)
    rainy_night = sun_altitude < 0 and (precipitation >= 10 or wetness >= 50)
    return dict(schema_version='environment_speed_bound/1.0', rainy_night=rainy_night,
        road_limit_kmh=road_limit_kmh, route_limit_kmh=route_limit_kmh,
        base_limit_kmh=base, max_speed_kmh=base * (5. / 6. if rainy_night else 1.),
        collision_risk_added=False)


def apply_environment_speed_bound(decision, bound, ego_speed_kmh, requested_goal_kmh=None):
    result = dict(decision)
    result.pop('environment_speed_constraint', None)
    before = result['target_speed_kmh']
    result['target_speed_kmh'] = min(float(before), bound['max_speed_kmh'])
    changed = result['target_speed_kmh'] != before
    if changed:
        for key in ('longitudinal_sequence_schema', 'target_acceleration_mps2', 'sequence_valid_until_s'):
            result.pop(key, None)
        if ego_speed_kmh >= result['target_speed_kmh']:
            result['allow_positive_acceleration'] = False
    if (isinstance(requested_goal_kmh, (int, float)) and not isinstance(requested_goal_kmh, bool)
            and math.isfinite(requested_goal_kmh) and requested_goal_kmh > bound['max_speed_kmh']):
        codes = []
        if requested_goal_kmh > bound['road_limit_kmh']:
            codes.append('road_speed_limit')
        if requested_goal_kmh > bound['route_limit_kmh']:
            codes.append('route_speed_cap')
        if bound['rainy_night']:
            codes.append('rainy_night_speed_cap')
        # The goal bound is independent of a temporary stop or obstacle speed cap.
        result['environment_speed_constraint'] = dict(
            requested_target_speed_kmh=requested_goal_kmh,
            effective_target_speed_kmh=bound['max_speed_kmh'], speed_constraint_codes=codes)
    return result, dict(bound, changed=changed)
