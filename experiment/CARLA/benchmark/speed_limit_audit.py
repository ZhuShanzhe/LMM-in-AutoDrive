"""Capture map speed-sign evidence without changing vehicle limits or metrics."""
import math


def capture_speed_signs(world, route, location_factory):
    signs = []
    for actor in sorted(world.get_actors().filter('traffic.speed_limit.*'), key=lambda a: a.id):
        pose = actor.get_transform()
        record = dict(actor_id=actor.id, type_id=actor.type_id,
                      position_m={k: float(getattr(pose.location, k)) for k in ('x', 'y', 'z')},
                      yaw_deg=float(pose.rotation.yaw), route_trigger_ranges_m=[])
        try:
            value = float(actor.type_id.rsplit('.', 1)[-1])
            if not math.isfinite(value) or value <= 0:
                raise ValueError('invalid speed sign value')
            record['sign_speed_kmh'] = value
            box = actor.trigger_volume
            record['trigger_extent_m'] = {k: float(getattr(box.extent, k)) for k in ('x', 'y', 'z')}
            active = None
            for point in route:
                inside = box.contains(location_factory(**{k: point[k] for k in ('x','y','z')}), pose)
                if inside:
                    if active is None:
                        active = [float(point['distance_m']), float(point['distance_m'])]
                        record['route_trigger_ranges_m'].append(active)
                    else:
                        active[1] = float(point['distance_m'])
                else:
                    active = None
        except (AttributeError, RuntimeError, ValueError) as error:
            record['capture_error'] = str(error)
        signs.append(record)
    return dict(scope='sampled_map_sign_triggers_not_complete_speed_limit_profile',
                signs=signs, sign_count=len(signs),
                limitations=['route_samples_can_miss_narrow_triggers',
                             'vehicle_extent_and_direction_not_evaluated',
                             'initial_vehicle_limit_may_be_default_not_map_evidence',
                             'no_speeding_metric_overrides'])
