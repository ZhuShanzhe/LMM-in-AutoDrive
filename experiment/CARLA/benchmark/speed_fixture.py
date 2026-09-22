"""Slice a source route with preparation distance for isolated speed checks."""
from copy import deepcopy
import math

from .catalog import ConfigError


def slice_speed_fixture(route, spec, world_map, location_factory, pre_roll_m=120):
    if not math.isfinite(pre_roll_m) or pre_roll_m < 0:
        raise ConfigError('invalid speed fixture preparation distance')
    if spec.get('requires_task_success') or spec.get('constraints'):
        raise ConfigError('speed slice cannot reconstruct prior task or interval evidence')
    activation = spec['activate_m']
    if activation == 0:
        return deepcopy(route), deepcopy(spec), 0.0
    preferred = max(route[0]['distance_m'], activation-pre_roll_m)
    candidates = [i for i,p in enumerate(route)
                  if preferred <= p['distance_m'] < activation]
    entry = None
    for i in candidates:
        point = route[i]
        waypoint = world_map.get_waypoint(location_factory(**{k:point[k] for k in ('x','y','z')}))
        if (waypoint is not None and not waypoint.is_junction
                and waypoint.road_id == point['road_id'] and waypoint.lane_id == point['lane_id']):
            entry = i
            break
    if entry is None:
        raise ConfigError('no matching non-junction preparation entry before speed task')
    offset = route[entry]['distance_m']
    local_route = deepcopy(route[entry:])
    for point in local_route:
        point['distance_m'] -= offset
    local_spec = deepcopy(spec)
    local_spec['activate_m'] -= offset
    if local_spec.get('end_route_s_m') is not None:
        local_spec['end_route_s_m'] -= offset
    return local_route, local_spec, offset
