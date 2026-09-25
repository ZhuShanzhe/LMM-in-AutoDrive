"""Truth-side lane-following corridor, independent of policy completion claims."""
import math

from .catalog import ConfigError
from .truth_capture import lane_key, angle_delta, prepare_lane_fixture


def trace_lane_change_keys(world_map, route, start_m, end_m, direction,
                           location_factory, entry_fixture):
    """Bind road-spanning lane IDs to one route-verified lane-change pair."""
    if not start_m < end_m <= route[-1]['distance_m']:
        raise ConfigError('lane-change corridor outside recorded route')
    entries = [entry_fixture['entry_lane_key']]
    targets = [entry_fixture['target_lane_key']]
    checked = 0
    verified_waypoint = None
    verified_distance = None
    for point in route:
        distance = point['distance_m']
        if not start_m <= distance <= end_m:
            continue
        location = location_factory(**{key:point[key] for key in ('x','y','z')})
        waypoint = world_map.get_waypoint(location)
        if waypoint is None:
            raise ConfigError('lane-change route waypoint missing')
        expected = f"{point['road_id']}:{point['section_id']}:{point['lane_id']}"
        if lane_key(waypoint) != expected:
            # Town junction connectors can occupy exactly the same coordinates.
            # They are not legal lane-change sites, so do not bind either lane
            # from a nearest-waypoint tie at such a point.
            if waypoint.is_junction:
                continue
            delta = distance - verified_distance if verified_distance is not None else 0
            successor = getattr(verified_waypoint, 'next', None)
            if not callable(successor) or not 0 < delta <= 15:
                raise ConfigError('lane-change route/map mismatch')
            candidates = []
            for adjustment in (0, .25, .5, -.25, -.5):
                options = successor(max(.05, delta + adjustment))
                if len(options) > 1:
                    raise ConfigError('lane-change route/map mismatch')
                candidates.extend(options)
            matching = [item for item in candidates if lane_key(item) == expected
                        and item.transform.location.distance(location) <= .75]
            if not matching:
                raise ConfigError('lane-change route/map mismatch')
            waypoint = min(matching, key=lambda item: item.transform.location.distance(location))
        verified_waypoint = waypoint
        verified_distance = distance
        if waypoint.is_junction:
            continue
        checked += 1
        try:
            pair = prepare_lane_fixture(world_map, location, direction, entry_waypoint=waypoint)
        except ConfigError:
            continue
        for key, values in (('entry_lane_key', entries), ('target_lane_key', targets)):
            if pair[key] not in values:
                values.append(pair[key])
    if not checked or set(entries) & set(targets):
        raise ConfigError('invalid lane-change corridor')
    return dict(entry_lane_keys=entries, target_lane_keys=targets)


def build_lane_corridor(world_map,route,start_m,location_factory,end_m=None):
    result=trace_lane_corridor(world_map,route,start_m,location_factory,end_m)
    if result['stop_reason']:
        raise ConfigError(result['stop_reason'])
    return result['lane_corridor']


def trace_lane_corridor(world_map,route,start_m,location_factory,end_m=None):
    """Return only the proven prefix; the caller must reject observations beyond it."""
    end_m=route[-1]['distance_m'] if end_m is None and route else end_m
    if end_m is None or not math.isfinite(end_m) or end_m<=start_m:
        raise ConfigError('lane corridor end must follow start')
    index=max((i for i,p in enumerate(route) if p['distance_m']<=start_m),default=-1)
    if index<0 or index>=len(route)-1:
        raise ConfigError('lane corridor start outside route')
    end_index=next((i for i in range(index+1,len(route)) if route[i]['distance_m']>=end_m),None)
    if end_index is None:
        raise ConfigError('lane corridor end outside route')
    points=route[index:end_index+1]
    waypoints=[]
    segments=[]
    stop_reason=None
    for offset,point in enumerate(points):
        location=location_factory(**{k:point[k] for k in ('x','y','z')})
        nearest=world_map.get_waypoint(location)
        wp=nearest
        expected=f"{point['road_id']}:{point['section_id']}:{point['lane_id']}"
        if waypoints:
            distance=point['distance_m']-points[offset-1]['distance_m']
            if not math.isfinite(distance) or not 0<distance<=15:
                raise ConfigError('lane corridor samples too sparse or unordered')
            successors=waypoints[-1].next(distance)
            if len(successors)>1:
                stop_reason='lane corridor has ambiguous forward topology'
                break
            candidates=list(successors)
            # GRP and CARLA waypoint sampling can straddle a road boundary by
            # a few decimeters. Search only immediate topological successors.
            if not any(lane_key(item)==expected and item.transform.location.distance(location)<=.75
                       for item in candidates):
                for adjustment in (.25,.5,.75,-.25,-.5):
                    candidates.extend(waypoints[-1].next(max(.05,distance+adjustment)))
            matching=[item for item in candidates if lane_key(item)==expected
                      and item.transform.location.distance(location)<=.75]
            if matching:
                wp=min(matching,key=lambda item:item.transform.location.distance(location))
            elif len(successors)==1:
                wp=successors[0]
            else:
                stop_reason='lane corridor has ambiguous forward topology'
                break
        if wp is None or str(wp.lane_type)!='Driving':
            raise ConfigError('lane corridor has no driving waypoint')
        if lane_key(wp)!=expected or wp.transform.location.distance(location)>.75:
            stop_reason='lane corridor route/map mismatch'
            break
        reference_yaw=point.get('yaw')
        if reference_yaw is None and nearest is not None and lane_key(nearest)==expected:
            reference_yaw=nearest.transform.rotation.yaw
        if reference_yaw is not None and abs(angle_delta(wp.transform.rotation.yaw,reference_yaw))>10:
            stop_reason='lane corridor heading mismatch'
            break
        if waypoints:
            segments.append(dict(start_m=points[offset-1]['distance_m'],end_m=point['distance_m'],
                                 lane_keys=list(dict.fromkeys([lane_key(waypoints[-1]),lane_key(wp)]))))
        waypoints.append(wp)
    if not segments:
        raise ConfigError(stop_reason or 'lane corridor has no verified segment')
    return dict(lane_corridor=segments,requested_end_m=end_m,
                verified_end_m=segments[-1]['end_m'],stop_reason=stop_reason)
