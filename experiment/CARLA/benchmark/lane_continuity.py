"""Truth-side lane-following corridor, independent of policy completion claims."""
import math

from .catalog import ConfigError
from .truth_capture import lane_key, angle_delta


def build_lane_corridor(world_map,route,start_m,location_factory,end_m=None):
    result=trace_lane_corridor(world_map,route,start_m,location_factory,end_m)
    if result['stop_reason']:
        raise ConfigError(result['stop_reason'])
    return result['lane_corridor']


def trace_speed_lane_corridor(world_map, route, start_m, location_factory, end_m):
    result = trace_lane_corridor(world_map, route, start_m, location_factory, end_m)
    if result['stop_reason'] in {'lane corridor route/map mismatch',
                                 'lane corridor heading mismatch'}:
        raise ConfigError(result['stop_reason'])
    return result


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
        if waypoints:
            distance=point['distance_m']-points[offset-1]['distance_m']
            if not math.isfinite(distance) or not 0<distance<=10:
                raise ConfigError('lane corridor samples too sparse or unordered')
            successors=waypoints[-1].next(distance)
            expected=f"{point['road_id']}:{point['section_id']}:{point['lane_id']}"
            if len(successors)>1 and (nearest is None or lane_key(nearest)!=expected):
                matching=[candidate for candidate in successors
                          if lane_key(candidate)==expected
                          and candidate.transform.location.distance(location)<=.75]
                if len(matching)==1:
                    successors=matching
            if len(successors)!=1:
                stop_reason='lane corridor has ambiguous forward topology'
                break
            wp=successors[0]
        if wp is None or str(wp.lane_type)!='Driving':
            raise ConfigError('lane corridor has no driving waypoint')
        expected=f"{point['road_id']}:{point['section_id']}:{point['lane_id']}"
        if (waypoints and nearest is not None and lane_key(nearest)==expected
                and (lane_key(wp)!=expected or wp.transform.location.distance(location)>.75)):
            for adjusted_distance in (distance-.5,distance+.5):
                if adjusted_distance<=0:
                    continue
                adjusted=waypoints[-1].next(adjusted_distance)
                if len(adjusted)>1:
                    stop_reason='lane corridor has ambiguous forward topology'
                    break
                if len(adjusted)==1 and lane_key(adjusted[0])==expected \
                        and adjusted[0].transform.location.distance(location)<=.75:
                    wp=adjusted[0]
                    break
            if stop_reason is not None:
                break
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
