"""Topology-backed isolated junction turns on the selected CARLA map."""
import math

from .catalog import ConfigError
from .route_audit import audit_route
from .truth_capture import angle_delta


def turn_evidence(route, direction):
    report=audit_route(route)
    if report['suspicious_gaps']:
        raise ConfigError('turn route contains discontinuities')
    if not report['junction_visits']:
        raise ConfigError('turn route contains no junction')
    first=report['junction_visits'][0]
    if not first['distinct_road_turn_candidate'] or first['direction']!=direction:
        raise ConfigError('first junction does not match requested distinct-road turn')
    return dict(direction=direction,legal=True,entry_road_key=str(first['entry_road_id']),
                exit_road_key=str(first['exit_road_id']),junction_start_m=first['start_m'],
                junction_end_m=first['end_m'],heading_change_deg=first['heading_change_deg'],
                legality_scope='directed_map_topology_not_signal_permission')


def waypoint_route(waypoints, distances=None):
    if distances is not None and len(waypoints)!=len(distances):
        raise ConfigError('route waypoints and distances must align')
    route=[]
    distance=0.
    for index,waypoint in enumerate(waypoints):
        pose=waypoint.transform
        p=dict(x=pose.location.x,y=pose.location.y,z=pose.location.z,yaw=pose.rotation.yaw,
               road_id=waypoint.road_id,section_id=waypoint.section_id,lane_id=waypoint.lane_id,
               is_junction=bool(waypoint.is_junction))
        if route:
            gap=math.sqrt(sum((p[k]-route[-1][k])**2 for k in ('x','y','z')))
            if gap<.05:
                continue
            distance+=gap
        p['distance_m']=distance if distances is None else float(distances[index])
        if route and p['distance_m']<=route[-1]['distance_m']:
            raise ConfigError('route distances must increase after waypoint deduplication')
        route.append(p)
    return route


def bind_route_turn(route, direction, activation_m, end_m):
    """Bind the next junction, never search ahead for a convenient matching turn."""
    report=audit_route(route)
    if report['suspicious_gaps']:
        raise ConfigError('turn route contains discontinuities')
    visits=[v for v in report['junction_visits'] if v['end_m']>=activation_m]
    if not visits:
        raise ConfigError('no next junction after task activation')
    visit=visits[0]
    if visit['start_m']<=activation_m:
        raise ConfigError('turn task activated inside or after junction entry')
    if visit['end_m']>=end_m:
        raise ConfigError('turn exit overlaps next task boundary')
    if not visit['distinct_road_turn_candidate'] or visit['direction']!=direction:
        raise ConfigError('next junction does not match requested distinct-road turn')
    return dict(direction=direction,legal=True,entry_road_key=str(visit['entry_road_id']),
                exit_road_key=str(visit['exit_road_id']),junction_start_m=visit['start_m'],
                junction_end_m=visit['end_m'],heading_change_deg=visit['heading_change_deg'],
                legality_scope='recorded_route_topology_not_signal_permission')


def build_turn_route(world_map, reference, direction, preferred_m):
    from agents.navigation.global_route_planner import GlobalRoutePlanner

    anchor=min(reference,key=lambda p:abs(p['distance_m']-preferred_m))
    candidates=[]
    for entry,exit_wp in world_map.get_topology():
        if not entry.is_junction:
            continue
        delta=angle_delta(exit_wp.transform.rotation.yaw,entry.transform.rotation.yaw)
        if not ((direction=='LEFT' and -145<delta<-45) or (direction=='RIGHT' and 45<delta<145)):
            continue
        position=entry.transform.location
        candidates.append(((position.x-anchor['x'])**2+(position.y-anchor['y'])**2,entry,exit_wp))
    candidates.sort(key=lambda v:(v[0],v[1].road_id,v[1].lane_id))
    planner=GlobalRoutePlanner(world_map,2.0)
    for _,entry,exit_wp in candidates:
        starts=sorted(entry.previous(25),key=lambda w:(w.road_id,w.lane_id,w.s))
        ends=sorted(exit_wp.next(60),key=lambda w:(w.road_id,w.lane_id,w.s))
        for start in starts:
            for end in ends:
                if start.is_junction or end.is_junction or start.road_id==end.road_id:
                    continue
                path=[w for w,_ in planner.trace_route(start.transform.location,end.transform.location)]
                if len(path)<3:
                    continue
                try:
                    evidence=turn_evidence(waypoint_route(path),direction)
                except ConfigError:
                    continue
                # A long continuation allows existing background-flow placement.
                cursor=path[-1]
                for _ in range(300):
                    options=cursor.next(5)
                    if not options:
                        break
                    cursor=min(options,key=lambda w:(abs(angle_delta(w.transform.rotation.yaw,
                                      cursor.transform.rotation.yaw)),w.road_id,w.lane_id,w.s))
                    path.append(cursor)
                route=waypoint_route(path)
                if route[-1]['distance_m']<1300:
                    continue
                audit_route(route)
                return route,evidence
    raise ConfigError('no directed-map junction turn could be constructed for requested direction')


def bind_straight_junction(route, world_map, location_factory, activation_m, end_m, max_heading_deg):
    report = audit_route(route)
    if report['suspicious_gaps']:
        raise ConfigError('junction route contains discontinuities')
    visits = [v for v in report['junction_visits'] if v['end_m'] >= activation_m]
    if not visits:
        raise ConfigError('no junction in remaining route')
    visit = visits[0]
    if not activation_m < visit['start_m'] < visit['end_m'] < end_m:
        raise ConfigError('straight junction not contained in task window')
    if abs(visit['heading_change_deg']) > max_heading_deg:
        raise ConfigError('next junction is not straight')
    ids = set()
    for point in route:
        if visit['start_m'] <= point['distance_m'] <= visit['end_m'] and point['is_junction']:
            waypoint = world_map.get_waypoint(location_factory(**{k:point[k] for k in ('x','y','z')}))
            if waypoint is None or not waypoint.is_junction:
                raise ConfigError('junction geometry differs from recorded route')
            ids.add(waypoint.junction_id)
    if len(ids) != 1:
        raise ConfigError('straight fixture must identify exactly one junction')
    return dict(direction='STRAIGHT',legal=True,junction_id=ids.pop(),
                entry_road_key=str(visit['entry_road_id']),exit_road_key=str(visit['exit_road_id']),
                junction_start_m=visit['start_m'],junction_end_m=visit['end_m'],
                legality_scope='recorded_route_topology_not_signal_permission')


def bind_junction_sequence(route, world_map, location_factory, steps, activation_m, end_m):
    """Bind every requested junction once, in route order, without skipping mismatches."""
    cursor=activation_m
    result={}
    for index,step in enumerate(steps):
        if step['kind'] not in {'turn','straight_junction'}:
            continue
        start=max(cursor,step.get('start_route_s_m',cursor))
        if step['kind']=='turn':
            evidence=bind_route_turn(route,step['direction'],start,end_m)
        else:
            evidence=bind_straight_junction(route,world_map,location_factory,start,end_m,
                                            step['max_heading_change_deg'])
        result[str(index)]=evidence
        cursor=evidence['junction_end_m']+1e-3
    return result
