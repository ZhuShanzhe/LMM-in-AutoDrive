"""Offline geometry preparation for passing and returning, not a driving policy."""
import copy
import math
from collections import Counter

from .catalog import ConfigError
from .truth_capture import RouteProjector, angle_delta, lane_key, prepare_lane_fixture


def check_forward_link(previous, current):
    a,b=previous.transform.location,current.transform.location
    distance=math.sqrt((b.x-a.x)**2+(b.y-a.y)**2+(b.z-a.z)**2)
    if not 0<distance<=8:
        raise ConfigError('lane samples duplicate or spatially discontinuous')
    successors=previous.next(distance)
    if len(successors)!=1:
        raise ConfigError('lane corridor has ambiguous forward topology')
    successor=successors[0]
    p=successor.transform.location
    error=math.sqrt((p.x-b.x)**2+(p.y-b.y)**2+(p.z-b.z)**2)
    if lane_key(successor)!=lane_key(current) or error>.75:
        raise ConfigError('adjacent samples are not forward-connected')
    if abs(angle_delta(successor.transform.rotation.yaw,current.transform.rotation.yaw))>10:
        raise ConfigError('lane link heading mismatch')


def build_overtake_geometry(world_map, route, location_factory, *, preferred_m,
                            length_m=280, max_anchor_shift_m=500):
    RouteProjector(route)
    if not all(math.isfinite(v) for v in (preferred_m,length_m,max_anchor_shift_m)):
        raise ConfigError('finite geometry parameters required')
    if length_m<120 or max_anchor_shift_m<0:
        raise ConfigError('overtake corridor too short or invalid search radius')
    candidates=sorted((i for i,p in enumerate(route)
                       if abs(p['distance_m']-preferred_m)<=max_anchor_shift_m),
                      key=lambda i:abs(route[i]['distance_m']-preferred_m))
    rejected=[]
    for index in candidates:
        start=route[index]['distance_m']
        end=next((j for j in range(index,len(route)) if route[j]['distance_m']>=start+length_m),None)
        if end is None:
            continue
        segment=route[index:end+1]
        target_path=[]
        lane_pairs=[]
        previous_entry=previous_target=None
        outgoing=returning=None
        try:
            for n,point in enumerate(segment):
                if n and point['distance_m']-segment[n-1]['distance_m']>5:
                    raise ConfigError('route samples exceed 5 m spacing')
                location=location_factory(**{k:point[k] for k in ('x','y','z')})
                entry=world_map.get_waypoint(location)
                if entry is None or (entry.road_id,entry.section_id,entry.lane_id)!=(
                        point['road_id'],point['section_id'],point['lane_id']):
                    raise ConfigError('route/map lane mismatch')
                pair=prepare_lane_fixture(world_map,location,'LEFT')
                target=entry.get_left_lane()
                back=prepare_lane_fixture(world_map,target.transform.location,'RIGHT')
                if back['target_lane_key']!=pair['entry_lane_key']:
                    raise ConfigError('right return does not reach original lane')
                if abs(angle_delta(entry.transform.rotation.yaw,point['yaw']))>10:
                    raise ConfigError('route heading differs from map')
                pos=entry.transform.location
                if math.sqrt((pos.x-point['x'])**2+(pos.y-point['y'])**2+(pos.z-point['z'])**2)>.75:
                    raise ConfigError('route deviates from lane center')
                if previous_entry is not None:
                    check_forward_link(previous_entry,entry)
                    check_forward_link(previous_target,target)
                previous_entry,previous_target=entry,target
                if outgoing is None:
                    outgoing,returning=pair,back
                lane_pairs.append(dict(distance_m=point['distance_m']-start,
                                       outgoing=pair,returning=back))
                pos=target.transform.location
                target_path.append(dict(x=pos.x,y=pos.y,z=pos.z,
                    yaw=target.transform.rotation.yaw,distance_m=point['distance_m']-start,
                    road_id=target.road_id,section_id=target.section_id,lane_id=target.lane_id))
        except ConfigError as error:
            rejected.append(dict(source_entry_m=start,reason=str(error)))
            continue
        local=copy.deepcopy(segment)
        for point in local:
            point['distance_m']-=start
        return dict(source_entry_m=start,source_end_m=segment[-1]['distance_m'],
            requested_anchor_m=preferred_m,anchor_shift_m=start-preferred_m,
            requested_anchor_inside=start<=preferred_m<=segment[-1]['distance_m'],
            remaining_after_requested_anchor_m=max(0,segment[-1]['distance_m']-max(start,preferred_m)),
            route=local,target_lane_route=target_path,outgoing=outgoing,returning=returning,
            lane_pairs=lane_pairs,
            rejected_candidates=rejected,geometry_ready=True,execution_supported=False,
            scope='sampled static lane legality only; no actors, dynamic gaps or driving validation')
    raise ConfigError('no legal pass-and-return corridor within search radius; '+
                      (str(dict(Counter(item['reason'] for item in rejected)))
                       if rejected else 'insufficient remaining route'))
