"""Repair discontinuous planner traces using actual directed lane successors."""
from collections import deque
import math


class RouteContinuityError(ValueError):
    pass


def _distance(a,b):
    x,y=a.transform.location,b.transform.location
    return math.sqrt((x.x-y.x)**2+(x.y-y.y)**2+(x.z-y.z)**2)


def _heading_delta(a,b):
    return abs((a-b+180.)%360.-180.)


def _connection_option(start,target,fallback):
    delta=(target.transform.rotation.yaw-start.transform.rotation.yaw+180.)%360.-180.
    if start.road_id==target.road_id or not 45.<=abs(delta)<150.:
        return fallback
    name='RIGHT' if delta>0. else 'LEFT'
    return name if isinstance(fallback,str) else getattr(type(fallback),name,fallback)


def _legal_lane_transition(a,b):
    if (getattr(a,'is_junction',False) or getattr(b,'is_junction',False)
            or (a.road_id,a.section_id)!=(b.road_id,b.section_id)
            or abs(a.lane_id-b.lane_id)!=1
            or _heading_delta(a.transform.rotation.yaw,b.transform.rotation.yaw)>15.
            or _distance(a,b)>20.):
        return False
    marking=str(getattr(a,'lane_change','None')).split('.')[-1]
    for direction,method in [('Left','get_left_lane'),('Right','get_right_lane')]:
        if marking not in {direction,'Both'}:
            continue
        getter=getattr(a,method,None)
        neighbor=getter() if callable(getter) else None
        if neighbor is not None and (neighbor.road_id,neighbor.section_id,neighbor.lane_id)==\
                (b.road_id,b.section_id,b.lane_id):
            origin,target=a.transform.location,b.transform.location
            yaw=math.radians(a.transform.rotation.yaw)
            forward=(target.x-origin.x)*math.cos(yaw)+(target.y-origin.y)*math.sin(yaw)
            return forward>0.
    return False


def _broken(a,b,sampling):
    length=_distance(a,b)
    if length<.05:
        return False
    if _legal_lane_transition(a,b):
        return False
    origin,target=a.transform.location,b.transform.location
    yaw=math.radians(a.transform.rotation.yaw)
    forward=(target.x-origin.x)*math.cos(yaw)+(target.y-origin.y)*math.sin(yaw)
    return length>max(10.,sampling*5.) or (length>sampling*1.5 and forward<-.2*length)


def _key(wp,sampling):
    location=wp.transform.location
    return (wp.road_id,wp.section_id,wp.lane_id,
        round(location.x/(sampling*.5)),round(location.y/(sampling*.5)),round(location.z/.25))


def _connect(start,target,sampling,max_distance):
    queue=deque([(start,[],0.)]);seen={_key(start,sampling)};examined=0
    while queue and examined<2000:
        current,path,travel=queue.popleft();examined+=1
        yaw=math.radians(current.transform.rotation.yaw)
        location,goal=current.transform.location,target.transform.location
        remaining_forward=(goal.x-location.x)*math.cos(yaw)+(goal.y-location.y)*math.sin(yaw)
        if (path and (current.road_id,current.section_id,current.lane_id)==
                (target.road_id,target.section_id,target.lane_id)
                and _distance(current,target)<=sampling*1.25
                and remaining_forward>=-.05
                and _heading_delta(current.transform.rotation.yaw,target.transform.rotation.yaw)<=25.
                and not _broken(current,target,sampling)):
            return path
        if travel>=max_distance:
            continue
        successors=sorted(current.next(sampling),key=lambda wp:_key(wp,sampling))
        for successor in successors:
            step=_distance(current,successor)
            if step<.05 or step>sampling*2.5 or _broken(current,successor,sampling):
                continue
            key=_key(successor,sampling)
            if key in seen:
                continue
            seen.add(key);queue.append((successor,path+[successor],travel+step))
    return None


def repair_route_continuity(route,sampling_m=2.):
    """No interpolation across missing roads, teleports or synthetic lane changes.

    A bounded backward search replaces an overshot junction trace with genuine
    map successors. An unrepairable trace is rejected before vehicle execution.
    """
    sampling=float(sampling_m)
    if not math.isfinite(sampling) or sampling<=0:
        raise ValueError('Positive finite route sampling required')
    result=[];repairs=[]
    route=list(route)
    if len(route)>=3:
        first,second,third=(item[0] for item in route[:3])
        a,b=first.transform.location,second.transform.location
        yaw=math.radians(first.transform.rotation.yaw)
        forward=(b.x-a.x)*math.cos(yaw)+(b.y-a.y)*math.sin(yaw)
        same_road=(first.road_id,first.section_id)==(second.road_id,second.section_id)
        near_lateral=(2.<_distance(first,second)<6. and abs(forward)<sampling)
        aligned=_heading_delta(first.transform.rotation.yaw,second.transform.rotation.yaw)<15.
        neighbor_match=False
        for method in ('get_left_lane','get_right_lane'):
            getter=getattr(first,method,None)
            neighbor=getter() if callable(getter) else None
            if neighbor is not None and (neighbor.road_id,neighbor.section_id,neighbor.lane_id)==\
                    (second.road_id,second.section_id,second.lane_id) and _distance(neighbor,second)<=max(sampling*2.5,4.5):
                neighbor_match=True
        # A spawn anchor may precede the planner's chosen lane by one lateral
        # sample. Spawn on the forward trace itself; never execute that edge.
        if same_road and near_lateral and aligned and neighbor_match and not _broken(second,third,sampling):
            route=route[1:]
            repairs.append(dict(source='map_adjacent_lane_spawn_anchor',removed_waypoints=1,
                added_waypoints=0,policy_truth_access=False))
    for input_index,item in enumerate(route):
        if not result:
            result.append(item);continue
        if _distance(result[-1][0],item[0])<.05:
            result[-1]=item
            continue
        if not _broken(result[-1][0],item[0],sampling):
            result.append(item);continue
        anchor=len(result)-1;rewind=0.;connection=None
        while anchor>=0 and rewind<=40.:
            # A local junction repair cannot use a long detour as a shortcut.
            connection=_connect(result[anchor][0],item[0],sampling,
                min(70.,rewind+_distance(result[-1][0],item[0])+12.))
            if connection is not None:
                break
            if anchor==0:
                break
            rewind+=_distance(result[anchor-1][0],result[anchor][0]);anchor-=1
        if connection is None:
            raise RouteContinuityError(f'No directed local connection at trace waypoint {input_index}')
        record=dict(input_index=input_index,rewind_m=rewind,
            removed_waypoints=len(result)-anchor-1,added_waypoints=len(connection),
            source='directed_map_waypoint_successors',policy_truth_access=False)
        option=_connection_option(result[anchor][0],item[0],item[1])
        result=result[:anchor+1]+[(wp,option) for wp in connection]
        if _distance(result[-1][0],item[0])>.05:
            result.append(item)
        repairs.append(record)
    if any(_broken(a[0],b[0],sampling) for a,b in zip(result,result[1:])):
        raise RouteContinuityError('Repaired trace remains discontinuous')
    return result,repairs
