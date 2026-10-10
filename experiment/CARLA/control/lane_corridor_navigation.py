"""Resolve a lane corridor fork from the public navigation route only."""
import bisect
import math


def navigation_reference(controller, road_map, location, progress_m, offset_m):
    context=getattr(controller,'route_context',None)
    plan=getattr(controller,'route_plan',None)
    if context is not None and plan:
        distances=list(context.distances_m)
        points=[item[0].transform.location for item in plan]
        values=[(p.x,p.y,p.z) for p in points]
    else:
        route=getattr(getattr(controller,'route_manager',None),'route',None)
        if not route:return None
        try:
            distances=[float(p['distance_m']) for p in route]
            values=[(float(p['x']),float(p['y']),float(p['z'])) for p in route]
        except (KeyError,TypeError,ValueError):return None
    if (len(distances)!=len(values) or len(distances)<2
            or not all(math.isfinite(x) for x in distances)
            or not all(math.isfinite(x) for p in values for x in p)
            or any(b<=a for a,b in zip(distances,distances[1:]))):return None
    wanted=float(progress_m)+float(offset_m)
    if not math.isfinite(wanted) or not distances[0]<=wanted<=distances[-1]:return None
    index=min(max(1,bisect.bisect_left(distances,wanted)),len(distances)-1)
    first,last=distances[index-1:index+1]
    if last-first>5.:return None
    fraction=(wanted-first)/(last-first)
    xyz=[a+fraction*(b-a) for a,b in zip(values[index-1],values[index])]
    return road_map.get_waypoint(type(location)(x=xyz[0],y=xyz[1],z=xyz[2]))


def select_navigation_successor(choices, reference, target_lane):
    if len(choices)==1:return choices[0]
    if reference is None or target_lane is None:return None
    selected=[candidate for candidate in choices if _matches_reference(candidate, reference)]
    return selected[0] if len(selected)==1 else None


def _matches_reference(candidate, reference):
    if (candidate.road_id,candidate.section_id)!=(reference.road_id,reference.section_id):return False
    # Lane signs are local to one OpenDRIVE road, not globally comparable.
    if candidate.lane_id*reference.lane_id<=0:return False
    yaw=abs((candidate.transform.rotation.yaw-reference.transform.rotation.yaw+180.)%360.-180.)
    if not math.isfinite(yaw) or yaw>10.:return False
    a=candidate.transform.location;b=reference.transform.location
    return (all(math.isfinite(v) for v in (a.x,a.y,a.z,b.x,b.y,b.z))
        and math.hypot(a.x-b.x,a.y-b.y)<=max(5.,2.*float(reference.lane_width))
        and abs(a.z-b.z)<=.25)


def select_navigation_branch(choices, reference, *, extension_m):
    """Resolve a fork-boundary sample by its later public-route continuation."""
    if (reference is None or isinstance(extension_m,bool)
            or not isinstance(extension_m,(int,float))
            or not math.isfinite(extension_m) or extension_m<=0.):return None
    selected=[]
    try:
        for candidate in choices:
            if any(_matches_reference(successor,reference) for successor in candidate.next(extension_m)):
                selected.append(candidate)
    except (AttributeError,RuntimeError,TypeError,ValueError):return None
    return selected[0] if len(selected)==1 else None
