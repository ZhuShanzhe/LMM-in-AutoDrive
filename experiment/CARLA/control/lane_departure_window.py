"""Static topology bound for starting, not completing, a lateral maneuver."""

import math


def relative_lane_rebind_allowed(document,state):
    if document is None:return True
    index=(state or {}).get('active_step_index')
    steps=(document.get('intent') or {}).get('steps') or []
    if not isinstance(index,int) or isinstance(index,bool) or not 0<=index<len(steps):return False
    step=steps[index];parameters=step.get('parameters') or {}
    return (step.get('action')=='CHANGE_LANE' and parameters.get('direction') in {'LEFT','RIGHT'}
        and set(parameters)<={'direction'}
        and not any(step.get(key) for key in ('target','target_ref','lane_ref','references','selectors')))


def lane_departure_window(current, target, *, horizon_m=20.):
    result=dict(allowed=False, source='static_lane_departure_topology',
                horizon_m=horizon_m, reason='departure_geometry_unavailable')
    if (current is None or target is None or not isinstance(horizon_m,(int,float))
            or isinstance(horizon_m,bool) or not math.isfinite(horizon_m) or horizon_m<5.):
        return result
    try:
        for distance in range(int(math.ceil(horizon_m))+1):
            for initial in (current,target):
                choices=initial.next(float(distance)) if distance else [initial]
                if len(choices)!=1:
                    return dict(result,reason='departure_branch_or_endpoint',distance_m=distance)
                waypoint=choices[0]
                if waypoint.is_junction:
                    return dict(result,reason='departure_junction_horizon',distance_m=distance)
                if int(waypoint.lane_id)*int(initial.lane_id)<=0:
                    return dict(result,reason='departure_direction_discontinuity',distance_m=distance)
    except (AttributeError,RuntimeError,TypeError,ValueError):
        return result
    return dict(result,allowed=True,reason='nonjunction_unique_lane_horizon')
