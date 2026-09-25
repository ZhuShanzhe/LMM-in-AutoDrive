"""Offline geometry evidence for route fixtures, not a traffic-legality certificate."""
import math

from .catalog import ConfigError, number
from .truth_capture import RouteProjector, angle_delta


def audit_route(route):
    RouteProjector(route)
    for point in route:
        number(point['yaw'],'yaw')
        if not isinstance(point['is_junction'],bool):
            raise ConfigError('junction boolean required for route audit')
        for key in ('road_id','section_id','lane_id'):
            if type(point[key]) is not int:
                raise ConfigError(f'{key}: integer topology ID required')
    visits=[]
    first=None
    for index,point in enumerate(route):
        if point['is_junction'] and first is None:
            first=index
        if first is not None and (not point['is_junction'] or index==len(route)-1):
            complete=first>0 and not point['is_junction']
            entry=route[max(0,first-1)]
            delta=angle_delta(point['yaw'],entry['yaw'])
            direction=('U_TURN' if abs(delta)>=150 else 'RIGHT' if delta>=45
                       else 'LEFT' if delta<=-45 else 'STRAIGHT_OR_SHALLOW')
            visits.append(dict(start_m=route[first]['distance_m'],end_m=point['distance_m'],
                entry_road_id=entry['road_id'],exit_road_id=point['road_id'],
                heading_change_deg=round(delta,3),direction=direction,complete=complete,
                distinct_road_turn_candidate=complete and entry['road_id']!=point['road_id'] and abs(delta)>=45))
            first=None
    lengths=[math.sqrt(sum((b[k]-a[k])**2 for k in ('x','y','z'))) for a,b in zip(route,route[1:])]
    jumps=[dict(start_m=a['distance_m'],end_m=b['distance_m'],geometric_m=round(length,3))
           for a,b,length in zip(route,route[1:],lengths)
           if length > max(15,2*(b['distance_m']-a['distance_m']))]
    return dict(schema_version='route_audit/1.0',points=len(route),
        declared_length_m=route[-1]['distance_m']-route[0]['distance_m'],
        polyline_length_m=sum(lengths),max_segment_m=max(lengths),
        suspicious_gaps=jumps,junction_visits=visits,
        distinct_road_turn_candidates=[v for v in visits if v['distinct_road_turn_candidate']],
        legal_turns_verified=False,lane_count_verified=False,scene_config_binding_verified=False,
        scope='recorded_route_geometry_only')
