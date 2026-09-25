"""Bind existing scenario actors and official crosswalk geometry for evaluation."""
import math
from types import SimpleNamespace

from .catalog import ConfigError, number
from .truth_capture import ActorBinding, RouteProjector, polygons_intersect, validate_polygon


def bind_task_roles(spec, actors, initial_progress):
    from .role_journal import required_role_types
    required=required_role_types(spec)
    bindings={}
    for role,expected in required.items():
        matches=[a for a in actors if a.attributes.get('role_name')==role and a.is_alive]
        if len(matches)!=1:
            raise ConfigError(f'role {role} requires one live actor, found {len(matches)}')
        actor=matches[0]
        if not actor.type_id.startswith(expected):
            raise ConfigError(f'role {role} has incompatible actor type {actor.type_id}')
        if role not in initial_progress:
            raise ConfigError(f'missing initial route progress for role {role}')
        bindings[role]=ActorBinding.from_actor(actor,initial_progress[role])
    if len({b.actor_id for b in bindings.values()})!=len(bindings):
        raise ConfigError('roles must bind distinct actors')
    return bindings


def closed_crosswalks(points):
    polygons=[]
    current=[]
    for point in points:
        xy=[float(point.x),float(point.y)]
        current.append(xy)
        if len(current)>=4 and math.dist(current[0],current[-1])<.05:
            polygons.append(current[:-1])
            current=[]
    if current:
        raise ConfigError('unterminated official crosswalk polygon')
    return polygons


def crosswalk_fixture(world_map, route, polygon_index, route_hint, clearance_m=2):
    if type(polygon_index) is not int or polygon_index<0 or not math.isfinite(clearance_m) or clearance_m<0:
        raise ConfigError('invalid crosswalk fixture parameters')
    polygons=closed_crosswalks(world_map.get_crosswalks())
    if polygon_index>=len(polygons):
        raise ConfigError('official crosswalk polygon unavailable')
    polygon=validate_polygon(polygons[polygon_index])
    projector=RouteProjector(route)
    hits=[]
    for point in route:
        if abs(point['distance_m']-route_hint)>80:
            continue
        yaw=math.radians(point['yaw'])
        footprint=[(point['x']+math.cos(yaw)*x-math.sin(yaw)*y,
                    point['y']+math.sin(yaw)*x+math.cos(yaw)*y)
                   for x,y in [(-2.5,-1.5),(2.5,-1.5),(2.5,1.5),(-2.5,1.5)]]
        if polygons_intersect(footprint,polygon):
            hits.append(point)
    if not hits:
        raise ConfigError('official crosswalk does not intersect this route near anchor')
    height=hits[0]['z']
    projected=[projector.project(SimpleNamespace(x=x,y=y,z=height),route_hint)['route_s_m']
               for x,y in polygon]
    stop_line=min(projected)-clearance_m
    if stop_line<=route[0]['distance_m']:
        raise ConfigError('insufficient approach before crosswalk')
    return dict(stop_line_route_s_m=stop_line,conflict_polygon_xy=[list(p) for p in polygon],
                geometry_source='official_crosswalk_polygon',polygon_index=polygon_index,
                route_anchor_m=route_hint)


def crosswalk_candidates(world_map, route, preferred_m, search_window_m=500):
    points=[p for p in route if abs(p['distance_m']-preferred_m)<=search_window_m]
    result=[]
    if not points:
        return result
    for index,polygon in enumerate(closed_crosswalks(world_map.get_crosswalks())):
        cx=sum(p[0] for p in polygon)/len(polygon)
        cy=sum(p[1] for p in polygon)/len(polygon)
        closest=min(points,key=lambda p:(p['x']-cx)**2+(p['y']-cy)**2)
        try:
            value=crosswalk_fixture(world_map,route,index,closest['distance_m'])
        except ConfigError:
            continue
        result.append(value)
    return sorted(result,key=lambda p:abs(p['route_anchor_m']-preferred_m))


def route_crossing_fixture(world_map, route, anchor_m, location_factory,
                           half_length_m=3.0, clearance_m=2.0):
    """A declared unmarked crossing footprint, not an invented official crosswalk."""
    from .truth_capture import angle_delta
    anchor_m=number(anchor_m,'crossing anchor')
    if number(half_length_m,'crossing half length')<=0 or number(clearance_m,'clearance')<0:
        raise ConfigError('invalid unmarked crossing dimensions')
    if not route or not route[0]['distance_m']<anchor_m<route[-1]['distance_m']:
        raise ConfigError('crossing anchor outside route')
    point=min(route,key=lambda p:abs(p['distance_m']-anchor_m))
    if abs(point['distance_m']-anchor_m)>5:
        raise ConfigError('route too sparse at crossing anchor')
    location=location_factory(**{k:point[k] for k in ('x','y','z')})
    waypoint=world_map.get_waypoint(location)
    if waypoint is None or waypoint.is_junction:
        raise ConfigError('unmarked crossing requires non-junction route lane')
    if (waypoint.road_id,waypoint.section_id,waypoint.lane_id)!=(point['road_id'],point['section_id'],point['lane_id']):
        raise ConfigError('crossing route lane differs from actual map')
    width=number(waypoint.lane_width,'crossing lane width')
    if width<=0 or abs(angle_delta(waypoint.transform.rotation.yaw,point['yaw']))>10:
        raise ConfigError('invalid crossing lane geometry')
    yaw=math.radians(point['yaw'])
    polygon=[(point['x']+math.cos(yaw)*x-math.sin(yaw)*y,
              point['y']+math.sin(yaw)*x+math.cos(yaw)*y)
             for x,y in [(-half_length_m,-width/2),(half_length_m,-width/2),
                         (half_length_m,width/2),(-half_length_m,width/2)]]
    stop_line=point['distance_m']-half_length_m-clearance_m
    if stop_line<=route[0]['distance_m']:
        raise ConfigError('insufficient crossing approach')
    return dict(stop_line_route_s_m=stop_line,
                conflict_polygon_xy=[list(p) for p in validate_polygon(polygon)],
                geometry_source='configured_unmarked_crossing_on_route_lane',
                requested_anchor_m=anchor_m,route_anchor_m=point['distance_m'],
                half_length_m=half_length_m,lane_width_m=width)


def bus_passenger_fixture(world_map, route, event, roles, location_factory):
    """Declared bus-stop approach lane area; not an official crossing."""
    passengers = event.get('passengers', [])
    configured = {item.get('role_name'): item for item in passengers}
    if (event.get('kind') != 'bus_stop' or len(configured) != len(passengers)
            or not roles or not set(roles) <= set(configured)):
        raise ConfigError('bus clearance requires explicit unique passenger roles')
    offsets = [number(configured[role].get('longitudinal_offset_m'), 'passenger offset') for role in roles]
    fixture = route_crossing_fixture(world_map, route,
        number(event.get('anchor_progress_m'), 'bus anchor'), location_factory,
        half_length_m=max(abs(value) for value in offsets) + 3.0)
    fixture.update(geometry_source='configured_bus_stop_route_lane_area',
                   passenger_roles=sorted(roles),
                   limitation='route-lane hazard area only; does not prove boarding or curb alignment')
    return fixture
