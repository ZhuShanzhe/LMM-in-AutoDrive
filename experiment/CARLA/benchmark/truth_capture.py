"""Snapshot-only benchmark observations; this module never ticks or controls CARLA."""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math

from .catalog import ConfigError, number


def lane_key(waypoint):
    return f'{waypoint.road_id}:{waypoint.section_id}:{waypoint.lane_id}'


def angle_delta(a, b):
    return (a - b + 180) % 360 - 180


class RouteProjector:
    """Project on a bounded route neighborhood, rejecting ambiguous overlapping laps."""

    def __init__(self, route):
        self.route = copy.deepcopy(route)
        if len(route) < 2:
            raise ConfigError('route needs at least two points')
        for point in route:
            for key in ('x', 'y', 'z', 'distance_m'):
                number(point[key], key)
        for a, b in zip(route, route[1:]):
            if b['distance_m'] <= a['distance_m'] or math.hypot(b['x']-a['x'], b['y']-a['y']) < 1e-5:
                raise ConfigError('route must have nondegenerate ordered segments')

    def project(self, location, hint, window=80.0):
        hint = number(hint, 'route hint')
        if not self.route[0]['distance_m'] <= hint <= self.route[-1]['distance_m']:
            raise ConfigError('route hint outside route')
        candidates = []
        for a, b in zip(self.route, self.route[1:]):
            if b['distance_m'] < hint-window or a['distance_m'] > hint+window:
                continue
            dx, dy = b['x']-a['x'], b['y']-a['y']
            ratio = max(0., min(1., ((location.x-a['x'])*dx + (location.y-a['y'])*dy)/(dx*dx+dy*dy)))
            s = a['distance_m'] + ratio*(b['distance_m']-a['distance_m'])
            error = math.sqrt((location.x-a['x']-ratio*dx)**2 +
                              (location.y-a['y']-ratio*dy)**2 +
                              (location.z-a['z']-ratio*(b['z']-a['z']))**2)
            candidates.append((error, s, math.degrees(math.atan2(dy, dx))))
        if not candidates:
            raise ConfigError('no route segment near hint')
        candidates.sort()
        best = candidates[0]
        if any(error <= best[0]+.25 and abs(s-best[1]) > 25 for error,s,yaw in candidates[1:]):
            raise ConfigError('ambiguous route projection')
        return dict(route_s_m=best[1], route_error_m=best[0], route_yaw_deg=best[2])


def validate_polygon(vertices):
    if not isinstance(vertices, list) or len(vertices) < 3:
        raise ConfigError('conflict zone needs a convex polygon')
    points = []
    for vertex in vertices:
        if len(vertex) != 2:
            raise ConfigError('polygon vertices must be XY pairs')
        points.append(tuple(number(v, 'polygon coordinate') for v in vertex))
    crosses = []
    for i, a in enumerate(points):
        b, c = points[(i+1)%len(points)], points[(i+2)%len(points)]
        cross = (b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0])
        if abs(cross) < 1e-8:
            raise ConfigError('degenerate conflict polygon')
        crosses.append(cross)
    if not (all(c>0 for c in crosses) or all(c<0 for c in crosses)):
        raise ConfigError('conflict polygon must be convex and ordered')
    # Convex ordered vertices must also form a simple polygon.
    for i,a in enumerate(points):
        b=points[(i+1)%len(points)]
        signs=[(b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0])
               for j,p in enumerate(points) if j not in (i,(i+1)%len(points))]
        if not (all(s>0 for s in signs) or all(s<0 for s in signs)):
            raise ConfigError('self-intersecting conflict polygon')
    return points


def polygons_intersect(first, second):
    """Separating-axis test; touching the boundary still counts as conflict."""
    for polygon in (first, second):
        for i, a in enumerate(polygon):
            b = polygon[(i+1)%len(polygon)]
            axis = (a[1]-b[1], b[0]-a[0])
            values = [[p[0]*axis[0]+p[1]*axis[1] for p in points] for points in (first,second)]
            if max(values[0]) < min(values[1]) or max(values[1]) < min(values[0]):
                return False
    return True


@dataclass(frozen=True)
class ActorBinding:
    actor_id: int
    half_length_m: float
    half_width_m: float
    initial_route_s_m: float
    offset_x_m: float = 0.
    offset_y_m: float = 0.
    box_yaw_deg: float = 0.

    def __post_init__(self):
        if isinstance(self.actor_id, bool) or not isinstance(self.actor_id, int) or self.actor_id <= 0:
            raise ConfigError('positive actor ID required')
        for key,value in asdict(self).items():
            number(value, key)
        if self.half_length_m <= 0 or self.half_width_m <= 0:
            raise ConfigError('positive bounding box dimensions required')

    @classmethod
    def from_actor(cls, actor, initial_route_s_m):
        # Only static identity/geometry is read from the live actor handle.
        box = actor.bounding_box
        return cls(actor.id, box.extent.x, box.extent.y, initial_route_s_m,
                   box.location.x, box.location.y, box.rotation.yaw)

    def footprint(self, transform):
        yaw = math.radians(transform.rotation.yaw)
        cx = transform.location.x + math.cos(yaw)*self.offset_x_m-math.sin(yaw)*self.offset_y_m
        cy = transform.location.y + math.sin(yaw)*self.offset_x_m+math.cos(yaw)*self.offset_y_m
        yaw += math.radians(self.box_yaw_deg)
        return [(cx+math.cos(yaw)*x-math.sin(yaw)*y, cy+math.sin(yaw)*x+math.cos(yaw)*y)
                for x,y in [(-self.half_length_m,-self.half_width_m),
                            (self.half_length_m,-self.half_width_m),
                            (self.half_length_m,self.half_width_m),
                            (-self.half_length_m,self.half_width_m)]]


def prepare_lane_fixture(world_map, location, direction, entry_waypoint=None):
    if direction not in {'LEFT','RIGHT'}:
        raise ConfigError('LEFT or RIGHT required')
    entry = entry_waypoint if entry_waypoint is not None else world_map.get_waypoint(location)
    if entry_waypoint is not None and entry.transform.location.distance(location) > .75:
        raise ConfigError('verified entry differs from route position')
    if entry is None or entry.is_junction or str(entry.lane_type) != 'Driving':
        raise ConfigError('entry must be a nonjunction driving lane')
    target = entry.get_left_lane() if direction == 'LEFT' else entry.get_right_lane()
    if target is None or str(target.lane_type) != 'Driving' or target.is_junction:
        raise ConfigError('adjacent driving lane missing')
    if target.road_id != entry.road_id or abs(angle_delta(target.transform.rotation.yaw, entry.transform.rotation.yaw)) >= 60:
        raise ConfigError('adjacent lane is not same-direction')
    if str(entry.lane_change) not in {'Both', direction.title()}:
        raise ConfigError('lane change not permitted at entry')
    return dict(direction=direction, legal=True, entry_lane_key=lane_key(entry),
                target_lane_key=lane_key(target))


class SnapshotTruthCollector:
    """Receive one frozen WorldSnapshot plus explicitly finalized safety/validity."""

    def __init__(self, spec, route, world_map, ego, roles, fixture, corridor_id):
        from .task_oracle import validate_spec
        self.spec = copy.deepcopy(validate_spec(spec))
        self.projector = RouteProjector(route)
        self.world_map = world_map
        self.bindings = {'ego': ego, **roles}
        if 'ego' in roles or len({b.actor_id for b in self.bindings.values()}) != len(self.bindings):
            raise ConfigError('actor bindings must be distinct')
        required = {s['target_role'] for s in spec['steps'] if 'target_role' in s}
        required.update(role for s in spec['steps'] for role in s.get('target_roles', []))
        if required-set(roles):
            raise ConfigError(f'missing actor bindings: {sorted(required-set(roles))}')
        if not isinstance(corridor_id,str) or not corridor_id:
            raise ConfigError('route corridor ID required')
        self.corridor_id = corridor_id
        self.fixture = copy.deepcopy(fixture)
        self.zones = {}
        for index,step in enumerate(spec['steps']):
            if step['kind'] in {'lane_change','turn','yield_pedestrian','wait_clear','straight_junction'}:
                value = self.fixture['steps'][str(index)]
                if step['kind'] in {'yield_pedestrian', 'wait_clear'}:
                    number(value['stop_line_route_s_m'], 'stop line')
                    roles = step.get('target_roles', [step.get('target_role')])
                    for role in roles:
                        polygon = validate_polygon(value['conflict_polygon_xy'])
                        if role in self.zones and self.zones[role] != polygon:
                            raise ConfigError('role has conflicting assessment zones')
                        self.zones[role] = polygon
        self.hints = {role:b.initial_route_s_m for role,b in self.bindings.items()}
        self.previous = {}
        self.last_frame = None
        self.last_time = None

    def collect(self, snapshot, safety, validity):
        frame = snapshot.frame
        now = number(snapshot.timestamp.elapsed_seconds, 'snapshot time')
        if isinstance(frame,bool) or not isinstance(frame,int) or frame < 0:
            raise ConfigError('invalid snapshot frame')
        if self.last_frame is not None and (frame <= self.last_frame or now <= self.last_time):
            raise ConfigError('non-monotonic snapshot')
        if self.last_time is not None and now-self.last_time > self.spec['max_frame_gap_s']+1e-8:
            raise ConfigError('snapshot gap')
        for packet in (safety,validity):
            if type(packet.get('frame')) is not int or packet['frame'] != frame or packet.get('complete') is not True:
                raise ConfigError('safety/validity not finalized for snapshot frame')
        for key in ('collisions','violations'):
            value=safety[key]
            if isinstance(value,bool) or not isinstance(value,int) or value<0:
                raise ConfigError('invalid safety count')
        if validity.get('valid') is not True:
            raise ConfigError(f"fixture invalid: {validity.get('reason', 'unspecified')}")
        result = dict(schema_version='task_truth/1.0', source='simulator_truth',
                      task_id=self.spec['task_id'], source_sha256=self.spec.get('source_sha256'),
                      frame=frame, sim_time_s=now, scenario_valid=True,
                      fixture=copy.deepcopy(self.fixture), actors={},
                      safety={k:safety[k] for k in ('collisions','violations')})
        updates, hints = {}, {}
        for role,binding in self.bindings.items():
            actor = snapshot.find(binding.actor_id)
            if actor is None:
                raise ConfigError(f'bound actor absent from snapshot: {role}')
            transform, velocity = actor.get_transform(), actor.get_velocity()
            location = transform.location
            for value in (location.x,location.y,location.z,transform.rotation.yaw,velocity.x,velocity.y,velocity.z):
                number(value,'actor kinematics')
            speed = math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2)
            if role in self.previous:
                old,old_speed = self.previous[role]
                distance = math.sqrt(sum((a-b)**2 for a,b in zip((location.x,location.y,location.z),old)))
                if distance > max(old_speed,speed)*(now-self.last_time)+3:
                    raise ConfigError(f'abnormal actor displacement: {role}')
            projection = self.projector.project(location,self.hints[role])
            waypoint = self.world_map.get_waypoint(location)
            if waypoint is None:
                raise ConfigError(f'map waypoint unavailable: {role}')
            lane_transform = waypoint.transform
            yaw = math.radians(lane_transform.rotation.yaw)
            lateral = -(location.x-lane_transform.location.x)*math.sin(yaw)+(location.y-lane_transform.location.y)*math.cos(yaw)
            in_corridor = (projection['route_error_m'] <= 15 and
                           abs(angle_delta(transform.rotation.yaw,projection['route_yaw_deg'])) < 90)
            route_yaw = math.radians(projection['route_yaw_deg'])
            origin_yaw = math.radians(transform.rotation.yaw)
            offset_x = math.cos(origin_yaw)*binding.offset_x_m-math.sin(origin_yaw)*binding.offset_y_m
            offset_y = math.sin(origin_yaw)*binding.offset_x_m+math.cos(origin_yaw)*binding.offset_y_m
            box_center_s = projection['route_s_m']+offset_x*math.cos(route_yaw)+offset_y*math.sin(route_yaw)
            box_angle = origin_yaw+math.radians(binding.box_yaw_deg)-route_yaw
            longitudinal_extent = abs(math.cos(box_angle))*binding.half_length_m+abs(math.sin(box_angle))*binding.half_width_m
            record = dict(actor_id=binding.actor_id, alive=True, teleported=False,
                          position_m=dict(x=location.x,y=location.y,z=location.z),
                          speed_kmh=speed*3.6, lane_key=lane_key(waypoint), road_key=str(waypoint.road_id),
                          in_junction=bool(waypoint.is_junction), lateral_error_m=lateral,
                          junction_id=getattr(waypoint, 'junction_id', None) if waypoint.is_junction else None,
                          heading_error_deg=angle_delta(transform.rotation.yaw,lane_transform.rotation.yaw),
                          yaw_deg=transform.rotation.yaw, half_length_m=binding.half_length_m,
                          box_center_route_s_m=box_center_s,
                          front_route_s_m=box_center_s+longitudinal_extent,
                          rear_route_s_m=box_center_s-longitudinal_extent,
                          route_corridor_id=self.corridor_id if in_corridor else 'outside:'+self.corridor_id,
                          **projection)
            if role in self.zones:
                record['in_conflict_zone'] = polygons_intersect(binding.footprint(transform), self.zones[role])
            if role == 'ego':
                from .lane_position import lane_position
                record['lane_position'] = lane_position(waypoint)
                result['ego'] = record
            else:
                result['actors'][role] = record
            hints[role] = projection['route_s_m']
            updates[role] = ((location.x,location.y,location.z),speed)
        self.previous, self.hints = updates,hints
        self.last_frame,self.last_time = frame,now
        return result
