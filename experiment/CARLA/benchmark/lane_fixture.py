"""Isolated lane-change geometry and conservative baseline trigger checks."""
import copy
import math

from .catalog import ConfigError
from .truth_capture import angle_delta, lane_key, prepare_lane_fixture


def select_lane_segment(world_map, route, direction, location_factory, *, preferred_m, length_m):
    candidates = sorted(range(len(route)), key=lambda i: abs(route[i]['distance_m']-preferred_m))
    for index in candidates:
        start = route[index]['distance_m']
        segment = [p for p in route[index:] if p['distance_m'] <= start+length_m+5]
        if not segment or segment[-1]['distance_m'] < start+length_m:
            continue
        fixtures, targets = [], []
        try:
            for p in segment:
                location = location_factory(x=p['x'], y=p['y'], z=p['z'])
                entry = world_map.get_waypoint(location)
                if entry is None or (entry.road_id,entry.section_id,entry.lane_id) != (
                        p['road_id'],p['section_id'],p['lane_id']):
                    raise ConfigError('route/map mismatch')
                value = prepare_lane_fixture(world_map,location,direction)
                if fixtures and value != fixtures[0]:
                    raise ConfigError('lane corridor changes identity')
                if abs(angle_delta(entry.transform.rotation.yaw,segment[0]['yaw'])) > 20:
                    raise ConfigError('lane corridor is too curved for this baseline')
                fixtures.append(value)
                target = entry.get_left_lane() if direction=='LEFT' else entry.get_right_lane()
                targets.append(target.transform.location)
        except ConfigError:
            continue
        local = copy.deepcopy(route[index:])
        for p in local:
            p['distance_m'] -= start
        return local, fixtures[0], targets, start
    raise ConfigError('no sufficiently long legal lane-change corridor on recorded route')


def gap_check(snapshot, ego, vehicles, world_map, fixture):
    """Test-baseline guard only; not a general lane-change safety planner."""
    state = snapshot.find(ego.id)
    if state is None:
        raise ConfigError('ego absent from snapshot')
    transform = state.get_transform()
    entry = world_map.get_waypoint(transform.location)
    if entry is None or lane_key(entry) != fixture['entry_lane_key']:
        return dict(clear=False,reason='entry_lane_changed',blocking_actor_ids=[])
    current = prepare_lane_fixture(world_map,transform.location,fixture['direction'])
    if current != fixture:
        return dict(clear=False,reason='fixture_changed',blocking_actor_ids=[])
    target = entry.get_left_lane() if fixture['direction']=='LEFT' else entry.get_right_lane()
    center = target.transform.location
    yaw = math.radians(target.transform.rotation.yaw)
    ux,uy = math.cos(yaw),math.sin(yaw)
    speed = state.get_velocity()
    ego_speed = max(0,speed.x*ux+speed.y*uy)
    blockers = []
    for vehicle in vehicles:
        if vehicle.id == ego.id:
            continue
        other = snapshot.find(vehicle.id)
        if other is None:
            continue
        pose = other.get_transform()
        if abs(pose.location.z-center.z) > 3:
            continue
        dx,dy = pose.location.x-center.x,pose.location.y-center.y
        longitudinal = dx*ux+dy*uy
        lateral = -dx*uy+dy*ux
        extent = vehicle.bounding_box.extent
        angle = math.radians(pose.rotation.yaw)-yaw
        half_width = abs(math.sin(angle))*extent.x+abs(math.cos(angle))*extent.y
        half_length = abs(math.cos(angle))*extent.x+abs(math.sin(angle))*extent.y
        if abs(lateral) > target.lane_width/2+half_width+.5:
            continue
        velocity = other.get_velocity()
        other_speed = velocity.x*ux+velocity.y*uy
        closing = max(0,ego_speed-other_speed) if longitudinal>=0 else max(0,other_speed-ego_speed)
        gap = abs(longitudinal)-half_length-ego.bounding_box.extent.x
        if gap < 15+closing*4:
            blockers.append(vehicle.id)
    return dict(clear=not blockers,reason='clear' if not blockers else 'target_lane_gap',
                blocking_actor_ids=blockers)
