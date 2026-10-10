"""Static terminal cross-section for commands that do not name a final lane."""
import math

from .catalog import ConfigError
from .truth_capture import lane_key


SOURCE = 'static_same_direction_terminal_lane_centers/1.0'


def terminal_lane_centers(world_map, endpoint, location_factory):
    base = world_map.get_waypoint(location_factory(**{k:endpoint[k] for k in ('x','y','z')}))
    if base is None or base.is_junction:
        raise ConfigError('terminal cross-section requires a non-junction driving road')
    yaw = float(base.transform.rotation.yaw)

    def valid(point):
        return (point is not None and not point.is_junction
                and point.road_id == base.road_id and point.section_id == base.section_id
                and str(point.lane_type).split('.')[-1] == 'Driving'
                and math.cos(math.radians(float(point.transform.rotation.yaw)-yaw)) > .5
                and float(point.lane_width) > 0.)

    if not valid(base):
        raise ConfigError('terminal reference is not a driving lane')
    points = [base]
    seen = {lane_key(base)}
    for direction in ('get_left_lane','get_right_lane'):
        current = base
        for _ in range(16):
            current = getattr(current,direction)()
            if not valid(current) or lane_key(current) in seen:
                break
            seen.add(lane_key(current));points.append(current)
        else:
            raise ConfigError('terminal lane enumeration exceeded bounded topology')
    return dict(source=SOURCE, lane_points=[dict(lane_key=lane_key(point),
        position_m={k:float(getattr(point.transform.location,k)) for k in ('x','y','z')},
        yaw_deg=float(point.transform.rotation.yaw),lane_width_m=float(point.lane_width)) for point in points])
