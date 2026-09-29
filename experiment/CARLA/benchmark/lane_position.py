"""Map-derived, ego-direction-relative lane numbering for assessment only."""
import math


def lane_position(waypoint):
    invalid = dict(valid=False, index_from_left=None, index_from_right=None, lane_count=None)
    if waypoint is None:
        return dict(invalid, reason='missing_waypoint')
    if waypoint.is_junction:
        return dict(invalid, reason='junction_has_no_stable_lane_ordinal')

    def driving(wp):
        return str(getattr(wp, 'lane_type', '')).split('.')[-1] == 'Driving'

    if not driving(waypoint):
        return dict(invalid, reason='not_a_driving_lane')
    try:
        origin_yaw = float(waypoint.transform.rotation.yaw)
        if not math.isfinite(origin_yaw):
            return dict(invalid, reason='invalid_lane_heading')
        seen = {(waypoint.road_id, waypoint.section_id, waypoint.lane_id)}
        counts = []
        for side in ('left', 'right'):
            count, current = 0, waypoint
            for _ in range(32):
                adjacent = getattr(current, 'get_' + side + '_lane')()
                if adjacent is None or not driving(adjacent):
                    break
                yaw = float(adjacent.transform.rotation.yaw)
                if not math.isfinite(yaw):
                    return dict(invalid, reason='invalid_adjacent_heading')
                if abs((yaw-origin_yaw+180) % 360-180) >= 45:
                    break
                key = (adjacent.road_id, adjacent.section_id, adjacent.lane_id)
                if key in seen or key[:2] != (waypoint.road_id, waypoint.section_id) or adjacent.is_junction:
                    return dict(invalid, reason='ambiguous_lateral_topology')
                seen.add(key)
                count += 1
                current = adjacent
            else:
                return dict(invalid, reason='lateral_topology_limit')
            counts.append(count)
        return dict(valid=True, reason=None, index_from_left=counts[0]+1,
                    index_from_right=counts[1]+1, lane_count=sum(counts)+1)
    except (AttributeError, TypeError, ValueError):
        return dict(invalid, reason='incomplete_map_lane_metadata')


def matches(position, side, index):
    from .catalog import ConfigError
    if isinstance(position, dict) and position.get('reason') == 'junction_has_no_stable_lane_ordinal':
        return False
    if not isinstance(position, dict) or position.get('valid') is not True:
        raise ConfigError('lane ordinal unavailable in current map observation')
    left, right, count = (position.get(key) for key in ('index_from_left','index_from_right','lane_count'))
    if any(type(value) is not int or value < 1 for value in (left,right,count)) or left+right != count+1:
        raise ConfigError('inconsistent lane ordinal observation')
    if index > count:
        raise ConfigError('requested lane ordinal does not exist')
    return position['index_from_' + side.lower()] == index
