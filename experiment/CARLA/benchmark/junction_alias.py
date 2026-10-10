"""Truth-side handling of statically proven, geometrically coincident IDs."""
import math


def coincident_junction_alias_matches(ego, segments, lateral_limit, heading_limit):
    if ego.get('in_junction') is not True:
        return False
    aliases={key for segment in segments
        if segment.get('junction_alias_source') == 'static_topology_coincident_within_0.05m_1deg'
        for key in segment.get('coincident_junction_lane_keys', []) if isinstance(key, str)}
    position = ego.get('route_s_m')
    if (isinstance(position, (int, float)) and not isinstance(position, bool)
            and math.isfinite(position)):
        for segment in segments:
            if segment.get('shared_entry_alias_source') in {
                    'static_topology_shared_entry_0.2m_5deg',
                    'static_topology_shared_entry_0.2m_10deg'}:
                for interval in segment.get('shared_entry_junction_lane_ranges', []):
                    first, last = interval.get('start_m'), interval.get('end_m')
                    if (all(isinstance(value, (int, float)) and not isinstance(value, bool)
                            and math.isfinite(value) for value in (first, last))
                            and first <= position <= last):
                        aliases.add(interval.get('lane_key'))
            if segment.get('junction_alias_source') != 'static_topology_coincident_within_0.05m_1deg':
                continue
            for interval in segment.get('coincident_junction_lane_ranges', []):
                first, last = interval.get('start_m'), interval.get('end_m')
                if (all(isinstance(value, (int, float)) and not isinstance(value, bool)
                        and math.isfinite(value) for value in (first, last))
                        and first <= position <= last):
                    aliases.add(interval.get('lane_key'))
    # Intersecting connectors can make nearest-waypoint ID discontinuous even
    # when the physical pose remains centered on the verified route.
    junction_projection = any(
        segment.get('junction_geometry_source') == 'static_topology_verified_route_projection'
        and ego.get('junction_id') in segment.get('verified_junction_ids', [])
        and isinstance(position, (int, float)) and not isinstance(position, bool)
        and math.isfinite(position)
        and segment['start_m'] <= position <= segment['end_m']
        for segment in segments)
    if ego.get('lane_key') not in aliases and not junction_projection:
        return False
    error, yaw, reference = (ego.get(key) for key in ('route_error_m', 'yaw_deg', 'route_yaw_deg'))
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in (error, yaw, reference)):
        return False
    delta=(yaw - reference + 180.) % 360. - 180.
    return 0. <= error <= lateral_limit and abs(delta) <= heading_limit
