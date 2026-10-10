"""Truth-side lane-following corridor, independent of policy completion claims."""
import math

from .catalog import ConfigError
from .truth_capture import lane_key, angle_delta, prepare_lane_fixture


def _resolve_overlapping_connectors(options,expected,location):
    matching=[wp for wp in options if lane_key(wp)==expected
              and wp.transform.location.distance(location)<=.75]
    if len(matching)!=1 or not options:return options
    reference=matching[0]
    if all(getattr(wp,'is_junction',False)
           and wp.transform.location.distance(reference.transform.location)<=.05
           and abs(angle_delta(wp.transform.rotation.yaw,reference.transform.rotation.yaw))<=1.
           for wp in options):
        return matching
    return options


def _converging_boundary_alias(connected,nearest,expected,location):
    if (nearest is None or lane_key(nearest)!=expected
            or nearest.transform.location.distance(location)>.05
            or connected.transform.location.distance(nearest.transform.location)>.75
            or abs(angle_delta(connected.transform.rotation.yaw,nearest.transform.rotation.yaw))>1.):
        return False
    if not all(callable(getattr(wp,'next',None)) for wp in (connected,nearest)):
        return False
    for distance in (1.,2.):
        left,right=connected.next(distance),nearest.next(distance)
        if (len(left)==len(right)==1 and lane_key(left[0])==lane_key(right[0])
                and left[0].transform.location.distance(right[0].transform.location)<=.75
                and abs(angle_delta(left[0].transform.rotation.yaw,right[0].transform.rotation.yaw))<=1.):
            return True
    return False


def _lane_pair_successor(previous,current,distance):
    if previous is None or not 0<distance<=15:return False
    successor=getattr(previous,'next',None)
    if not callable(successor):return False
    for adjustment in (0,.25,.5,-.25,-.5):
        options=successor(max(.05,distance+adjustment))
        if len(options)>1:return False
        if len(options)==1:
            candidate=options[0]
            if (lane_key(candidate)==lane_key(current)
                    and candidate.transform.location.distance(current.transform.location)<=.75
                    and abs(angle_delta(candidate.transform.rotation.yaw,current.transform.rotation.yaw))<=10):
                return True
    return False


def _continued_lane_change_pair(world_map,waypoint,direction,fixture,previous,distance):
    candidates=[waypoint]
    opposite=getattr(waypoint,'get_right_lane' if direction=='LEFT' else 'get_left_lane',None)
    if callable(opposite):candidates.append(opposite())
    matches=[]
    for entry in candidates:
        if entry is None:continue
        try:
            pair=prepare_lane_fixture(world_map,entry.transform.location,direction,entry_waypoint=entry)
        except ConfigError:
            continue
        target=entry.get_left_lane() if direction=='LEFT' else entry.get_right_lane()
        if previous is None:
            connected=(pair['entry_lane_key']==fixture['entry_lane_key']
                and pair['target_lane_key']==fixture['target_lane_key'])
        else:
            connected=(_lane_pair_successor(previous[0],entry,distance)
                and _lane_pair_successor(previous[1],target,distance))
        if connected:matches.append((pair,entry,target))
    if len(matches)!=1:raise ConfigError('lane-change pair has no unique continuous successor')
    return matches[0]


def trace_lane_change_keys(world_map, route, start_m, end_m, direction,
                           location_factory, entry_fixture, *, stop_before_junction=False):
    """Bind road-spanning lane IDs to one route-verified lane-change pair."""
    if not start_m < end_m <= route[-1]['distance_m']:
        raise ConfigError('lane-change corridor outside recorded route')
    entries = [entry_fixture['entry_lane_key']]
    targets = [entry_fixture['target_lane_key']]
    checked = 0
    verified_waypoint = None
    verified_distance = None
    previous_pair=None
    pair_distance=None
    for point in route:
        distance = point['distance_m']
        if not start_m <= distance <= end_m:
            continue
        location = location_factory(**{key:point[key] for key in ('x','y','z')})
        waypoint = world_map.get_waypoint(location)
        if waypoint is None:
            raise ConfigError('lane-change route waypoint missing')
        if stop_before_junction and (waypoint.is_junction or point.get('is_junction') is True):
            end_m = distance
            break
        expected = f"{point['road_id']}:{point['section_id']}:{point['lane_id']}"
        if lane_key(waypoint) != expected:
            # Town junction connectors can occupy exactly the same coordinates.
            # They are not legal lane-change sites, so do not bind either lane
            # from a nearest-waypoint tie at such a point.
            if waypoint.is_junction:
                continue
            delta = distance - verified_distance if verified_distance is not None else 0
            successor = getattr(verified_waypoint, 'next', None)
            if not callable(successor) or not 0 < delta <= 15:
                raise ConfigError('lane-change route/map mismatch')
            candidates = []
            for adjustment in (0, .25, .5, -.25, -.5):
                options = successor(max(.05, delta + adjustment))
                if len(options) > 1:
                    raise ConfigError('lane-change route/map mismatch')
                candidates.extend(options)
            matching = [item for item in candidates if lane_key(item) == expected
                        and item.transform.location.distance(location) <= .75]
            if not matching:
                raise ConfigError('lane-change route/map mismatch')
            waypoint = min(matching, key=lambda item: item.transform.location.distance(location))
        verified_waypoint = waypoint
        verified_distance = distance
        if waypoint.is_junction:
            continue
        checked += 1
        if stop_before_junction:
            pair,entry,target=_continued_lane_change_pair(world_map,waypoint,direction,
                entry_fixture,previous_pair,distance-pair_distance if pair_distance is not None else 0)
            previous_pair=(entry,target)
            pair_distance=distance
            for key,values in (('entry_lane_key',entries),('target_lane_key',targets)):
                if pair[key] not in values:values.append(pair[key])
            continue
        try:
            pair = prepare_lane_fixture(world_map, location, direction, entry_waypoint=waypoint)
        except ConfigError:
            continue
        for key, values in (('entry_lane_key', entries), ('target_lane_key', targets)):
            if pair[key] not in values:
                values.append(pair[key])
    if not checked or set(entries) & set(targets):
        raise ConfigError('invalid lane-change corridor')
    result = dict(entry_lane_keys=entries, target_lane_keys=targets)
    if stop_before_junction:
        result['lane_change_window_end_m'] = end_m
    return result


def build_lane_corridor(world_map,route,start_m,location_factory,end_m=None,*,follow_published_route=False):
    result=trace_lane_corridor(world_map,route,start_m,location_factory,end_m,
        follow_published_route=follow_published_route)
    if result['stop_reason']:
        raise ConfigError(result['stop_reason'])
    return result['lane_corridor']


def trace_speed_lane_corridor(world_map, route, start_m, location_factory, end_m):
    result = trace_lane_corridor(world_map, route, start_m, location_factory, end_m)
    if result['stop_reason'] in {'lane corridor route/map mismatch',
                                 'lane corridor heading mismatch'}:
        raise ConfigError(result['stop_reason'])
    return result


def _published_successor(options,expected,location,reference_yaw):
    if reference_yaw is None:return options
    matching=[candidate for candidate in options
        if lane_key(candidate)==expected
        and candidate.transform.location.distance(location)<=.75
        and abs(angle_delta(candidate.transform.rotation.yaw,reference_yaw))<=10]
    return matching if len(matching)==1 else options


def _published_connector_entry(previous, current, previous_point, point, following,
                               location_factory):
    """Resolve a coincident entry alias through a proven downstream successor."""
    if (not getattr(current,'is_junction',False) or following is None
            or getattr(current,'junction_id',None) is None or following.get('yaw') is None
            or not callable(getattr(previous,'next',None))):return None
    expected=f"{following['road_id']}:{following['section_id']}:{following['lane_id']}"
    if lane_key(current)==expected:return None
    distance=following['distance_m']-previous_point['distance_m']
    back=following['distance_m']-point['distance_m']
    if not 0<back<=15 or not back<distance<=30:return None
    future=location_factory(**{k:following[k] for k in ('x','y','z')})
    origin=location_factory(**{k:point[k] for k in ('x','y','z')})
    options=[]
    for adjustment in (0,.25,.5,-.25,-.5):
        options.extend(previous.next(max(.05,distance+adjustment)))
    downstream=[wp for wp in options if lane_key(wp)==expected
        and wp.transform.location.distance(future)<=.75
        and abs(angle_delta(wp.transform.rotation.yaw,following['yaw']))<=10.]
    candidates=[]
    for wp in downstream:
        if not callable(getattr(wp,'previous',None)):continue
        for adjustment in (0,.01,.025,.05,-.01,-.025,-.05):
            for entry in wp.previous(max(.05,back+adjustment)):
                if (lane_key(entry)==expected and getattr(entry,'is_junction',False)
                        and entry.junction_id==current.junction_id
                        and entry.transform.location.distance(origin)<=.05
                        and entry.transform.location.distance(current.transform.location)<=.05
                        and abs(angle_delta(entry.transform.rotation.yaw,current.transform.rotation.yaw))<=1.):
                    candidates.append(entry)
    if not candidates:return None
    return min(candidates,key=lambda wp:wp.transform.location.distance(origin))


def _coincident_connector_prefix(previous, expected, start_m, length_m,
                                 *, position_limit=.05, heading_limit=1.):
    """Bind only the measured shared prefix, not an entire turning connector."""
    if not 0 < length_m <= 15.:
        return []
    def matches_at(offset):
        if offset <= 0:
            return set()
        options = previous.next(offset)
        reference = [wp for wp in options if lane_key(wp) == expected]
        if len(reference) != 1 or not getattr(reference[0], 'is_junction', False):
            return set()
        base = reference[0]
        matches = set()
        for candidate in options:
            key = lane_key(candidate)
            if key == expected or not getattr(candidate, 'is_junction', False):
                continue
            # Different branches accumulate different arc lengths. Compare
            # their locally closest centers, not equal arc-length samples.
            adjusted = candidate
            for _ in range(2):
                angle = math.radians(adjusted.transform.rotation.yaw)
                dx = base.transform.location.x - adjusted.transform.location.x
                dy = base.transform.location.y - adjusted.transform.location.y
                shift = dx * math.cos(angle) + dy * math.sin(angle)
                method = getattr(adjusted, 'next' if shift >= 0 else 'previous', None)
                if not callable(method) or abs(shift) < 1e-5 or abs(shift) > .75:
                    break
                values = [wp for wp in method(abs(shift)) if lane_key(wp) == key
                          and getattr(wp, 'is_junction', False)]
                if len(values) != 1:
                    break
                projected = values[0]
                if projected.transform.location.distance(base.transform.location) >= adjusted.transform.location.distance(base.transform.location):
                    break
                adjusted = projected
            if (adjusted.transform.location.distance(base.transform.location) <= position_limit
                    and abs(angle_delta(adjusted.transform.rotation.yaw,base.transform.rotation.yaw)) <= heading_limit):
                matches.add(key)
        return matches

    def transition(key, left, right, entering):
        for _ in range(12):
            middle = (left + right) / 2.
            present = key in matches_at(middle)
            if present == entering:
                right = middle
            else:
                left = middle
        return right if entering else left

    intervals = {}; active = {}; previous_offset = 0.
    offsets = [index * .05 for index in range(1, int(length_m / .05) + 1)]
    if not offsets or offsets[-1] < length_m:
        offsets.append(length_m)
    for offset in offsets:
        matches = matches_at(offset)
        for key in set(active) - matches:
            first, last = active.pop(key)
            last = start_m + transition(key, previous_offset, offset, False)
            if last > first:
                intervals.setdefault(key, []).append((first, last))
        for key in matches:
            first = active[key][0] if key in active else start_m + transition(key, previous_offset, offset, True)
            active[key] = (first, start_m + offset)
        previous_offset = offset
    for key, (first, last) in active.items():
        if last > first:
            intervals.setdefault(key, []).append((first, last))
    return [dict(lane_key=key, start_m=first, end_m=last)
            for key, ranges in intervals.items() for first, last in ranges]


def trace_lane_corridor(world_map,route,start_m,location_factory,end_m=None,*,follow_published_route=False):
    """Return only the proven prefix; the caller must reject observations beyond it."""
    end_m=route[-1]['distance_m'] if end_m is None and route else end_m
    if end_m is None or not math.isfinite(end_m) or end_m<=start_m:
        raise ConfigError('lane corridor end must follow start')
    index=max((i for i,p in enumerate(route) if p['distance_m']<=start_m),default=-1)
    if index<0 or index>=len(route)-1:
        raise ConfigError('lane corridor start outside route')
    end_index=next((i for i in range(index+1,len(route)) if route[i]['distance_m']>=end_m),None)
    if end_index is None:
        raise ConfigError('lane corridor end outside route')
    requested_anchor_m=route[index]['distance_m']
    anchor=route[index]
    location=location_factory(**{k:anchor[k] for k in ('x','y','z')})
    nearest=world_map.get_waypoint(location)
    expected=f"{anchor['road_id']}:{anchor['section_id']}:{anchor['lane_id']}"
    if nearest is None or lane_key(nearest)!=expected:
        # Resolve an overlapping boundary only through a nearby verified
        # predecessor and the same successor checks used for the corridor.
        for predecessor in range(index-1,-1,-1):
            point=route[predecessor]
            if requested_anchor_m-point['distance_m']>15:break
            loc=location_factory(**{k:point[k] for k in ('x','y','z')})
            wp=world_map.get_waypoint(loc)
            key=f"{point['road_id']}:{point['section_id']}:{point['lane_id']}"
            if (wp is not None and lane_key(wp)==key
                    and wp.transform.location.distance(loc)<=.75):
                index=predecessor
                break
    points=route[index:end_index+1]
    waypoints=[]
    equivalent_predecessors=[]
    segments=[]
    stop_reason=None
    for offset,point in enumerate(points):
        location=location_factory(**{k:point[k] for k in ('x','y','z')})
        nearest=world_map.get_waypoint(location)
        wp=nearest
        expected=f"{point['road_id']}:{point['section_id']}:{point['lane_id']}"
        if waypoints:
            distance=point['distance_m']-points[offset-1]['distance_m']
            if not math.isfinite(distance) or not 0<distance<=15:
                raise ConfigError('lane corridor samples too sparse or unordered')
            successors=waypoints[-1].next(distance)
            connected_options=list(successors)
            # Carry only topologically proven, coincident junction aliases.
            # GRP can change the ID at the shared endpoint before the paths split.
            alias_matches=[]
            for predecessor in equivalent_predecessors:
                if not callable(getattr(predecessor,'next',None)):
                    continue
                for adjustment in (0,.25,.5,.75,-.25,-.5):
                    options=predecessor.next(max(.05,distance+adjustment))
                    connected_options.extend(options)
                    alias_matches.extend(item for item in options
                        if lane_key(item)==expected
                        and item.transform.location.distance(location)<=.75)
            if alias_matches:
                successors=[min(alias_matches,key=lambda item:item.transform.location.distance(location))]
            if len(successors)>1:
                successors=_resolve_overlapping_connectors(successors,expected,location)
            if len(successors)>1 and follow_published_route:
                successors=_published_successor(successors,expected,location,point.get('yaw'))
            if len(successors)>1 and (nearest is None or lane_key(nearest)!=expected):
                matching=[candidate for candidate in successors
                          if lane_key(candidate)==expected
                          and candidate.transform.location.distance(location)<=.75]
                if len(matching)==1:
                    successors=matching
            if len(successors)>1:
                stop_reason='lane corridor has ambiguous forward topology'
                break
            candidates=list(successors)
            # GRP and CARLA waypoint sampling can straddle a road boundary by
            # a few decimeters. Search only immediate topological successors.
            if not any(lane_key(item)==expected and item.transform.location.distance(location)<=.75
                       for item in candidates):
                for adjustment in (.25,.5,.75,-.25,-.5):
                    adjusted=waypoints[-1].next(max(.05,distance+adjustment))
                    connected_options.extend(adjusted)
                    if len(adjusted)>1:
                        adjusted=_resolve_overlapping_connectors(adjusted,expected,location)
                    if len(adjusted)>1 and follow_published_route:
                        adjusted=_published_successor(adjusted,expected,location,point.get('yaw'))
                    if len(adjusted)>1:
                        stop_reason='lane corridor has ambiguous forward topology'
                        break
                    candidates.extend(adjusted)
            if stop_reason is not None:
                break
            matching=[item for item in candidates if lane_key(item)==expected
                      and item.transform.location.distance(location)<=.75]
            if matching:
                wp=min(matching,key=lambda item:item.transform.location.distance(location))
            elif len(successors)==1:
                wp=successors[0]
                # A short road stub at a junction exit can share the sampled
                # endpoint. Require a unique immediate topological convergence.
                if _converging_boundary_alias(wp,nearest,expected,location):
                    wp=nearest
            else:
                stop_reason='lane corridor has ambiguous forward topology'
                break
        if wp is None or str(wp.lane_type)!='Driving':
            raise ConfigError('lane corridor has no driving waypoint')
        entry_alias=None
        if follow_published_route and waypoints and offset+1<len(points):
            published_entry=(nearest if nearest is not None and lane_key(nearest)==expected
                and nearest.transform.location.distance(location)<=.05 else wp)
            replacement=_published_connector_entry(waypoints[-1],published_entry,points[offset-1],point,
                points[offset+1],location_factory)
            if replacement is not None:
                entry_alias=published_entry;wp=replacement
        if (lane_key(wp)!=expected and entry_alias is None) or wp.transform.location.distance(location)>.75:
            stop_reason='lane corridor route/map mismatch'
            break
        reference_yaw=point.get('yaw')
        if reference_yaw is None and nearest is not None and lane_key(nearest)==expected:
            reference_yaw=nearest.transform.rotation.yaw
        if reference_yaw is not None and abs(angle_delta(wp.transform.rotation.yaw,reference_yaw))>10:
            stop_reason='lane corridor heading mismatch'
            break
        # Chord lengths on curved connectors differ from OpenDRIVE arc length.
        # Re-anchor on the same proven lane to prevent accumulated sample drift.
        if (nearest is not None and lane_key(nearest)==lane_key(wp)
                and nearest.transform.location.distance(location)<=.75
                and nearest.transform.location.distance(wp.transform.location)<=.75
                and abs(angle_delta(nearest.transform.rotation.yaw,wp.transform.rotation.yaw))<=10):
            wp=nearest
        current_aliases=[]
        if waypoints and getattr(wp,'is_junction',False):
            seen={lane_key(wp)}
            for candidate in connected_options:
                key=lane_key(candidate)
                if (key not in seen and getattr(candidate,'is_junction',False)
                        and candidate.transform.location.distance(wp.transform.location)<=.05
                        and abs(angle_delta(candidate.transform.rotation.yaw,wp.transform.rotation.yaw))<=1.):
                    current_aliases.append(candidate)
                    seen.add(key)
        if waypoints:
            segment=dict(start_m=points[offset-1]['distance_m'],end_m=point['distance_m'],
                lane_keys=list(dict.fromkeys([lane_key(waypoints[-1]),lane_key(wp)])))
            if entry_alias is not None:
                segment.update(published_entry_alias=lane_key(entry_alias),
                    entry_alias_source='connected_downstream_successor_shared_entry_0.05m_1deg')
            junction_ids = sorted({int(item.junction_id) for item in (waypoints[-1], wp)
                if getattr(item, 'is_junction', False)
                and getattr(item, 'junction_id', None) is not None})
            if junction_ids:
                segment.update(verified_junction_ids=junction_ids,
                    junction_geometry_source='static_topology_verified_route_projection')
            aliases=list(dict.fromkeys(lane_key(candidate)
                for candidate in equivalent_predecessors + current_aliases))
            if aliases:
                segment.update(coincident_junction_lane_keys=aliases,
                    junction_alias_source='static_topology_coincident_within_0.05m_1deg')
            if len({lane_key(option) for option in connected_options}) > 1:
                ranges = _coincident_connector_prefix(waypoints[-1], lane_key(wp),
                    points[offset-1]['distance_m'], point['distance_m']-points[offset-1]['distance_m'])
                if ranges:
                    segment.update(coincident_junction_lane_ranges=ranges,
                        junction_alias_source='static_topology_coincident_within_0.05m_1deg')
                # Near a branch entry, nearest-waypoint identity can select the
                # other connector while the vehicle stays on the proven route.
                # This bounded prefix is not permission to follow that branch.
                entry_ranges = _coincident_connector_prefix(waypoints[-1], lane_key(wp),
                    points[offset-1]['distance_m'], point['distance_m']-points[offset-1]['distance_m'],
                    position_limit=.2, heading_limit=10.)
                if entry_ranges:
                    segment.update(shared_entry_junction_lane_ranges=entry_ranges,
                        shared_entry_alias_source='static_topology_shared_entry_0.2m_10deg')
            segments.append(segment)
        equivalent_predecessors=current_aliases
        waypoints.append(wp)
    if not segments:
        raise ConfigError(stop_reason or 'lane corridor has no verified segment')
    segments=[segment for segment in segments if segment['end_m']>requested_anchor_m]
    for segment in segments:
        segment['end_m']=min(segment['end_m'],end_m)
    if not segments:
        raise ConfigError(stop_reason or 'lane corridor has no verified segment')
    return dict(lane_corridor=segments,requested_end_m=end_m,
                verified_end_m=segments[-1]['end_m'],stop_reason=stop_reason)
