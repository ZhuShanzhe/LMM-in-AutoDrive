"""Bind an ordered lane-change pair and a later turn to actual map geometry.

This is scenario construction, never an input to the learned decision policy.
The resolved configuration is shared by execution and independent assessment.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


def remap_commands(commands, anchors):
    """Preserve command content and ordering under a monotone distance map."""
    anchors = sorted(anchors)
    if len(anchors) < 2 or any(not math.isfinite(float(v)) for pair in anchors for v in pair) or any(
        b[0] <= a[0] or b[1] <= a[1] for a, b in zip(anchors, anchors[1:])
    ):
        raise ValueError('Task bindings must be strictly monotone')

    def distance(value):
        value = float(value)
        if not anchors[0][0] <= value <= anchors[-1][0]:
            raise ValueError('Command distance is outside the route')
        for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
            if value <= x1:
                return round(y0 + (value-x0)*(y1-y0)/(x1-x0), 3)
        return anchors[-1][1]

    result = deepcopy(commands)
    for command in result:
        for key in ('announce_at_m', 'activate_at_m'):
            if key in command:
                command[key] = distance(command[key])
        if command.get('route_directive'):
            command['route_directive']['distance_m'] = distance(
                command['route_directive']['distance_m'])
    return result


def bind_task_geometry(world, source_path, output_path):
    import carla
    from scenarios.basic.voice_control_5km import BasicVoiceControl5KmScenario

    source_path, output_path = Path(source_path), Path(output_path)
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    commands = source['commands']
    lanes = sorted((c for c in commands if c.get('action') in
                    ('lane_change_left', 'lane_change_right')),
                   key=lambda c: c.get('activate_at_m', c['announce_at_m']))
    turns = [c for c in commands if c.get('route_directive')]
    if (len(lanes) != 2 or len(turns) != 1 or
            [c['action'] for c in lanes] != ['lane_change_left', 'lane_change_right'] or
            turns[0]['route_directive']['action'] not in ('turn_left', 'turn_right')):
        raise ValueError('Geometry binding requires a lane-change pair and one subsequent turn')
    positions = [float(c.get('activate_at_m', c['announce_at_m'])) for c in lanes]
    turn_position = float(turns[0]['route_directive']['distance_m'])
    length = float(source['route']['length_m'])
    if not 0 < positions[0] < positions[1] < turn_position < length:
        raise ValueError('Unsupported task order for geometry binding')
    gap = positions[1]-positions[0]
    preparation = float(source['route'].get('lane_change_preparation_m', 500))
    step = float(source['route'].get('corridor_sample_step_m', 50))
    minimum_lanes = int(source['route'].get('minimum_same_direction_lanes', 3))
    world_map = world.get_map()
    scenario = BasicVoiceControl5KmScenario(world, config_path=str(source_path))
    attempts = 0
    best = None
    # Bounded map search, independent of Town IDs, event IDs, and actor truth.
    for fraction in (.3, .4, .5, .6, .7, .8):
        for spawn_index, spawn in enumerate(world_map.get_spawn_points()):
            waypoint = world_map.get_waypoint(spawn.location)
            if waypoint is None or scenario._same_direction_lane_count(waypoint) < minimum_lanes:
                continue
            waypoint = scenario._select_initial_lane(waypoint)
            if waypoint is None or waypoint.is_junction:
                continue
            attempts += 1
            route = scenario.route_manager.build_route(
                waypoint.transform.location, length, source['route'].get('step_m', 5),
                [dict(turns[0]['route_directive'], distance_m=length*fraction,
                      id=turns[0]['id'])])
            applied = scenario.route_manager.applied_directives
            if not applied or scenario.route_manager.route_length_m < length:
                continue
            turn_m = float(applied[0]['applied_distance_m'])
            start = None
            next_sample = 0.
            lane_positions = None
            for point in route:
                d = float(point['distance_m'])
                if d < next_sample:
                    continue
                next_sample = d + step
                wp = world_map.get_waypoint(carla.Location(**{k: point[k] for k in ('x', 'y', 'z')}))
                if (wp is None or wp.is_junction or
                        scenario._same_direction_lane_count(wp) < minimum_lanes):
                    start = None
                    continue
                if start is None:
                    start = d
                first = start + preparation + step
                second = first + gap
                if d >= second and d + 200 < turn_m:
                    first = min(max(positions[0], first), d-gap)
                    proposal = (first, first+gap)
                    if lane_positions is None or abs(first-positions[0]) < abs(lane_positions[0]-positions[0]):
                        lane_positions = proposal
            if lane_positions is None:
                continue
            cost = sum(abs(a-b) for a,b in zip(positions,lane_positions)) + abs(turn_m-turn_position)
            if best is not None and cost >= best[0]:
                continue
            bound = deepcopy(source)
            anchors = [(0., 0.), (positions[0], lane_positions[0]),
                       (positions[1], lane_positions[1]), (turn_position, turn_m), (length, length)]
            bound['commands'] = remap_commands(commands, anchors)
            bound['route'].update(start_spawn_index=spawn_index, strict_start_spawn=True,
                                  spawn_backoff_m=0.)
            scenario.config = bound
            try:
                scenario._select_spawn_point_and_route([
                    dict(c['route_directive'], id=c['id']) for c in bound['commands']
                    if c.get('route_directive')])
            except RuntimeError:
                scenario.config = source
                continue
            bound['geometry_binding'] = dict(
                source_sha256=hashlib.sha256(source_bytes).hexdigest(),
                map_sha256=hashlib.sha256(world_map.to_opendrive().encode()).hexdigest(),
                map_name=world_map.name, attempts=attempts, distance_anchors=anchors,
                scope='map_bound_task_layout_not_original_fixed_mileage_benchmark',
                preflight=deepcopy(scenario.route_preflight), displacement_cost_m=cost)
            best = (cost, bound)
            scenario.config = source
    if best is not None:
        best[1]['geometry_binding']['search_candidates'] = attempts
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(best[1], ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        return output_path
    raise RuntimeError(f'No valid map binding after {attempts} candidates; original constraints retained')
