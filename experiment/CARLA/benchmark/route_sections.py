"""Diagnostic route sections, independent of policy and assessment truth."""
from bisect import bisect_left
import math


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def event_extent(event, length):
    positions = []
    keys = {'activate_at_m', 'activate_progress_m', 'anchor_progress_m', 'distance_m',
            'resolve_after_m', 'resolve_progress_m', 'start_s_m', 'end_s_m',
            'taper_start_s_m', 'taper_end_s_m', 's_m'}

    def visit(value, prefix=''):
        if isinstance(value, dict):
            for key, item in value.items():
                path = prefix + '.' + key if prefix else key
                if key in keys and finite(item):
                    positions.append((min(length, max(0., float(item))), path))
                elif isinstance(item, (dict, list)) and key not in ('variants', 'ground_truth', 'safety'):
                    visit(item, path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, prefix + f'[{index}]')
    visit(event)
    if not positions:
        raise ValueError('No route position for event ' + event['id'])
    explicit_end = next((event[k] for k in ('resolve_after_m', 'resolve_progress_m', 'end_s_m')
                         if finite(event.get(k))), None)
    return dict(event_id=event['id'], kind=event.get('kind', event.get('scenario', 'event')),
                start_m=min(p[0] for p in positions),
                focus_end_m=max(p[0] for p in positions),
                explicit_end_m=float(explicit_end) if explicit_end is not None else None,
                retain_after_completion=bool(event.get('retain_after_completion', False)),
                end_is_completion_criterion=False, positions=positions)


def geometry_features(samples, length):
    """Record real topology and shape transitions, not inferred traffic safety."""
    if len(samples) < 2 or any(not finite(p.get('s_m')) for p in samples):
        raise ValueError('Ordered geometry samples required')
    if any(b['s_m'] <= a['s_m'] for a, b in zip(samples, samples[1:])):
        raise ValueError('Geometry progress must be strictly increasing')
    features = []
    active = {}
    for index, point in enumerate(samples):
        previous = samples[max(0, index-1)]
        ds = point['s_m'] - previous['s_m']
        dyaw = (point['yaw_deg'] - previous['yaw_deg'] + 180.) % 360. - 180.
        curvature = abs(math.radians(dyaw)) / ds if ds > 0 else 0.
        grade = abs(point['z'] - previous['z']) / ds if ds > 0 else 0.
        states = dict(junction=bool(point['is_junction']), curve=curvature >= .012,
                      grade=grade >= .03, narrow_lane=point['lane_width_m'] < 2.8,
                      traffic_signal=bool(point.get('traffic_signal_ids')))
        for kind, present in states.items():
            if present and kind not in active:
                active[kind] = point['s_m']
            if not present and kind in active:
                start = active.pop(kind)
                end = min(length, point['s_m'])
                if end > start:
                    features.append(dict(kind=kind, start_m=start, end_m=end))
        if index and point['same_direction_lanes'] != previous['same_direction_lanes']:
            s = min(length, point['s_m'])
            features.append(dict(kind='lane_count_change', start_m=s, end_m=s,
                before=previous['same_direction_lanes'], after=point['same_direction_lanes']))
    for kind, start in active.items():
        if length > start:
            features.append(dict(kind=kind, start_m=start, end_m=length))
    return sorted(features, key=lambda x: (x['start_m'], x['end_m'], x['kind']))


def partition(catalog, samples, *, maximum_section_m=400., warmup_m=150., postroll_m=50.,
              failures=()):
    length = catalog.route_length_m
    if not all(finite(x) and x > 0 for x in (maximum_section_m, warmup_m, postroll_m)):
        raise ValueError('Positive finite section parameters required')
    features = geometry_features(samples, length)
    events = [event_extent(e, length) for e in catalog.events]
    anchors = []

    def anchor(s, kind, identity, **extra):
        s = round(min(length, max(0., float(s))), 3)
        anchors.append(dict(s_m=s, kind=kind, source_id=identity, **extra))

    anchor(0., 'route_start', catalog.scene_id)
    anchor(length, 'route_end', catalog.scene_id)
    tasks = []
    for index, task in enumerate(catalog.tasks):
        next_start = catalog.tasks[index+1].activate_m if index+1 < len(catalog.tasks) else length
        end = task.end_m if task.end_m is not None else next_start
        row = dict(task_id=task.task_id, instruction=task.instruction, announce_m=task.announce_m,
            activate_m=task.activate_m, diagnostic_end_m=end,
            end_source='explicit_window' if task.end_m is not None else 'next_task_boundary',
            end_is_completion_criterion=False, required_event_states=task.required_event_states,
            linked_event_ids=list(task.linked_event_ids))
        tasks.append(row)
        anchor(task.announce_m, 'task_announce', task.task_id)
        anchor(task.activate_m, 'task_activate', task.task_id)
        if task.end_m is not None:
            anchor(task.end_m, 'task_window_end', task.task_id)
        directive = task.source_command.get('route_directive') or {}
        if finite(directive.get('distance_m')):
            anchor(directive['distance_m'], 'route_directive', task.task_id)
    for event in events:
        for s, field in event['positions']:
            anchor(s, 'event_position', event['event_id'], field=field)
    for index, feature in enumerate(features):
        for side in ('start', 'end'):
            anchor(feature[side + '_m'], 'geometry_' + side, f'geometry_{index:03d}',
                   geometry_kind=feature['kind'])
    for failure in failures:
        if finite(failure.get('s_m')) and 0 <= failure['s_m'] <= length:
            anchor(failure['s_m'], 'historical_failure', failure['source'],
                   route_equivalence='not_verified')

    # Keep exact semantic anchors; nearby geometry changes remain sub-breakpoints
    # rather than creating dozens of tiny actor/map-transition test sections.
    cuts = {0., length}
    for row in anchors:
        if row['kind'] in ('task_activate', 'task_window_end', 'route_directive'):
            cuts.add(row['s_m'])
    for event in events:
        cuts.update((round(event['start_m'], 3), round(event['focus_end_m'], 3)))
    for feature in features:
        if feature['kind'] in ('junction', 'curve', 'grade', 'narrow_lane', 'traffic_signal'):
            proposals = (max(0., feature['start_m']-40.), min(length, feature['end_m']+40.))
        else:
            proposals = (feature['start_m'],)
        for s in proposals:
            s = round(s, 3)
            if min(abs(s-c) for c in cuts) >= 60.:
                cuts.add(s)
    original_cuts = sorted(cuts)
    for start, end in zip(original_cuts, original_cuts[1:]):
        count = math.ceil((end-start)/maximum_section_m)
        for part in range(1, count):
            cuts.add(round(start+(end-start)*part/count, 3))
    cuts = sorted(cuts)
    event_by_id = {event['event_id']: event for event in events}
    positions = [point['s_m'] for point in samples]
    sections = []
    for number, (start, end) in enumerate(zip(cuts, cuts[1:]), 1):
        linked_tasks = [t for t in tasks if t['announce_m'] < end and t['diagnostic_end_m'] > start]
        linked_events = [e for e in events if e['start_m'] <= end and e['focus_end_m'] >= start]
        geometry = [f for f in features if f['start_m'] <= end and f['end_m'] >= start]
        replay_start = max(0., start-warmup_m)
        dependencies = {}
        for task in linked_tasks:
            replay_start = min(replay_start, max(0., task['announce_m']-warmup_m))
            dependencies.update(task['required_event_states'])
            for ref in task['linked_event_ids']:
                linked_events.extend([event_by_id[ref]])
        for ref in dependencies:
            replay_start = min(replay_start, max(0., event_by_id[ref]['start_m']-warmup_m))
        for event in linked_events:
            replay_start = min(replay_start, max(0., event['start_m']-warmup_m))
        point_index = max(0, bisect_left(positions, replay_start)-1)
        # A cold pose in a junction is not a valid substitute for its approach.
        while point_index and samples[point_index]['is_junction']:
            point_index -= 1
        pose = samples[point_index]
        replay_start = min(replay_start, pose['s_m'])
        section_id = catalog.scene_id.replace('scene_', 'S') + f'-{number:03d}'
        local_failures = [f for f in failures if finite(f.get('s_m')) and start <= f['s_m'] < end]
        section = dict(section_id=section_id, start_m=start, end_m=end,
            task_ids=[t['task_id'] for t in linked_tasks],
            event_ids=sorted({e['event_id'] for e in linked_events} | set(dependencies)),
            geometry_kinds=sorted({f['kind'] for f in geometry}),
            replay_start_m=round(replay_start, 3), stop_after_m=min(length, end+postroll_m),
            minimum_sensor_warmup_s=5., minimum_memory_warmup_s=30.,
            entry_mode='replay_from_task_and_event_origin',
            cold_start_allowed_for=['geometry', 'output_contract', 'pid_actuation'],
            cold_start_valid_for_task_completion=False,
            prerequisites=dependencies,
            persistent_actor_event_ids=[e['event_id'] for e in events
                if e['retain_after_completion'] and e['start_m'] < start],
            state_to_preserve=['ordered_plan_and_step', 'event_actor_state', 'ego_speed_and_lane',
                'target_tracks', 'temporal_memory', 'traffic_signal_state'],
            start_pose={key:pose[key] for key in ('x','y','z','yaw_deg','road_id','lane_id')},
            start_pose_route_m=pose['s_m'], historical_failures=local_failures,
            sub_breakpoints=[row for row in anchors if start <= row['s_m'] < end],
            checks=['collision_and_rule_violations', 'output_schema_and_finiteness',
                'task_order_and_observed_completion', 'event_response_and_recovery',
                'progress_and_stop_reason', 'latency_p95_and_max'])
        sections.append(section)
    phase_cuts = sorted({0., length} | {t['activate_m'] for t in tasks})
    phases = []
    for index, (start, end) in enumerate(zip(phase_cuts, phase_cuts[1:]), 1):
        children = [s for s in sections if s['start_m'] < end and s['end_m'] > start]
        phases.append(dict(phase_id=catalog.scene_id.replace('scene_', 'S')+f'-P{index:02d}',
            start_m=start, end_m=end, section_ids=[s['section_id'] for s in children],
            task_ids=sorted({t for s in children for t in s['task_ids']}),
            event_ids=sorted({e for s in children for e in s['event_ids']}),
            replay_start_m=min(s['replay_start_m'] for s in children),
            stop_after_m=max(s['stop_after_m'] for s in children)))
    report = dict(schema_version='diagnostic_route_sections/1.0', scene_id=catalog.scene_id,
        map_name=catalog.map_name, route_length_m=length, source_file=catalog.source_file,
        source_sha256=catalog.source_sha256, maximum_section_m=maximum_section_m,
        warmup_m=warmup_m, postroll_m=postroll_m, tasks=tasks, events=events,
        geometry_features=features, breakpoints=sorted(anchors, key=lambda row:row['s_m']),
        phases=phases, sections=sections, assessment_passed=False,
        scope='Diagnostic route partition; not a change to scenario or policy')
    validate_sections(report)
    return report


def validate_sections(report):
    sections = report['sections']
    if not sections or sections[0]['start_m'] != 0. or sections[-1]['end_m'] != report['route_length_m']:
        raise ValueError('Sections must cover the entire route')
    if len({s['section_id'] for s in sections}) != len(sections):
        raise ValueError('Duplicate section id')
    for i, section in enumerate(sections):
        if not all(finite(section[k]) for k in ('start_m','end_m','replay_start_m','stop_after_m')):
            raise ValueError('Finite section distances required')
        if not 0 <= section['replay_start_m'] <= section['start_m'] < section['end_m'] <= section['stop_after_m'] <= report['route_length_m']:
            raise ValueError('Invalid section/replay bounds')
        if i and sections[i-1]['end_m'] != section['start_m']:
            raise ValueError('Section gap or overlap')
    phases = report.get('phases') or []
    if phases:
        if phases[0]['start_m'] != 0. or phases[-1]['end_m'] != report['route_length_m']:
            raise ValueError('Phases must cover the entire route')
        for i, phase in enumerate(phases):
            expected = [s['section_id'] for s in sections
                        if s['start_m'] < phase['end_m'] and s['end_m'] > phase['start_m']]
            if phase['section_ids'] != expected or (i and phases[i-1]['end_m'] != phase['start_m']):
                raise ValueError('Invalid phase section references or phase gap')
    return True
