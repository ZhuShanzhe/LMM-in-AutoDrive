"""Independent truth-only task assessment. Never expose these inputs to policy."""
from __future__ import annotations

import copy
import json

from .catalog import CONFIG_ROOT, ConfigError, number
from . import extended_criteria
from .coverage import profile_coverage, instruction_status


KINDS = {'speed', 'lane_change', 'turn', 'yield_pedestrian', 'overtake', 'passed_target', 'progress', 'destination'} | extended_criteria.KINDS
TERMINAL = {'SUCCESS', 'FAILURE', 'SCENE_INVALID', 'TIMEOUT'}


def constraint_spec(spec, constraint):
    return {**{k:v for k,v in spec.items() if k not in {'steps','constraints'}}, 'steps':[constraint]}


def load_profile(catalog, task):
    if getattr(catalog, 'geometry_binding', None):
        from .catalog import load_catalog
        from continuous.task_geometry_binding import remap_commands
        original = load_catalog(catalog.scene_id)
        original_task = original.select(task.task_id)[0]
        profile = load_profile(original, original_task)
        if profile is None:
            return None
        anchors = catalog.geometry_binding['distance_anchors']

        def bind_distances(value):
            if isinstance(value, list):
                return [bind_distances(item) for item in value]
            if not isinstance(value, dict):
                return value
            result = {}
            for key, item in value.items():
                if key in {'activate_m', 'end_route_s_m', 'start_route_s_m', 'from_route_s_m'}:
                    item = remap_commands([dict(announce_at_m=item)], anchors)[0]['announce_at_m']
                else:
                    item = bind_distances(item)
                result[key] = item
            return result

        profile = bind_distances(profile)
        profile['source_sha256'] = catalog.source_sha256
        return validate_spec(profile)
    path = CONFIG_ROOT / 'benchmark' / f'{task.task_id}.json'
    if not path.is_file():
        return None
    profile = validate_spec(json.loads(path.read_text(encoding='utf-8')))
    if (profile.get('source_sha256') != catalog.source_sha256
            or profile.get('scene_id') != catalog.scene_id
            or profile['task_id'] != task.task_id
            or profile['activate_m'] != task.activate_m):
        return None
    earlier={t.task_id for t in catalog.tasks if t.activate_m<task.activate_m}
    if not set(profile.get('requires_task_success',[]))<=earlier:
        raise ConfigError('task prerequisites must refer to earlier tasks in the same scene')
    return profile


def validate_spec(spec):
    if not isinstance(spec, dict) or spec.get('schema_version') != 'task_oracle/1.0':
        raise ConfigError('expected task_oracle/1.0')
    if not isinstance(spec.get('task_id'), str) or not spec['task_id']:
        raise ConfigError('task_id required')
    profile_coverage(spec)
    dependencies=spec.get('requires_task_success',[])
    if (not isinstance(dependencies,list) or
        any(not isinstance(d,str) or not d or d==spec['task_id'] for d in dependencies) or
        len(set(dependencies))!=len(dependencies)):
        raise ConfigError('invalid task prerequisites')
    for key in ('activate_m', 'timeout_s', 'max_frame_gap_s'):
        value = number(spec.get(key), key)
        if value < 0 or (key != 'activate_m' and value == 0):
            raise ConfigError(f'{key}: invalid bound')
    budget = spec.get('violation_budget', 0)
    if 'end_route_s_m' in spec and number(spec['end_route_s_m'],'end_route_s_m')<=spec['activate_m']:
        raise ConfigError('task distance window must end after activation')
    if number(spec.get('max_route_error_m',15),'max_route_error_m') <= 0:
        raise ConfigError('positive route departure threshold required')
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ConfigError('violation_budget must be a nonnegative integer')
    constraints = spec.get('constraints', [])
    if not isinstance(constraints, list):
        raise ConfigError('constraints must be a list')
    ids = set()
    for constraint in constraints:
        if not isinstance(constraint, dict) or constraint.get('kind') != 'maintain_interval':
            raise ConfigError('parallel constraints currently require maintain_interval')
        key = constraint.get('id')
        if not isinstance(key, str) or not key or key in ids:
            raise ConfigError('constraint IDs must be nonempty and unique')
        ids.add(key)
        validate_spec(constraint_spec(spec, constraint))
    steps = spec.get('steps')
    if not isinstance(steps, list) or not steps:
        raise ConfigError('nonempty steps required')
    required = {
        'speed': ('target_kmh', 'tolerance_kmh', 'hold_s'),
        'lane_change': ('hold_s', 'max_lateral_error_m', 'max_heading_error_deg'),
        'turn': ('hold_s', 'min_heading_change_deg', 'max_lateral_error_m'),
        'yield_pedestrian': ('stop_hold_s', 'stopped_kmh', 'clear_hold_s'),
        'overtake': ('rear_clearance_m', 'hold_s'),
        'passed_target': ('rear_clearance_m', 'hold_s'),
        'progress': ('distance_m',),
        'destination': ('max_distance_m','max_remaining_m'),
    }
    for step in steps:
        if not isinstance(step, dict) or step.get('kind') not in KINDS:
            raise ConfigError('unsupported criterion')
        kind = step['kind']
        if 'start_route_s_m' in step and number(step['start_route_s_m'],'start_route_s_m') < 0:
            raise ConfigError('negative step start progress')
        if kind in extended_criteria.KINDS:
            extended_criteria.validate(step)
            if kind == 'maintain_interval':
                if step['from_route_s_m'] < max(spec['activate_m'], step.get('start_route_s_m', 0)):
                    raise ConfigError('interval begins before task/step activation')
                if 'end_route_s_m' in spec and step['until_route_s_m'] >= spec['end_route_s_m']:
                    raise ConfigError('interval completion must precede task deadline boundary')
            continue
        for key in required[kind]:
            if number(step.get(key), key) < 0:
                raise ConfigError(f'{key}: negative threshold')
        for key in ('hold_s', 'stop_hold_s', 'clear_hold_s', 'rear_clearance_m', 'distance_m',
                    'min_heading_change_deg'):
            if key in step and step[key] <= 0:
                raise ConfigError(f'{key}: must be positive')
        if kind in {'yield_pedestrian', 'overtake', 'passed_target'} and not step.get('target_role'):
            raise ConfigError('target_role required')
        if kind == 'passed_target' and not dependencies:
            raise ConfigError('passed_target confirmation requires prior task evidence')
        if kind == 'yield_pedestrian' and step.get('require_stop', True) is False:
            if number(step.get('min_speed_drop_kmh'), 'min_speed_drop_kmh') <= 0:
                raise ConfigError('positive speed decrease required for rolling yield')
        if kind=='speed':
            if 'keep_lane' in step and not isinstance(step['keep_lane'],bool):
                raise ConfigError('keep_lane must be boolean')
            for key in ('max_lateral_error_m','max_heading_error_deg'):
                if key in step and (not step.get('keep_lane') or number(step[key],key)<=0):
                    raise ConfigError('speed lane bounds require keep_lane and positive thresholds')
        if kind=='destination' and any(step[k]<=0 for k in ('max_distance_m','max_remaining_m')):
            raise ConfigError('destination tolerances must be positive')
        if kind=='destination' and 'keep_lane' in step:
            if not isinstance(step['keep_lane'],bool):
                raise ConfigError('keep_lane must be boolean')
            if step['keep_lane']:
                for key in ('max_lateral_error_m','max_heading_error_deg'):
                    if number(step.get(key),key)<=0:
                        raise ConfigError('positive destination lane bounds required')
        if kind == 'lane_change' and step.get('direction') not in {'LEFT', 'RIGHT'}:
            raise ConfigError('lane_change direction required')
        if kind == 'turn' and step.get('direction') not in {'LEFT', 'RIGHT', 'U_TURN'}:
            raise ConfigError('turn direction required')
    return spec


class TaskOracle:
    """One task, ordered criteria, monotonic frames and immutable final result."""

    def __init__(self, spec):
        self.spec = copy.deepcopy(validate_spec(spec))
        self.status = 'WAITING'
        self.reason = None
        self.index = 0
        self.evidence = []
        self.last = None
        self.started = None
        self.baseline = (0, 0)
        self.state = {}
        self.roles = {}
        self.fixtures = {}
        self.last_counts = None
        self.constraints = {c['id']:TaskOracle(constraint_spec(self.spec,c))
                            for c in self.spec.get('constraints',[])}
        self.constraint_reported = set()
        self.original_lane = None
        self.requires_original_lane = any(s['kind']=='return_original_lane' for s in self.spec['steps'])

    def result(self):
        result = dict(task_id=self.spec['task_id'], status=self.status, reason=self.reason,
                    completed_steps=self.index, total_steps=len(self.spec['steps']),
                    evidence=list(self.evidence))
        result['coverage']=profile_coverage(self.spec)
        result['instruction_status']=instruction_status(self.status,result['coverage'])
        if self.constraints:
            result['constraints'] = {key:value.result() for key,value in self.constraints.items()}
        return result

    def finish(self, status, reason, frame=None):
        if self.status not in TERMINAL:
            self.status, self.reason = status, reason
            self.evidence.append(dict(event=status, reason=reason, frame=frame))
        return self.result()

    def end_of_stream(self):
        # Truncated capture is not evidence of a model timeout or success.
        return self.finish('SCENE_INVALID', 'observation_stream_ended')

    def update(self, obs):
        if self.status in TERMINAL:
            return self.result()
        try:
            return self._update(obs)
        except (KeyError, TypeError, ValueError) as error:
            return self.finish('SCENE_INVALID', f'invalid_observation: {error}')

    def _update(self, obs):
        if obs['schema_version'] != 'task_truth/1.0' or obs['source'] != 'simulator_truth':
            raise ConfigError('independent simulator truth required')
        if 'source_sha256' in self.spec:
            if obs.get('source_sha256') != self.spec['source_sha256'] or obs.get('task_id') != self.spec['task_id']:
                raise ConfigError('task/config provenance mismatch')
        frame = obs['frame']
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ConfigError('invalid frame')
        now = number(obs['sim_time_s'], 'sim_time_s')
        if self.last is not None:
            if frame <= self.last[0] or now <= self.last[1]:
                raise ConfigError('non-monotonic observation')
            if now - self.last[1] > self.spec['max_frame_gap_s'] + 1e-8:
                raise ConfigError('observation gap')
        self.last = (frame, now)
        if obs['scenario_valid'] is not True:
            return self.finish('SCENE_INVALID', obs.get('invalid_reason', 'fixture_invalid'), frame)
        ego = obs['ego']
        progress = number(ego['route_s_m'], 'route_s_m')
        speed = number(ego['speed_kmh'], 'speed_kmh')
        if speed < 0:
            raise ConfigError('negative speed')
        if 'route_error_m' in ego and number(ego['route_error_m'],'route_error_m') > self.spec.get('max_route_error_m',15):
            return self.finish('FAILURE','ego_route_departure',frame)
        counts = tuple(obs['safety'][k] for k in ('collisions', 'violations'))
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts):
            raise ConfigError('invalid safety counters')
        if self.last_counts is not None and any(v < b for v, b in zip(counts, self.last_counts)):
            raise ConfigError('safety counters decreased')
        self.last_counts = counts
        if counts[0] > self.baseline[0]:
            return self.finish('FAILURE', 'collision', frame)
        if counts[1] - self.baseline[1] > self.spec.get('violation_budget', 0):
            return self.finish('FAILURE', 'violation_budget_exceeded', frame)
        if self.status == 'WAITING':
            if progress < self.spec['activate_m']:
                return self.result()
            if progress - self.spec['activate_m'] > 20:
                raise ConfigError('activation window skipped')
            for dependency in self.spec.get('requires_task_success',[]):
                proof=obs['fixture'].get('prerequisites',{}).get(dependency,{})
                completed=proof.get('completion_frame')
                if (proof.get('status')!='SUCCESS' or proof.get('source')!='independent_task_oracle' or
                    proof.get('source_sha256')!=self.spec.get('source_sha256') or
                    type(completed) is not int or not 0<=completed<frame):
                    raise ConfigError('missing independent prerequisite success: '+dependency)
            self.status, self.started = 'RUNNING', now
            if self.requires_original_lane:
                if (not isinstance(ego.get('lane_key'),str) or not ego['lane_key'] or
                    not isinstance(ego.get('road_key'),str) or not ego['road_key'] or ego.get('in_junction') is not False):
                    raise ConfigError('original lane requires non-junction task entry')
                self.original_lane=dict(lane_key=ego['lane_key'],road_key=ego['road_key'],frame=frame)
                self.evidence.append(dict(event='ORIGINAL_LANE_BOUND',**self.original_lane))
            self.evidence.append(dict(event='START', frame=frame, sim_time_s=now))
        if self.requires_original_lane and (ego['road_key']!=self.original_lane['road_key'] or ego['in_junction']):
            raise ConfigError('original lane continuation across road/junction not verified')
        if now - self.started > self.spec['timeout_s']:
            return self.finish('TIMEOUT', 'task_deadline', frame)
        if 'end_route_s_m' in self.spec and progress>=self.spec['end_route_s_m']:
            return self.finish('TIMEOUT','task_distance_window_closed',frame)
        for key, oracle in self.constraints.items():
            constraint_obs = dict(obs)
            fixture = dict(obs.get('fixture', {}))
            fixture['steps'] = {'0':fixture.get('constraints', {}).get(key, {})}
            constraint_obs['fixture'] = fixture
            result = oracle.update(constraint_obs)
            if oracle.status in TERMINAL and key not in self.constraint_reported:
                self.evidence.append(dict(event='CONSTRAINT_RESULT',constraint_id=key,
                                          status=oracle.status,reason=oracle.reason,frame=frame))
                self.constraint_reported.add(key)
            if oracle.status in {'FAILURE','SCENE_INVALID','TIMEOUT'}:
                return self.finish(oracle.status, f"constraint:{key}:{result['reason']}", frame)
        if self.index == len(self.spec['steps']):
            if all(c.status=='SUCCESS' for c in self.constraints.values()):
                return self.finish('SUCCESS','all_criteria_satisfied',frame)
            return self.result()
        step = self.spec['steps'][self.index]
        if progress < step.get('start_route_s_m',0):
            return self.result()
        if self._criterion(step, obs, now):
            self.evidence.append(dict(event='STEP_SUCCESS', step=self.index,
                                     kind=step['kind'], frame=frame, sim_time_s=now))
            self.index += 1
            self.state = {}
            if self.index == len(self.spec['steps']) and all(c.status=='SUCCESS' for c in self.constraints.values()):
                return self.finish('SUCCESS', 'all_criteria_satisfied', frame)
        return self.result()

    def _hold(self, key, condition, now, duration):
        if not condition:
            self.state.pop(key, None)
            return False
        start = self.state.setdefault(key, now)
        return now - start + 1e-8 >= duration

    def _target(self, step, obs):
        role = step['target_role']
        target = obs['actors'][role]
        identity = target['actor_id']
        if not isinstance(identity, (int, str)) or isinstance(identity, bool) or not str(identity):
            raise ConfigError('invalid target identity')
        if target['alive'] is not True or target.get('teleported', False):
            raise ConfigError('target missing or teleported')
        if self.roles.setdefault(role, identity) != identity:
            raise ConfigError('bound target replaced')
        return target

    def _fixture(self, obs):
        value = obs['fixture']['steps'][str(self.index)]
        if self.fixtures.setdefault(self.index, copy.deepcopy(value)) != value:
            raise ConfigError('fixture changed during task')
        return value

    def _criterion(self, step, obs, now):
        ego, state = obs['ego'], self.state
        speed = number(ego['speed_kmh'], 'speed_kmh')
        kind = step['kind']
        progress = number(ego['route_s_m'], 'route_s_m')
        if kind in extended_criteria.KINDS:
            return extended_criteria.evaluate(self, step, obs, now)
        if kind == 'speed':
            condition = abs(speed - step['target_kmh']) <= step['tolerance_kmh']
            if step.get('keep_lane', False):
                if not isinstance(ego['lane_key'], str) or not ego['lane_key'] or not isinstance(ego['in_junction'], bool):
                    raise ConfigError('invalid lane observation')
                evidence=obs.get('fixture',{}).get('steps',{}).get(str(self.index),{})
                if 'lane_corridor' in evidence:
                    fixture=self._fixture(obs)
                    corridor=fixture['lane_corridor']
                    if not isinstance(corridor,list) or not corridor:
                        raise ConfigError('invalid speed lane corridor')
                    if 'verified_end_m' in fixture:
                        verified=number(fixture['verified_end_m'],'verified_end_m')
                        requested=number(fixture['requested_end_m'],'requested_end_m')
                        reason=fixture.get('stop_reason')
                        if (abs(verified-number(corridor[-1]['end_m'],'corridor end'))>1e-3
                                or verified>requested or reason not in
                                {None,'lane corridor has ambiguous forward topology'}):
                            raise ConfigError('invalid speed lane corridor proof')
                        if progress>verified+1e-3 and reason is not None:
                            self.finish('TIMEOUT','speed_lane_proof_window_closed',obs['frame'])
                            return False
                    segments=[s for s in corridor if s['start_m']<=progress<=s['end_m']]
                    if not segments:
                        raise ConfigError('speed lane corridor does not cover current progress')
                    allowed={key for s in segments for key in s['lane_keys']}
                    condition &= ego['lane_key'] in allowed
                else:
                    initial = state.setdefault('lane', ego['lane_key'])
                    condition &= ego['lane_key'] == initial and not ego['in_junction']
                for key,measurement in [('max_lateral_error_m','lateral_error_m'),
                                        ('max_heading_error_deg','heading_error_deg')]:
                    if key in step:
                        condition &= abs(number(ego[measurement],measurement))<=step[key]
            return self._hold('stable', condition, now, step['hold_s'])
        if kind == 'lane_change':
            if not isinstance(ego['in_junction'], bool):
                raise ConfigError('invalid junction flag')
            fixture = self._fixture(obs)
            if fixture['direction'] != step['direction'] or fixture['legal'] is not True:
                raise ConfigError('invalid lane-change fixture')
            if any(not isinstance(value, str) or not value for value in (
                    ego['lane_key'], fixture['entry_lane_key'], fixture['target_lane_key'])):
                raise ConfigError('invalid lane identity')
            entries=fixture.get('entry_lane_keys',[fixture['entry_lane_key']])
            targets=fixture.get('target_lane_keys',[fixture['target_lane_key']])
            if (not isinstance(entries,list) or not isinstance(targets,list) or not entries or not targets
                    or any(not isinstance(k,str) or not k for k in entries+targets)
                    or fixture['entry_lane_key'] not in entries or fixture['target_lane_key'] not in targets
                    or set(entries)&set(targets)):
                raise ConfigError('invalid continuous lane corridor')
            if not state:
                if ego['lane_key'] not in entries:
                    raise ConfigError('incorrect entry lane')
                if fixture['entry_lane_key'] == fixture['target_lane_key']:
                    raise ConfigError('identical entry and target lanes')
                state['entered'] = True
            condition = (ego['lane_key'] in targets and not ego['in_junction']
                         and abs(number(ego['lateral_error_m'], 'lateral_error_m')) <= step['max_lateral_error_m']
                         and abs(number(ego['heading_error_deg'], 'heading_error_deg')) <= step['max_heading_error_deg'])
            return self._hold('stable', condition, now, step['hold_s'])
        if kind == 'turn':
            if not isinstance(ego['in_junction'], bool):
                raise ConfigError('invalid junction flag')
            fixture = self._fixture(obs)
            if fixture['direction'] != step['direction'] or fixture['legal'] is not True:
                raise ConfigError('invalid turn fixture')
            if any(not isinstance(value, str) or not value for value in (
                    ego['road_key'], fixture['entry_road_key'], fixture['exit_road_key'])):
                raise ConfigError('invalid road identity')
            if not state:
                if ego['road_key'] != fixture['entry_road_key'] or ego['in_junction']:
                    raise ConfigError('incorrect turn entry')
                state.update(entry_yaw=number(ego['yaw_deg'], 'yaw_deg'), saw_junction=False)
            state['saw_junction'] |= ego['in_junction'] is True
            delta = (number(ego['yaw_deg'], 'yaw_deg') - state['entry_yaw'] + 180) % 360 - 180
            angle = step['min_heading_change_deg']
            # CARLA yaw increases clockwise: positive is right.
            directed = {'LEFT': delta <= -angle, 'RIGHT': delta >= angle,
                        'U_TURN': abs(delta) >= angle}[step['direction']]
            condition = (state['saw_junction'] and not ego['in_junction'] and directed
                         and ego['road_key'] == fixture['exit_road_key']
                         and fixture['entry_road_key'] != fixture['exit_road_key']
                         and abs(number(ego['lateral_error_m'], 'lateral_error_m')) <= step['max_lateral_error_m'])
            return self._hold('stable', condition, now, step['hold_s'])
        if kind == 'yield_pedestrian':
            target = self._target(step, obs)
            entry_speed = state.setdefault('entry_speed', speed)
            conflict = target['in_conflict_zone']
            if not isinstance(conflict, bool):
                raise ConfigError('conflict flag must be geometric boolean')
            stop_line = number(self._fixture(obs)['stop_line_route_s_m'], 'stop_line')
            if conflict:
                state['seen_conflict'] = True
                state.pop('clear', None)
                if number(ego.get('front_route_s_m',progress),'front_route_s_m') > stop_line:
                    self.finish('FAILURE', 'crossed_stop_line_while_pedestrian_conflict', obs['frame'])
                    return False
                yielding = speed <= step['stopped_kmh']
                if step.get('require_stop', True) is False:
                    yielding |= entry_speed - speed >= step['min_speed_drop_kmh']
                if self._hold('stopped', yielding, now, step['stop_hold_s']):
                    state['yielded'] = True
                return False
            return self._hold('clear', state.get('seen_conflict', False) and state.get('yielded', False),
                              now, step['clear_hold_s'])
        if kind in {'overtake', 'passed_target'}:
            target = self._target(step, obs)
            if not isinstance(ego['route_corridor_id'], str) or not ego['route_corridor_id']:
                raise ConfigError('invalid route corridor identity')
            if target['route_corridor_id'] != ego['route_corridor_id']:
                raise ConfigError('target left shared route corridor')
            separation = number(target['route_s_m'], 'target route_s_m') - progress
            if number(ego['half_length_m'], 'ego half length') <= 0 or number(target['half_length_m'], 'target half length') <= 0:
                raise ConfigError('positive actor dimensions required')
            state['was_ahead'] = state.get('was_ahead', False) or separation > 0
            rear = number(ego.get('rear_route_s_m',progress-ego['half_length_m']),'ego rear')
            front = number(target.get('front_route_s_m',target['route_s_m']+target['half_length_m']),'target front')
            gap = rear-front
            return self._hold('passed', (kind == 'passed_target' or state['was_ahead'])
                              and gap >= step['rear_clearance_m'], now, step['hold_s'])
        if kind == 'progress':
            start = state.setdefault('start_progress', progress)
            return progress - start >= step['distance_m']
        if kind == 'destination':
            fixture=self._fixture(obs)
            if step.get('keep_lane'):
                corridor=fixture['lane_corridor']
                segments=[s for s in corridor if s['start_m']<=progress<=s['end_m']]
                if not segments:
                    raise ConfigError('destination lane corridor does not cover current progress')
                allowed={key for s in segments for key in s['lane_keys']}
                if (ego['lane_key'] not in allowed or
                    abs(number(ego['lateral_error_m'],'lateral_error_m'))>step['max_lateral_error_m'] or
                    abs(number(ego['heading_error_deg'],'heading_error_deg'))>step['max_heading_error_deg']):
                    self.finish('FAILURE','left_destination_lane_corridor',obs['frame'])
                    return False
            end=number(fixture['route_end_s_m'],'route_end_s_m')
            if end<=self.spec['activate_m']:
                raise ConfigError('destination must follow task activation')
            point=fixture['position_m']
            actual=ego['position_m']
            squared=sum((number(actual[k],'ego '+k)-number(point[k],'destination '+k))**2
                        for k in ('x','y','z'))
            return (end-step['max_remaining_m']<=progress<=end+step['max_remaining_m'] and
                    squared<=step['max_distance_m']**2)
        raise ConfigError('unsupported criterion')
