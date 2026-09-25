"""Reusable compound-task criteria consuming assessment observations only."""
from .catalog import ConfigError, number


KINDS = {'speed_ceiling', 'speed_change', 'follow_distance', 'wait_clear', 'straight_junction', 'maintain_interval', 'lane_position', 'lane_hold', 'guarded_lane_change', 'yield_cut_in', 'return_original_lane'}


def validate(step):
    kind = step['kind']
    required = {
        'speed_ceiling': ('max_speed_kmh', 'hold_s'),
        'speed_change': ('delta_kmh', 'hold_s'),
        'follow_distance': ('min_gap_m', 'time_headway_s', 'hold_s', 'min_travel_m'),
        'wait_clear': ('stopped_kmh', 'stop_hold_s', 'clear_hold_s'),
        'straight_junction': ('max_heading_change_deg', 'max_lateral_error_m', 'hold_s'),
        'maintain_interval': ('until_route_s_m', 'max_sample_distance_m'),
        'lane_position': ('hold_s', 'max_lateral_error_m', 'max_heading_error_deg'),
        'lane_hold': ('hold_s', 'max_lateral_error_m', 'max_heading_error_deg'),
        'guarded_lane_change': ('min_front_gap_m','min_rear_gap_m','min_ttc_s','gap_hold_s',
                                'hold_s','max_lateral_error_m','max_heading_error_deg',
                                'stop_gap_m','stopped_kmh'),
        'yield_cut_in': ('min_speed_drop_kmh','min_gap_m','min_ttc_s','hold_s',
                         'max_lateral_error_m','max_heading_error_deg','stopped_kmh'),
        'return_original_lane': ('hold_s','max_lateral_error_m','max_heading_error_deg'),
    }
    for key in required[kind]:
        value = number(step.get(key), key)
        if value < 0 or (key != 'stopped_kmh' and value == 0):
            raise ConfigError(f'{key}: invalid criterion threshold')
    if kind == 'guarded_lane_change':
        if step.get('direction') not in {'LEFT','RIGHT'}:
            raise ConfigError('guarded lane change direction required')
        roles=[step.get(key) for key in ('front_role','rear_role','obstacle_role')]
        if any(not isinstance(r,str) or not r for r in roles) or len(set(roles)) != 3:
            raise ConfigError('three distinct gap/obstacle roles required')
        if step.get('target_roles') != roles:
            raise ConfigError('target_roles must list front, rear, obstacle roles in order')
    if kind == 'maintain_interval':
        start = number(step.get('from_route_s_m'), 'from_route_s_m')
        if start < 0 or step['until_route_s_m'] <= start:
            raise ConfigError('interval must have positive length')
        if type(step.get('keep_lane', False)) is not bool:
            raise ConfigError('keep_lane must be boolean')
        if 'max_speed_kmh' not in step and not step.get('keep_lane') and 'lane_ordinal' not in step:
            raise ConfigError('interval requires speed or lane constraint')
        if 'max_speed_kmh' in step and number(step['max_speed_kmh'], 'max_speed_kmh') <= 0:
            raise ConfigError('positive speed ceiling required')
        if step.get('keep_lane') or 'lane_ordinal' in step:
            for key in ('max_lateral_error_m', 'max_heading_error_deg'):
                if number(step.get(key), key) <= 0:
                    raise ConfigError('positive lane geometry thresholds required')
    if kind == 'lane_position' or 'lane_ordinal' in step:
        if step.get('lane_side') not in {'LEFT', 'RIGHT'} or type(step.get('lane_ordinal')) is not int or step['lane_ordinal'] < 1:
            raise ConfigError('lane ordinal requires LEFT/RIGHT and a positive integer')
    if kind == 'speed_change' and step.get('direction') not in {'INCREASE', 'DECREASE'}:
        raise ConfigError('speed_change requires INCREASE or DECREASE')
    if kind in {'follow_distance','yield_cut_in'} and not isinstance(step.get('target_role'), str):
        raise ConfigError('follow_distance requires a target role')
    if kind in {'follow_distance','yield_cut_in'} and not step['target_role']:
        raise ConfigError('empty target role')
    if kind == 'wait_clear':
        for key in ('require_stop', 'require_observed_conflict'):
            if key in step and type(step[key]) is not bool:
                raise ConfigError(key + ' must be boolean')
        if step.get('require_stop', True) and not step.get('require_observed_conflict', True):
            raise ConfigError('required conflict stop needs observed conflict')
        roles = step.get('target_roles')
        if (not isinstance(roles, list) or not roles or
                any(not isinstance(r, str) or not r for r in roles) or len(set(roles)) != len(roles)):
            raise ConfigError('wait_clear requires unique nonempty target roles')
    if kind == 'straight_junction' and step['max_heading_change_deg'] >= 45:
        raise ConfigError('straight junction heading bound must be less than 45 degrees')


def evaluate(oracle, step, obs, now):
    ego, state = obs['ego'], oracle.state
    speed = number(ego['speed_kmh'], 'speed_kmh')
    progress = number(ego['route_s_m'], 'route_s_m')
    kind = step['kind']
    if kind == 'return_original_lane':
        origin=oracle.original_lane
        if origin is None:
            raise ConfigError('original lane was not recorded at task entry')
        if not state:
            if ego['lane_key']==origin['lane_key']:
                oracle.finish('FAILURE','already_returned_before_return_step',obs['frame'])
                return False
            state['started_outside_original']=True
        condition=(ego['lane_key']==origin['lane_key'] and not ego['in_junction']
                   and abs(number(ego['lateral_error_m'],'lateral error'))<=step['max_lateral_error_m']
                   and abs(number(ego['heading_error_deg'],'heading error'))<=step['max_heading_error_deg'])
        return oracle._hold('returned',condition,now,step['hold_s'])
    if kind == 'yield_cut_in':
        target=oracle._target(step,obs)
        if not state:
            state.update(entry_lane=ego['lane_key'],entry_road=ego['road_key'],entry_speed=speed)
        if ego['road_key']!=state['entry_road'] or ego['in_junction']:
            raise ConfigError('cut-in criterion needs a continuous non-junction lane segment')
        if ego['lane_key']!=state['entry_lane']:
            oracle.finish('FAILURE','left_lane_during_cut_in_yield',obs['frame'])
            return False
        if target['route_corridor_id']!=ego['route_corridor_id']:
            raise ConfigError('cut-in target outside shared route corridor')
        gap=number(target['rear_route_s_m'],'target rear')-number(ego['front_route_s_m'],'ego front')
        same_lane=target['lane_key']==ego['lane_key']
        if 'merge_time' not in state:
            if not same_lane:
                state['seen_adjacent']=True
                return False
            if not state.get('seen_adjacent'):
                raise ConfigError('cut-in transition not observed before target entered ego lane')
            if gap<=0:
                raise ConfigError('cut-in occurred behind or overlapping ego')
            state['merge_time']=now
            oracle.evidence.append(dict(event='CUT_IN_LANE_ENTRY',frame=obs['frame'],sim_time_s=now,
                                        target_actor_id=target['actor_id'],net_gap_m=gap))
        if not same_lane:
            raise ConfigError('cut-in target departed ego lane before assessment completed')
        reduced=state['entry_speed']-speed>=step['min_speed_drop_kmh'] or speed<=step['stopped_kmh']
        if reduced and 'response_time' not in state:
            state['response_time']=now
            oracle.evidence.append(dict(event='MEASURED_YIELD_RESPONSE',frame=obs['frame'],
                response_after_lane_entry_s=now-state['merge_time'],
                timing_scope='physical_speed_response_not_model_inference_latency'))
        closing=max(0,(speed-number(target['speed_kmh'],'target speed'))/3.6)
        safe=(gap>=step['min_gap_m'] and (closing==0 or gap/closing>=step['min_ttc_s']))
        centered=(abs(number(ego['lateral_error_m'],'lateral error'))<=step['max_lateral_error_m']
                  and abs(number(ego['heading_error_deg'],'heading error'))<=step['max_heading_error_deg'])
        return oracle._hold('yielded',reduced and safe and centered,now,step['hold_s'])
    if kind == 'guarded_lane_change':
        fixture=oracle._fixture(obs)
        if fixture.get('legal') is not True or fixture.get('direction') != step['direction']:
            raise ConfigError('legal adjacent-lane fixture required')
        entry,target_lane=fixture['entry_lane_key'],fixture['target_lane_key']
        if entry==target_lane or not entry or not target_lane:
            raise ConfigError('distinct lane identities required')
        front,rear,obstacle=[oracle._target({'target_role':step[key]},obs)
                             for key in ('front_role','rear_role','obstacle_role')]
        if any(a['route_corridor_id'] != ego['route_corridor_id'] for a in (front,rear,obstacle)):
            raise ConfigError('gap actors left shared route corridor')
        if front['lane_key'] != target_lane or rear['lane_key'] != target_lane:
            raise ConfigError('gap reference vehicles not in target lane')
        if obstacle['lane_key'] != entry:
            raise ConfigError('obstacle no longer blocks entry lane')
        if not state:
            if ego['lane_key'] != entry or ego['in_junction']:
                raise ConfigError('guarded lane change must start in entry lane')
            state['entered']=True
        ego_front=number(ego['front_route_s_m'],'ego front')
        ego_rear=number(ego['rear_route_s_m'],'ego rear')
        front_gap=number(front['rear_route_s_m'],'front rear')-ego_front
        rear_gap=ego_rear-number(rear['front_route_s_m'],'rear front')
        front_closing=max(0,(speed-number(front['speed_kmh'],'front speed'))/3.6)
        rear_closing=max(0,(number(rear['speed_kmh'],'rear speed')-speed)/3.6)
        clear=(front_gap>=step['min_front_gap_m'] and rear_gap>=step['min_rear_gap_m']
               and (front_closing==0 or front_gap/front_closing>=step['min_ttc_s'])
               and (rear_closing==0 or rear_gap/rear_closing>=step['min_ttc_s']))
        stable_gap=oracle._hold('gap',clear,now,step['gap_hold_s'])
        lateral=abs(number(ego['lateral_error_m'],'lateral error'))
        merging=ego['lane_key']!=entry or lateral>step['max_lateral_error_m']
        if merging and not state.get('merge_started'):
            if not stable_gap:
                oracle.finish('FAILURE','entered_lane_before_verified_gap',obs['frame'])
                return False
            state['merge_started']=True
            oracle.evidence.append(dict(event='GAP_ACCEPTED',frame=obs['frame'],
                                        front_gap_m=front_gap,rear_gap_m=rear_gap))
        if state.get('merge_started') and not clear:
            oracle.finish('FAILURE','gap_became_unsafe_during_merge',obs['frame'])
            return False
        if not state.get('merge_started') and not clear:
            obstacle_gap=number(obstacle['rear_route_s_m'],'obstacle rear')-ego_front
            if obstacle_gap<step['stop_gap_m'] and speed>step['stopped_kmh']:
                oracle.finish('FAILURE','failed_to_stop_before_blockage',obs['frame'])
                return False
            state['observed_blocked']=True
        centered=(state.get('merge_started',False) and ego['lane_key']==target_lane
                  and not ego['in_junction'] and lateral<=step['max_lateral_error_m']
                  and abs(number(ego['heading_error_deg'],'heading error'))<=step['max_heading_error_deg'])
        return oracle._hold('merged',centered,now,step['hold_s'])
    if kind == 'lane_hold':
        lane = ego['lane_key']
        if not isinstance(lane, str) or not lane or type(ego['in_junction']) is not bool:
            raise ConfigError('invalid lane observation')
        if 'entry_lane' not in state:
            if ego['in_junction']:
                raise ConfigError('lane hold must start outside junction')
            state['entry_lane'] = lane
        condition = (not ego['in_junction'] and lane == state['entry_lane']
                     and abs(number(ego['lateral_error_m'], 'lateral error')) <= step['max_lateral_error_m']
                     and abs(number(ego['heading_error_deg'], 'heading error')) <= step['max_heading_error_deg'])
        return oracle._hold('stable_lane', condition, now, step['hold_s'])
    if kind == 'lane_position':
        from .lane_position import matches
        condition = (matches(ego.get('lane_position'), step['lane_side'], step['lane_ordinal'])
                     and abs(number(ego['lateral_error_m'], 'lateral error')) <= step['max_lateral_error_m']
                     and abs(number(ego['heading_error_deg'], 'heading error')) <= step['max_heading_error_deg'])
        return oracle._hold('lane_position', condition, now, step['hold_s'])
    if kind == 'maintain_interval':
        start, end = step['from_route_s_m'], step['until_route_s_m']
        previous = state.get('last_progress')
        if previous is not None and abs(progress - previous) > step['max_sample_distance_m']:
            raise ConfigError('interval observation distance gap')
        state['last_progress'] = progress
        if progress < start:
            if state.get('entered'):
                oracle.finish('FAILURE', 'reversed_out_of_required_interval', obs['frame'])
            return False
        if not state.get('entered'):
            if progress >= end or progress - start > step['max_sample_distance_m']:
                raise ConfigError('required interval entry not observed')
            state['entered'] = True
            state['entry_frame'] = obs['frame']
        if 'max_speed_kmh' in step and speed > step['max_speed_kmh']:
            oracle.finish('FAILURE', 'interval_speed_ceiling_exceeded', obs['frame'])
            return False
        if 'lane_ordinal' in step:
            from .lane_position import matches
            if (not matches(ego.get('lane_position'), step['lane_side'], step['lane_ordinal']) or
                    abs(number(ego['lateral_error_m'], 'lateral error')) > step['max_lateral_error_m'] or
                    abs(number(ego['heading_error_deg'], 'heading error')) > step['max_heading_error_deg']):
                oracle.finish('FAILURE', 'interval_lane_ordinal_violated', obs['frame'])
                return False
        if step.get('keep_lane'):
            corridor = oracle._fixture(obs)['lane_corridor']
            # At the first sample past the boundary retain the final segment's lane.
            at = min(progress, end)
            segments = [s for s in corridor if s['start_m'] <= at <= s['end_m']]
            if not segments:
                raise ConfigError('required interval outside verified lane corridor')
            allowed = {key for s in segments for key in s['lane_keys']}
            if (ego['lane_key'] not in allowed or
                    abs(number(ego['lateral_error_m'], 'lateral error')) > step['max_lateral_error_m'] or
                    abs(number(ego['heading_error_deg'], 'heading error')) > step['max_heading_error_deg']):
                oracle.finish('FAILURE', 'interval_lane_constraint_violated', obs['frame'])
                return False
        return progress >= end and obs['frame'] > state['entry_frame']

    if kind == 'speed_ceiling':
        return oracle._hold('under_ceiling', speed <= step['max_speed_kmh'], now, step['hold_s'])

    if kind == 'speed_change':
        entry = state.setdefault('entry_speed_kmh', speed)
        change = speed - entry
        if step['direction'] == 'DECREASE':
            change = -change
        return oracle._hold('changed', change >= step['delta_kmh'], now, step['hold_s'])

    if kind == 'follow_distance':
        target = oracle._target(step, obs)
        if not ego.get('route_corridor_id') or target['route_corridor_id'] != ego['route_corridor_id']:
            raise ConfigError('following target left shared route corridor')
        front = number(ego['front_route_s_m'], 'ego front')
        rear = number(target['rear_route_s_m'], 'target rear')
        gap = rear - front
        if gap < 0:
            oracle.finish('FAILURE', 'following_target_no_longer_ahead', obs['frame'])
            return False
        required = step['min_gap_m'] + max(0, speed) / 3.6 * step['time_headway_s']
        safe = gap >= required
        if not safe:
            state.pop('safe_start_progress', None)
        else:
            state.setdefault('safe_start_progress', progress)
        held = oracle._hold('safe', safe, now, step['hold_s'])
        # A parked ego cannot satisfy a sustained-following task.
        return held and progress - state['safe_start_progress'] >= step['min_travel_m']

    if kind == 'wait_clear':
        targets = [oracle._target({'target_role': role}, obs) for role in step['target_roles']]
        flags = [target['in_conflict_zone'] for target in targets]
        if any(not isinstance(flag, bool) for flag in flags):
            raise ConfigError('conflict occupancy requires geometric booleans')
        fixture = oracle._fixture(obs)
        line = number(fixture['stop_line_route_s_m'], 'stop line')
        front = number(ego['front_route_s_m'], 'ego front')
        if any(flags):
            state['observed_conflict'] = True
            state.pop('clear', None)
            if front > line:
                oracle.finish('FAILURE', 'crossed_stop_line_while_group_conflict', obs['frame'])
                return False
            if oracle._hold('stopped', speed <= step['stopped_kmh'], now, step['stop_hold_s']):
                state['stopped_for_conflict'] = True
            return False
        # Missing/replaced actors are invalid, not evidence that the road cleared.
        oracle._hold('stopped', False, now, step['stop_hold_s'])
        ready = ((not step.get('require_observed_conflict', True) or state.get('observed_conflict', False))
                 and (not step.get('require_stop', True) or state.get('stopped_for_conflict', False)))
        return oracle._hold('clear', ready, now, step['clear_hold_s'])

    if kind == 'straight_junction':
        fixture = oracle._fixture(obs)
        if fixture.get('legal') is not True or fixture.get('direction') != 'STRAIGHT':
            raise ConfigError('verified straight-junction fixture required')
        entry, exit_road = fixture['entry_road_key'], fixture['exit_road_key']
        if any(not isinstance(v, str) or not v for v in (entry, exit_road, ego['road_key'])):
            raise ConfigError('invalid junction road identity')
        inside = ego['in_junction']
        if not isinstance(inside, bool):
            raise ConfigError('invalid junction flag')
        if not state:
            if inside or ego['road_key'] != entry:
                raise ConfigError('straight junction must start before entry')
            state.update(entry_yaw=number(ego['yaw_deg'], 'entry yaw'), saw_junction=False)
        if inside:
            junction = ego.get('junction_id')
            if junction is None or junction != fixture.get('junction_id'):
                raise ConfigError('unexpected or missing junction identity')
            state['saw_junction'] = True
        delta = abs((number(ego['yaw_deg'], 'yaw') - state['entry_yaw'] + 180) % 360 - 180)
        condition = (state['saw_junction'] and not inside and ego['road_key'] == exit_road
                     and delta <= step['max_heading_change_deg']
                     and abs(number(ego['lateral_error_m'], 'lateral error')) <= step['max_lateral_error_m'])
        return oracle._hold('exited', condition, now, step['hold_s'])
    raise ConfigError('unsupported extended criterion')
