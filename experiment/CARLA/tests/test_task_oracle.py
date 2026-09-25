import copy
import json

import pytest

from benchmark.catalog import ConfigError, load_catalog
from benchmark.planning import build_plan
from benchmark.task_oracle import TaskOracle
from benchmark.__main__ import main


SPEED = dict(kind='speed', target_kmh=40, tolerance_kmh=2, hold_s=1, keep_lane=True)
LANE = dict(kind='lane_change', direction='LEFT', hold_s=1,
            max_lateral_error_m=.35, max_heading_error_deg=5)
TURN = dict(kind='turn', direction='RIGHT', hold_s=1,
            min_heading_change_deg=45, max_lateral_error_m=.5)
YIELD = dict(kind='yield_pedestrian', target_role='ped', stop_hold_s=1,
             stopped_kmh=.5, clear_hold_s=1)
PASS = dict(kind='overtake', target_role='lead', rear_clearance_m=8, hold_s=1)


def spec(*steps):
    return dict(schema_version='task_oracle/1.0', task_id='test', activate_m=0,
                timeout_s=10, max_frame_gap_s=1, steps=list(steps))


def observation(t, **ego):
    state = dict(route_s_m=0, speed_kmh=40, lane_key='a', road_key='r1',
                 in_junction=False, lateral_error_m=0, heading_error_deg=0,
                 yaw_deg=0, route_corridor_id='main', half_length_m=2)
    state.update(ego)
    return dict(schema_version='task_truth/1.0', source='simulator_truth', frame=int(t*20),
        sim_time_s=t, scenario_valid=True, ego=state, safety=dict(collisions=0, violations=0),
        actors=dict(ped=dict(actor_id=10, alive=True, in_conflict_zone=True),
                    lead=dict(actor_id=11, alive=True, route_s_m=20, half_length_m=2,
                              route_corridor_id='main')),
        fixture=dict(steps={'0': dict(entry_lane_key='a', target_lane_key='b', direction='LEFT', legal=True)}))


def test_speed_requires_continuous_measured_hold():
    oracle = TaskOracle(spec(SPEED))
    for t, speed in [(0,40),(.5,40),(1,20),(1.5,40),(2,40)]:
        assert oracle.update(observation(t, speed_kmh=speed))['status'] == 'RUNNING'
    assert oracle.update(observation(2.5))['status'] == 'SUCCESS'


@pytest.mark.parametrize('proof,expected', [(True,'SUCCESS'), (False,'SCENE_INVALID')])
def test_passed_target_confirmation_requires_prior_evidence_not_second_overtake(proof,expected):
    profile=spec(dict(kind='passed_target',target_role='lead',rear_clearance_m=8,hold_s=1))
    profile.update(requires_task_success=['earlier'],source_sha256='test-source')
    oracle=TaskOracle(profile)
    for t in (1,2):
        row=observation(t)
        row.update(source_sha256='test-source',task_id='test')
        row['actors']['lead']['route_s_m']=-20
        if proof:
            row['fixture']['prerequisites']={'earlier':dict(status='SUCCESS',source='independent_task_oracle',
                source_sha256='test-source',completion_frame=0)}
        result=oracle.update(row)
    assert result['status']==expected


@pytest.mark.parametrize('second_lane,expected',[('b','SUCCESS'),('adjacent','RUNNING')])
def test_speed_uses_verified_corridor_across_road_and_junction(second_lane,expected):
    oracle=TaskOracle(spec(SPEED))
    corridor=[dict(start_m=0,end_m=5,lane_keys=['a']),
              dict(start_m=5,end_m=10,lane_keys=['b'])]
    def obs(t,progress,lane):
        row=observation(t,route_s_m=progress,lane_key=lane,in_junction=True)
        row['fixture']['steps']['0']={'lane_corridor':corridor}
        return row
    oracle.update(obs(0,0,'a'))
    assert oracle.update(obs(1,6,second_lane))['status']==expected


def test_speed_corridor_still_enforces_geometric_bounds():
    oracle=TaskOracle(spec(dict(SPEED,max_lateral_error_m=.35,max_heading_error_deg=5)))
    for t in range(3):
        row=observation(t,lateral_error_m=.36)
        row['fixture']['steps']['0']={'lane_corridor':[dict(start_m=0,end_m=10,lane_keys=['a'])]}
        assert oracle.update(row)['status']=='RUNNING'


@pytest.mark.parametrize('second_progress,expected',[(4,'SUCCESS'),(6,'TIMEOUT')])
def test_unfinished_speed_task_cannot_cross_unverified_corridor(second_progress,expected):
    oracle=TaskOracle(spec(SPEED))
    for t,progress in [(0,0),(1,second_progress)]:
        row=observation(t,route_s_m=progress)
        row['fixture']['steps']['0']={'lane_corridor':[dict(start_m=0,end_m=5,lane_keys=['a'])],
                                    'verified_end_m':5,'requested_end_m':15,
                                    'stop_reason':'ambiguous forward topology'}
        result=oracle.update(row)
    assert result['status']==expected


def test_speed_corridor_missing_at_task_entry_is_scene_invalid():
    oracle=TaskOracle(spec(SPEED))
    row=observation(0,route_s_m=6)
    row['fixture']['steps']['0']={'lane_corridor':[dict(start_m=0,end_m=5,lane_keys=['a'])]}
    assert oracle.update(row)['status']=='SCENE_INVALID'


@pytest.mark.parametrize('field,value',[('lateral_error_m',.6),('heading_error_deg',8)])
def test_speed_with_lane_constraint_rejects_off_center_vehicle(field,value):
    step=dict(SPEED,max_lateral_error_m=.35,max_heading_error_deg=5)
    oracle=TaskOracle(spec(step))
    for t in range(3):
        assert oracle.update(observation(t,**{field:value}))['status']=='RUNNING'
    assert oracle.update(observation(3))['status']=='RUNNING'
    assert oracle.update(observation(4))['status']=='SUCCESS'


def test_speed_cannot_complete_after_next_command_boundary():
    bounded=dict(spec(SPEED),end_route_s_m=50)
    oracle=TaskOracle(bounded)
    oracle.update(observation(0,route_s_m=0))
    result=oracle.update(observation(1,route_s_m=50))
    assert result['status']=='TIMEOUT'
    assert result['reason']=='task_distance_window_closed'


@pytest.mark.parametrize('progress,position,expected',[
    (99,{'x':99,'y':0,'z':0},'SUCCESS'),
    (99,{'x':99,'y':10,'z':0},'RUNNING'),
    (99,{'x':99,'y':0,'z':10},'RUNNING'),
    (10,{'x':100,'y':0,'z':0},'RUNNING'),
    (110,{'x':100,'y':0,'z':0},'RUNNING'),
])
def test_destination_requires_progress_and_physical_arrival(progress,position,expected):
    oracle=TaskOracle(spec({'kind':'destination','max_distance_m':3,'max_remaining_m':3}))
    def obs(t,s,point):
        value=observation(t,route_s_m=s,position_m=point)
        value['fixture']['steps']['0']={'route_end_s_m':100,'position_m':{'x':100,'y':0,'z':0}}
        return value
    oracle.update(obs(0,0,{'x':0,'y':0,'z':0}))
    assert oracle.update(obs(1,progress,position))['status']==expected


def test_destination_missing_position_is_not_success():
    oracle=TaskOracle(spec({'kind':'destination','max_distance_m':3,'max_remaining_m':3}))
    value=observation(0)
    value['fixture']['steps']['0']={'route_end_s_m':100,'position_m':{'x':100,'y':0,'z':0}}
    assert oracle.update(value)['status']=='SCENE_INVALID'


def test_destination_lane_departure_is_terminal_even_if_vehicle_returns():
    step=dict(kind='destination',max_distance_m=3,max_remaining_m=3,keep_lane=True,
              max_lateral_error_m=.35,max_heading_error_deg=5)
    oracle=TaskOracle(spec(step))
    def obs(t,progress,lane):
        value=observation(t,route_s_m=progress,lane_key=lane,position_m={'x':progress,'y':0,'z':0})
        value['fixture']['steps']['0']={'route_end_s_m':100,'position_m':{'x':100,'y':0,'z':0},
            'lane_corridor':[{'start_m':0,'end_m':100,'lane_keys':['a']}]}
        return value
    oracle.update(obs(0,0,'a'))
    assert oracle.update(obs(1,10,'b'))['status']=='FAILURE'
    assert oracle.update(obs(2,99,'a'))['status']=='FAILURE'


def test_speed_profiles_match_source_speed_and_next_announcement():
    from benchmark.task_oracle import load_profile
    catalog=load_catalog('scene_1')
    seen=[]
    for task in catalog.tasks:
        profile=load_profile(catalog,task)
        if profile is None or profile['steps'][0]['kind']!='speed':
            continue
        step=profile['steps'][0]
        assert step['target_kmh']==task.source_command['target_speed_kmh']
        later=[t.announce_m for t in catalog.tasks if t.announce_m>task.activate_m]
        assert profile['end_route_s_m']==min(later)
        seen.append(task.task_id)
    assert len(seen)==11


@pytest.mark.parametrize('change',[{'status':'RUNNING'}, {'source':'policy'},
    {'completion_frame':20},{'source_sha256':'wrong'}])
def test_prerequisite_cannot_use_unfinished_policy_or_future_claim(change):
    contract=dict(spec(SPEED),requires_task_success=['turn'],source_sha256='same')
    oracle=TaskOracle(contract)
    value=observation(1)
    value.update(source_sha256='same',task_id='test')
    value['fixture']['prerequisites']={'turn':dict(status='SUCCESS',source='independent_task_oracle',
        source_sha256='same',completion_frame=1)}
    value['fixture']['prerequisites']['turn'].update(change)
    assert oracle.update(value)['status']=='SCENE_INVALID'


def test_independent_prerequisite_allows_followup():
    oracle=TaskOracle(dict(spec(SPEED),requires_task_success=['turn']))
    for t in (1,2):
        value=observation(t)
        value['fixture']['prerequisites']={'turn':dict(status='SUCCESS',
            source='independent_task_oracle',completion_frame=1)}
        result=oracle.update(value)
    assert result['status']=='SUCCESS'


def test_isolated_followup_plan_exposes_missing_prior_task():
    plan=build_plan(load_catalog('scene_1'),'c10_keep_35')
    assert plan['missing_selected_prerequisites']==['c07_turn_left']
    assert 'prior_task_evidence_required' in plan['blockers']
    full=build_plan(load_catalog('scene_1'),'all')
    assert full['missing_selected_prerequisites']==[]


def test_instruction_or_policy_completion_does_not_imply_success():
    oracle = TaskOracle(spec(SPEED))
    for t in range(11):
        frame = observation(t, speed_kmh=0)
        frame.update(command_emitted=True, policy_status='SUCCESS', event_state='RESOLVED')
        assert oracle.update(frame)['status'] == 'RUNNING'
    assert oracle.update(observation(11, speed_kmh=0))['status'] == 'TIMEOUT'


def test_lane_change_requires_correct_lane_and_stable_center():
    oracle = TaskOracle(spec(LANE))
    oracle.update(observation(0))
    oracle.update(observation(1, lane_key='b', lateral_error_m=1))
    assert oracle.update(observation(2, lane_key='b'))['status'] == 'RUNNING'
    assert oracle.update(observation(3, lane_key='b'))['status'] == 'SUCCESS'


def test_wrong_lane_never_completes():
    oracle = TaskOracle(spec(LANE))
    oracle.update(observation(0))
    for t in range(1,4):
        assert oracle.update(observation(t,lane_key='c'))['status'] == 'RUNNING'


def test_registered_right_lane_profile_matches_source_and_direction():
    from benchmark.task_oracle import load_profile
    catalog=load_catalog('scene_1')
    task=next(t for t in catalog.tasks if t.task_id=='c05_change_right')
    profile=load_profile(catalog,task)
    assert profile is not None and profile['activate_m']==1450
    assert profile['steps'][0]['direction']=='RIGHT'


def test_right_lane_change_requires_correct_direction_and_stable_center():
    def right(t,**ego):
        value=observation(t,**ego)
        value['fixture']['steps']['0']['direction']='RIGHT'
        return value
    oracle=TaskOracle(spec(dict(LANE,direction='RIGHT')))
    oracle.update(right(0))
    assert oracle.update(right(1,lane_key='c'))['status']=='RUNNING'
    assert oracle.update(right(2,lane_key='b',lateral_error_m=.6))['status']=='RUNNING'
    assert oracle.update(right(3,lane_key='b'))['status']=='RUNNING'
    assert oracle.update(right(4,lane_key='b'))['status']=='SUCCESS'
    wrong=TaskOracle(spec(dict(LANE,direction='RIGHT')))
    assert wrong.update(observation(0))['status']=='SCENE_INVALID'


def test_connected_lane_identity_can_cross_road_boundary():
    oracle=TaskOracle(spec(LANE))
    def obs(t,key):
        value=observation(t,lane_key=key)
        value['fixture']['steps']['0'].update(entry_lane_keys=['a','a_next'],target_lane_keys=['b','b_next'])
        return value
    oracle.update(obs(0,'a'))
    oracle.update(obs(1,'b'))
    assert oracle.update(obs(2,'b_next'))['status']=='SUCCESS'


def test_step_location_gate_does_not_evaluate_wrong_entry_before_arrival():
    oracle=TaskOracle(spec(dict(LANE,start_route_s_m=20)))
    assert oracle.update(observation(0,lane_key='approach',route_s_m=0))['status']=='RUNNING'
    assert oracle.update(observation(1,route_s_m=20))['status']=='RUNNING'
    oracle.update(observation(2,lane_key='b',route_s_m=25))
    assert oracle.update(observation(3,lane_key='b',route_s_m=30))['status']=='SUCCESS'


def turn_obs(t, **ego):
    obs = observation(t, **ego)
    obs['fixture']['steps']['0'] = dict(direction='RIGHT', legal=True,
                                      entry_road_key='r1', exit_road_key='r2')
    return obs


def test_turn_requires_junction_and_new_road():
    oracle = TaskOracle(spec(TURN))
    oracle.update(turn_obs(0))
    oracle.update(turn_obs(1, in_junction=True,yaw_deg=40))
    oracle.update(turn_obs(2, road_key='r2',yaw_deg=90))
    assert oracle.update(turn_obs(3,road_key='r2',yaw_deg=90))['status'] == 'SUCCESS'


@pytest.mark.parametrize('yaw,road', [(90,'r1'),(-90,'r2'),(10,'r2')])
def test_curve_wrong_turn_or_small_bend_not_success(yaw,road):
    oracle = TaskOracle(spec(TURN))
    oracle.update(turn_obs(0))
    oracle.update(turn_obs(1,in_junction=True))
    oracle.update(turn_obs(2,road_key=road,yaw_deg=yaw))
    assert oracle.update(turn_obs(3,road_key=road,yaw_deg=yaw))['status'] == 'RUNNING'


def pedestrian_obs(t, conflict=True, **ego):
    obs=observation(t,speed_kmh=0,**ego)
    obs['actors']['ped']['in_conflict_zone']=conflict
    obs['fixture']['steps']['0']=dict(stop_line_route_s_m=5)
    return obs


def test_wait_stop_then_real_clear():
    oracle=TaskOracle(spec(YIELD))
    oracle.update(pedestrian_obs(0))
    oracle.update(pedestrian_obs(1))
    oracle.update(pedestrian_obs(2,False))
    assert oracle.update(pedestrian_obs(3,False))['status']=='SUCCESS'


@pytest.mark.parametrize('modification', [dict(alive=False),dict(teleported=True),dict(actor_id=999)])
def test_disappearing_or_replaced_pedestrian_invalidates_scene(modification):
    oracle=TaskOracle(spec(YIELD))
    oracle.update(pedestrian_obs(0))
    frame=pedestrian_obs(1,False)
    frame['actors']['ped'].update(modification)
    assert oracle.update(frame)['status']=='SCENE_INVALID'


def test_clear_without_observed_conflict_is_not_success():
    oracle=TaskOracle(spec(YIELD))
    for t in range(4):
        assert oracle.update(pedestrian_obs(t,False))['status']=='RUNNING'


def test_crossing_occupied_stop_line_fails():
    oracle=TaskOracle(spec(YIELD))
    assert oracle.update(pedestrian_obs(0,route_s_m=6))['status']=='FAILURE'


def test_overtake_requires_front_to_rear_and_bumper_clearance():
    oracle=TaskOracle(spec(PASS))
    oracle.update(observation(0))
    oracle.update(observation(1,route_s_m=25))
    assert oracle.update(observation(2,route_s_m=32))['status']=='RUNNING'
    assert oracle.update(observation(3,route_s_m=34))['status']=='SUCCESS'


def test_already_behind_target_does_not_count_as_overtake():
    profile=spec(PASS)
    profile['activate_m']=40
    oracle=TaskOracle(profile)
    for t in range(4):
        assert oracle.update(observation(t,route_s_m=40))['status']=='RUNNING'


def test_sequence_cannot_skip_speed_step():
    oracle=TaskOracle(spec(SPEED,PASS))
    for t in range(4):
        oracle.update(observation(t,route_s_m=t*20,speed_kmh=20))
    assert oracle.index==0


def test_sequence_requires_separate_ordered_observations():
    oracle=TaskOracle(spec(SPEED,dict(kind='progress',distance_m=10)))
    oracle.update(observation(0))
    oracle.update(observation(1))
    assert oracle.index==1
    oracle.update(observation(2,route_s_m=30))
    assert oracle.update(observation(3,route_s_m=40))['status']=='SUCCESS'


@pytest.mark.parametrize('change', [dict(sim_time_s=float('nan')),dict(frame=True),
                                  dict(source='policy'),dict(scenario_valid=False)])
def test_bad_truth_invalidates_scene(change):
    oracle=TaskOracle(spec(SPEED))
    frame=observation(0)
    frame.update(change)
    assert oracle.update(frame)['status']=='SCENE_INVALID'


def test_gaps_duplicates_and_incomplete_stream_are_not_success():
    for next_time in (0,2):
        oracle=TaskOracle(spec(SPEED))
        oracle.update(observation(0))
        assert oracle.update(observation(next_time))['status']=='SCENE_INVALID'
    oracle=TaskOracle(spec(SPEED))
    oracle.update(observation(0))
    assert oracle.end_of_stream()['status']=='SCENE_INVALID'


def test_collision_has_priority_over_completion_and_result_latches():
    oracle=TaskOracle(spec(SPEED))
    oracle.update(observation(0))
    frame=observation(1)
    frame['safety']['collisions']=1
    assert oracle.update(frame)['status']=='FAILURE'
    assert oracle.update(observation(2))['status']=='FAILURE'


def test_fixture_mutation_is_rejected():
    oracle=TaskOracle(spec(LANE))
    oracle.update(observation(0))
    frame=observation(1,lane_key='c')
    frame['fixture']['steps']['0']['target_lane_key']='c'
    assert oracle.update(frame)['status']=='SCENE_INVALID'


def test_plan_preserves_dependencies_and_does_not_claim_execution():
    catalog=load_catalog('scene_2')
    plan=build_plan(catalog,'4',seed=123)
    assert plan['required_event_states']['s2_t05_cmd_04']=={'crosswalk_pedestrian':'RESOLVED'}
    assert plan['preparation_start_m']==830
    assert not plan['execution_supported']
    assert not plan['history_equivalent_to_full_run']
    assert 'source_command' not in plan['model_commands'][0]
    assert len(plan['fixture_review_events'])==len(catalog.events)


def test_cli_truth_evaluation(tmp_path):
    profile=tmp_path/'profile.json'
    frames=tmp_path/'truth.jsonl'
    output=tmp_path/'result.json'
    profile.write_text(json.dumps(spec(SPEED)),encoding='utf-8')
    frames.write_text('\n'.join(json.dumps(observation(t)) for t in (0,1)),encoding='utf-8')
    assert main(['evaluate','--spec',str(profile),'--observations',str(frames),'--output',str(output)])==0
    assert json.loads(output.read_text())['status']=='SUCCESS'
    assert main(['plan','--scene','scene_3','--task','1','--output',str(output)])==0


def test_invalid_spec_rejected():
    bad=copy.deepcopy(SPEED)
    bad['hold_s']=0
    with pytest.raises(ConfigError):
        TaskOracle(spec(bad))


def test_first_frame_collision_is_not_hidden_by_baseline():
    frame=observation(0)
    frame['safety']['collisions']=1
    assert TaskOracle(spec(SPEED)).update(frame)['status']=='FAILURE'


def test_late_start_is_not_valid_task_entry():
    assert TaskOracle(spec(SPEED)).update(observation(0,route_s_m=30))['status']=='SCENE_INVALID'


def test_registered_profiles_match_current_catalog():
    from benchmark.catalog import CONFIG_ROOT
    for path in (CONFIG_ROOT/'benchmark').glob('*.json'):
        profile=json.loads(path.read_text(encoding='utf-8'))
        TaskOracle(profile)
        catalog=load_catalog(profile['scene_id'])
        assert profile['source_sha256']==catalog.source_sha256
        assert catalog.select(profile['task_id'])[0].activate_m==profile['activate_m']


def test_rolling_yield_does_not_require_full_stop():
    criterion=dict(YIELD, require_stop=False, min_speed_drop_kmh=5)
    oracle=TaskOracle(spec(criterion))
    for t,speed,conflict in [(0,40,True),(1,30,True),(2,30,True),(3,30,False),(4,30,False)]:
        frame=pedestrian_obs(t,conflict)
        frame['ego']['speed_kmh']=speed
        result=oracle.update(frame)
    assert result['status']=='SUCCESS'


def test_complete_three_step_sequence():
    oracle=TaskOracle(spec(YIELD,LANE,PASS))
    for t in range(4):
        oracle.update(pedestrian_obs(t,conflict=t<2))
    for t in (4,5,6):
        frame=observation(t,lane_key='a' if t==4 else 'b')
        frame['fixture']['steps']['1']=frame['fixture']['steps']['0']
        oracle.update(frame)
    for t,progress in [(7,0),(8,40),(9,42)]:
        result=oracle.update(observation(t,route_s_m=progress))
    assert result['status']=='SUCCESS'
    assert [e['kind'] for e in result['evidence'] if e['event']=='STEP_SUCCESS']==[
        'yield_pedestrian','lane_change','overtake']


def test_plan_binds_only_matching_explicit_profiles():
    plan=build_plan(load_catalog('scene_1'),'1')
    assert list(plan['oracle_profiles'])==['c01_depart_45']
    assert not plan['missing_oracle_profiles']
    assert not plan['execution_supported']


def test_profile_provenance_is_checked():
    profile=spec(SPEED)
    profile['source_sha256']='abc'
    assert TaskOracle(profile).update(observation(0))['status']=='SCENE_INVALID'
