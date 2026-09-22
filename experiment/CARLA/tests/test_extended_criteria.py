import pytest

from benchmark.catalog import ConfigError
from benchmark.task_oracle import TaskOracle


def spec(step):
    return dict(schema_version='task_oracle/1.0',task_id='test',activate_m=0,
                timeout_s=10,max_frame_gap_s=1,steps=[step])


def observation(t, **changes):
    ego=dict(route_s_m=0,speed_kmh=40,lane_key='a',road_key='r1',in_junction=False,
             lateral_error_m=0,heading_error_deg=0,yaw_deg=0,route_corridor_id='main',half_length_m=2)
    ego.update(changes)
    return dict(schema_version='task_truth/1.0',source='simulator_truth',frame=int(t*20),
                sim_time_s=t,scenario_valid=True,ego=ego,safety=dict(collisions=0,violations=0),
                actors=dict(ped=dict(actor_id=10,alive=True,in_conflict_zone=True),
                            lead=dict(actor_id=11,alive=True,route_s_m=20,half_length_m=2,
                                      route_corridor_id='main')),fixture=dict(steps={}))


def test_relative_deceleration_requires_real_drop_and_hold():
    oracle = TaskOracle(spec(dict(kind='speed_change', direction='DECREASE', delta_kmh=10, hold_s=1)))
    for t, speed in [(0,40),(1,35),(2,30),(3,30)]:
        result = oracle.update(observation(t,speed_kmh=speed))
    assert result['status'] == 'SUCCESS'


@pytest.mark.parametrize('speed,expected', [(20,'SUCCESS'),(32,'SUCCESS'),(33,'RUNNING')])
def test_speed_ceiling_does_not_require_acceleration_to_limit(speed,expected):
    oracle=TaskOracle(spec(dict(kind='speed_ceiling',max_speed_kmh=32,hold_s=1)))
    for t in (0,1):
        result=oracle.update(observation(t,speed_kmh=speed))
    assert result['status']==expected


def test_ceiling_then_interval_cannot_finish_by_parking():
    profile=spec(dict(kind='speed_ceiling',max_speed_kmh=32,hold_s=1))
    profile['steps'].append(dict(kind='maintain_interval',from_route_s_m=5,until_route_s_m=20,
                                 max_sample_distance_m=5,max_speed_kmh=32))
    oracle=TaskOracle(profile)
    for t in range(6):
        result=oracle.update(observation(t,speed_kmh=0))
    assert result['completed_steps']==1
    assert result['status']=='RUNNING'


def test_following_requires_motion_not_just_waiting():
    oracle = TaskOracle(spec(dict(kind='follow_distance',target_role='lead',min_gap_m=3,
                                 time_headway_s=1,hold_s=1,min_travel_m=5)))
    for t, progress in [(0,0),(1,0),(2,5)]:
        row=observation(t,speed_kmh=10,route_s_m=progress,front_route_s_m=progress+2)
        row['actors']['lead']['rear_route_s_m']=progress+15
        result=oracle.update(row)
        assert result['status']==('SUCCESS' if t==2 else 'RUNNING')


@pytest.mark.parametrize('missing', [False, True])
def test_group_wait_does_not_treat_missing_actor_as_clear(missing):
    oracle=TaskOracle(spec(dict(kind='wait_clear',target_roles=['ped','second'],
                               stopped_kmh=.5,stop_hold_s=1,clear_hold_s=1)))
    for t in range(4):
        row=observation(t,speed_kmh=0,front_route_s_m=2)
        row['fixture']['steps']['0']={'stop_line_route_s_m':5}
        row['actors']['ped']['in_conflict_zone']=t<2
        row['actors']['second']=dict(actor_id=12,alive=True,in_conflict_zone=t<2)
        if t>=2 and missing:
            row['actors']['second']['alive']=False
        result=oracle.update(row)
    assert result['status']==('SCENE_INVALID' if missing else 'SUCCESS')


@pytest.mark.parametrize('exit_yaw,expected', [(3,'SUCCESS'),(90,'RUNNING')])
def test_straight_requires_observed_expected_junction_and_straight_exit(exit_yaw,expected):
    oracle=TaskOracle(spec(dict(kind='straight_junction',max_heading_change_deg=15,
                               max_lateral_error_m=.5,hold_s=1)))
    for t in range(4):
        row=observation(t,road_key='r1' if t<2 else 'r2',in_junction=t==1,
                        junction_id=42 if t==1 else None,yaw_deg=exit_yaw if t>=2 else 0)
        row['fixture']['steps']['0']=dict(direction='STRAIGHT',legal=True,junction_id=42,
                                         entry_road_key='r1',exit_road_key='r2')
        result=oracle.update(row)
    assert result['status']==expected


def test_invalid_group_role_definition_rejected():
    with pytest.raises(ConfigError):
        TaskOracle(spec(dict(kind='wait_clear',target_roles=['ped','ped'],stopped_kmh=.5,
                             stop_hold_s=1,clear_hold_s=1)))


@pytest.mark.parametrize('occupied,expected', [(False, 'SUCCESS'), (True, 'FAILURE')])
def test_clear_confirmation_does_not_require_unnecessary_stop(occupied, expected):
    oracle=TaskOracle(spec(dict(kind='wait_clear',target_roles=['ped'],
                               require_stop=False,require_observed_conflict=False,
                               stopped_kmh=.5,stop_hold_s=1,clear_hold_s=1)))
    for t in range(2):
        row=observation(t,speed_kmh=20,front_route_s_m=6)
        row['fixture']['steps']['0']={'stop_line_route_s_m':5}
        row['actors']['ped']['in_conflict_zone']=occupied
        result=oracle.update(row)
    assert result['status']==expected


def test_rolling_wait_must_observe_conflict_when_required():
    oracle=TaskOracle(spec(dict(kind='wait_clear',target_roles=['ped'],require_stop=False,
                               stopped_kmh=.5,stop_hold_s=1,clear_hold_s=1)))
    for t in range(3):
        row=observation(t,speed_kmh=10,front_route_s_m=2)
        row['fixture']['steps']['0']={'stop_line_route_s_m':5}
        row['actors']['ped']['in_conflict_zone']=False
        result=oracle.update(row)
    assert result['status']=='RUNNING'


def interval(**overrides):
    step=dict(kind='maintain_interval',from_route_s_m=0,until_route_s_m=30,
              max_sample_distance_m=12,max_speed_kmh=40)
    step.update(overrides)
    return TaskOracle(spec(step))


def test_interval_does_not_finish_after_brief_compliance():
    oracle=interval()
    for t in range(4):
        result=oracle.update(observation(t,route_s_m=t*10,speed_kmh=35))
        assert result['status']==('SUCCESS' if t==3 else 'RUNNING')


def test_interval_speed_violation_cannot_be_erased_by_later_slowdown():
    oracle=interval()
    oracle.update(observation(0,speed_kmh=35))
    assert oracle.update(observation(1,route_s_m=10,speed_kmh=41))['status']=='FAILURE'
    assert oracle.update(observation(2,route_s_m=20,speed_kmh=30))['status']=='FAILURE'


def test_interval_rejects_teleport_over_remaining_section():
    oracle=interval()
    oracle.update(observation(0))
    assert oracle.update(observation(1,route_s_m=30))['status']=='SCENE_INVALID'


def test_interval_does_not_accept_already_passed_entry():
    assert interval().update(observation(0,route_s_m=15))['status']=='SCENE_INVALID'


def test_interval_lane_violation_fails_not_resets_hold():
    oracle=interval(keep_lane=True,max_lateral_error_m=.35,max_heading_error_deg=5)
    for t,lane in [(0,'a'),(1,'b')]:
        row=observation(t,route_s_m=t*10,lane_key=lane)
        row['fixture']['steps']['0']={'lane_corridor':[dict(start_m=0,end_m=30,lane_keys=['a'])]}
        result=oracle.update(row)
    assert result['status']=='FAILURE'


def test_interval_cannot_end_on_timeout_boundary():
    definition=spec(dict(kind='maintain_interval',from_route_s_m=0,until_route_s_m=30,
                         max_sample_distance_m=12,max_speed_kmh=40))
    definition['end_route_s_m']=30
    with pytest.raises(ConfigError):
        TaskOracle(definition)


def test_lane_hold_uses_step_entry_lane_and_continuous_centering():
    oracle=TaskOracle(spec(dict(kind='lane_hold',hold_s=1,max_lateral_error_m=.35,
                               max_heading_error_deg=5)))
    for t,lane in [(0,'a'),(1,'b'),(2,'a'),(3,'a')]:
        result=oracle.update(observation(t,lane_key=lane))
        assert result['status']==('SUCCESS' if t==3 else 'RUNNING')


def test_scene2_new_profiles_match_current_source():
    from benchmark.catalog import load_catalog
    from benchmark.task_oracle import load_profile
    catalog=load_catalog('scene_2')
    for task_id in ('s2_t05_cmd_01','s2_t05_cmd_09','s2_t05_cmd_11'):
        profile=load_profile(catalog,catalog.select(task_id)[0])
        assert profile is not None
        assert len(profile['steps'])==3
