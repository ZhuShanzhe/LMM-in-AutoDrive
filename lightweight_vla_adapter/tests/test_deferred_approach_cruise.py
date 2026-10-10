import pytest
from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime


def context():
    runtime=DrivingPlanRuntime()
    runtime.frame_id='f20';runtime.timestamp_s=1.
    runtime.document=dict(request_id='change',intent=dict(steps=[dict(step_id='lane',
        action='CHANGE_LANE',parameters=dict(direction='LEFT'))]))
    runtime.state=dict(plan_status='ACTIVE',active_step_index=0,
        step_states=[dict(status='WAITING')])
    decision=dict(decision_status='BLOCKED',action='stop',target_speed_kmh=0.,
        emergency=False,blocked_reason_codes=['passing_target_not_acquired'])
    risk=dict(frame_id='f20',risk_level='low',behavior_policy=dict(
        disposition='CLEAR',physical_stop=False,max_speed_kmh=30.,
        lidar=dict(valid=True,clear_distance_m=35.,sensor_frame=20,observed_at_s=1.)),
        physical_forward_radar=dict(sensor_frame=20,obstacle_candidate_count=0,nearest_distance_m=None))
    evidence=dict(request_id='change',frame_id='f20',condition='CURRENT_LANE_MOTION',
        valid=True,satisfied=True,source='fresh_lidar_radar_driving_corridor',
        observed_at_s=1.,valid_until_s=1.25,sensor_frame=20)
    return runtime,decision,risk,evidence


def test_observed_long_clearance_allows_nominal_approach_but_not_a_lane_change():
    runtime,decision,risk,evidence=context()
    result=runtime.deferred_lane_navigation(decision,risk,evidence,cruise_speed_kmh=30.)
    assert result['target_speed_kmh']==30.
    assert result['action']=='keep_lane'
    assert result['target_lane'] is None and result['target_location'] is None
    assert runtime.state['step_states'][0]['status']=='WAITING'


@pytest.mark.parametrize('bad',['short','radar_obstacle','radar_old','lidar_old','absolute_stop'])
def test_short_or_inconsistent_observation_never_upgrades_creep_speed(bad):
    runtime,decision,risk,evidence=context()
    lidar=risk['behavior_policy']['lidar'];radar=risk['physical_forward_radar']
    if bad=='short':lidar['clear_distance_m']=8.
    if bad=='radar_obstacle':radar.update(obstacle_candidate_count=1,nearest_distance_m=8.)
    if bad=='radar_old':radar['sensor_frame']=19
    if bad=='lidar_old':lidar['observed_at_s']=.8
    if bad=='absolute_stop':risk['behavior_policy']['absolute_stop_active']=True
    result=runtime.deferred_lane_navigation(decision,risk,evidence,cruise_speed_kmh=30.)
    assert result['target_speed_kmh']<=10.


@pytest.mark.parametrize('target',[0.,5.,15.,20.])
def test_explicit_prior_speed_target_is_not_erased_by_approach_cruise(target):
    runtime,decision,risk,evidence=context()
    runtime.document['intent']['steps'].insert(0,dict(step_id='slow',action='ADJUST_SPEED',
        parameters=dict(target_speed_kmh=target)))
    runtime.state['active_step_index']=1
    runtime.state['step_states'].insert(0,dict(status='COMPLETED'))
    result=runtime.deferred_lane_navigation(decision,risk,evidence,cruise_speed_kmh=30.)
    assert result['target_speed_kmh']<=target


def test_temporary_relative_slowdown_does_not_lock_pending_cruise_to_one_kmh():
    runtime,decision,risk,evidence=context()
    runtime.document['intent']['steps'].insert(0,dict(step_id='slow',action='ADJUST_SPEED',
        parameters=dict(change='DECREASE',delta_speed_kmh=5)))
    runtime.state['active_step_index']=1
    runtime.state['step_states'].insert(0,dict(status='COMPLETED',resolved_target_speed_kmh=.9))
    assert runtime.deferred_lane_navigation(decision,risk,evidence,
        cruise_speed_kmh=30.)['target_speed_kmh']==30.


def test_started_merge_and_physical_stop_still_reject_any_approach_cruise():
    runtime,decision,risk,evidence=context()
    assert runtime.deferred_lane_navigation(decision,risk,evidence,
        cruise_speed_kmh=30.,maneuver_started=True)==decision
    risk['behavior_policy']['physical_stop']=True
    assert runtime.deferred_lane_navigation(decision,risk,evidence,cruise_speed_kmh=30.)==decision


def test_newer_completed_absolute_speed_replaces_an_older_target():
    runtime,decision,risk,evidence=context()
    for index,value in enumerate((5.,25.)):
        runtime.document['intent']['steps'].insert(index,dict(step_id=str(index),action='SET_SPEED',
            parameters=dict(target_speed_kmh=value)))
        runtime.state['step_states'].insert(index,dict(status='COMPLETED'))
    runtime.state['active_step_index']=2
    assert runtime.deferred_lane_navigation(decision,risk,evidence,
        cruise_speed_kmh=30.)['target_speed_kmh']==25.
