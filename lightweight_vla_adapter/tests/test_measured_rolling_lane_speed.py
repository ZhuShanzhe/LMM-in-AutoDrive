import math
import pytest
from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime
from lightweight_vla_adapter.src.lane_risk_contract import LaneRiskContract


@pytest.mark.parametrize('front,rear,occupied,legal,expected',[
    (20.,10.,False,True,15.),(30.,15.,False,True,15.),
    (19.9,10.,False,True,8.),(20.,9.9,False,True,8.),
    (None,10.,False,True,8.),(math.nan,10.,False,True,8.),
    (True,10.,False,True,8.),(20.,10.,True,True,8.),
    (20.,10.,False,False,8.),
])
def test_rolling_cap_requires_continuously_measured_full_lane_geometry(front,rear,occupied,legal,expected):
    contract=LaneRiskContract()
    evidence=dict(valid=True,occupied=occupied,observed=True,
        source='same_frame_lidar_target_lane_geometry',front_horizon_m=front,rear_horizon_m=rear)
    for frame,time in enumerate([0.,.2,.4,.6,.8],1):
        risk=contract.update(dict(frame_id='f1'),'left',dict(source='camera',risk_level='low'),None,
            timestamp_s=time,sensor_frame=frame,frame_age_s=.05,speed_mps=1.,
            physical_evidence=evidence,road_legal=legal)
    assert risk['lane_change']['left']['speed_cap_kmh']==expected
    assert risk['lane_change']['left']['is_safe']==(not occupied and legal)


def fixture():
    runtime=DrivingPlanRuntime();runtime.frame_id='f1'
    runtime.document=dict(intent=dict(steps=[
        dict(action='ADJUST_SPEED',parameters=dict(change='DECREASE',delta_kmh=5.)),
        dict(action='CHANGE_LANE',parameters=dict(direction='LEFT'))]))
    runtime.state=dict(active_step_index=1,step_states=[dict(status='COMPLETED'),dict(status='ACTIVE')])
    decision=dict(action='lane_change_left',source_step_action='CHANGE_LANE',frame_id='f1',
        decision_status='READY',emergency=False,target_speed_kmh=0.)
    risk=dict(frame_id='f1',target_lane_contract=dict(direction='left',can_execute=True),
        behavior_policy=dict(disposition='CLEAR',physical_stop=False,max_speed_kmh=30.),
        lane_change=dict(left=dict(speed_cap_kmh=15.)),task_feasibility=dict(
            CHANGE_LANE_LEFT=dict(can_execute=True),PATH_CLEAR=dict(can_execute=True)))
    return runtime,decision,risk


@pytest.mark.parametrize('explicit,expected',[(None,15.),(5.,5.),(3.,3.),(0.,0.)])
def test_relative_slowdown_does_not_become_permanent_merge_speed(explicit,expected):
    runtime,decision,risk=fixture()
    if explicit is not None:runtime.document['intent']['steps'][0]['parameters']['target_speed_kmh']=explicit
    result=runtime.start_authorized_lane_motion(decision,risk,cruise_speed_kmh=30.)
    assert result['target_speed_kmh']==expected


@pytest.mark.parametrize('failure',['stop','blocked','path','lane','physical','stale','absolute_stop'])
def test_rolling_lane_speed_never_bypasses_stop_or_missing_permission(failure):
    runtime,decision,risk=fixture()
    if failure=='stop':decision['action']='stop'
    if failure=='blocked':decision['decision_status']='BLOCKED'
    if failure=='path':risk['task_feasibility']['PATH_CLEAR']['can_execute']=False
    if failure=='lane':risk['task_feasibility']['CHANGE_LANE_LEFT']['can_execute']=False
    if failure=='physical':risk['behavior_policy']['physical_stop']=True
    if failure=='stale':risk['frame_id']='old'
    if failure=='absolute_stop':risk['behavior_policy']['absolute_stop_active']=True
    assert runtime.start_authorized_lane_motion(decision,risk,cruise_speed_kmh=30.)==decision
