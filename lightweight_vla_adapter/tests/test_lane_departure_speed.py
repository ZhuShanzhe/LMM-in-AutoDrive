import pytest
from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime


@pytest.mark.parametrize('failure',[None,'blocked','emergency','no_path','no_lane','wrong_direction',
    'stale','physical_stop','unknown_motion','zero_cap','nan_cap','missing_cap','stop_action','wrong_source'])
def test_lane_departure_needs_both_corridors_and_never_bypasses_a_stop(failure):
    runtime=DrivingPlanRuntime();runtime.frame_id='f1'
    decision=dict(action='lane_change_left',source_step_action='CHANGE_LANE',
        frame_id='f1',decision_status='READY',emergency=False,target_speed_kmh=0.)
    risk=dict(frame_id='f1',target_lane_contract=dict(direction='left',can_execute=True),
        behavior_policy=dict(disposition='CLEAR',physical_stop=False,max_speed_kmh=30.),
        lane_change=dict(left=dict(speed_cap_kmh=8.)),task_feasibility=dict(
            CHANGE_LANE_LEFT=dict(can_execute=True),PATH_CLEAR=dict(can_execute=True)))
    if failure=='blocked':decision['decision_status']='BLOCKED'
    if failure=='emergency':decision['emergency']=True
    if failure=='no_path':risk['task_feasibility']['PATH_CLEAR']['can_execute']=False
    if failure=='no_lane':risk['task_feasibility']['CHANGE_LANE_LEFT']['can_execute']=False
    if failure=='wrong_direction':risk['target_lane_contract']['direction']='right'
    if failure=='stale':risk['frame_id']='old'
    if failure=='physical_stop':risk['behavior_policy']['physical_stop']=True
    if failure=='unknown_motion':risk['behavior_policy']['disposition']='OBSERVATION_INSUFFICIENT'
    if failure=='zero_cap':risk['behavior_policy']['max_speed_kmh']=0.
    if failure=='nan_cap':risk['behavior_policy']['max_speed_kmh']=float('nan')
    if failure=='missing_cap':risk['lane_change']['left']={}
    if failure=='stop_action':decision['action']='stop'
    if failure=='wrong_source':decision['source_step_action']='STOP'
    result=runtime.start_authorized_lane_motion(decision,risk)
    assert (result.get('reason')=='sensor_authorized_low_speed_lane_departure')==(failure is None)
    if failure is None:assert result['target_speed_kmh']==5.
    else:assert result==decision
