from copy import deepcopy
import pytest
from lightweight_vla_adapter.src.safety_adjusted_speed import safety_adjusted_speed_evidence
from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime
from lightweight_vla_adapter.tests.test_driving_plan_runtime import documents


def execution(frame='f'):
    return dict(observation_frame_id=frame,source_step_id='step_1',action='keep_lane',
        effective_target_speed_kmh=23.,speed_target_status='CONSTRAINED',
        speed_constraint_codes=['trajectory_stability_speed_cap'],
        pid=dict(lateral_error_m=.1,heading_error_deg=1.,emergency_latched=False))


@pytest.mark.parametrize('fault',[None,'stale','stop','unknown','off_lane','zero','accelerating','missing_geometry'])
def test_adjusted_speed_requires_active_tracking_or_physical_braking(fault):
    e=execution();a=-1.;actual=27.5
    if fault=='stale':e['observation_frame_id']='old'
    if fault=='stop':e['action']='stop'
    if fault=='unknown':e['speed_constraint_codes']=['learned_visual_risk_high']
    if fault=='off_lane':e['pid']['lateral_error_m']=.5
    if fault=='zero':actual=0.
    if fault=='accelerating':a=.1
    if fault=='missing_geometry':e['pid']={}
    r=safety_adjusted_speed_evidence(e,frame_id='f',timestamp_s=1.,requested_kmh=30.,
        actual_kmh=actual,acceleration_mps2=a)
    assert (r is not None)==(fault is None)


@pytest.mark.parametrize('enabled',[False,True])
def test_runtime_can_finish_safe_low_speed_without_changing_requested_goal(enabled):
    d,w,r=documents();d['intent']['steps'][0]['parameters']={'target_speed_mps':30./3.6}
    runtime=DrivingPlanRuntime(speed_completion_hold_s=2.,speed_completion_tolerance_kmh=2.,
        safety_adjusted_completion=enabled)
    for i in range(24):
        frame=f'f{i}';w['frame_id']=r['frame_id']=frame
        runtime.prepare(d,frame_id=frame,timestamp_s=i*.1,speed_mps=23./3.6,
            execution_state=execution(frame))
        runtime.advance(w,r)
    assert (runtime.state['step_states'][0]['status']=='COMPLETED')==enabled
    assert runtime.document['intent']['steps'][0]['parameters']['target_speed_mps']==30./3.6
    if enabled:assert 'observed_safety_adjusted_speed_completion'in runtime.state['step_states'][0]['reason_codes']


def test_strict_band_and_valid_braking_form_one_continuous_engineering_window():
    d,w,r=documents();d['intent']['steps'][0]['parameters']={'target_speed_mps':30./3.6}
    runtime=DrivingPlanRuntime(speed_completion_hold_s=2.,speed_completion_tolerance_kmh=2.,
        safety_adjusted_completion=True)
    for i in range(23):
        frame=f'f{i}';w['frame_id']=r['frame_id']=frame
        v=28.5 if i<18 else 28.-.15*(i-17)
        e=execution(frame)
        if i<18:e.update(effective_target_speed_kmh=30.,speed_constraint_codes=[],speed_target_status='REACHABLE')
        runtime.prepare(d,frame_id=frame,timestamp_s=i*.1,speed_mps=v/3.6,execution_state=e)
        runtime.advance(w,r)
    assert runtime.state['step_states'][0]['status']=='COMPLETED'
    assert runtime.last_speed_completion_evidence['mode']=='SAFETY_ADJUSTED'


def test_sensor_capped_speed_completion_accepts_measured_safe_undertracking_within_three_kmh():
    d,w,r=documents();d['intent']['steps'][0]['parameters']={'target_speed_mps':22.46/3.6}
    runtime=DrivingPlanRuntime(speed_completion_hold_s=2.,speed_completion_tolerance_kmh=3.,
        safety_adjusted_completion=True)
    for i in range(24):
        frame=f'c{i}';now=i*.1;w['frame_id']=r['frame_id']=frame
        e=execution(frame)
        e.update(action='decelerate',effective_target_speed_kmh=15.5,
            speed_target_status='REACHABLE',speed_constraint_codes=[],
            risk_motion_permission=dict(schema_version='cautious_motion/2.0',
                source='sensor_risk_behavior_policy',observed_at_s=now-.05,
                valid_until_s=now+.15,max_speed_kmh=15.5))
        runtime.prepare(d,frame_id=frame,timestamp_s=now,speed_mps=13.3/3.6,
            execution_state=e)
        runtime.advance(w,r)
    assert runtime.state['step_states'][0]['status']=='COMPLETED'
    assert runtime.last_speed_completion_evidence['constraint_codes']==['sensor_verified_cautious_speed_cap']
