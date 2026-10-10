from copy import deepcopy
import pytest
from control.observed_conditions import observed_condition_readiness
from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime


def context(condition='CURRENT_LANE_MOTION'):
    document=dict(request_id='p',intent=dict(steps=[dict(step_id='motion',action='CHECK',parameters=dict(condition=condition))]))
    risk=dict(frame_id='f',risk_level='low',risk_evidence=dict(is_fresh=True,sensor_frame=10),
        physical_forward_radar=dict(sensor_frame=10,obstacle_candidate_count=0,nearest_distance_m=None),
        behavior_policy=dict(disposition='CLEAR',physical_stop=False,max_speed_kmh=15.8,
            allow_positive_acceleration=True,current_speed_required_clearance_m=8.5631687,
            lidar=dict(valid=True,sensor_frame=10,observed_at_s=1.,clear_distance_m=9.091)))
    return document,risk


def test_bounded_motion_is_not_mislabeled_as_semantic_path_clear():
    document,risk=context()
    observed=observed_condition_readiness(document,risk,frame_id='f',timestamp_s=1.05)['motion']
    assert observed['satisfied']
    assert observed['semantic_scope']=='bounded_current_lane_motion_not_semantic_path_clear'
    document['intent']['steps'][0]['parameters']['condition']='PATH_CLEAR'
    assert not observed_condition_readiness(document,risk,frame_id='f',timestamp_s=1.05)['motion']['satisfied']
    runtime=DrivingPlanRuntime();runtime.frame_id='f';runtime.timestamp_s=1.05
    runtime.document=dict(request_id='p',intent=dict(steps=[dict(step_id='merge',action='CHANGE_LANE',parameters=dict(direction='LEFT'))]))
    runtime.state=dict(plan_status='ACTIVE',active_step_index=0,step_states=[dict(status='WAITING')])
    decision=dict(decision_status='BLOCKED',action='stop',target_speed_kmh=0.,emergency=False)
    result=runtime.deferred_lane_navigation(decision,risk,observed)
    assert result['action']=='keep_lane' and result['target_lane'] is None
    assert result['target_speed_kmh']==10.
    assert result['deferred_lane_change']


@pytest.mark.parametrize('fault',['expired','missing_support','wrong_sensor','occupied','short','absolute_stop',
    'physical_stop','missing_required','negative_required','nan_required'])
def test_bounded_motion_never_skips_physical_or_freshness_constraints(fault):
    document,risk=context();policy=risk['behavior_policy'];lidar=policy['lidar']
    if fault=='expired':lidar['observed_at_s']=0.
    if fault=='missing_support':lidar['valid']=False
    if fault=='wrong_sensor':risk['physical_forward_radar']['sensor_frame']=9
    if fault=='occupied':risk['physical_forward_radar'].update(obstacle_candidate_count=1,nearest_distance_m=7.)
    if fault=='short':lidar['clear_distance_m']=8.
    if fault=='absolute_stop':policy['absolute_stop_active']=True
    if fault=='physical_stop':policy['physical_stop']=True
    if fault=='missing_required':policy.pop('current_speed_required_clearance_m')
    if fault=='negative_required':policy['current_speed_required_clearance_m']=-1.
    if fault=='nan_required':policy['current_speed_required_clearance_m']=float('nan')
    assert not observed_condition_readiness(document,risk,frame_id='f',timestamp_s=1.05)['motion']['satisfied']


@pytest.mark.parametrize('fault',[None,'required','scope','sensor','cap'])
def test_actual_bounded_motion_output_crosses_every_execution_gate(fault):
    from lightweight_vla_adapter.tests.fixtures import integration_documents,proposal
    from lightweight_vla_adapter.src.safety_bridge import gate_vla_proposal
    from lightweight_vla_adapter.src.lane_risk_contract import enforce_requested_lane
    from scene_understanding.src.control_decision import build_control_decision
    intent,world,alignment,risk=integration_documents()
    canonical=build_control_decision(intent,world,alignment,risk)
    canonical.update(decision_status='BLOCKED',action='stop',target_lane=None,target_speed_kmh=0.,blocked_reason_codes=['passing_target_not_acquired'])
    document,motion=context();risk.update({k:v for k,v in motion.items() if k!='frame_id'})
    risk['risk_evidence']['decision_timestamp_s']=1.05
    risk['task_feasibility']=dict(PATH_CLEAR=dict(can_execute=True))
    risk['lane_change']['left'].update(is_safe=False,reason_codes=['passing_target_not_acquired'])
    document['request_id']=intent['request_id']
    observed=observed_condition_readiness(document,risk,frame_id=risk['frame_id'],timestamp_s=1.05)['motion']
    runtime=DrivingPlanRuntime();runtime.document=intent;runtime.frame_id=world['frame_id'];runtime.timestamp_s=1.05
    runtime.state=dict(plan_status='ACTIVE',active_step_index=0,step_states=[dict(status='WAITING')])
    canonical=runtime.deferred_lane_navigation(canonical,risk,observed)
    assert canonical.get('deferred_lane_change')
    if fault=='required':canonical['deferred_current_lane_evidence']['required_clearance_m']=.5
    if fault=='scope':canonical['deferred_current_lane_evidence']['semantic_scope']='ignore_obstacles'
    if fault=='sensor':canonical['deferred_current_lane_evidence']['sensor_frame']=9
    if fault=='cap':risk['behavior_policy']['max_speed_kmh']=5.
    if fault:
        with pytest.raises(ValueError):gate_vla_proposal(proposal(action='accelerate'),canonical,risk)
        return
    gated=gate_vla_proposal(proposal(action='accelerate'),canonical,risk)
    bounded,reason=enforce_requested_lane(gated,risk,'left')
    final,_=runtime.enforce_execution(bounded,canonical,risk=risk)
    assert final['action']=='keep_lane' and final['target_lane'] is None
    assert final['target_speed_kmh']==10. and reason=='deferred_current_lane_only'
