import numpy as np
import pytest
from lightweight_vla_adapter.src.lane_risk_contract import LaneRiskContract, measured_lane_clearance
from lightweight_vla_adapter.src.task_feasibility import predicate, task_permission
from lightweight_vla_adapter.src.task_feasibility import requested_lane_for_task


def test_low_risk_without_verified_clearance_is_not_path_authority():
    from lightweight_vla_adapter.src.task_feasibility import attach_motion_predicates
    risk=dict(risk_level='low',frame_id='current',risk_evidence=dict(is_fresh=True),
        behavior_policy=dict(disposition='MUST_STOP',physical_stop=False))
    assert not task_permission(attach_motion_predicates(risk),'PATH_CLEAR')


@pytest.mark.parametrize('level',['low','medium','high'])
def test_authorized_lane_survives_a_longitudinal_deceleration_advisory(level):
    from lightweight_vla_adapter.tests.fixtures import integration_documents,proposal
    from lightweight_vla_adapter.src.safety_bridge import gate_vla_proposal
    from scene_understanding.src.control_decision import build_control_decision
    from control.generic_instruction_fsm import GenericInstructionFSM,ParsedInstruction
    intent,world,alignment,risk=integration_documents()
    canonical=build_control_decision(intent,world,alignment,risk)
    risk.update(risk_level=level,recommended_action='decelerate',
        target_lane_contract=dict(can_execute=True),
        task_feasibility=dict(PATH_CLEAR=predicate('PATH_CLEAR',True)))
    fsm=GenericInstructionFSM.__new__(GenericInstructionFSM);fsm.default_speed_kmh=30.
    parsed=ParsedInstruction(parsed_intent='CHANGE_LANE_LEFT',requested_lane_direction='left')
    envelope=fsm.canonical_decision(parsed,frame_id=risk['frame_id'],request_id=canonical['request_id'],
        risk=risk,ego_speed_kmh=8.)
    assert envelope['action']=='lane_change_left'
    decision=gate_vla_proposal(proposal(action='decelerate',target_speed_kmh=8.),envelope,risk)
    assert decision['action']=='lane_change_left' and decision['target_speed_kmh']<=8.


@pytest.mark.parametrize('level',['medium','high'])
def test_bounded_lane_permission_can_restart_from_rest(level):
    from lightweight_vla_adapter.src.risk_behavior import RiskBehaviorPolicy,apply_behavior_decision
    from lightweight_vla_adapter.src.longitudinal_contract import enforce_longitudinal_contract
    from lightweight_vla_adapter.tests.test_risk_behavior import risk,clearance
    policy=RiskBehaviorPolicy()
    for i in range(16):
        now=1.+i*.1;r=risk(str(i),now,level)
        r.update(source='event_memory_vla')
        r['risk_evidence']['sensor_frame']=i;r['physical_forward_radar']['sensor_frame']=i
        e=clearance();e.update(sensor_frame=i,observed_at_s=now,clear_distance_m=40.)
        out=policy.apply(r,e,frame_id=r['frame_id'],timestamp_s=now,ego_speed_kmh=0.,
            parsed_intent='CHANGE_LANE_LEFT',task_conditioned=True)
    out['target_lane_contract']=dict(can_execute=True)
    decision=dict(frame_id=out['frame_id'],action='lane_change_left',decision_status='READY',
        target_lane='left',target_speed_kmh=8.,source_step_action='CHANGE_LANE',blocked_reason_codes=[])
    bounded,_=apply_behavior_decision(decision,out,allow_restart=True)
    issued,diagnostics=enforce_longitudinal_contract(bounded,out,0.)
    assert issued['action']=='lane_change_left' and issued['target_speed_kmh']==8.
    assert issued['allow_positive_acceleration']
    assert diagnostics['cautious_motion_authorized']


@pytest.mark.parametrize('level',['low','medium','high'])
def test_task_predicate_is_independent_of_learned_grade(level):
    gate=LaneRiskContract()
    for i in range(9):
        result=gate.update({},'left',dict(risk_level=level,source='learned_left_camera_visual_risk_head'),
            np.zeros((4,64,64)),timestamp_s=i*.1,sensor_frame=i,frame_age_s=0.,speed_mps=0.,
            physical_evidence=dict(valid=True,observed=True,occupied=False),road_legal=True)
    assert result['target_lane_contract']['can_execute']
    assert result['target_lane_contract']['visual']['risk_level']==level
    assert result['lane_change']['left']['speed_cap_kmh']==8.


@pytest.mark.parametrize('evidence',[dict(valid=True,observed=True,occupied=True),
    dict(valid=True,observed=False,occupied=False),dict(valid=False,observed=False,occupied=None)])
def test_no_authority_from_missing_or_occupied_measurement(evidence):
    gate=LaneRiskContract()
    for i in range(9):
        result=gate.update({},'left',dict(risk_level='low'),np.zeros((4,64,64)),
            timestamp_s=i*.1,sensor_frame=i,frame_age_s=0.,speed_mps=0.,physical_evidence=evidence,road_legal=True)
    assert not result['target_lane_contract']['can_execute']


@pytest.mark.parametrize('legal,status',[(None,'UNKNOWN'),(False,'BLOCKED')])
def test_clear_lane_without_map_permission_cannot_execute(legal,status):
    gate=LaneRiskContract()
    for i in range(9):
        result=gate.update({},'left',dict(risk_level='low'),np.zeros((4,64,64)),
            timestamp_s=i*.1,sensor_frame=i,frame_age_s=0.,speed_mps=0.,
            physical_evidence=dict(valid=True,observed=True,occupied=False),road_legal=legal)
    assert not result['target_lane_contract']['can_execute']
    assert result['task_feasibility']['CHANGE_LANE_LEFT']['status']==status


def test_emergency_and_physical_stop_keep_priority():
    assert not task_permission(dict(target_lane_contract=dict(can_execute=True),
        behavior_policy=dict(physical_stop=True)),'CHANGE_LANE')
    assert not task_permission(dict(task_feasibility=dict(PATH_CLEAR=predicate('PATH_CLEAR',True)),
        recommended_action='emergency_brake'),'PATH_CLEAR')


@pytest.mark.parametrize('direction',['left','right'])
def test_check_predicate_probes_its_own_target_lane(direction):
    document=dict(intent=dict(steps=[dict(action='CHECK',parameters=dict(
        condition=direction.upper()+'_LANE_SAFE'))]))
    assert requested_lane_for_task(document,None,None)==direction
    assert requested_lane_for_task(document,dict(active_step_index=None),None) is None


@pytest.mark.parametrize('fault',[None,'obstacle','stale','crossing'])
def test_model_grade_remains_advisory_but_physical_hazards_remain_authoritative(fault):
    from lightweight_vla_adapter.src.risk_behavior import RiskBehaviorPolicy,cautious_permission
    from lightweight_vla_adapter.tests.test_risk_behavior import risk,clearance
    policy=RiskBehaviorPolicy()
    for i in range(16):
        now=1.+i*.1;r=risk(str(i),now,'high')
        r.update(source='event_memory_vla',prediction_provenance=dict(residual_applied=True))
        r['risk_evidence']['sensor_frame']=i;r['physical_forward_radar']['sensor_frame']=i
        e=clearance();e.update(sensor_frame=i,observed_at_s=now,clear_distance_m=40.)
        if fault=='obstacle':e['nearest_obstacle_m']=2.
        if fault=='stale':e['observed_at_s']=now-.4
        out=policy.apply(r,e,frame_id=r['frame_id'],timestamp_s=now,ego_speed_kmh=0.,
            parsed_intent='CHANGE_LANE_LEFT',task_conditioned=True,crossing_hazard=fault=='crossing')
    permission=cautious_permission(out)
    assert bool(permission)==(fault is None)
    if permission:assert permission['max_speed_kmh']==10. and out['risk_level']=='high'


@pytest.mark.parametrize('yaw',[0.,35.,90.,180.])
def test_lane_road_support_is_rotation_and_height_invariant(yaw):
    theta=np.deg2rad(yaw);rot=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    road=np.array([[-10.,3.5,10.],[0.,3.5,10.],[10.,3.5,10.],[30.,3.5,10.]])
    ground=np.array([[x,3.5+y,10.] for x in np.arange(-9.9,20.,.5) for y in [-.5,0.,.5]])
    road[:,:2]=road[:,:2]@rot.T;ground[:,:2]=ground[:,:2]@rot.T
    result=measured_lane_clearance(ground,road,[0.,0.,10.])
    assert result['valid'] and result['observed'] and not result['occupied']
    obstacle=np.array([5.,3.5,11.]);obstacle[:2]=obstacle[:2]@rot.T
    assert measured_lane_clearance(np.vstack([ground,obstacle]),road,[0.,0.,10.])['occupied']
