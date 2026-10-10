import numpy as np
import pytest
from control.passing_conditions import (PassingMemory, require_acquired_vehicle_before_lane_change,
    apply_passing_target_acquisition, pending_vehicle_passing_lane, passing_target_binding_allowed)


def doc():
    return dict(request_id='request',intent=dict(steps=[
        dict(step_id='merge',action='CHANGE_LANE',parameters=dict(direction='LEFT')),
        dict(step_id='pass',action='OVERTAKE',parameters=dict(condition='SLOW_VEHICLE'),depends_on=['merge'])]))


def observe(memory,frame,*,allowed=True,lateral=0,ego=0,speed=3.):
    now=frame*.05
    memory.update([dict(view='front',local_id=1,category='vehicle',
        position=np.array([20.+speed*now,lateral,1.]),surface=np.array([[20.+speed*now,lateral,1.]]))],
        frame=frame,timestamp_s=now,ego_xy=[ego,0],heading=[1,0])
    return memory.proofs(doc(),frame_id=str(frame),frame=frame,timestamp_s=now,
        observed_at_s=now,ego_xy=[ego,0],heading=[1,0],ego_rear_m=2.4,
        corridor_clear=True,passing_origin_side='RIGHT' if lateral else None,
        target_binding_allowed=allowed)['pass']


def test_moving_slow_vehicle_can_be_acquired_but_is_not_already_passed():
    memory=PassingMemory()
    for f in range(1,35):proof=observe(memory,f)
    assert proof['target_id']
    assert not proof['satisfied']
    assert proof['target_observation']['binding_origin']=='measured_before_maneuver'


def test_stationary_parked_vehicle_is_not_a_moving_slow_vehicle():
    memory=PassingMemory()
    for f in range(1,35):proof=observe(memory,f,speed=0.)
    assert proof['target_id'] is None
    assert not proof['satisfied']


def test_new_adjacent_vehicle_cannot_be_bound_after_maneuver():
    memory=PassingMemory()
    for f in range(1,12):proof=observe(memory,f,allowed=False,lateral=3.5)
    assert proof['target_id'] is None
    assert proof['reason']=='target_not_acquired_before_maneuver'
    assert not proof['satisfied']


def test_existing_pre_maneuver_identity_is_retained_after_merge():
    memory=PassingMemory()
    for f in range(1,35):proof=observe(memory,f)
    identity=proof['target_id']
    assert identity
    proof=observe(memory,35,allowed=False)
    assert proof['target_id']==identity


@pytest.mark.parametrize('proof',[{},dict(valid=True,target_id='target'),
    dict(valid=False,target_id='target',target_observation=dict(measured_in_current_frame=True)),
    dict(valid=True,target_id='target',target_observation=dict(measured_in_current_frame=False))])
def test_lane_merge_waits_for_valid_current_target(proof):
    result=require_acquired_vehicle_before_lane_change(dict(can_execute=True),
        doc()['intent']['steps'],'merge',{'pass':proof},maneuver_started=False)
    assert not result['can_execute']
    assert 'passing_target_not_acquired' in result['reason_codes']


@pytest.mark.parametrize('original',[True,False])
def test_target_acquisition_never_overrides_existing_lane_safety(original):
    proof=dict(valid=True,target_id='target',target_observation=dict(measured_in_current_frame=True))
    result=require_acquired_vehicle_before_lane_change(dict(can_execute=original),
        doc()['intent']['steps'],'merge',{'pass':proof},maneuver_started=False)
    assert result['can_execute']==original


def test_temporary_target_occlusion_does_not_cancel_already_started_merge():
    result=require_acquired_vehicle_before_lane_change(dict(can_execute=True),
        doc()['intent']['steps'],'merge',{},maneuver_started=True)
    assert result['can_execute']


def test_all_lane_authority_consumers_agree_when_referent_missing():
    from copy import deepcopy
    from lightweight_vla_adapter.src.task_feasibility import task_permission
    risk=dict(frame_id='f',target_lane_contract=dict(direction='left',can_execute=True),
        lane_change=dict(left=dict(is_safe=True,reason_codes=[]),right=dict(is_safe=True,reason_codes=[])),
        task_feasibility=dict(CHANGE_LANE_LEFT=dict(can_execute=True)),behavior_policy={})
    before=deepcopy(risk)
    result=apply_passing_target_acquisition(risk,doc()['intent']['steps'],'merge',{},maneuver_started=False)
    assert risk==before
    assert not result['target_lane_contract']['can_execute']
    assert not result['lane_change']['left']['is_safe']
    assert not task_permission(result,'CHANGE_LANE',frame_id='f')
    assert not task_permission(result,'CHANGE_LANE_LEFT',frame_id='f')
    assert result['lane_change']['right']['is_safe']


def test_real_configured_plan_cannot_issue_merge_without_target():
    import json
    from pathlib import Path
    from scene2_runtime_interface import build_scheduled_driving_intent
    from scene_understanding.src.control_decision import build_control_decision
    root=Path(__file__).resolve().parents[1]
    command=next(c for c in json.loads((root/'configs/scene_2_town05_runtime.json').read_text())['commands']
        if c['id']=='s2_t05_cmd_03')
    document=build_scheduled_driving_intent(command,1,950.,1.)
    steps=document['intent']['steps'];merge=next(s for s in steps if s['action']=='CHANGE_LANE')
    risk=dict(frame_id='f',risk_level='low',recommended_action='maintain_speed',reason_codes=[],
        target_lane_contract=dict(direction='left',can_execute=True),
        lane_change=dict(left=dict(is_safe=True,reason_codes=[]),right=dict(is_safe=True,reason_codes=[])))
    risk=apply_passing_target_acquisition(risk,steps,merge['step_id'],{},maneuver_started=False)
    single={**document,'intent':{**document['intent'],'steps':[merge]}}
    alignment=dict(request_id=document['request_id'],world_state_frame_id='f',parse_status='VALID',
        step_alignments=[dict(step_id=merge['step_id'],alignment_required=False,alignment_success=True)])
    decision=build_control_decision(single,dict(frame_id='f',ego=dict(speed_mps=2.)),alignment,risk)
    assert decision['decision_status']=='BLOCKED'
    assert decision['action'] not in {'lane_change_left','lane_change_right'}
    assert 'passing_target_not_acquired' in decision['blocked_reason_codes']


@pytest.mark.parametrize('action,allowed',[('YIELD',False),('WAIT',False),('ADJUST_SPEED',False),
    ('CHANGE_LANE',True),('OVERTAKE',True)])
def test_warming_tracks_does_not_bind_during_prior_unrelated_phases(action,allowed):
    steps=[dict(step_id='current',action=action)]
    contract=dict(road_legal=True,departure_window=dict(allowed=True))
    assert passing_target_binding_allowed(steps,'current',maneuver_started=False,origin_side=None,lane_contract=contract)==allowed
    assert not passing_target_binding_allowed(steps,'current',maneuver_started=True,origin_side=None)


def test_next_lane_is_guarded_before_speed_completion_advances_into_it():
    steps=[dict(step_id='slow',action='ADJUST_SPEED'),*doc()['intent']['steps']]
    assert pending_vehicle_passing_lane(steps,'slow')=='merge'
    assert pending_vehicle_passing_lane(steps,'pass') is None
    risk=dict(lane_change=dict(left=dict(is_safe=True,reason_codes=[])),
        target_lane_contract=dict(direction='left',can_execute=True))
    gate=pending_vehicle_passing_lane(steps,'slow')
    result=apply_passing_target_acquisition(risk,steps,gate,{},maneuver_started=False)
    assert not result['lane_change']['left']['is_safe']


def test_indirect_dependent_overtake_is_also_guarded():
    steps=doc()['intent']['steps']
    steps.insert(1,dict(step_id='check',action='CHECK',depends_on=['merge']))
    steps[-1]['depends_on']=['check']
    assert pending_vehicle_passing_lane(steps,'merge')=='merge'
    result=require_acquired_vehicle_before_lane_change(dict(can_execute=True),steps,'merge',{},maneuver_started=False)
    assert not result['can_execute']


@pytest.mark.parametrize('contract',[None,{},dict(road_legal=False),
    dict(road_legal=True,departure_window=dict(allowed=False))])
def test_target_binding_waits_until_legal_nonjunction_departure_corridor(contract):
    assert not passing_target_binding_allowed(doc()['intent']['steps'],'merge',
        maneuver_started=False,origin_side=None,lane_contract=contract)


def test_crossing_traffic_is_not_a_slow_leading_vehicle():
    memory=PassingMemory()
    for f in range(1,12):
        now=f*.05
        memory.update([dict(view='front',local_id=1,category='vehicle',position=np.array([20.,6.*now,1.]),
            surface=np.array([[20.,6.*now,1.]]))],frame=f,timestamp_s=now,ego_xy=[0,0],heading=[1,0])
        proof=memory.proofs(doc(),frame_id=str(f),frame=f,timestamp_s=now,observed_at_s=now,
            ego_xy=[0,0],heading=[1,0],ego_rear_m=2.4,corridor_clear=True)['pass']
    assert proof['target_id'] is None
