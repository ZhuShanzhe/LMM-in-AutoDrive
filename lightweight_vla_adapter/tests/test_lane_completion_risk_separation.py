from copy import deepcopy

import pytest
from lightweight_vla_adapter.src.driving_plan_runtime import DrivingPlanRuntime
from lightweight_vla_adapter.src.maneuver_evidence import issued_lane_arrival
from lightweight_vla_adapter.tests.fixtures import integration_documents


def documents(next_action='STOP'):
    document, world, _, risk = integration_documents(parser_action='CHANGE_LANE')
    first = document['intent']['steps'][0]
    first.update(parameters={'direction': 'LEFT'}, completion={'type': 'LANE_CHANGE_COMPLETED'}, on_blocked='WAIT')
    second = deepcopy(first)
    second.update(step_id='step_2', action=next_action,
                  parameters={'direction': 'RIGHT'} if next_action == 'CHANGE_LANE' else {},
                  depends_on=[first['step_id']], trigger={'type': 'AFTER_STEP', 'step_id': first['step_id']})
    document['intent']['steps'].append(second)
    return document, world, risk


def execution(document, frame):
    return dict(observation_frame_id=frame, source_step_id='step_1',
                pid=dict(lane_change_completed=True, current_lane_ref=dict(road_id=35, section_id=0, lane_id=3),
                         target_lane_id=3, lane_change_command_id=document['request_id']+':step_1',
                         lateral_error_m=.04, heading_error_deg=-3.44, emergency_latched=False))


@pytest.mark.parametrize('next_action', ['STOP', 'CHANGE_LANE'])
@pytest.mark.parametrize('risk_level', ['medium', 'high'])
def test_held_issued_arrival_survives_risk_pause_but_next_action_is_still_gated(next_action, risk_level):
    document, world, risk = documents(next_action)
    runtime = DrivingPlanRuntime()
    for index in range(9):
        frame = f'f{index}'
        world['frame_id'] = risk['frame_id'] = frame
        risk['risk_level'] = 'low' if index == 0 else risk_level
        risk['recommended_action'] = 'maintain_speed' if index == 0 else 'emergency_brake' if risk_level == 'high' else 'monitor'
        risk['lane_change'] = {side: dict(is_safe=index == 0, reason_codes=[] if index == 0 else ['visual_risk_blocks_lane_change'])
                               for side in ('left', 'right')}
        runtime.prepare(document, frame_id=frame, timestamp_s=index*.1, speed_mps=0.,
                        execution_state=execution(document, frame))
        decision = runtime.advance(world, risk)
    assert runtime.state['step_states'][0]['status'] == 'COMPLETED'
    assert runtime.state['active_step_id'] == 'step_2'
    issued, _ = runtime.enforce_execution(decision, decision, risk=risk)
    assert issued['action'] in {'stop', 'emergency_brake'}
    assert issued['target_speed_kmh'] == 0.
    assert len(runtime.completed_maneuver_facts) == 1


@pytest.mark.parametrize('fault', ['stale', 'wrong_step', 'wrong_command', 'unissued', 'wrong_lane',
                                  'off_center', 'wrong_heading', 'emergency', 'missing_road'])
def test_waiting_lane_step_cannot_complete_from_unqualified_execution(fault):
    document, world, risk = documents()
    runtime = DrivingPlanRuntime()
    for index in range(10):
        frame = f'f{index}'
        world['frame_id'] = risk['frame_id'] = frame
        risk['risk_level'] = 'medium'
        risk['recommended_action'] = 'monitor'
        risk['lane_change'] = {side: dict(is_safe=False, reason_codes=['visual_risk_blocks_lane_change'])
                               for side in ('left', 'right')}
        observed = execution(document, frame)
        pid = observed['pid']
        if fault == 'stale': observed['observation_frame_id'] = 'old'
        if fault == 'wrong_step': observed['source_step_id'] = 'step_2'
        if fault == 'wrong_command': pid['lane_change_command_id'] = 'different:step_1'
        if fault == 'unissued': pid['lane_change_completed'] = False
        if fault == 'wrong_lane': pid['target_lane_id'] = 2
        if fault == 'off_center': pid['lateral_error_m'] = .36
        if fault == 'wrong_heading': pid['heading_error_deg'] = 5.1
        if fault == 'emergency': pid['emergency_latched'] = True
        if fault == 'missing_road': pid['current_lane_ref'].pop('road_id')
        assert not issued_lane_arrival(document['request_id'], 'step_1', observed, frame)
        runtime.prepare(document, frame_id=frame, timestamp_s=index*.1, speed_mps=0., execution_state=observed)
        runtime.advance(world, risk)
    assert runtime.state['step_states'][0]['status'] != 'COMPLETED'
    assert not runtime.completed_maneuver_facts
