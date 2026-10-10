"""Focused terminal-output and radar crawl contract regressions."""
import io
import json
from types import SimpleNamespace

import pytest

from control.command_dispatch import CompletionCommandQueue, CommandDispatchHalted
from control.generic_temporal_risk_supervisor import GenericTemporalRiskSupervisor
from control.pid_controller import EgoPIDController
from universal_vla_controller import UniversalVLAController
from lightweight_vla_adapter.src.longitudinal_contract import (
    enforce_active_instruction_contract, enforce_longitudinal_contract,
)


def make_risk(frame=10, distance=30., now=1., observed=1.):
    return dict(risk_level='high', recommended_action='emergency_brake',
        probabilities={'high': .99}, risk_score=.99,
        physical_forward_radar=dict(schema_version='physical_front_radar/1.0',
            sensor_frame=frame, emergency_distance_m=4., caution_distance_m=15.,
            nearest_distance_m=distance, obstacle_candidate_count=0 if distance is None else 1,
            nearest_relative_velocity_mps=0.),
        risk_evidence=dict(is_fresh=True, sensor_frame=frame, observed_at_s=observed,
            decision_timestamp_s=now, valid_until_s=observed + .3))


def supervise(risk):
    supervisor = GenericTemporalRiskSupervisor()
    decision = dict(action='emergency_brake', target_speed_kmh=0., emergency=True,
                    decision_status='READY', allow_positive_acceleration=False)
    for frame in range(1, 4):
        risk['physical_forward_radar']['sensor_frame'] = frame
        risk['risk_evidence']['sensor_frame'] = frame
        issued, _ = supervisor.apply(decision, dict(action='keep_lane', target_speed_kmh=40.),
            risk, parsed_intent='KEEP_LANE', requested_lane_direction=None,
            target_lane_risk=None, stationary_elapsed_s=3., resume_active=False,
            resume_speed_kmh=40., frame=frame)
    return issued


@pytest.mark.parametrize('speed', [0., .01, 5.])
@pytest.mark.parametrize('distance', [30., None])
def test_authorized_crawl_survives_all_longitudinal_contract_passes(speed, distance):
    risk = make_risk(distance=distance)
    issued = supervise(risk)
    assert issued['reason'] == 'stationary_high_risk_radar_guarded_crawl'
    for _ in range(3):
        issued, diag = enforce_longitudinal_contract(issued, risk, speed)
        assert diag['guarded_crawl_authorized']
        assert issued['target_speed_kmh'] == 6.
        assert issued['allow_positive_acceleration']
        assert issued['longitudinal_control_mode'] == 'GUARDED_CRAWL'


@pytest.mark.parametrize('fault', ['stale', 'future', 'mismatched_frame', 'emergency_distance',
    'closing', 'blocked', 'red_light', 'explicit_stop', 'emergency', 'forged', 'no_permission'])
def test_crawl_cannot_bypass_safety_or_instruction(fault):
    risk = make_risk()
    issued = supervise(risk)
    if fault == 'stale': risk['risk_evidence']['decision_timestamp_s'] = 2.
    if fault == 'future': risk['risk_evidence']['decision_timestamp_s'] = .5
    if fault == 'mismatched_frame': risk['physical_forward_radar']['sensor_frame'] = 99
    if fault == 'emergency_distance': risk['physical_forward_radar']['nearest_distance_m'] = 2.
    if fault == 'closing': risk['physical_forward_radar']['nearest_relative_velocity_mps'] = -10.
    if fault == 'blocked': issued['decision_status'] = 'BLOCKED'
    if fault == 'red_light': issued.update(action='stop', reason='red_light', target_speed_kmh=0.)
    if fault == 'explicit_stop': issued, _ = enforce_active_instruction_contract(issued, 'STOP', None, 0.)
    if fault == 'emergency': issued['emergency'] = True
    if fault == 'forged': issued['risk_motion_permission']['max_speed_kmh'] = 40.
    if fault == 'no_permission': issued.pop('risk_motion_permission')
    result, diag = enforce_longitudinal_contract(issued, risk, 0.)
    assert not diag['guarded_crawl_authorized']
    assert result['target_speed_kmh'] == 0.
    assert not result['allow_positive_acceleration']
    assert 'risk_motion_permission' not in result


def test_command_timeout_is_one_terminal_event_not_a_runtime_fallback():
    queue = CompletionCommandQueue([dict(text='Drive.', driving_intent=dict(
        request_id='request-1', intent=dict(steps=[dict(action='KEEP_LANE')])))] , timeout_s=1.)
    queue.select(0., 0.)
    controller = UniversalVLAController.__new__(UniversalVLAController)
    controller.command_queue = queue
    controller.fsm = SimpleNamespace(active_command=lambda commands, progress: {})
    controller.commands = []
    controller._progress_m = lambda: 10.
    controller.driving_plan = None
    controller.world = SimpleNamespace(get_snapshot=lambda: SimpleNamespace(
        timestamp=SimpleNamespace(elapsed_seconds=2.)))
    decisions = []
    controller.route_controller = SimpleNamespace(set_high_level_decision=decisions.append)
    controller._stream = io.StringIO()
    for frame in range(10, 20):
        controller._decide(frame)
    assert queue.failed == 'TIMEOUT' and queue.index == 0
    assert len(queue.events) == 2 and len(decisions) == 10
    records = [json.loads(line) for line in controller._stream.getvalue().splitlines()]
    assert len(records) == 1 and records[0]['status'] == 'command_dispatch_halted'
    assert controller.overlay()['emergency'] is False
    assert all(d['decision_status'] == 'BLOCKED' and d['target_speed_kmh'] == 0. for d in decisions)
    with pytest.raises(CommandDispatchHalted):
        queue.select(10., 3.)
    assert len(queue.events) == 2


@pytest.mark.parametrize('speed', [5.8, 6., 7., 10.])
def test_guarded_crawl_ceiling_dominates_smoothed_throttle(speed):
    controller = EgoPIDController.__new__(EgoPIDController)
    controller._speed_integral = 4.
    control = SimpleNamespace(throttle=.5, brake=0.)
    controller._enforce_guarded_crawl_ceiling(control, 6., speed)
    assert control.throttle == 0. and controller._speed_integral == 0.
    assert control.brake >= .1 if speed > 6. else control.brake == 0.


def test_guarded_crawl_launch_does_not_use_nominal_full_throttle():
    controller = EgoPIDController.__new__(EgoPIDController)
    controller._speed_integral = 0.
    control = SimpleNamespace(throttle=.85, brake=0.)
    controller._enforce_guarded_crawl_ceiling(control, 6., 0.)
    assert control.throttle == .45 and control.brake == 0.
