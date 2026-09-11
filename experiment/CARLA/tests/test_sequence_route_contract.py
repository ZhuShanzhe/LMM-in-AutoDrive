from types import SimpleNamespace
from control.generic_route_pid import GenericRoutePID


def route(now=10.):
    value=GenericRoutePID.__new__(GenericRoutePID)
    value._default_speed_kmh=30.
    value.world=SimpleNamespace(get_snapshot=lambda:SimpleNamespace(timestamp=SimpleNamespace(elapsed_seconds=now)))
    value._lane_change_legal=lambda:True
    value._junction_or_road_transition_ahead=lambda:False
    value._lane_transition_ahead=lambda:False
    value._route_target=lambda:None
    value.route_manager=None
    value.fixed_delta_seconds=.05
    value._pid=SimpleNamespace(run_step=lambda intent,dt:(intent,None))
    return value


def decision(**updates):
    d=dict(action='accelerate',target_speed_kmh=20.,emergency=False,
           longitudinal_sequence_schema='longitudinal_sequence/1.0',target_acceleration_mps2=1.,sequence_valid_until_s=10.3)
    d.update(updates);return d


def test_generic_route_transmits_acceleration_and_expiry():
    r=route();r.set_high_level_decision(decision())
    assert r.run_step()['target_acceleration_mps2']==1.


def test_expired_sequence_stops_instead_of_reusing_acceleration():
    r=route(10.4);r.set_high_level_decision(decision())
    out=r.run_step()
    assert out['action']=='stop' and out['target_speed_kmh']==0.
    assert 'target_acceleration_mps2' not in out


def test_new_safety_decision_discards_old_sequence():
    r=route();r.set_high_level_decision(decision())
    r.set_high_level_decision(dict(action='emergency_brake',target_speed_kmh=0.,emergency=True))
    out=r.run_step()
    assert out['emergency'] and 'target_acceleration_mps2' not in out


def test_route_cap_cannot_keep_positive_feedforward():
    r=route();r.set_high_level_decision(decision(target_speed_kmh=40.))
    assert 'target_acceleration_mps2' not in r.run_step()


def test_speed_cap_cannot_discard_command_expiry():
    r=route(10.4);r.set_high_level_decision(decision(target_speed_kmh=40.))
    out=r.run_step()
    assert out['action']=='stop' and out['target_speed_kmh']==0.
    assert 'target_acceleration_mps2' not in out


def test_missing_or_nonfinite_sequence_metadata_fails_closed():
    for patch in ({'sequence_valid_until_s':None},{'target_acceleration_mps2':float('nan')},
                  {'target_acceleration_mps2':4.}):
        r=route();r.set_high_level_decision(decision(**patch))
        assert r.run_step()['target_speed_kmh']==0.


def test_rewritten_nonsequence_decision_keeps_common_deadline():
    r=route(10.4)
    r.set_high_level_decision(dict(action='keep_lane',target_speed_kmh=12.,control_valid_until_s=10.3))
    assert r.run_step()['target_speed_kmh']==0.


def test_earliest_of_control_and_sequence_deadlines_wins():
    r=route(10.2)
    r.set_high_level_decision(decision(control_valid_until_s=10.1))
    assert r.run_step()['target_speed_kmh']==0.


def test_risk_speed_cap_semantics_reach_pid():
    r=route();r.set_high_level_decision(dict(action='decelerate',target_speed_kmh=2.,
        allow_positive_acceleration=False,longitudinal_control_mode='RISK_SPEED_CAP'))
    intent=r.run_step()
    assert not intent['allow_positive_acceleration']
    assert intent['longitudinal_control_mode']=='RISK_SPEED_CAP'
