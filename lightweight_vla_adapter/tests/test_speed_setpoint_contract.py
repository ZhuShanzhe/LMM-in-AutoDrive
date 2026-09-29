from lightweight_vla_adapter.src.longitudinal_contract import (
    enforce_active_instruction_contract,
)


def decision(**changes):
    result=dict(decision_status='READY',action='accelerate',target_speed_kmh=42.,
        emergency=False,blocked_reason_codes=[],allow_positive_acceleration=True)
    result.update(changes)
    return result


def test_ready_set_speed_uses_explicit_absolute_setpoint():
    result,diagnostics=enforce_active_instruction_contract(
        decision(),'SET_SPEED',45.,ego_speed_kmh=40.)
    assert result['target_speed_kmh']==45.
    assert diagnostics['speed_contract_mode']=='ABSOLUTE_SETPOINT'


def test_speed_setpoint_does_not_override_safety_deceleration():
    result,_=enforce_active_instruction_contract(
        decision(decision_status='BLOCKED',action='decelerate',target_speed_kmh=12.,
            blocked_reason_codes=['risk_requires_deceleration'],
            allow_positive_acceleration=False),
        'SET_SPEED',45.,ego_speed_kmh=40.)
    assert result['action']=='decelerate'
    assert result['target_speed_kmh']==12.


def test_non_set_speed_intent_remains_an_upper_bound():
    result,diagnostics=enforce_active_instruction_contract(
        decision(),'KEEP_LANE',45.,ego_speed_kmh=40.)
    assert result['target_speed_kmh']==42.
    assert diagnostics['speed_contract_mode']=='UPPER_BOUND'
