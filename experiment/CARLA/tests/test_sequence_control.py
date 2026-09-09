from control.pid_controller import EgoPIDController


def controller():
    result = EgoPIDController(None, None)
    result._get_speed_kmh = lambda: 0.
    result._curvature_speed_cap = lambda intent: 100.
    result._lateral_control = lambda intent, dt: 0.
    result._lateral_intent_for_control = lambda intent: intent
    return result


def request(**updates):
    result = dict(action='accelerate', target_speed_kmh=.72,
                  longitudinal_sequence_schema='longitudinal_sequence/1.0',
                  target_acceleration_mps2=2.)
    result.update(updates)
    return result


def test_sequence_acceleration_reaches_low_level_control():
    value = controller()
    command, _ = value.run_step(request(), .05)
    assert command.throttle > .5
    assert command.brake == 0.


def test_emergency_overrides_positive_sequence_and_resets_history():
    value = controller()
    value.run_step(request(), .05)
    command, _ = value.run_step(request(emergency=True), .05)
    assert command.throttle == 0.
    assert command.brake == 1.
    assert value._sequence_tracker.previous_speed is None


def test_downstream_speed_cap_disables_acceleration_feedforward():
    value = controller()
    value._curvature_speed_cap = lambda intent: 0.
    command, _ = value.run_step(request(), .05)
    assert command.throttle == 0.
    assert value._sequence_tracker.previous_speed is None


def test_legacy_request_does_not_create_sequence_tracker():
    value = controller()
    value.run_step(dict(action='accelerate', target_speed_kmh=.72), .05)
    assert not hasattr(value, '_sequence_tracker')
