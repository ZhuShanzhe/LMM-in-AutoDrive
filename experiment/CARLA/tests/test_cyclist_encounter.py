import pytest

from control.cyclist_encounter import CyclistEncounterWindow


DOCUMENT = dict(request_id='cyclist-test', intent=dict(steps=[
    dict(step_id='pass', action='PASS_BY', parameters=dict(condition='CYCLIST'))]))


def observe(memory, time, progress, tracks=None, **overrides):
    arguments = dict(frame_id=int(time*20), sensor_frame=int(time*20), timestamp_s=time,
        observed_at_s=time, progress_m=progress, ego_speed_kmh=18.,
        expected_speed_kmh=18., heading=[1.,0.], contact_count=0)
    arguments.update(overrides)
    return memory.observe(DOCUMENT, tracks or {}, **arguments)['pass']


def track(frame=0, **overrides):
    value = dict(category='cyclist', frame=frame, ahead_m=12., lateral_m=2.,
                 velocity_valid=True, velocity=[2.,0.])
    value.update(overrides)
    return {'cyclist-1':value}


def armed():
    memory = CyclistEncounterWindow()
    result = observe(memory, 0., 100., track())
    assert result['encounter_window']['predicted_encounter_s'] == 4.
    assert result['satisfied'] is False
    assert memory.needs_scan(DOCUMENT) is False
    return memory


def test_first_observation_then_no_continuous_tracking_required():
    memory = armed()
    for frame in range(1, 122):
        result = observe(memory, frame*.05, 100.+frame*.2)
    assert result['satisfied'] is True
    assert result['completed_event_fact']['completion_semantics'] == 'TIME_WINDOW_NO_CONTACT'
    assert result['measured_rear_clearance_required'] is False


def test_cannot_complete_before_the_predicted_window_and_hold_end():
    memory = armed()
    for frame in range(1, 119):
        result = observe(memory, frame*.05, 100.+frame*.2)
    assert result['satisfied'] is False


def test_discrete_route_waypoints_do_not_pause_a_moving_vehicle_timer():
    memory = armed()
    for frame in range(1, 122):
        result = observe(memory, frame*.05, 100.+(frame//10)*2.)
    assert result['satisfied'] is True
    assert result['encounter_window']['finish_after_s'] == pytest.approx(6.)


def test_stopped_time_does_not_count_as_successful_passing():
    memory = armed()
    for frame in range(1, 201):
        result = observe(memory, frame*.05, 100., ego_speed_kmh=0.)
    assert result['satisfied'] is False
    assert result['encounter_window']['finish_after_s'] == pytest.approx(16.)


def test_contact_prevents_completion_even_if_target_is_no_longer_visible():
    memory = armed()
    for frame in range(1, 201):
        result = observe(memory, frame*.05, 100.+frame*.2, contact_count=int(frame>=50))
    assert result['satisfied'] is False
    assert result['reason'] == 'cyclist_encounter_contact_detected'


@pytest.mark.parametrize('overrides', [dict(contact_count=None), dict(observed_at_s=-1.),
                                     dict(progress_m=float('nan')), dict(ego_speed_kmh=0.)])
def test_missing_contact_stale_sensor_or_no_motion_cannot_complete(overrides):
    memory = armed()
    result = observe(memory, 10., 150., **overrides)
    assert result['satisfied'] is False


@pytest.mark.parametrize('overrides', [dict(velocity=[8.,0.]), dict(ahead_m=200.),
    dict(lateral_m=12.), dict(velocity_valid=False), dict(category='vehicle'), dict(frame=-1)])
def test_no_window_without_an_observed_possible_cyclist_encounter(overrides):
    memory = CyclistEncounterWindow()
    result = observe(memory, 0., 100., track(**overrides))
    assert result['encounter_window'] is None
    assert memory.needs_scan(DOCUMENT) is True


def test_different_command_cannot_inherit_a_previous_timer():
    memory = armed()
    changed = dict(DOCUMENT, request_id='next-command')
    assert memory.needs_scan(changed) is True


def test_long_observation_gap_is_excluded_from_countdown():
    memory = armed()
    result = observe(memory, 10., 130.)
    assert result['satisfied'] is False
    assert result['encounter_window']['finish_after_s'] == pytest.approx(16.)


@pytest.mark.parametrize('hold,horizon', [(0,45), (2,0), (float('nan'),45), (2,True)])
def test_invalid_window_parameters_rejected(hold,horizon):
    with pytest.raises(ValueError):
        CyclistEncounterWindow(hold,horizon)
