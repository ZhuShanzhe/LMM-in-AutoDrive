import pytest

from run_emergency_response_6km import (
    EmergencyEventScheduler, build_parser, diagnostic_command_window, diagnostic_start_index, validate_args,
)


@pytest.mark.parametrize('start', [-1, float('nan'), float('inf'), 100])
def test_invalid_start_rejected(start):
    with pytest.raises(ValueError):
        diagnostic_start_index([0, 50, 100], start)


def test_start_preserves_original_route_indices():
    distances = [0, 50, 100, 150]
    assert diagnostic_start_index(distances, 103) == 2
    assert distances == [0, 50, 100, 150]


def test_start_at_zero_preserves_all_commands():
    commands = [dict(id='a', end_progress_m=100)]
    assert diagnostic_command_window(commands, 0) == commands


def test_cold_start_excludes_expired_commands_not_current_or_future():
    commands = [dict(id='a', end_progress_m=100), dict(id='b', end_progress_m=200),
                dict(id='c', end_progress_m=300)]
    assert [item['id'] for item in diagnostic_command_window(commands, 100)] == ['b', 'c']
    assert len(commands) == 3


@pytest.mark.parametrize('value', ['-1', 'nan', 'inf'])
def test_parser_validation_rejects_bad_start(value):
    args = build_parser().parse_args(['--diagnostic-start-progress-m', value])
    with pytest.raises(ValueError):
        validate_args(args)


@pytest.mark.parametrize('start', [1550, 2698, 4538, 5050])
def test_expired_cold_events_skipped_without_callbacks(tmp_path, start):
    from unittest.mock import Mock
    handler = Mock()
    events = [dict(id='old', scenario='cut_in', activate_at_m=1250, resolve_after_m=1550),
              dict(id='future', scenario='future', activate_at_m=5500, resolve_after_m=5600)]
    scheduler = EmergencyEventScheduler(events, output_path=tmp_path/'timeline.jsonl',
        event_handler=handler, initial_route_s_m=start)
    scheduler.update(route_s_m=start, simulation_frame=10, elapsed_s=1)
    assert scheduler.state_snapshot() == {'old': 'SKIPPED', 'future': 'PENDING'}
    assert scheduler.summary()['RESOLVED'] == 0
    assert scheduler.summary()['SKIPPED'] == 1
    handler.on_activate.assert_not_called()
    handler.on_resolve.assert_not_called()


def test_current_event_not_skipped_at_cold_start(tmp_path):
    from unittest.mock import Mock
    handler = Mock()
    event = dict(id='current', scenario='cut_in', activate_at_m=1250, resolve_after_m=1550)
    scheduler = EmergencyEventScheduler([event], output_path=tmp_path/'timeline.jsonl',
        event_handler=handler, initial_route_s_m=1240)
    scheduler.update(route_s_m=1250, simulation_frame=10, elapsed_s=1)
    assert scheduler.state_snapshot()['current'] == 'ACTIVE'
    handler.on_activate.assert_called_once()
    handler.on_resolve.assert_not_called()


@pytest.mark.parametrize('start', [-1, float('nan'), float('inf')])
def test_scheduler_rejects_invalid_initial_progress(tmp_path, start):
    with pytest.raises(ValueError):
        EmergencyEventScheduler([], output_path=tmp_path/'timeline.jsonl', initial_route_s_m=start)
