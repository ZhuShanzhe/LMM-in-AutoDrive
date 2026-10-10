import pytest

from benchmark.fragment_completion import remember_completed_plans, fragment_tail_stop_reason


@pytest.mark.parametrize('completed,now,progress,expected', [
    (None, 100., 200., None),
    ((10., 100.), 12., 210., None),
    ((10., 100.), 20., 199., None),
    ((10., 100.), 13., 200., 'parsed_plan_completed_with_diagnostic_tail'),
])
def test_fragment_tail_requires_elapsed_time_and_actual_progress(completed, now, progress, expected):
    assert fragment_tail_stop_reason(completed, now, progress) == expected


def test_completed_plan_fact_survives_cruise_handoff():
    selected = ['s2_t05_cmd_03']
    completed = remember_completed_plans(set(), selected, 's2_t05_cmd_03-configured', 'COMPLETED')
    assert remember_completed_plans(completed, selected, 'cruise', None) == set(selected)


def test_unrelated_and_active_plans_do_not_complete_selected_task():
    assert not remember_completed_plans(set(), ['selected'], 'other', 'COMPLETED')
    assert not remember_completed_plans(set(), ['selected'], 'selected-configured', 'ACTIVE')
