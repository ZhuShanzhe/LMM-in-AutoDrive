"""Cyclist configuration uses existing safety counters, not a passing proof."""
from benchmark.catalog import load_catalog
from benchmark.task_oracle import TaskOracle, load_profile


def cyclist_profile():
    catalog = load_catalog('scene_2')
    return load_profile(catalog, catalog.select('s2_t05_cmd_07')[0])


def observation(profile, frame, progress, collisions=0):
    return dict(
        schema_version='task_truth/1.0', source='simulator_truth',
        source_sha256=profile['source_sha256'], task_id=profile['task_id'],
        frame=frame, sim_time_s=frame*.05, scenario_valid=True,
        ego=dict(route_s_m=progress, speed_kmh=12, route_error_m=0),
        safety=dict(collisions=collisions, violations=0), actors={}, fixture={})


def test_no_cyclist_identity_or_rear_clearance_is_required():
    profile = cyclist_profile()
    oracle = TaskOracle(profile)
    for frame in range(496):
        result = oracle.update(observation(profile, frame, 1760+frame))
    assert result['status'] == 'SUCCESS'
    assert result['coverage']['status'] == 'PARTIAL'


def test_stationary_vehicle_does_not_pass_by_avoiding_collision():
    profile = cyclist_profile()
    oracle = TaskOracle(profile)
    for frame in range(200):
        result = oracle.update(observation(profile, frame, 1760))
    assert result['status'] == 'RUNNING'


def test_collision_during_traversal_still_fails():
    profile = cyclist_profile()
    oracle = TaskOracle(profile)
    oracle.update(observation(profile, 0, 1760))
    result = oracle.update(observation(profile, 1, 1761, collisions=1))
    assert result['status'] == 'FAILURE'
    assert result['reason'] == 'collision'


def test_updated_commands_do_not_wait_for_passed_confirmation():
    catalog = load_catalog('scene_2')
    for task_id in ('s2_t05_cmd_07', 's2_t05_cmd_08'):
        command = catalog.select(task_id)[0].source_command
        assert 'CONFIRM:CYCLIST_PASSED' not in command['steps']
        assert not any(step.startswith('CHANGE_LANE') for step in command['steps'])
    assert catalog.select('s2_t05_cmd_07')[0].source_command['steps'] == ['PASS_BY:CYCLIST','PROCEED']
    previous = load_profile(catalog, catalog.select('s2_t05_cmd_03')[0])
    assert any(step['kind'] == 'overtake' for step in previous['steps'])


def test_all_scene2_profiles_match_updated_source():
    catalog = load_catalog('scene_2')
    assert len(catalog.tasks) == 15
    assert all(load_profile(catalog, task) is not None for task in catalog.tasks)
