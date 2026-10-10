import json
from benchmark.catalog import CONFIG_ROOT, load_catalog
from benchmark.task_oracle import load_profile


def test_ambient_following_gap_does_not_remove_required_scene_actors():
    config = json.loads((CONFIG_ROOT/'scene_2_town05_runtime.json').read_text())
    assert config['traffic']['global_distance_to_leading_vehicle_m'] == 15.
    assert config['traffic']['following_sources_enabled'] is False
    assert config['traffic']['vehicles'] == 70
    assert config['traffic']['buses'] == 0
    assert any(event['kind']=='bus_stop' for event in config['special_events'])
    assert {event['id'] for event in config['special_events']} == {
        'slow_vehicle', 'compound_slow_vehicle', 'crosswalk_pedestrian',
        'bus_stop_passengers', 'slow_cyclist'}
    assert config['acceptance']['collision_count_equals'] == 0


def test_scene2_speed_text_steps_and_independent_profiles_agree():
    catalog = load_catalog('scene_2')
    for identity in (1, 2, 11, 13, 14, 15):
        task = catalog.select(f's2_t05_cmd_{identity:02}')[0]
        command = task.source_command
        assert '30公里每小时' in command['text']
        assert command['text'] == command['spoken_text']
        assert 'SET_SPEED:8.33mps' in command['steps']
        speeds = [step['target_kmh'] for step in load_profile(catalog, task)['steps']
                  if step['kind'] == 'speed']
        assert speeds == [30]

    cruise = catalog.select('s2_t05_cmd_10')[0]
    command = cruise.source_command
    assert command['text'] == command['spoken_text']
    assert '遵守道路限速' in command['text']
    assert '根据路况调整车速' in command['text']
    assert command['steps'] == ['CHECK:PATH_CLEAR', 'PROCEED']
    criteria = load_profile(catalog, cruise)['steps']
    assert criteria == [dict(kind='speed_ceiling', max_speed_kmh=30, hold_s=2),
                        dict(kind='progress', distance_m=30)]

    straight = catalog.select('s2_t05_cmd_09')[0]
    source = straight.source_command
    assert source['text'] == source['spoken_text']
    assert source['steps'] == ['CHECK:PATH_CLEAR', 'KEEP_LANE', 'PROCEED']
    assert [step['kind'] for step in load_profile(catalog, straight)['steps']] == [
        'speed_ceiling', 'lane_hold', 'progress']
    for identity in ('s2_t05_cmd_03',):
        command = catalog.select(identity)[0].source_command
        assert any(step.startswith('CHANGE_LANE:') for step in command['steps'])
    cyclist = catalog.select('s2_t05_cmd_07')[0]
    assert cyclist.source_command['steps'] == ['PASS_BY:CYCLIST','PROCEED']
    assert [step['kind'] for step in load_profile(catalog, cyclist)['steps']] == ['progress']


def test_later_corridor_speed_tasks_do_not_add_unrequired_turns():
    catalog = load_catalog('scene_2')
    for identity in (11, 14):
        task = catalog.select(f's2_t05_cmd_{identity:02}')[0]
        steps = task.source_command['steps']
        assert steps == ['CHECK:PATH_CLEAR','SET_SPEED:8.33mps','KEEP_LANE']
        expected=['speed','lane_hold','progress']
        assert [s['kind'] for s in load_profile(catalog, task)['steps']] == expected


def test_no_duplicate_return_or_unsupported_bus_pull_over():
    catalog = load_catalog('scene_2')
    bus = catalog.select('s2_t05_cmd_05')[0].source_command
    after_cyclist = catalog.select('s2_t05_cmd_08')[0].source_command
    assert bus['steps'][0] == 'KEEP_LANE:RIGHT'
    assert 'PULL_OVER:RIGHT' not in bus['steps']
    assert 'CHANGE_LANE:RIGHT_WHEN_SAFE' not in after_cyclist['steps']
    assert after_cyclist['steps'] == ['KEEP_LANE','PROCEED']
    assert 'CONFIRM:CYCLIST_PASSED' not in after_cyclist['steps']


def test_post_junction_right_lane_goal_does_not_request_nonexistent_adjacent_lane():
    catalog = load_catalog('scene_2')
    task = catalog.select('s2_t05_cmd_04')[0]
    command = task.source_command
    assert command['text'] == command['spoken_text']
    assert command['steps'] == [
        'WAIT:PEDESTRIAN_CLEAR', 'KEEP_LANE:RIGHT', 'SET_SPEED:8.33mps']
    criteria = load_profile(catalog, task)['steps']
    assert [step['kind'] for step in criteria] == ['wait_clear', 'lane_position', 'speed']
    assert criteria[1]['lane_side'] == 'RIGHT'
    assert criteria[1]['lane_ordinal'] == 1
    overtaking = catalog.select('s2_t05_cmd_03')[0].source_command
    assert 'CHANGE_LANE:LEFT_WHEN_SAFE' in overtaking['steps']
    assert 'OVERTAKE:SLOW_VEHICLE' in overtaking['steps']


def test_scene1_turn_activation_precedes_its_route_directive():
    catalog = load_catalog('scene_1')
    task = catalog.select('c07_turn_right')[0]
    assert task.announce_m < task.activate_m < task.source_command['route_directive']['distance_m']
    assert task.activate_m == 3050.
    assert load_profile(catalog, task)['activate_m'] == task.activate_m


def test_lane_change_starts_in_the_available_lane_window():
    catalog = load_catalog('scene_2')
    task = catalog.select('s2_t05_cmd_12')[0]
    assert task.activate_m == 4596.
    assert load_profile(catalog, task)['activate_m'] == task.activate_m


def test_all_scheduled_lane_changes_wait_for_fresh_safe_clearance():
    catalog = load_catalog('scene_2')
    for task in catalog.tasks:
        for step in task.source_command['steps']:
            if step.startswith('CHANGE_LANE:'):
                assert step.endswith('_WHEN_SAFE'), (task.task_id, step)


def test_junction_confirmation_uses_measured_route_corridor():
    catalog = load_catalog('scene_2')
    task = catalog.select('s2_t05_cmd_14')[0]
    assert task.activate_m == 6050.
    assert load_profile(catalog, task)['activate_m'] == task.activate_m
    assert task.source_command['steps'][0] == 'CHECK:PATH_CLEAR'


def test_final_route_task_reaches_endpoint_without_extra_multi_junction_sequence():
    catalog = load_catalog('scene_2')
    task = catalog.select('s2_t05_cmd_15')[0]
    assert task.source_command['steps'][-1] == 'PROCEED:FINISH'
    assert task.source_command['execution_timeout_s'] == 600
    assert load_profile(catalog, task)['timeout_s'] == 600
    steps = task.source_command['steps']
    assert steps == ['CHECK:PATH_CLEAR','SET_SPEED:8.33mps','PROCEED:FINISH']
    assert task.activate_m == 6964.
    criteria = load_profile(catalog, task)['steps']
    assert [s['kind'] for s in criteria] == ['speed','destination']


def test_corridor_speed_recovery_keeps_current_lane():
    catalog=load_catalog('scene_2')
    task=catalog.select('s2_t05_cmd_11')[0]
    steps=task.source_command['steps']
    assert steps == ['CHECK:PATH_CLEAR','SET_SPEED:8.33mps','KEEP_LANE']
    criteria=load_profile(catalog,task)['steps']
    assert [s['kind'] for s in criteria] == ['speed','lane_hold','progress']


def test_turn_speed_recovery_window_continues_beyond_next_signal():
    catalog = load_catalog('scene_2')
    task = catalog.select('s2_t05_cmd_11')[0]
    next_task = catalog.select('s2_t05_cmd_12')[0]
    assert load_profile(catalog, task)['end_route_s_m'] == 4560
    assert 4560 < next_task.activate_m
    assert load_profile(catalog, task)['steps'][0]['kind'] == 'speed'


def test_scene3_command_windows_do_not_end_merge_or_exit_early():
    catalog = load_catalog('scene_3')
    source = json.loads((CONFIG_ROOT / catalog.source_file).read_text())
    events = {event['id']: event for event in source['events']}
    passage = catalog.select('scene3_pass_work_zone')[0]
    resume = catalog.select('scene3_resume_normal_driving')[0]
    assert passage.activate_m >= events['scene3_cone_taper']['closure']['taper_end_s_m']
    assert resume.activate_m >= events['scene3_work_zone_exit']['distance_m']
    for task in catalog.tasks:
        assert load_profile(catalog, task)['activate_m'] == task.activate_m
    assert source['surface_and_visibility']['wet_speed_limit_kmh'] == 50.


def test_route_flow_respects_global_following_distance():
    import ast
    from types import SimpleNamespace
    source = (CONFIG_ROOT.parent / 'scenarios/complex/town05_scene2.py').read_text()
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == 'distance_to_leading_vehicle'
             and len(node.args) == 2]
    configured = [node for node in calls if 'global_distance_to_leading_vehicle_m' in ast.unparse(node)]
    assert len(configured) == 1
    expression = ast.Expression(configured[0].args[1])
    for config, expected in (({}, 15.), ({'global_distance_to_leading_vehicle_m': 15.}, 15.),
                             ({'global_distance_to_leading_vehicle_m': 22.}, 22.),
                             ({'global_distance_to_leading_vehicle_m': 1.}, 5.)):
        assert eval(compile(expression, '<following-distance>', 'eval'),
                    {'self': SimpleNamespace(config=config)}) == expected
