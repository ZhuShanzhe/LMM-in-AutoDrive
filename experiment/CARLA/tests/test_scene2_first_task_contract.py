from benchmark.catalog import load_catalog
from benchmark.task_oracle import load_profile
from benchmark.turn_fixture import bind_junction_sequence


def test_scene2_first_instruction_and_oracle_request_right_turn():
    catalog = load_catalog('scene_2')
    task = catalog.select('s2_t05_cmd_01')[0]
    command = task.source_command
    assert '右转' in command['text']
    assert command['spoken_text'] == command['text']
    assert command['steps'] == ['KEEP_LANE', 'SET_SPEED:12.50mps', 'TURN:RIGHT', 'KEEP_LANE']
    profile = load_profile(catalog, task)
    assert profile['steps'][1]['kind'] == 'turn'
    assert profile['steps'][1]['direction'] == 'RIGHT'
    assert profile['steps'][0]['target_kmh'] == 45
    assert profile['end_route_s_m'] == 350
    assert profile['timeout_s'] == 90
    assert profile['violation_budget'] == 0


def test_all_scene2_profiles_remain_bound_to_current_config():
    catalog = load_catalog('scene_2')
    assert len(catalog.tasks) == 15
    assert all(load_profile(catalog, task) is not None for task in catalog.tasks)


def test_first_task_does_not_search_past_the_next_junction(monkeypatch):
    from benchmark import turn_fixture
    seen = []
    monkeypatch.setattr(turn_fixture, 'bind_route_turn',
        lambda route, direction, start, end: seen.append((direction,start,end)) or
        dict(junction_end_m=321.675859))
    catalog = load_catalog('scene_2')
    profile = load_profile(catalog, catalog.select('s2_t05_cmd_01')[0])
    bound = bind_junction_sequence([], None, None, profile['steps'], 0, 350)
    assert seen == [('RIGHT', 0, 350)]
    assert list(bound) == ['1']
