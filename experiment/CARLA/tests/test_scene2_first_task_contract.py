from benchmark.catalog import load_catalog
from benchmark.task_oracle import load_profile
from benchmark.turn_fixture import bind_junction_sequence


def test_scene2_first_instruction_and_oracle_request_right_turn():
    catalog = load_catalog('scene_2')
    task = catalog.select('s2_t05_cmd_01')[0]
    command = task.source_command
    assert '右转' in command['text']
    assert command['spoken_text'] == command['text']
    assert command['steps'] == ['KEEP_LANE', 'SET_SPEED:8.33mps', 'TURN:RIGHT', 'KEEP_LANE']
    profile = load_profile(catalog, task)
    assert profile['steps'][1]['kind'] == 'turn'
    assert profile['steps'][1]['direction'] == 'RIGHT'
    assert profile['steps'][0]['target_kmh'] == 30
    assert profile['end_route_s_m'] == 380
    assert profile['timeout_s'] == 180
    assert profile['violation_budget'] == 0


def test_all_scene2_profiles_remain_bound_to_current_config():
    catalog = load_catalog('scene_2')
    assert len(catalog.tasks) == 15
    assert all(load_profile(catalog, task) is not None for task in catalog.tasks)


def test_scene2_completion_budgets_include_signal_wait_without_relaxing_violations():
    import math
    catalog=load_catalog('scene_2')
    for task in catalog.tasks:
        profile=load_profile(catalog,task)
        assert profile['timeout_s']>=180
        end=profile.get('end_route_s_m')
        if end is not None:
            baseline=(end-profile['activate_m'])/(30/3.6)+120
            assert profile['timeout_s']>=math.ceil(baseline/30)*30
        assert profile['violation_budget']==0
        assert task.source_command['execution_timeout_s']==profile['timeout_s']


def test_scene2_runtime_schedule_preserves_per_command_timeout():
    import json
    from pathlib import Path
    from run_complex_avoidance_town05 import build_vla_command_schedule
    from control.command_dispatch import CompletionCommandQueue
    config=json.loads((Path(__file__).parents[1]/'configs/scene_2_town05_runtime.json').read_text())
    commands=build_vla_command_schedule(config)
    assert [c['execution_timeout_s'] for c in commands]==[c['execution_timeout_s'] for c in config['commands']]
    queue=CompletionCommandQueue(commands)
    queue.select(0,0)
    queue.select(0,121)
    assert queue.status()['failure'] is None and queue.status()['execution_timeout_s']==180


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
