from copy import deepcopy
import pytest
from continuous.task_geometry_binding import remap_commands


def test_distance_binding_preserves_content_order_and_source():
    commands = [dict(id='lane', announce_at_m=1260, activate_at_m=1300,
                     action='lane_change_left', target_speed_kmh=50,
                     voice_text='When safe, change left.'),
                dict(id='turn', announce_at_m=2985, activate_at_m=3085,
                     action='turn_left', route_directive=dict(distance_m=3085, action='turn_left'))]
    original = deepcopy(commands)
    bound = remap_commands(commands, [(0, 0), (1300, 550), (1450, 700), (3085, 1640), (5000, 5000)])
    assert commands == original
    assert bound[0]['activate_at_m'] == 550
    assert bound[0]['announce_at_m'] < bound[0]['activate_at_m']
    assert bound[1]['route_directive']['distance_m'] == 1640
    assert bound[0]['target_speed_kmh'] == 50
    assert bound[0]['voice_text'] == commands[0]['voice_text']
    assert bound[0]['activate_at_m'] < bound[1]['announce_at_m']


@pytest.mark.parametrize('anchors', [[(0, 0), (10, 0)], [(0, 0), (0, 10)], [(0, 0)], [(0, 0), (10, float('nan'))]])
def test_nonmonotone_bindings_rejected(anchors):
    with pytest.raises(ValueError):
        remap_commands([], anchors)


def test_command_outside_route_rejected():
    with pytest.raises(ValueError):
        remap_commands([dict(announce_at_m=101)], [(0, 0), (100, 100)])


def test_assessment_binding_keeps_thresholds_and_rejects_semantic_changes(tmp_path):
    import hashlib
    import json
    from types import SimpleNamespace
    from benchmark.catalog import CONFIG_ROOT, ConfigError, load_catalog, load_episode_catalog
    from benchmark.task_oracle import load_profile
    original = load_catalog('scene_1')
    source = json.loads((CONFIG_ROOT/original.source_file).read_text())
    anchors = [[0, 0], [1300, 550], [1450, 700], [3085, 1640], [5000, 5000]]
    source['commands'] = remap_commands(source['commands'], anchors)
    source['route'].update(start_spawn_index=40, strict_start_spawn=True, spawn_backoff_m=0.)
    source['geometry_binding'] = dict(source_sha256=original.source_sha256,
        map_sha256=hashlib.sha256(b'map').hexdigest(), distance_anchors=anchors)
    path = tmp_path/'bound.json'
    path.write_text(json.dumps(source))
    world_map = SimpleNamespace(to_opendrive=lambda:'map')
    bound = load_episode_catalog('scene_1', path, world_map)
    assert len(bound.tasks) == len(original.tasks) == 15
    for task in bound.tasks:
        profile = load_profile(bound, task)
        previous = load_profile(original, original.select(task.task_id)[0])
        assert profile is not None
        assert profile['activate_m'] == task.activate_m
        assert profile['timeout_s'] == previous['timeout_s']
        assert profile['violation_budget'] == previous['violation_budget']
        assert profile['steps'] == previous['steps']
    source['commands'][0]['target_speed_kmh'] = 5
    path.write_text(json.dumps(source))
    with pytest.raises(ConfigError, match='semantics'):
        load_episode_catalog('scene_1', path, world_map)


def test_launcher_explicitly_binds_scene1_and_external_scene2(tmp_path):
    import runpy
    from pathlib import Path
    build = runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/run_challenge_x86.py'))['build_command']
    assert '--bind-task-geometry' in build('scene1',tmp_path,tmp_path,tmp_path,'localhost',2000,'cpu',20)
    assert '--external-ego-control' in build('scene2',tmp_path,tmp_path,tmp_path,'localhost',2000,'cpu',20)


def test_incomplete_assessment_is_not_a_startup_failure(tmp_path):
    import json
    import runpy
    from pathlib import Path
    inspect = runpy.run_path(str(Path(__file__).resolve().parents[1]/'tools/run_challenge_x86.py'))['runner_result']
    assert inspect(tmp_path, 3)['status'] == 'runner_failed'
    root = tmp_path/'run/benchmark'
    root.mkdir(parents=True)
    (root/'run_outcome.json').write_text(json.dumps(dict(exit_code=3, status='INCOMPLETE_EVIDENCE')))
    (root.parent/'vla_control_decisions.jsonl').write_text('{"error":"sensor_not_ready"}\n')
    assert not inspect(tmp_path, 3)['closed_loop_started']
    (root.parent/'vla_control_decisions.jsonl').write_text('{"simulation_frame":100}\n')
    result = inspect(tmp_path, 3)
    assert result['closed_loop_started']
    assert result['status'] == 'assessment_incomplete'
    assert result['runner_exit_code'] == 3
