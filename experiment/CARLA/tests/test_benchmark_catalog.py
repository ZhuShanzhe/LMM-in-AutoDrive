import json
import hashlib
from pathlib import Path

import pytest

from benchmark.catalog import CONFIG_ROOT, SCENES, ConfigError, load_catalog, validate_catalog
from benchmark.__main__ import main


def altered(tmp_path, scene, change):
    filename = SCENES[scene][0]
    raw = json.loads((CONFIG_ROOT / filename).read_text(encoding='utf-8'))
    change(raw)
    (tmp_path / filename).write_text(json.dumps(raw), encoding='utf-8')
    return tmp_path


@pytest.mark.parametrize('scene,count,length', [('scene_1', 15, 5000), ('scene_2', 15, 8000), ('scene_3', 8, 6000)])
def test_existing_configs(scene, count, length):
    catalog = load_catalog(scene)
    assert len(catalog.tasks) == count
    assert catalog.route_length_m == length
    assert catalog.map_name.startswith('Town')
    assert len(catalog.source_sha256) == 64
    assert validate_catalog(catalog)['config_valid']
    assert not validate_catalog(catalog)['benchmark_ready']


@pytest.mark.parametrize('scene', ('scene_1', 'scene_2', 'scene_3'))
def test_catalog_source_hash_is_line_ending_independent(tmp_path, scene):
    filename = SCENES[scene][0]
    source = (CONFIG_ROOT / filename).read_bytes().replace(b'\r\n', b'\n')
    (tmp_path / filename).write_bytes(source.replace(b'\n', b'\r\n'))
    expected = hashlib.sha256(source).hexdigest()
    assert load_catalog(scene, tmp_path).source_sha256 == expected
    assert load_catalog(scene).source_sha256 == expected
    (tmp_path / filename).write_bytes(source + b' ')
    assert load_catalog(scene, tmp_path).source_sha256 != expected


def test_activation_order_preserves_source_and_text():
    catalog = load_catalog('scene_1')
    task = catalog.select('4')[0]
    assert task.task_id == 'c08_keep_50'
    assert task.source_order == 8
    assert task.instruction == task.source_command['voice_text']
    assert catalog.select(task.task_id) == (task,)
    assert task.end_m is None


def test_scene1_turn_schedule_and_profile_bindings():
    from benchmark.task_oracle import load_profile
    catalog = load_catalog('scene_1')
    tasks = {task.task_id: task for task in catalog.tasks}
    turn = tasks['c07_turn_left']
    assert turn.activate_m == turn.source_command['route_directive']['distance_m']
    assert tasks['c11_accelerate_50'].announce_m < tasks['c09_reduce_for_turn'].announce_m
    assert tasks['c09_reduce_for_turn'].announce_m < turn.announce_m < turn.activate_m
    assert turn.activate_m < tasks['c10_keep_35'].announce_m < tasks['c12_keep_50'].announce_m
    for task in catalog.tasks:
        profile = load_profile(catalog, task)
        assert profile['activate_m'] == task.activate_m
        if 'end_route_s_m' in profile:
            next_announcement = min(t.announce_m for t in catalog.tasks if t.announce_m > task.activate_m)
            assert profile['end_route_s_m'] == next_announcement


def test_scene1_runtime_cannot_finish_before_destination_oracle():
    from benchmark.task_oracle import load_profile
    catalog = load_catalog('scene_1')
    raw = json.loads((CONFIG_ROOT / catalog.source_file).read_text(encoding='utf-8'))
    profile = load_profile(catalog, catalog.select('c15_keep_to_goal')[0])
    step = profile['steps'][0]
    tolerance = raw['route']['goal_tolerance_m']
    assert 0 < tolerance <= min(step['max_distance_m'], step['max_remaining_m'])


def test_scene1_uses_validated_background_traffic_without_preview_commands():
    raw = json.loads((CONFIG_ROOT / SCENES['scene_1'][0]).read_text(encoding='utf-8'))
    preview = json.loads((CONFIG_ROOT / 'basic_voice_traffic_preview.json').read_text(encoding='utf-8'))
    flow = raw['traffic']
    assert flow == preview['traffic']
    assert flow['enabled'] and flow['maintenance_mode'] == 'route_density'
    assert len(flow['vehicles']) == 96
    assert flow['density_max_actors'] == 128
    assert flow['density_ahead_start_m'] > flow['lifecycle_protected_radius_m']
    assert not flow['ignore_traffic_lights'] and not flow['ignore_traffic_signs']
    assert len(raw['commands']) == 15
    assert raw['route']['start_spawn_index'] == 323


def test_scene3_links_and_overlap_are_not_silently_dropped():
    catalog = load_catalog('scene_3')
    task = catalog.select('scene3_worker_crossing')[0]
    assert task.linked_event_ids == ('scene3_temporary_pedestrian',)
    assert any(i['code'] == 'overlapping_tasks' for i in validate_catalog(catalog)['issues'])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1, True, '800'])
def test_invalid_distance(tmp_path, value):
    root = altered(tmp_path, 'scene_1', lambda c: c['commands'][0].update(announce_at_m=value))
    with pytest.raises(ConfigError):
        load_catalog('scene_1', root)


def test_duplicate_ids(tmp_path):
    root = altered(tmp_path, 'scene_2', lambda c: c['commands'].append(c['commands'][0]))
    with pytest.raises(ConfigError, match='duplicate'):
        load_catalog('scene_2', root)


def test_dangling_dependency(tmp_path):
    root = altered(tmp_path, 'scene_2', lambda c: c['commands'][0].update(requires_event_states={'absent': 'RESOLVED'}))
    with pytest.raises(ConfigError, match='dependency'):
        load_catalog('scene_2', root)


def test_dangling_voice_reference(tmp_path):
    root = altered(tmp_path, 'scene_3', lambda c: c['events'][0].update(voice_command_id='absent'))
    with pytest.raises(ConfigError, match='unknown voice'):
        load_catalog('scene_3', root)


def test_cli_exports_selection_without_claiming_execution(tmp_path):
    output = tmp_path / 'manifest.json'
    assert main(['manifest', '--scene', 'scene_2', '--task', '3', '--output', str(output)]) == 0
    result = json.loads(output.read_text(encoding='utf-8'))
    assert len(result['tasks']) == 1
    assert result['tasks'][0]['task_id'] == 's2_t05_cmd_03'
    assert not result['execution_supported']
    assert main(['validate', '--scene', 'scene_1', '--require-ready', '--output', str(output)]) == 1
    assert main(['list', '--scene', 'scene_1', '--task', '0']) == 2
