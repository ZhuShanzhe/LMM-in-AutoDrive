import hashlib
import json
from pathlib import Path

from benchmark.catalog import load_catalog
from benchmark.task_oracle import load_profile


def test_corridor_commands_and_diagnostic_geometry_share_current_source():
    root = Path(__file__).resolve().parents[1] / 'configs'
    folder = root / 'diagnostic_sections/scene_2'
    sections = json.loads((folder / 'sections.json').read_text())
    points = json.loads((folder / 'geometry.json').read_text())
    catalog = load_catalog('scene_2')
    assert sections['source_sha256'] == catalog.source_sha256
    assert sections['route_geometry_sha256'] == hashlib.sha256(
        json.dumps(points, sort_keys=True).encode()).hexdigest()
    assert points[-1]['s_m'] >= 7995
    config = json.loads((root / catalog.source_file).read_text())
    for identity in ('s2_t05_cmd_09', 's2_t05_cmd_11'):
        task = catalog.select(identity)[0]
        assert load_profile(catalog, task) is not None
        command = next(c for c in config['commands'] if c['id'] == identity)
        assert not any(s.startswith(('TURN:', 'CHANGE_LANE:')) for s in command['steps'])
