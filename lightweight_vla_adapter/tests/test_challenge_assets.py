import hashlib
import importlib.util
from pathlib import Path

import pytest

MODULE = Path(__file__).parents[1]/'scripts/prepare_challenge_runtime.py'
spec = importlib.util.spec_from_file_location('prepare_challenge_runtime', MODULE)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def manifest(data):
    return {'schema_version': 'challenge_runtime_assets/1.0', 'files': [
        {'path': 'model.pt', 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}]}


def test_verify_exact_asset(tmp_path):
    (tmp_path/'model.pt').write_bytes(b'model')
    assert prepare.verify_assets(tmp_path, manifest(b'model')) == ['model.pt']


def test_modified_or_missing_asset_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        prepare.verify_assets(tmp_path, manifest(b'model'))
    (tmp_path/'model.pt').write_bytes(b'wrong')
    with pytest.raises(ValueError, match='checksum'):
        prepare.verify_assets(tmp_path, manifest(b'model'))


def test_parent_escape_rejected(tmp_path):
    value = manifest(b'model')
    value['files'][0]['path'] = '../model.pt'
    with pytest.raises(ValueError, match='relative'):
        prepare.verify_assets(tmp_path, value)


@pytest.mark.parametrize('town', ['Town01', 'Town04', 'Town05'])
def test_portable_map_and_model_paths(tmp_path, town):
    root = Path(__file__).parents[2]
    config = prepare.resolve_config(root, tmp_path, root/f'lightweight_vla_adapter/configs/maps/{town}_traffic_heads.json')
    assert config['event_memory_checkpoint'] == str(tmp_path/'challenge/phase_recovery_v1.pt')
    assert config['sequence_execution']['operation_interval_s'] == .5
    assert config['simulation_only'] is True
    assert town in config['traffic_signal_observer']['static_map_path']
