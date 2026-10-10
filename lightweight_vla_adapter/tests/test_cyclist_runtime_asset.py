import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
spec = importlib.util.spec_from_file_location('cyclist_runtime_prepare',
    ROOT/'lightweight_vla_adapter/scripts/prepare_challenge_runtime.py')
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
MAP = ROOT/'lightweight_vla_adapter/configs/maps/Town05_traffic_heads.json'


def expert_config(path, digest):
    return dict(enabled=True, relative_weights=str(path), sha256=digest)


def test_optional_bicycle_asset_is_verified_and_resolved(tmp_path):
    target = tmp_path/'challenge/bike.pt'
    target.parent.mkdir()
    target.write_bytes(b'bicycle')
    digest = hashlib.sha256(b'bicycle').hexdigest()
    value = prepare.resolve_config(ROOT, tmp_path, MAP,
        cyclist_expert=expert_config('challenge/bike.pt', digest))
    assert value['pedestrian_condition_observer']['cyclist_expert'] == {
        'weights': str(target.resolve()), 'sha256': digest}
    assert value['cyclist_condition_expert']['enabled'] is True


def test_wrong_bicycle_asset_checksum_rejected(tmp_path):
    (tmp_path/'bike.pt').write_bytes(b'wrong')
    with pytest.raises(ValueError, match='checksum'):
        prepare.resolve_config(ROOT, tmp_path, MAP,
            cyclist_expert=expert_config('bike.pt', hashlib.sha256(b'bicycle').hexdigest()))


@pytest.mark.parametrize('path', ['../bike.pt', '/tmp/bike.pt'])
def test_optional_bicycle_asset_cannot_escape_model_root(tmp_path, path):
    with pytest.raises(ValueError, match='relative'):
        prepare.resolve_config(ROOT, tmp_path, MAP, cyclist_expert=expert_config(path, '0'*64))


def test_symlink_escape_rejected(tmp_path):
    models = tmp_path/'models'
    models.mkdir()
    outside = tmp_path/'outside.pt'
    outside.write_bytes(b'bicycle')
    (models/'bike.pt').symlink_to(outside)
    with pytest.raises(ValueError, match='within model root'):
        prepare.resolve_config(ROOT, models, MAP,
            cyclist_expert=expert_config('bike.pt', hashlib.sha256(b'bicycle').hexdigest()))


@pytest.mark.parametrize('flag,value', [('--cyclist-weights', 'bike.pt'), ('--cyclist-sha256', '0'*64)])
def test_cli_requires_weight_and_checksum_together(monkeypatch, flag, value):
    monkeypatch.setattr('sys.argv', ['prepare', '--output', 'runtime.json', flag, value])
    with pytest.raises(SystemExit):
        prepare.main()
