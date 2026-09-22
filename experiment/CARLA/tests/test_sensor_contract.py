import json
from pathlib import Path

import pytest

from control.sensor_contract import resolve_sensor_contract


def test_legacy_sensor_arguments_preserved():
    assert resolve_sensor_contract({}, ('front',), False) == {
        'cameras': ['front'], 'lidar': False, 'source': 'runner_arguments'}


def test_challenge_config_supplies_all_required_sensors():
    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / 'lightweight_vla_adapter/configs/challenge_signal_generalization.json').read_text())
    result = resolve_sensor_contract(config, ('front',), False)
    assert result['cameras'] == ['front', 'left', 'right', 'rear']
    assert result['lidar'] is True
    assert result['source'] == 'model_config.sensor_requirements'


@pytest.mark.parametrize('requirements', [
    {}, {'cameras': [], 'lidar': True},
    {'cameras': ['front', 'front'], 'lidar': True},
    {'cameras': ['overhead'], 'lidar': True},
    {'cameras': ['front'], 'lidar': 'false'},
])
def test_invalid_contract_rejected(requirements):
    with pytest.raises(ValueError):
        resolve_sensor_contract({'sensor_requirements': requirements}, ('front',), False)
