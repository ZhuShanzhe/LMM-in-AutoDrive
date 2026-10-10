import ast
from pathlib import Path

import pytest

from carla_multiview_sensor import resolve_physical_lidar_sampling


def test_physical_channels_are_not_network_bev_channels():
    assert resolve_physical_lidar_sampling({'lidar_channels': 4})['lidar_channels'] == 32
    assert resolve_physical_lidar_sampling({'lidar_channels': 4, 'physical_lidar_sampling': {
        'channels': 128, 'range_m': 100., 'points_per_second': 448000}}) == {
            'lidar_channels': 128, 'lidar_range_m': 100., 'lidar_points_per_second': 448000}


@pytest.mark.parametrize('sampling', [None, [], {'channels': True}, {'channels': 1.5},
    {'range_m': float('nan')}, {'range_m': 0}, {'points_per_second': -1}, {'channel': 128}])
def test_invalid_sampling_is_rejected(sampling):
    with pytest.raises((ValueError, TypeError)):
        resolve_physical_lidar_sampling({'physical_lidar_sampling': sampling})


def test_controller_passes_resolved_physical_sampling_to_rig():
    source = Path(__file__).resolve().parents[1] / 'universal_vla_controller.py'
    tree = ast.parse(source.read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'SynchronizedMultiviewCameraRig']
    assert len(calls) == 1
    assert any(k.arg is None and isinstance(k.value, ast.Attribute)
               and k.value.attr == 'physical_lidar_sampling' for k in calls[0].keywords)
