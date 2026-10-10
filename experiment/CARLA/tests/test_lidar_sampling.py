import pytest

from carla_multiview_sensor import physical_lidar_attributes


def test_dense_sampling_is_default():
    assert physical_lidar_attributes(.05, 32, 80) == {
        'sensor_tick': '0.05', 'channels': '32', 'range': '80.0',
        'points_per_second': '224000', 'rotation_frequency': '20.0'}


def test_dense_sampling_does_not_change_scan_frequency_or_channels():
    old = physical_lidar_attributes(.05, 32, 80, 56000)
    dense = physical_lidar_attributes(.05, 32, 80, 224000)
    assert dense.pop('points_per_second') == '224000'
    old.pop('points_per_second')
    assert dense == old


@pytest.mark.parametrize('tick,channels,distance,rate', [
    (0, 32, 80, 56000), (float('nan'), 32, 80, 56000),
    (.05, 0, 80, 56000), (.05, 32.5, 80, 56000),
    (.05, 32, 0, 56000), (.05, 32, 80, -1),
    (.05, 32, 80, True), (.05, 32, 80, 224000.5),
    (.05, 32, 80, float('inf')), (.05, float('nan'), 80, 56000)])
def test_invalid_sampling_rejected(tick, channels, distance, rate):
    with pytest.raises(ValueError):
        physical_lidar_attributes(tick, channels, distance, rate)
