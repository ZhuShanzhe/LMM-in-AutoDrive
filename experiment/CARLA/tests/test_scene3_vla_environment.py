import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

CARLA_DIR = Path(__file__).resolve().parents[1]
if str(CARLA_DIR) not in sys.path:
    sys.path.insert(0, str(CARLA_DIR))

from universal_vla_controller import environment_feature_tensor


class FakeWorld:
    def get_weather(self):
        return SimpleNamespace(
            cloudiness=95.0,
            precipitation=55.0,
            precipitation_deposits=80.0,
            wind_intensity=25.0,
            sun_azimuth_angle=280.0,
            sun_altitude_angle=-12.0,
            fog_density=35.0,
            fog_distance=80.0,
            fog_falloff=0.2,
            wetness=100.0,
            scattering_intensity=1.0,
            mie_scattering_scale=0.8,
        )


class FakeEgo:
    def get_speed_limit(self):
        return 60.0


def test_unified_environment_contract_is_14_dim():
    features = environment_feature_tensor(FakeWorld())
    assert tuple(features.shape) == (1, 14)
    assert float(features[0, 11]) == pytest.approx(0.8)


def test_v3_environment_adds_road_and_control_speed_caps():
    features = environment_feature_tensor(
        FakeWorld(), FakeEgo(), 30.0
    )
    assert tuple(features.shape) == (1, 14)
    assert float(features[0, 11]) == pytest.approx(0.8)
    assert float(features[0, 12]) == pytest.approx(0.6)
    assert float(features[0, 13]) == pytest.approx(0.3)
