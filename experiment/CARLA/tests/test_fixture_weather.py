import json
from types import SimpleNamespace as NS
import pytest
from benchmark.catalog import CONFIG_ROOT, load_catalog, ConfigError
from benchmark.weather import apply_source_weather


class Weather(NS):
    pass


Weather.ClearNoon=Weather(sun_altitude_angle=75,precipitation=0)


class World:
    def set_weather(self, weather):
        self.weather=weather
    def get_weather(self):
        return self.weather


@pytest.mark.parametrize('scene,sun,rain', [('scene_1',75,0),('scene_2',5,0),('scene_3',-15,80)])
def test_registered_source_weather_preserved(scene,sun,rain):
    catalog=load_catalog(scene)
    source=json.loads((CONFIG_ROOT/catalog.source_file).read_text(encoding='utf-8'))
    result=apply_source_weather(World(),source,Weather)
    assert result['applied']['sun_altitude_angle']==sun
    assert result['applied']['precipitation']==rain


def test_weather_mismatch_is_not_silently_accepted():
    world=World()
    world.get_weather=lambda: Weather(sun_altitude_angle=75)
    with pytest.raises(ConfigError,match='differs'):
        apply_source_weather(world,{'weather':{'sun_altitude_angle':-15}},Weather)


@pytest.mark.parametrize('value',[None, 'missing', {'preset':'rain'}, {'wetness':float('nan')}])
def test_invalid_weather_rejected(value):
    with pytest.raises(ConfigError):
        apply_source_weather(World(),{'weather':value},Weather)
