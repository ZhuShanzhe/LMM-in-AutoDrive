"""Apply source-scene weather without silently substituting a sunny preset."""
import math

from .catalog import ConfigError


FIELDS = ('cloudiness', 'precipitation', 'precipitation_deposits', 'wind_intensity',
          'sun_azimuth_angle', 'sun_altitude_angle', 'fog_density', 'fog_distance',
          'wetness', 'fog_falloff', 'scattering_intensity', 'mie_scattering_scale',
          'rayleigh_scattering_scale', 'dust_storm')


def apply_source_weather(world, source, weather_type, advance_world=None):
    requested = source.get('weather', source.get('environment', {}).get('weather'))
    if isinstance(requested, str):
        if not requested or requested.startswith('_'):
            raise ConfigError('invalid weather preset')
        weather = getattr(weather_type, requested, None)
        if weather is None or not hasattr(weather, 'sun_altitude_angle'):
            raise ConfigError('unknown weather preset: '+requested)
        expected = {key: float(getattr(weather, key)) for key in FIELDS if hasattr(weather, key)}
    elif isinstance(requested, dict):
        if set(requested) - set(FIELDS) - {'preset'}:
            raise ConfigError('unsupported weather fields')
        expected = {key: value for key, value in requested.items() if key != 'preset'}
        if not expected:
            raise ConfigError('named custom weather requires explicit parameters')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               for v in expected.values()):
            raise ConfigError('weather parameters must be finite numbers')
        weather = weather_type(**expected)
    else:
        raise ConfigError('source scene must specify weather')
    world.set_weather(weather)
    if advance_world is not None:
        advance_world()
    actual_weather = world.get_weather()
    actual = {key: float(getattr(actual_weather, key)) for key in FIELDS if hasattr(actual_weather, key)}
    mismatch = [key for key, value in expected.items()
                if key not in actual or not math.isclose(actual[key], value, rel_tol=1e-5, abs_tol=1e-4)]
    if mismatch:
        raise ConfigError('applied weather differs from source: '+', '.join(mismatch))
    return dict(requested=requested, applied=actual, verified_fields=sorted(expected),
                scope='CARLA weather parameters, not visual appearance or road friction verification')
