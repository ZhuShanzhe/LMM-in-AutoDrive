"""Causal, mask-aware physical radar features; never accepts simulator labels."""

from collections import deque
import math

import numpy as np


SCHEMA_VERSION = 'risk_motion_observation/1.0'
FIELDS = ('sensor_valid', 'obstacle_present', 'distance_80m', 'radial_velocity_40mps',
          'closing_speed_40mps', 'ttc_20s', 'closing_return_present',
          'closing_distance_80m', 'closing_return_speed_40mps', 'closing_ttc_20s')
FEATURE_NAMES = tuple(f'{direction}_{field}' for direction in ('front', 'rear') for field in FIELDS)


def finite_number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def encode_motion_observation(front, rear, *, frame, timestamp_s, max_age_frames=1):
    """Encode physical radar summaries. Negative radial velocity means closing.

    A fresh sensor with no detection differs from an absent/stale sensor.
    Ranges and speeds from different returns are never combined into a TTC.
    """
    if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
        raise ValueError('frame must be a nonnegative integer')
    if finite_number(timestamp_s) is None or timestamp_s < 0 or max_age_frames < 0:
        raise ValueError('Invalid timestamp or maximum age')
    values, masks, provenance = [], [], {}
    for direction, observation in (('front', front), ('rear', rear)):
        observation = observation or {}
        x, mask = [0.] * len(FIELDS), [False] * len(FIELDS)
        mask[0] = True
        source_frame = observation.get('sensor_frame')
        valid_frame = isinstance(source_frame, int) and not isinstance(source_frame, bool)
        valid = (observation.get('schema_version') == f'physical_{direction}_radar/1.0'
                 and valid_frame and 0 <= frame - source_frame <= max_age_frames)
        provenance[direction] = {'sensor_frame': source_frame, 'valid': bool(valid),
                                 'corridor_filtered': bool(observation.get('route_corridor_filter_applied', False))}
        if valid:
            x[0] = 1.
            mask[1] = mask[6] = True
            distance = finite_number(observation.get('nearest_distance_m'))
            velocity = finite_number(observation.get('nearest_relative_velocity_mps'))
            if distance is not None and distance >= 0:
                x[1], x[2], mask[2] = 1., min(distance / 80., 1.), True
                if velocity is not None:
                    closing = max(-velocity, 0.)
                    x[3:6] = [max(-1., min(velocity / 40., 1.)), min(closing / 40., 1.),
                              min(distance / closing, 20.) / 20. if closing > 1e-3 else 1.]
                    mask[3:6] = [True] * 3
            closing_distance = finite_number(observation.get('nearest_closing_distance_m'))
            closing_speed = finite_number(observation.get('nearest_closing_velocity_mps'))
            if closing_distance is not None and closing_distance >= 0 and closing_speed is not None and closing_speed > 0:
                x[6:10] = [1., min(closing_distance / 80., 1.), min(closing_speed / 40., 1.),
                           min(closing_distance / closing_speed, 20.) / 20.]
                mask[7:10] = [True] * 3
        values.extend(x)
        masks.extend(mask)
    return {'schema_version': SCHEMA_VERSION, 'frame': frame, 'timestamp_s': float(timestamp_s),
            'feature_names': list(FEATURE_NAMES), 'values': values, 'valid_mask': masks,
            'provenance': provenance, 'source': 'physical_radar_only'}


class CausalMotionHistory:
    """Fixed padded history; reset on episode changes or discontinuous time."""

    def __init__(self, length=4, max_gap_s=.3, sample_interval_s=.1):
        if length < 1 or sample_interval_s <= 0 or max_gap_s < sample_interval_s:
            raise ValueError('Invalid history dimensions')
        self.length, self.max_gap_s = length, max_gap_s
        self.sample_interval_s = sample_interval_s
        self.rows = deque(maxlen=length)
        self.episode = None

    def push(self, observation, *, episode_id):
        if not episode_id or observation.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Explicit episode and observation schema required')
        if len(observation['values']) != len(FEATURE_NAMES) or len(observation['valid_mask']) != len(FEATURE_NAMES):
            raise ValueError('Invalid motion feature dimensions')
        if tuple(observation.get('feature_names', ())) != FEATURE_NAMES:
            raise ValueError('Motion feature order does not match the schema')
        if not all(finite_number(v) is not None for v in observation['values']):
            raise ValueError('Motion values must be finite')
        if finite_number(observation.get('timestamp_s')) is None or observation['timestamp_s'] < 0:
            raise ValueError('Motion timestamp must be finite and nonnegative')
        if isinstance(observation.get('frame'), bool) or not isinstance(observation.get('frame'), int) or observation['frame'] < 0:
            raise ValueError('Invalid motion frame')
        if episode_id != self.episode:
            self.rows.clear()
            self.episode = episode_id
        if self.rows:
            previous = self.rows[-1]
            if observation['frame'] == previous['frame']:
                return False
            dt = observation['timestamp_s'] - previous['timestamp_s']
            if observation['frame'] < previous['frame'] or not 0 < dt <= self.max_gap_s:
                self.rows.clear()
            elif dt < self.sample_interval_s - 1e-6:
                return False
        self.rows.append({'frame': observation['frame'], 'timestamp_s': observation['timestamp_s'],
                          'values': list(observation['values']), 'valid_mask': list(observation['valid_mask'])})
        return True

    def tensors(self):
        values = np.zeros((self.length, len(FEATURE_NAMES)), dtype=np.float32)
        mask = np.zeros_like(values, dtype=np.bool_)
        steps = np.zeros(self.length, dtype=np.bool_)
        for index, row in enumerate(self.rows, start=self.length - len(self.rows)):
            values[index] = row['values']
            mask[index] = row['valid_mask']
            steps[index] = True
        return values, mask, steps
