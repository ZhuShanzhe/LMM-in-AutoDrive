"""Opt-in receding sequence execution; safety responses never wait for dwell."""

import copy
import math


class SequenceExecutionPolicy:
    def __init__(self, interval_s=1., forecast_steps=10, reference_mode='mean_acceleration'):
        if not math.isfinite(interval_s) or interval_s < .5 or not 1 <= forecast_steps <= 30:
            raise ValueError('Expected >=0.5 second normal dwell and 1..30 forecast samples')
        self.interval_s, self.forecast_steps = interval_s, forecast_steps
        if reference_mode not in ('mean_acceleration','speed_endpoint'):
            raise ValueError('Unknown sequence reference mode')
        self.reference_mode = reference_mode
        self.reset()

    def reset(self):
        self.episode = None
        self.previous_time = None
        self.operation_time = None
        self.acceleration = 0.
        self.anchor_speed = 0.
        self.command = None
        self.diagnostics = {}

    def update(self, proposal, sequence, *, timestamp_s, episode_id, speed_kmh, desired_speed_kmh, risk):
        speed = sequence['speed_mps']
        accel = sequence['acceleration_mps2']
        if (sequence.get('schema_version') != 'longitudinal_sequence/1.0' or len(speed) != 30 or len(accel) != 30
                or abs(sequence.get('dt_s',0)-.1)>1e-9
                or not all(math.isfinite(x) for x in [*speed,*accel,timestamp_s,speed_kmh,desired_speed_kmh])):
            raise ValueError('Invalid sequence contract')
        same_episode = self.episode == episode_id
        gap = same_episode and self.previous_time is not None and timestamp_s-self.previous_time > .35
        previous_operation = self.operation_time
        if not same_episode or (self.previous_time is not None and not 0 < timestamp_s-self.previous_time <= .35):
            self.reset()
            if gap:
                # Discard stale acceleration without allowing a faster routine restart.
                self.operation_time = previous_operation
                self.anchor_speed = max(0.,speed_kmh/3.6)
        self.episode, self.previous_time = episode_id, timestamp_s
        immediate = risk.get('risk_level') != 'low' or risk.get('recommended_action') in ('decelerate','emergency_brake') or accel[0] <= -3.
        new_operation = self.operation_time is None or timestamp_s-self.operation_time >= self.interval_s-1e-6
        requested = sum(accel[:self.forecast_steps])/self.forecast_steps
        if self.reference_mode == 'speed_endpoint' and not immediate:
            requested = (speed[self.forecast_steps-1]-max(0.,speed_kmh/3.6))/(self.forecast_steps*.1)
        if immediate:
            requested = min(requested,accel[0],0.)
        if immediate or new_operation:
            self.acceleration = max(-8.,min(3.,requested))
            self.operation_time = timestamp_s
            self.anchor_speed = max(0.,speed_kmh/3.6)
        current = max(0.,speed_kmh/3.6)
        ceiling = max(current,max(0.,desired_speed_kmh)/3.6)
        elapsed = min(self.interval_s,max(0.,timestamp_s-self.operation_time)+.1)
        target = min(ceiling,max(0.,self.anchor_speed+elapsed*self.acceleration))
        effective = self.acceleration
        if (target >= ceiling and effective > 0.) or (target <= 0. and effective < 0.):
            effective = 0.
        action = 'accelerate' if effective>.3 else 'decelerate' if effective<-.3 else 'keep_lane'
        if self.acceleration<=-6.: action='emergency_brake'
        if target<=.05 and effective<=0.: action='stop'
        result=copy.deepcopy(proposal)
        result.update(action=action,target_speed_kmh=target*3.6,target_lane=None)
        self.command=dict(longitudinal_sequence_schema='longitudinal_sequence/1.0',
                          target_acceleration_mps2=effective,sequence_valid_until_s=timestamp_s+.3)
        self.diagnostics=dict(operation_updated=bool(immediate or new_operation),
                              safety_bypass=bool(immediate),operation_time_s=self.operation_time,
                              selected_acceleration_mps2=self.acceleration,applied_acceleration_mps2=effective,
                              anchor_speed_mps=self.anchor_speed,reference_speed_mps=target,
                              observation_gap_reset=bool(gap),
                              reference_mode=self.reference_mode,
                              interval_s=self.interval_s,forecast_steps=self.forecast_steps)
        return result
