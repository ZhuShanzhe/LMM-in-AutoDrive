"""Opt-in low-level acceleration feedforward with velocity/acceleration feedback."""

import math


class SequenceLongitudinalTracker:
    def __init__(self, integral_limit=8.):
        if not math.isfinite(integral_limit) or integral_limit <= 0:
            raise ValueError('Expected a finite positive integral limit')
        self.integral_limit = integral_limit
        self.reset()

    def reset(self):
        self.previous_speed = None
        self.acceleration = 0.
        self.integral = 0.

    def step(self, speed_mps, acceleration_mps2, current_speed_mps, dt):
        if not all(math.isfinite(x) for x in (speed_mps,acceleration_mps2,current_speed_mps,dt)) or not .001 <= dt <= .2:
            raise ValueError('Invalid sequence tracking input')
        if self.previous_speed is not None:
            measured=(current_speed_mps-self.previous_speed)/dt
            self.acceleration=.7*self.acceleration+.3*max(-15.,min(15.,measured))
        self.previous_speed=current_speed_mps
        reference=max(-8.,min(3.,acceleration_mps2+.4*(speed_mps-current_speed_mps)))
        error=reference-self.acceleration
        candidate=max(-self.integral_limit,min(self.integral_limit,self.integral+error*dt))
        candidate_effort=reference+.5*error+.25*candidate
        # Retain learned drag compensation; freeze only when it worsens actuator saturation.
        if not ((candidate_effort>3. and error>0.) or (candidate_effort < -7. and error<0.)):
            self.integral=candidate
        effort=reference+.5*error+.25*self.integral
        if speed_mps<=.05 and current_speed_mps<=.1 and reference<=0:
            self.integral=0.
            return 0.,.3
        return (min(1.,max(0.,effort/3.)), min(1.,max(0.,-effort/7.)))
