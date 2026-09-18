"""Enforce fresh, route-associated traffic-control observations at execution."""

from copy import deepcopy
import math


class TrafficControlContract:
    def __init__(self):
        self.last_blocked=None
        self.green_since=None

    def apply(self,decision,observation,*,timestamp_s,ego_speed_mps):
        result=deepcopy(decision)
        info=dict(source='traffic_control_contract/1.0',observation=deepcopy(observation),
            active=False,changed=False,policy_safety_credit=False)
        if (observation is None or not observation.get('applicable')) and self.last_blocked is not None:
            observation=dict(applicable=True,signal_id=self.last_blocked,state='UNKNOWN',confidence=0.,
                timestamp_s=timestamp_s,stop_distance_m=0.,source='lost_unreleased_signal')
            info['observation']=deepcopy(observation)
        if observation is None:
            info['status']='unavailable'
            return result,info
        if not observation.get('applicable'):
            info['status']='no_route_associated_signal'
            return result,info
        distance=float(observation['stop_distance_m'])
        age=timestamp_s-float(observation['timestamp_s'])
        if not math.isfinite(distance) or not math.isfinite(age):
            raise ValueError('Nonfinite traffic control observation')
        signal=str(observation['signal_id'])
        fresh=0. <= age <= .25
        state=observation.get('state','UNKNOWN') if fresh else 'UNKNOWN'
        confidence=float(observation.get('confidence',0.))
        green=state=='GREEN' and confidence>=.6
        if green:
            if self.green_since is None or self.green_since[0]!=signal:
                self.green_since=(signal,timestamp_s)
            released=timestamp_s-self.green_since[1]>=.3
        else:
            self.green_since=None
            released=False
        if released:
            info['status']='observed_green_release'
            self.last_blocked=None
            return result,info
        self.last_blocked=signal
        remaining=max(0.,distance-4.-max(0.,ego_speed_mps)*.4)
        cap=math.sqrt(2.*2.*remaining)*3.6
        if distance<=5.:
            cap=0.
        info.update(active=True,status='wait_for_observed_green',state=state,speed_cap_kmh=cap)
        requested=float(result.get('target_speed_kmh',0.))
        result['target_speed_kmh']=min(requested,cap)
        emergency=result.get('action')=='emergency_brake' or bool(result.get('emergency'))
        if cap==0. and not emergency:
            result.update(action='stop',emergency=False,target_lane=None,target_location=None)
        if result['target_speed_kmh']<requested or cap==0.:
            result['allow_positive_acceleration']=False
            result['reason']='traffic_signal_stopline_constraint'
            for key in ('longitudinal_sequence_schema','target_acceleration_mps2','sequence_valid_until_s'):
                result.pop(key,None)
        info['changed']=result!=decision
        return result,info
