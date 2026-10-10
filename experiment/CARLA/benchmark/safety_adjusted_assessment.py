"""Supplementary engineering assessment; never rewrites strict task results."""
from copy import deepcopy

from .task_oracle import TaskOracle
from lightweight_vla_adapter.src.safety_adjusted_speed import safety_adjusted_speed_evidence, finite


class SafetyAdjustedTaskOracle(TaskOracle):
    def __init__(self,spec,*,max_safety_wait_s=120.):
        if not finite(max_safety_wait_s) or not 0<=max_safety_wait_s<=120.:
            raise ValueError('Safety wait budget must be between 0 and 120 seconds')
        self.max_safety_wait_s=max_safety_wait_s
        self.safety_wait_s=0.;self.previous_wait=False
        self.previous_motion=None;self.acceleration=None
        self.adjustments=[];self.raw_time=None
        super().__init__(spec)

    @staticmethod
    def justified_wait(obs):
        record=obs.get('execution_evidence') or {}
        decision=record.get('control_decision') or {}
        if not (obs['ego']['speed_kmh']<=.5 and decision.get('action')in{'stop','emergency_brake'}
                and decision.get('frame_id')==f"carla_{obs['frame']}"):
            return False
        signal=record.get('traffic_control_contract') or {};light=signal.get('observation')or{}
        stop_distance=light.get('stop_distance_m')
        light_wait=(decision.get('reason')=='traffic_signal_stopline_constraint'
            and light.get('applicable')is True and light.get('state')in{'RED','YELLOW'}
            and light.get('stop_line_source')=='static_hdmap_stop_line'
            and finite(stop_distance) and 0<=stop_distance<=5.
            and finite(light.get('confidence')) and light['confidence']>=.8)
        risk=record.get('risk') or {};policy=risk.get('behavior_policy')or{}
        evidence=risk.get('risk_evidence')or{};now=obs['sim_time_s']
        lidar=policy.get('lidar')or{};radar=risk.get('physical_forward_radar')or{}
        observed=lidar.get('observed_at_s')
        required=policy.get('current_speed_required_clearance_m',policy.get('required_clearance_m'))
        distances=(lidar.get('nearest_obstacle_m'),radar.get('nearest_distance_m'))
        close_obstacle=finite(required) and any(finite(d) and 0<=d<=required for d in distances)
        closing=radar.get('nearest_closing_velocity_mps');gap=radar.get('nearest_closing_distance_m')
        closing_risk=finite(closing) and closing>0 and finite(gap) and 0<=gap/closing<=5.
        crossing=any(actor.get('in_conflict_zone')is True for actor in obs.get('actors',{}).values())
        physical_wait=(policy.get('physical_stop')is True
            and risk.get('frame_id')==decision['frame_id'] and evidence.get('is_fresh')is True
            and finite(observed) and 0<=now-observed<=.25
            and (close_obstacle or closing_risk or crossing))
        return light_wait or physical_wait

    def _update(self,obs):
        now=obs['sim_time_s'];speed=obs['ego']['speed_kmh'];self.raw_time=now
        waiting=self.justified_wait(obs)
        if self.previous_motion is not None:
            previous_time,previous_speed=self.previous_motion;dt=now-previous_time
            self.acceleration=(speed-previous_speed)/3.6/dt if 0<dt<=.25 else None
            if self.started is not None and waiting and self.previous_wait and 0<dt<=.25:
                self.safety_wait_s=min(self.max_safety_wait_s,self.safety_wait_s+dt)
        self.previous_motion=(now,speed);self.previous_wait=waiting
        return super()._update(obs)

    def _deadline_elapsed(self,obs,now):
        return now-self.started-self.safety_wait_s

    def _criterion(self,step,obs,now):
        relative=step['kind']=='speed_change' and step.get('direction')=='INCREASE'
        if step['kind']!='speed' and not relative:return super()._criterion(step,obs,now)
        actual=obs['ego']['speed_kmh']
        entry=self.state.setdefault('entry_speed_kmh',actual)if relative else None
        target=entry+step['delta_kmh']if relative else step['target_kmh']
        tolerance=2. if relative else step['tolerance_kmh']
        strict=actual-entry>=step['delta_kmh']if relative else abs(actual-target)<=tolerance
        record=obs.get('execution_evidence')or{}
        execution=deepcopy(record.get('previous_pid_execution')or{})
        # Lane geometry is independently measured by the simulator truth collector.
        execution.setdefault('pid',{}).update({k:obs['ego'][k]for k in
            ('lateral_error_m','heading_error_deg')})
        adjustment=safety_adjusted_speed_evidence(execution,frame_id=f"carla_{obs['frame']}",
            timestamp_s=now,requested_kmh=target,actual_kmh=actual,
            tolerance_kmh=tolerance,acceleration_mps2=self.acceleration)
        trial=step
        if not strict and adjustment is not None:
            trial=dict(step,target_kmh=actual)
            self.state['safety_adjustment_seen']=adjustment
        if not strict and adjustment is None:self.state.pop('safety_adjustment_seen',None)
        passed=(self._hold('changed',strict or adjustment is not None,now,step['hold_s'])
            if relative else super()._criterion(trial,obs,now))
        if passed and self.state.get('safety_adjustment_seen'):
            self.adjustments.append(dict(frame=obs['frame'],step_index=self.index,
                **self.state['safety_adjustment_seen']))
        return passed

    def result(self):
        result=super().result()
        raw=0. if self.started is None or self.raw_time is None else self.raw_time-self.started
        result.update(evaluation_mode='SAFETY_ADJUSTED_ENGINEERING',official_acceptance=False,
            strict_speed_target_unchanged=True,raw_elapsed_s=raw,
            justified_safety_wait_s=self.safety_wait_s,active_elapsed_s=max(0.,raw-self.safety_wait_s),
            maximum_safety_wait_s=self.max_safety_wait_s,speed_adjustments=list(self.adjustments))
        return result
