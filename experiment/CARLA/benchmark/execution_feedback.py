"""Measured execution feedback for the simulator baseline; not task acceptance."""
import math


class LaneExecutionFeedback:
    def __init__(self):
        self.command_id=None
        self.since=None
        self.last_time=None
        self.completed=set()

    def update(self, intent, state, time_s):
        command=intent.get('command_id')
        if intent.get('action') not in ('lane_change_left','lane_change_right') or not command:
            self.command_id=self.since=self.last_time=None
            return None
        if command in self.completed:
            return None
        if command!=self.command_id:
            self.command_id=command
            self.since=self.last_time=None
        continuous=(self.last_time is None or 0<time_s-self.last_time<=.15+1e-8)
        valid=(isinstance(time_s,(int,float)) and math.isfinite(time_s) and
               state.get('lane_change_command_id')==command and
               state.get('target_lane_id') is not None and
               state.get('current_lane_id')==state.get('target_lane_id') and
               state.get('in_junction') is False)
        for key,limit in [('lateral_error_m',.35),('heading_error_deg',5.)]:
            value=state.get(key)
            valid &= isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and abs(value)<=limit
        if not valid or not continuous:
            self.since=None
        if valid and continuous:
            if self.since is None:
                self.since=time_s
            if time_s-self.since>=2.-1e-8:
                self.completed.add(command)
                return dict(command_id=command,status='completed',sim_time_s=time_s,
                            scope='controller_execution_feedback_not_independent_assessment',
                            stable_s=time_s-self.since)
        self.last_time=time_s
        return None
