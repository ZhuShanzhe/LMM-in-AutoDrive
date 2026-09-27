"""Completion-gated command dispatch without simulator assessment truth."""
from copy import deepcopy
import math


def resolve_command_dispatch_mode(commands, configured_mode=None):
    """Choose completion gating only when every command carries a usable plan."""

    if configured_mode is not None:
        if configured_mode not in ('route_latest', 'completion_serial'):
            raise ValueError('unknown command dispatch mode')
        return configured_mode
    commands = list(commands)
    if commands and all(
        isinstance(command.get('driving_intent'), dict)
        and isinstance(command['driving_intent'].get('request_id'), str)
        and bool(command['driving_intent']['request_id'])
        and bool(command['driving_intent'].get('intent', {}).get('steps'))
        for command in commands
    ):
        return 'completion_serial'
    return 'route_latest'


class CompletionCommandQueue:
    def __init__(self, commands, timeout_s=120, safety_wait_timeout_s=300):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('positive command timeout required')
        self.timeout_s=timeout_s
        if not math.isfinite(safety_wait_timeout_s) or safety_wait_timeout_s <= 0:
            raise ValueError('positive safety wait timeout required')
        self.safety_wait_timeout_s=safety_wait_timeout_s
        self.execution_elapsed_s=0.
        self.safety_wait_elapsed_s=0.
        self.commands=deepcopy(list(commands))
        seen=set()
        previous=-math.inf
        for command in self.commands:
            document=command.get('driving_intent') or {}
            identity=document.get('request_id')
            trigger=next((command[k] for k in ('trigger_progress_m','activate_at_m','announce_at_m') if k in command),0)
            if (not isinstance(identity,str) or not identity or identity in seen or
                    not document.get('intent',{}).get('steps')):
                raise ValueError('serial dispatch requires distinct structured instruction requests')
            if isinstance(trigger,bool) or not isinstance(trigger,(int,float)) or not math.isfinite(trigger) or trigger<previous:
                raise ValueError('serial command triggers must be finite and ordered')
            command['_dispatch_trigger_m']=float(trigger)
            end=next((command[k] for k in ('end_progress_m','deactivate_at_m') if command.get(k) is not None),None)
            if end is not None and (isinstance(end,bool) or not isinstance(end,(int,float))
                                    or not math.isfinite(end) or end<=trigger):
                raise ValueError('command end progress must be finite and greater than trigger')
            command['_dispatch_end_m']=end
            previous=trigger
            seen.add(identity)
        self.index=0
        self.current=None
        self.started=None
        self.last_time=None
        self.failed=None
        self.events=[]

    def select(self, progress_m, timestamp_s, feedback=None):
        if not math.isfinite(progress_m) or not math.isfinite(timestamp_s):
            raise ValueError('finite dispatch progress/time required')
        if self.last_time is not None and timestamp_s < self.last_time:
            raise ValueError('dispatch time regressed')
        previous_time=self.last_time
        self.last_time=timestamp_s
        if self.failed:
            raise ValueError('command dispatch halted: '+self.failed)
        if self.current is not None:
            identity=self.current['driving_intent']['request_id']
            end=self.commands[self.index]['_dispatch_end_m']
            matching=bool(feedback and feedback.get('request_id')==identity)
            observed=feedback.get('observed_at_s') if matching else None
            fresh=(isinstance(observed,(int,float)) and not isinstance(observed,bool)
                   and math.isfinite(observed) and previous_time is not None
                   and previous_time-1e-6 <= observed <= timestamp_s
                   and timestamp_s-observed <= .5)
            elapsed=max(0.,timestamp_s-previous_time) if previous_time is not None else 0.
            if fresh and feedback.get('safety_wait') is True and feedback.get('plan_status')=='ACTIVE':
                self.safety_wait_elapsed_s+=elapsed
            else:
                self.execution_elapsed_s+=elapsed
            if end is not None and progress_m>=end:
                self.failed='ACTIVE_COMMAND_WINDOW_EXPIRED'
            elif self.execution_elapsed_s>=self.timeout_s:
                self.failed='TIMEOUT'
            elif self.safety_wait_elapsed_s>=self.safety_wait_timeout_s:
                self.failed='SAFETY_WAIT_TIMEOUT'
            if not self.failed and feedback and feedback.get('request_id')==identity:
                status=feedback.get('plan_status')
                if status=='COMPLETED':
                    self.events.append(dict(request_id=identity,status='COMPLETED',timestamp_s=timestamp_s))
                    self.current=None
                    self.index+=1
                elif status in ('FAILED','CANCELLED'):
                    self.failed=status
            if not self.failed and self.current is not None:
                if self.safety_wait_elapsed_s>=self.safety_wait_timeout_s:
                    self.failed='SAFETY_WAIT_TIMEOUT'
                elif self.execution_elapsed_s>=self.timeout_s:
                    self.failed='TIMEOUT'
            if self.failed:
                self.events.append(dict(request_id=identity,status=self.failed,timestamp_s=timestamp_s,
                                        progress_m=progress_m,end_progress_m=end))
                raise ValueError('command dispatch halted: '+self.failed)
        if self.current is None and self.index<len(self.commands):
            candidate=self.commands[self.index]
            if progress_m>=candidate['_dispatch_trigger_m']:
                end=candidate['_dispatch_end_m']
                if end is not None and progress_m>=end:
                    self.failed='MISSED_COMMAND_WINDOW'
                    self.events.append(dict(request_id=candidate['driving_intent']['request_id'],
                                            status=self.failed,timestamp_s=timestamp_s))
                    raise ValueError('command dispatch halted: '+self.failed)
                self.current=deepcopy(candidate)
                self.current.pop('_dispatch_trigger_m')
                self.current.pop('_dispatch_end_m')
                self.started=timestamp_s
                self.execution_elapsed_s=0.
                self.safety_wait_elapsed_s=0.
                self.events.append(dict(request_id=self.current['driving_intent']['request_id'],
                                        status='RUNNING',timestamp_s=timestamp_s,progress_m=progress_m))
        return deepcopy(self.current) if self.current else {'text':'Continue driving safely in the current lane.'}

    def status(self):
        return dict(mode='completion_serial',completed_commands=self.index,total_commands=len(self.commands),
                    active_request_id=self.current['driving_intent']['request_id'] if self.current else None,
                    failure=self.failed,events=deepcopy(self.events),
                    execution_elapsed_s=self.execution_elapsed_s,
                    safety_wait_elapsed_s=self.safety_wait_elapsed_s,
                    feedback_scope='model_execution_plan; not independent task assessment')
