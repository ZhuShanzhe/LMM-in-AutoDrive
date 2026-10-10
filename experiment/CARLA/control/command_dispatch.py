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


class CommandDispatchHalted(ValueError):
    """A terminal task outcome, distinct from invalid input or runtime failure."""


class CompletionCommandQueue:
    def __init__(self, commands, timeout_s=120, safety_wait_timeout_s=300,
                 speed_target_supersession=False, failure_policy='HALT'):
        if failure_policy not in ('HALT', 'RETIRE_AND_CONTINUE'):
            raise ValueError('unknown command failure policy')
        self.failure_policy=failure_policy
        if not isinstance(speed_target_supersession,bool):
            raise ValueError('speed target supersession must be boolean')
        self.speed_target_supersession=speed_target_supersession
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
            budget=command.get('execution_timeout_s',timeout_s)
            if isinstance(budget,bool) or not isinstance(budget,(int,float)) or not math.isfinite(budget) or budget<=0:
                raise ValueError('positive finite per-command execution timeout required')
            command['_dispatch_timeout_s']=float(budget)
            previous=trigger
            seen.add(identity)
        self.index=0
        self.current=None
        self.started=None
        self.last_time=None
        self.failed=None
        self.events=[]

    def _retire_failed(self, command, timestamp_s, progress_m, feedback=None):
        steps=command['driving_intent']['intent']['steps']
        # An explicit stop mission must not turn into cruise after a timeout.
        protected={'STOP','EMERGENCY_BRAKE','CANCEL','PULL_OVER'}
        if (self.failure_policy!='RETIRE_AND_CONTINUE'
                or any(step.get('action') in protected for step in steps)):
            return False
        self.events[-1].update(
            disposition='RETIRED_FOR_ROUTE_CONTINUITY',
            manual_review_status='NOT_REVIEWED',
            original_plan=deepcopy(command['driving_intent']),
            prior_plan_state=deepcopy((feedback or {}).get('plan_state'))
                if (feedback or {}).get('request_id')==command['driving_intent']['request_id'] else None,
            execution_elapsed_s=self.execution_elapsed_s,
            safety_wait_elapsed_s=self.safety_wait_elapsed_s)
        self.current=None
        self.index+=1
        self.failed=None
        self.started=None
        self.execution_elapsed_s=0.
        self.safety_wait_elapsed_s=0.
        return True

    def _can_supersede_speed(self, progress_m, feedback):
        if not self.speed_target_supersession or self.index+1 >= len(self.commands):
            return False
        candidate=self.commands[self.index+1]
        if progress_m < candidate['_dispatch_trigger_m']:
            return False
        end=candidate['_dispatch_end_m']
        if end is not None and progress_m >= end:
            return False
        upcoming=candidate['driving_intent'].get('intent',{}).get('steps',[])
        if not any(step.get('action') in {'SET_SPEED','ADJUST_SPEED'} for step in upcoming):
            return False
        state=feedback.get('plan_state') or {}
        steps=state.get('step_states') or []
        original=self.current['driving_intent']['intent']['steps']
        if (state.get('request_id') != feedback.get('request_id')
                or state.get('plan_status') != 'ACTIVE' or len(steps) != len(original)
                or not steps or state.get('active_step_index') != len(steps)-1
                or state.get('active_step_id') != steps[-1].get('step_id')):
            return False
        if any(saved.get('step_id') != planned.get('step_id') or saved.get('action') != planned.get('action')
               for saved,planned in zip(steps,original)):
            return False
        return (all(step.get('status') == 'COMPLETED' for step in steps[:-1])
                and steps[-1].get('action') in {'SET_SPEED','ADJUST_SPEED'}
                and steps[-1].get('status') in {'ACTIVE','WAITING'})

    def select(self, progress_m, timestamp_s, feedback=None):
        if not math.isfinite(progress_m) or not math.isfinite(timestamp_s):
            raise ValueError('finite dispatch progress/time required')
        if self.last_time is not None and timestamp_s < self.last_time:
            raise ValueError('dispatch time regressed')
        previous_time=self.last_time
        self.last_time=timestamp_s
        if self.failed:
            raise CommandDispatchHalted('command dispatch halted: '+self.failed)
        if self.current is not None:
            identity=self.current['driving_intent']['request_id']
            end=self.commands[self.index]['_dispatch_end_m']
            timeout=self.commands[self.index]['_dispatch_timeout_s']
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
            elif self.execution_elapsed_s>=timeout:
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
            if (not self.failed and self.current is not None and fresh
                    and self._can_supersede_speed(progress_m,feedback)):
                candidate=self.commands[self.index+1]
                self.events.append(dict(request_id=identity,status='SUPERSEDED',timestamp_s=timestamp_s,
                    progress_m=progress_m,reason='new_explicit_speed_target',
                    superseded_by=candidate['driving_intent']['request_id'],
                    prior_plan_state=deepcopy(feedback['plan_state'])))
                self.current=None
                self.index+=1
            if not self.failed and self.current is not None:
                if self.safety_wait_elapsed_s>=self.safety_wait_timeout_s:
                    self.failed='SAFETY_WAIT_TIMEOUT'
                elif self.execution_elapsed_s>=timeout:
                    self.failed='TIMEOUT'
            if self.failed:
                self.events.append(dict(request_id=identity,status=self.failed,timestamp_s=timestamp_s,
                                        progress_m=progress_m,end_progress_m=end))
                if not self._retire_failed(self.current,timestamp_s,progress_m,feedback):
                    raise CommandDispatchHalted('command dispatch halted: '+self.failed)
        while self.current is None and self.index<len(self.commands):
            candidate=self.commands[self.index]
            if progress_m<candidate['_dispatch_trigger_m']:
                break
            if progress_m>=candidate['_dispatch_trigger_m']:
                end=candidate['_dispatch_end_m']
                if end is not None and progress_m>=end:
                    self.failed='MISSED_COMMAND_WINDOW'
                    self.events.append(dict(request_id=candidate['driving_intent']['request_id'],
                                            status=self.failed,timestamp_s=timestamp_s))
                    if self._retire_failed(candidate,timestamp_s,progress_m):
                        continue
                    raise CommandDispatchHalted('command dispatch halted: '+self.failed)
                self.current=deepcopy(candidate)
                self.current.pop('_dispatch_trigger_m')
                self.current.pop('_dispatch_end_m')
                self.current.pop('_dispatch_timeout_s')
                self.started=timestamp_s
                self.execution_elapsed_s=0.
                self.safety_wait_elapsed_s=0.
                self.events.append(dict(request_id=self.current['driving_intent']['request_id'],
                                        status='RUNNING',timestamp_s=timestamp_s,progress_m=progress_m))
                break
        return deepcopy(self.current) if self.current else {'text':'Continue driving safely in the current lane.'}

    def status(self):
        return dict(mode='completion_serial',completed_commands=sum(e['status']=='COMPLETED' for e in self.events),
                    failure_policy=self.failure_policy,
                    retired_commands=sum(e.get('disposition')=='RETIRED_FOR_ROUTE_CONTINUITY' for e in self.events),
                    superseded_commands=sum(e['status']=='SUPERSEDED' for e in self.events),
                    speed_target_supersession_enabled=self.speed_target_supersession,total_commands=len(self.commands),
                    active_request_id=self.current['driving_intent']['request_id'] if self.current else None,
                    failure=self.failed,events=deepcopy(self.events),
                    execution_elapsed_s=self.execution_elapsed_s,
                    execution_timeout_s=self.commands[self.index]['_dispatch_timeout_s'] if self.current else None,
                    safety_wait_elapsed_s=self.safety_wait_elapsed_s,
                    feedback_scope='model_execution_plan; not independent task assessment')
