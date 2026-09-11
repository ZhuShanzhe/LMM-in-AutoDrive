"""Task lifecycle memory, separate from physical tracking and actuator authority."""

from copy import deepcopy
from collections import deque


class TaskEventMemory:
    def __init__(self, maximum_events=16):
        self.maximum_events = maximum_events
        self.events = {}
        self.transitions = deque(maxlen=32)

    def activate(self, event_id, kind, goal, timestamp_s):
        if event_id in self.events:
            if self.events[event_id]['kind']!=kind or self.events[event_id]['goal']!=goal:
                raise ValueError('Changed instruction requires a new event identity')
            return
        if len(self.events)>=self.maximum_events:
            raise RuntimeError('Active event capacity exceeded; do not silently forget unfinished tasks')
        self.events[event_id] = dict(event_id=event_id,kind=kind,goal=deepcopy(goal),
            started_s=timestamp_s,phase='ACTIVE',current_behavior=None,previous_behavior=None,
            boundary_observation=None)

    def behavior_changed(self, event_id, behavior, observation, timestamp_s):
        event = self.events[event_id]
        if event['current_behavior']==behavior:
            return False
        previous = event['current_behavior']
        if (previous is not None and behavior!='EMERGENCY_BRAKE'
                and timestamp_s-event.get('boundary_time_s',timestamp_s)<.5-1e-6):
            return False
        self.transitions.append(dict(event_id=event_id,previous_behavior=previous,
            next_behavior=behavior,timestamp_s=timestamp_s,previous_behavior_status='SUPERSEDED'))
        target=observation.get('selected') or {}
        context=dict(target_ref=target.get('track_id'),target_last_seen_s=target.get('measured_at_s'),
            observation_status=observation.get('status','UNKNOWN'))
        event.update(current_behavior=behavior,previous_behavior=previous,
            boundary_observation=context,boundary_time_s=timestamp_s)
        return True

    def finish(self, event_id, *, reason, timestamp_s):
        if reason not in ('COMPLETED','CANCELLED','REPLACED'):
            raise ValueError('Explicit lifecycle reason required')
        if event_id in self.events:
            self.transitions.append(dict(event_id=event_id,status=reason,timestamp_s=timestamp_s))
            del self.events[event_id]

    def decision_context(self, timestamp_s=None, target_history=None):
        if timestamp_s is not None:
            retained={e['track_id']:e['last_seen_s'] for e in (target_history or {}).get('entries',[])}
            for event in self.events.values():
                context=event.get('boundary_observation') or {}
                last_seen=retained.get(context.get('target_ref'),context.get('target_last_seen_s'))
                if last_seen is not None and timestamp_s-last_seen>30.+1e-6:
                    context.update(target_ref=None,target_last_seen_s=None,observation_status='EXPIRED')
        # Historical behavior is explanatory only and never an executable command.
        return dict(schema_version='task_event_memory/1.0',active_events=deepcopy(list(self.events.values())),
            recent_transitions=deepcopy(list(self.transitions)),authority='DECISION_CONTEXT_ONLY')
