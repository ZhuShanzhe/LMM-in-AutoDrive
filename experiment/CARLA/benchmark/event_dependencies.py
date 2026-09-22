"""Static actor-lifetime audit; never substitutes event status for task success."""


def audit_event_dependencies(spec, events):
    by_role={}
    for event in events:
        for role in event.get('ground_truth',{}).get('actor_roles',[]):
            by_role.setdefault(role,[]).append(event)
    findings=[]
    required=[]
    preceding_activation=spec['activate_m']
    for index,step in enumerate(spec['steps']):
        roles=set(step.get('target_roles',[]))
        if step.get('target_role'):
            roles.add(step['target_role'])
        for role in sorted(roles):
            required.append(role)
            matches=by_role.get(role,[])
            if len(matches)!=1:
                findings.append(dict(step=index,role=role,reason='missing_or_ambiguous_event_binding'))
                continue
            event=matches[0]
            activate=event.get('activate_progress_m',event.get('activate_at_m',event.get('anchor_progress_m',0)-80))
            preceding_activation=max(preceding_activation,activate)
            end=event.get('resolve_progress_m',event.get('resolve_after_m'))
            # An event can finish while its physical actor remains available.
            if end is not None and end<=preceding_activation and event.get('retain_after_completion') is not True:
                findings.append(dict(step=index,role=role,event_id=event['id'],
                    reason='role_retired_before_ordered_step_can_start',resolve_m=end,
                    earliest_start_m=preceding_activation))
    return dict(compatible=not findings,findings=findings,
                required_preserved_roles=sorted(set(required)),
                scope='static_lifetime_only_not_geometry_or_behavior_proof')
