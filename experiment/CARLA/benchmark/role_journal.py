"""Stable task actor identities; all kinematics come from the frozen snapshot."""
from dataclasses import asdict

from .catalog import ConfigError
from .truth_capture import ActorBinding


def required_role_types(spec):
    result={}
    for step in spec['steps']:
        roles=set(step.get('target_roles',[]))
        if step.get('target_role'):
            roles.add(step['target_role'])
        prefix='walker.pedestrian.' if step['kind'] in {'yield_pedestrian','wait_clear'} else 'vehicle.'
        for role in roles:
            if role in result and result[role]!=prefix:
                raise ConfigError(f'inconsistent task role type: {role}')
            result[role]=prefix
    return result


def configured_role_anchors(events):
    """Use per-actor route coordinates when present; never borrow ego progress for static actors."""
    from .catalog import number
    candidates={}
    for event in events:
        explicit={}
        objects=list(event.get('workers',[]))+list(event.get('zone',{}).get('work_vehicle_spawns',[]))
        objects += [event.get('actor',{}),event.get('blockage',{})]
        for actor in objects:
            role=actor.get('role_name')
            anchor=next((actor[k] for k in ('start_s_m','spawn_s_m','s_m') if k in actor),None)
            if role and anchor is not None:
                explicit[role]=number(anchor,'actor route anchor')
        for role in event.get('ground_truth',{}).get('actor_roles',[]):
            anchor=explicit.get(role,event.get('anchor_progress_m'))
            if anchor is not None:
                candidates.setdefault(role,[]).append(number(anchor,'event route anchor'))
    return {role:values[0] for role,values in candidates.items() if len(values)==1}


class RoleJournal:
    def __init__(self, profiles, anchor_hints=None):
        self.expected = {}
        self.bindings = {}
        self.errors = {}
        self.progress = dict(anchor_hints or {})
        for profile in profiles:
            for role,prefix in required_role_types(profile).items():
                if role in self.expected and self.expected[role] != prefix:
                    raise ConfigError(f'inconsistent task role type: {role}')
                self.expected[role] = prefix

    def capture(self, snapshot, actors, projector, route_hint, pack):
        matches = {role: [] for role in self.expected}
        for actor in actors:
            role = actor.attributes.get('role_name')
            if role in matches and snapshot.find(actor.id) is not None:
                matches[role].append(actor)
        states, packed = {}, {}
        for role, prefix in self.expected.items():
            found = matches[role]
            binding = self.bindings.get(role)
            if len(found) > 1:
                self.errors.setdefault(role, 'duplicate_role')
            elif found:
                actor = found[0]
                if not actor.type_id.startswith(prefix):
                    self.errors.setdefault(role, 'wrong_actor_type')
                elif binding is not None and actor.id != binding.actor_id:
                    self.errors.setdefault(role, 'actor_identity_changed')
                elif binding is None and role not in self.errors:
                    location = snapshot.find(actor.id).get_transform().location
                    progress = projector.project(location, self.progress.get(role,route_hint))['route_s_m']
                    binding = ActorBinding.from_actor(actor, progress)
                    self.bindings[role] = binding
            record = dict(candidate_ids=sorted(a.id for a in found))
            if binding is not None:
                record['binding'] = asdict(binding)
                frozen = snapshot.find(binding.actor_id)
                if frozen is not None:
                    packed[str(binding.actor_id)] = pack(frozen)
                    projection=projector.project(frozen.get_transform().location,
                                                  self.progress.get(role,binding.initial_route_s_m))
                    self.progress[role]=projection['route_s_m']
                    record.update(projection)
                else:
                    self.errors.setdefault(role, 'bound_actor_missing')
            if role in self.errors:
                record.update(status='INVALID', reason=self.errors[role])
            elif binding is None:
                record.update(status='NOT_SPAWNED', reason='no_matching_actor_in_snapshot')
            elif not found:
                self.errors[role] = 'bound_actor_metadata_missing'
                record.update(status='INVALID', reason=self.errors[role])
            else:
                record.update(status='BOUND')
            states[role] = record
        return states, packed
