"""Coordinate background traffic during scripted crossings, never ego permission."""
import math
from .crosswalk_start_gate import crossing_start_observation


def prune_unrelated_crossing_blockers(start,target,vehicles,*,owned_ids,excluded_ids,protected_roles):
    """Remove only owned non-event background bodies blocking a staged crossing."""
    owned=set(owned_ids);excluded=set(excluded_ids);protected=set(protected_roles)
    removed=[]
    for actor in vehicles:
        role=actor.attributes.get('role_name','')
        if (not actor.is_alive or actor.id not in owned or actor.id in excluded
                or role in protected or role in {'hero','ego'}):
            continue
        if crossing_start_observation(start,target,[actor])['clear']:
            continue
        pose=actor.get_transform();box=actor.bounding_box
        record=dict(actor_id=int(actor.id),role=role,position_xyz_m=[pose.location.x,pose.location.y,pose.location.z],
            yaw_deg=pose.rotation.yaw,half_length_m=box.extent.x,half_width_m=box.extent.y,
            reason='unrelated_background_body_intersects_staged_crossing',policy_input=False,
            source='authorized_scene_simplification')
        if not actor.destroy():raise RuntimeError('Failed to remove conflicting background actor')
        removed.append(record)
    return removed


def approaching_crossing(start, target, actor):
    pose=actor.get_transform()
    position=pose.location
    if abs(position.z-(start.z+target.z)*.5)>3.:
        return False
    dx,dy=target.x-start.x,target.y-start.y
    length2=dx*dx+dy*dy
    if length2<.01:raise ValueError('Degenerate crossing')
    fraction=max(0.,min(1.,((position.x-start.x)*dx+(position.y-start.y)*dy)/length2))
    nearest=(start.x+fraction*dx,start.y+fraction*dy)
    yaw=math.radians(pose.rotation.yaw)
    forward=(nearest[0]-position.x)*math.cos(yaw)+(nearest[1]-position.y)*math.sin(yaw)
    lateral=abs(-(nearest[0]-position.x)*math.sin(yaw)+(nearest[1]-position.y)*math.cos(yaw))
    speed=actor.get_velocity()
    speed=math.hypot(speed.x,speed.y)
    if not all(math.isfinite(v) for v in (forward,lateral,speed)):
        raise ValueError('Invalid NPC crossing kinematics')
    # Anticipate braking before the crossing, including vehicles stopped at a signal.
    distance=max(20.,min(55.,speed*5.+speed*speed/6.+5.))
    direct=0.<forward<=distance and lateral<=actor.bounding_box.extent.y+1.
    turning=0.<forward<=15. and math.hypot(nearest[0]-position.x,nearest[1]-position.y)<=15.
    return direct or turning


class CrossingTrafficPriority:
    def __init__(self):
        self.held={}

    def update(self,start,target,vehicles,manager,*,excluded_ids=(),finished=False,crossing_active=False):
        excluded=set(excluded_ids)
        for identity,(actor,_) in list(self.held.items()):
            if not actor.is_alive:
                del self.held[identity]
        if finished:
            restored=[]
            for identity,(actor,speed) in list(self.held.items()):
                if actor.is_alive:
                    manager.set_desired_speed(actor,speed)
                    restored.append(identity)
            self.held.clear()
            return dict(held_actor_ids=[],restored_actor_ids=restored,
                source='scenario_npc_priority_not_ego_policy',policy_input=False)
        bodies=list(vehicles)
        released_to_clear=[]
        for actor in bodies:
            if actor.id in excluded or not actor.is_alive:
                continue
            if actor.id in self.held:
                if not crossing_active and not crossing_start_observation(start,target,[actor])['clear']:
                    manager.set_desired_speed(actor,self.held.pop(actor.id)[1])
                    released_to_clear.append(actor.id)
                    continue
                manager.set_desired_speed(actor,0.)
                continue
            if (crossing_start_observation(start,target,[actor])['clear']
                    and approaching_crossing(start,target,actor)):
                # Restore a recorded legal ambient cruise target after the crossing.
                restore=max(1.,float(actor.get_speed_limit())*.9)
                if not math.isfinite(restore):raise ValueError('Invalid NPC speed limit')
                self.held[actor.id]=(actor,restore)
                manager.set_desired_speed(actor,0.)
        settled=all(math.hypot(actor.get_velocity().x,actor.get_velocity().y)<=.3
            for actor,_ in self.held.values() if actor.is_alive)
        return dict(held_actor_ids=sorted(self.held),held_vehicles_settled=settled,
            released_to_clear_actor_ids=released_to_clear,
            restore_target_kmh={str(k):v[1] for k,v in self.held.items()},
            source='scenario_npc_priority_not_ego_policy',policy_input=False)
