import math
from types import SimpleNamespace as NS
from scenarios.crosswalk_priority import CrossingTrafficPriority, approaching_crossing, prune_unrelated_crossing_blockers


def actor(identity, x=0.,y=-12.,yaw=90.,z=0.):
    pose=NS(location=NS(x=x,y=y,z=z),rotation=NS(yaw=yaw))
    def vertices(transform):
        c,s=math.cos(math.radians(yaw)),math.sin(math.radians(yaw))
        return [NS(x=x+c*a-s*b,y=y+s*a+c*b,z=z+h)
            for a in (-2.,2.) for b in (-1.,1.) for h in (0.,1.)]
    return NS(id=identity,is_alive=True,attributes={'role_name':f'background-{identity}'},
        destroy=lambda:True,bounding_box=NS(extent=NS(x=2.,y=1.),get_world_vertices=vertices),
        get_transform=lambda:pose,get_velocity=lambda:NS(x=0.,y=0.,z=0.),get_speed_limit=lambda:30.)


def geometry():return NS(x=-5.,y=0.,z=0.),NS(x=5.,y=0.,z=0.)


def test_background_approach_brakes_and_restores_legal_cruise():
    start,target=geometry()
    calls=[]
    manager=NS(set_desired_speed=lambda a,v:calls.append((a.id,v)))
    priority=CrossingTrafficPriority()
    car=actor(10)
    first=priority.update(start,target,[car],manager)
    assert first['held_actor_ids']==[10] and calls[-1]==(10,0.)
    assert first['policy_input'] is False
    last=priority.update(start,target,[car],manager,finished=True)
    assert last['restored_actor_ids']==[10] and calls[-1]==(10,27.)
    assert not priority.held


def test_ego_scripted_actors_and_body_already_crossing_are_not_held():
    start,target=geometry();calls=[]
    priority=CrossingTrafficPriority()
    manager=NS(set_desired_speed=lambda a,v:calls.append((a.id,v)))
    priority.update(start,target,[actor(1),actor(2),actor(3,y=0.)],manager,excluded_ids=[1,2])
    assert calls==[]


def test_moving_away_parallel_and_other_height_are_not_crossing_approaches():
    start,target=geometry()
    assert not approaching_crossing(start,target,actor(1,yaw=-90.))
    assert not approaching_crossing(start,target,actor(2,x=60.))
    assert not approaching_crossing(start,target,actor(3,z=10.))


def test_hold_is_continuous_until_crossing_finishes():
    start,target=geometry();calls=[]
    priority=CrossingTrafficPriority()
    manager=NS(set_desired_speed=lambda a,v:calls.append((a.id,v)))
    car=actor(5)
    priority.update(start,target,[car],manager)
    priority.update(start,target,[car],manager)
    assert calls==[(5,0.),(5,0.)]


def test_overshooting_npc_clears_crossing_before_walker_launch():
    start,target=geometry();calls=[]
    priority=CrossingTrafficPriority()
    manager=NS(set_desired_speed=lambda a,v:calls.append((a.id,v)))
    priority.update(start,target,[actor(7)],manager)
    result=priority.update(start,target,[actor(7,y=0.)],manager,crossing_active=False)
    assert result['released_to_clear_actor_ids']==[7]
    assert 7 not in priority.held and calls[-1]==(7,27.)


def test_scene_pruning_preserves_event_ego_unowned_and_nonblocking_traffic():
    start,target=geometry()
    values=[actor(i,y=0.) for i in range(1,6)]+[actor(6,y=-12.)]
    values[1].attributes['role_name']='task_lead'
    values[3].attributes['role_name']='hero'
    removed=prune_unrelated_crossing_blockers(start,target,values,
        owned_ids=[1,2,3,4,6],excluded_ids=[1],protected_roles=['task_lead'])
    assert [value['actor_id'] for value in removed]==[3]
    assert removed[0]['policy_input'] is False


def test_destroyed_background_actor_is_removed_from_priority_memory():
    start,target=geometry();calls=[]
    priority=CrossingTrafficPriority()
    manager=NS(set_desired_speed=lambda a,v:calls.append((a.id,v)))
    car=actor(9)
    priority.update(start,target,[car],manager)
    car.is_alive=False
    result=priority.update(start,target,[],manager)
    assert result['held_actor_ids']==[] and not priority.held
    assert calls==[(9,0.)]
