from types import SimpleNamespace as Obj

from control.background_spacing import maintain_rear_spacing


def actor(identity,x,y=0.,direction=1.):
    pose=Obj(location=Obj(x=x,y=y),get_forward_vector=lambda:Obj(x=direction,y=0.))
    return Obj(id=identity,is_alive=True,get_transform=lambda:pose)


def test_only_same_direction_background_behind_ego_gets_larger_spacing():
    calls=[]
    manager=Obj(distance_to_leading_vehicle=lambda vehicle,distance:calls.append((vehicle.id,distance)))
    ego=actor(1,0.)
    count=maintain_rear_spacing(manager,ego,[actor(2,-30.),actor(3,20.),
        actor(4,-30.,direction=-1.),actor(5,-30.,y=12.)],60.)
    assert count==1 and calls==[(2,60.)]


def test_disabled_spacing_has_no_actor_access():
    assert maintain_rear_spacing(None,None,[],0.)==0
