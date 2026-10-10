from types import SimpleNamespace as NS
from scenarios.complex.sidewalk_patrol import SidewalkPatrol, sidewalk_path


def test_patrol_bounces_without_teleporting():
    controls=[]
    location=NS(x=4.0,y=0.0,z=0.0)
    actor=NS(is_alive=True,get_location=lambda:location,apply_control=controls.append)
    points=tuple(NS(x=x,y=0.0,z=0.0) for x in (0.,4.))
    patrol=SidewalkPatrol(actor,points,1.3,1)
    patrol.update(NS(WalkerControl=lambda **kw:NS(**kw),Vector3D=lambda **kw:NS(**kw)))
    assert patrol.index==0 and patrol.direction==-1
    assert controls[-1].speed==1.3 and controls[-1].direction.x==-1


def test_displaced_patrol_does_not_cut_across_road():
    controls=[]
    location=NS(x=2.0,y=3.0,z=0.0)
    actor=NS(is_alive=True,get_location=lambda:location,apply_control=controls.append)
    patrol=SidewalkPatrol(actor,tuple(NS(x=x,y=0.,z=0.) for x in (0.,4.)),1.2,1)
    carla=NS(WalkerControl=lambda **kw:NS(**kw),Vector3D=lambda **kw:NS(**kw))
    patrol.update(carla)
    assert patrol.disabled and controls[-1].speed==0
    location.y=0
    patrol.update(carla)
    assert controls[-1].speed==0


def test_sidewalk_path_rejects_road_and_fork():
    road=NS(lane_type='Driving')
    assert sidewalk_path(road)==[]
    anchor=NS(lane_type='Sidewalk',transform=NS(location=NS(x=0.,y=0.,z=0.)))
    child=NS(lane_type='Sidewalk',transform=NS(location=NS(x=4.,y=0.,z=0.)),next=lambda _:[],previous=lambda _:[])
    anchor.next=lambda _:[child,child]
    anchor.previous=lambda _:[road]
    assert sidewalk_path(anchor)==[anchor]


def test_connected_sidewalk_is_bounded_in_both_directions():
    points=[NS(lane_type='Sidewalk',transform=NS(location=NS(x=float(x),y=0.,z=0.))) for x in (-4,0,4)]
    for i,point in enumerate(points):
        point.next=lambda _,i=i:points[i+1:i+2]
        point.previous=lambda _,i=i:points[i-1:i] if i else []
    assert sidewalk_path(points[1])==points
