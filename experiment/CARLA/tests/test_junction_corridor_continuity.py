import math
from types import SimpleNamespace as NS
import pytest
from scene3_town05_route import continuous_corridor_waypoint


def waypoint(x,y,lane,road,junction,yaw=0.,z=0.):
    loc=NS(x=x,y=y,z=z)
    loc.distance=lambda other:math.sqrt((x-other.x)**2+(y-other.y)**2+(z-other.z)**2)
    return NS(road_id=road,section_id=0,lane_id=lane,is_junction=junction,lane_type='Driving',
        transform=NS(location=loc,rotation=NS(yaw=yaw)),get_left_lane=lambda:None,get_right_lane=lambda:None)


@pytest.mark.parametrize('yaw',[0.,35.,90.,180.])
def test_lane_rank_does_not_reset_at_junction(yaw):
    angle=math.radians(yaw)
    rot=lambda x,y:(math.cos(angle)*x-math.sin(angle)*y,math.sin(angle)*x+math.cos(angle)*y)
    previous=waypoint(*rot(0.,0.),-2,1,False,yaw)
    nominal=waypoint(*rot(2.,-3.5),-1,2,True,yaw)
    connected=waypoint(*rot(2.,0.),-2,2,True,yaw)
    nominal.get_right_lane=lambda:connected
    connected.get_left_lane=lambda:nominal
    previous.next=lambda distance:[connected]
    assert continuous_corridor_waypoint(previous,nominal,2.) is connected
    assert continuous_corridor_waypoint(connected,nominal,0.) is connected


def test_unproven_or_different_level_neighbor_is_not_substituted():
    previous=waypoint(0.,0.,-2,1,False)
    nominal=waypoint(2.,-3.5,-1,2,True)
    member=waypoint(2.,0.,-2,2,True,z=10.)
    nominal.get_right_lane=lambda:member
    previous.next=lambda distance:[waypoint(2.,0.,-2,2,True)]
    assert continuous_corridor_waypoint(previous,nominal,2.) is nominal
