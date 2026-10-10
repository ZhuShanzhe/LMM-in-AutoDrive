import math
from types import SimpleNamespace as NS
import pytest
from benchmark.lane_continuity import _published_connector_entry


class Location:
    def __init__(self,x=0.,y=0.,z=0.):self.x,self.y,self.z=x,y,z
    def distance(self,other):return math.sqrt((self.x-other.x)**2+(self.y-other.y)**2+(self.z-other.z)**2)


def waypoint(road,x=0.,yaw=0.,junction=8):
    return NS(road_id=road,section_id=0,lane_id=1,is_junction=True,junction_id=junction,
        transform=NS(location=Location(x),rotation=NS(yaw=yaw)))


@pytest.mark.parametrize('failure',[None,'disconnected','different_junction','not_coincident','different_heading'])
def test_entry_alias_needs_connected_coincident_downstream_proof(failure):
    current=waypoint(10)
    entry=waypoint(11,x=.02 if failure!='not_coincident' else .2,
        yaw=0. if failure!='different_heading' else 5.,junction=8 if failure!='different_junction' else 9)
    following=waypoint(11 if failure!='disconnected' else 12,x=2.)
    following.previous=lambda distance:[entry]
    previous=NS(next=lambda distance:[following])
    before=dict(distance_m=0.)
    at=dict(distance_m=2.,x=0.,y=0.,z=0.)
    after=dict(distance_m=4.,x=2.,y=0.,z=0.,yaw=0.,road_id=11,section_id=0,lane_id=1)
    result=_published_connector_entry(previous,current,before,at,after,Location)
    assert (result is not None)==(failure is None)

