from types import SimpleNamespace as NS
from benchmark.pedestrian_fixture import PedestrianFixture


def test_yield_baseline_requires_observed_conflict_then_clearance():
    fixture=PedestrianFixture.__new__(PedestrianFixture)
    fixture.spec={'activate_m':10,'steps':[{'clear_hold_s':.5}]}
    fixture.ped_started=False
    fixture.seen_conflict=False
    fixture.clear_since=None
    fixture.events=[]
    fixture.role='worker'
    fixture.source_event={'safety':{'minimum_trigger_distance_m':75}}
    fixture.crossing={'route_anchor_m':80,'conflict_polygon_xy':[(0,0),(2,0),(2,2),(0,2)]}
    fixture.bindings={'worker':NS(footprint=lambda pose:pose)}
    starts=[]
    fixture.walker=NS(actor=NS(id=2),start=lambda:starts.append(True),update=lambda:None)
    speeds=[]
    fixture.tm=NS(set_desired_speed=lambda ego,speed:speeds.append(speed))
    outside=[(5,5),(6,5),(6,6),(5,6)]
    inside=[(.5,.5),(1,.5),(1,1),(.5,1)]
    def snapshot(frame,seconds,polygon):
        return NS(frame=frame,timestamp=NS(elapsed_seconds=seconds),
                  find=lambda identity:NS(get_transform=lambda:polygon))
    fixture.tick(snapshot(1,0,outside),None,10,None)
    assert speeds[-1]==0 and starts==[True]
    fixture.tick(snapshot(2,1,outside),None,10,None)
    assert speeds[-1]==0
    fixture.tick(snapshot(3,2,inside),None,10,None)
    assert fixture.seen_conflict and speeds[-1]==0
    fixture.tick(snapshot(4,3,outside),None,10,None)
    assert speeds[-1]==0
    result=fixture.tick(snapshot(5,3.5,outside),None,10,None)
    assert result['pedestrian_clear'] and speeds[-1]==30
