from types import SimpleNamespace as NS
from benchmark.scene3_preflight import audit_worker_endpoints
from scene3_town05_route import validate_scene3_event_anchors


def waypoint(y):
    return NS(lane_width=3.5,transform=NS(location=NS(x=0,y=y),rotation=NS(yaw=0)))


def event():
    return {'scenario':'temporary_worker_crossing','distance_m':3300,'workers':[
        {'role_name':'worker','start_s_m':3330,'start_lane_id':-4,'destination_lane_id':-2}]}


def test_center_lane_destination_is_rejected_before_spawn():
    adapter=NS(route_waypoint=lambda s:waypoint(0),
               get_waypoint_xodr=lambda road,lane,s:waypoint(4 if lane==-4 else 0))
    assert not audit_worker_endpoints(adapter,event())[0]['valid']


def test_opposite_side_endpoints_cross_route_lane():
    adapter=NS(route_waypoint=lambda s:waypoint(0),
               get_waypoint_xodr=lambda road,lane,s:waypoint(4 if lane==-4 else -4))
    assert audit_worker_endpoints(adapter,event())[0]['valid']


def test_anchor_validation_uses_actual_worker_position():
    calls=[]
    context=NS(adapter=NS(validate_anchor=lambda s,lanes:calls.append((s,lanes))))
    validate_scene3_event_anchors(context,[event()])
    assert calls==[(3330,(-4,-2))]
