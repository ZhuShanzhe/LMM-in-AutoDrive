from types import SimpleNamespace as NS

import pytest

from emergency_scene_3_events import EmergencySceneActorRuntime


def runtime(x=0):
    obj=object.__new__(EmergencySceneActorRuntime)
    obj._carla=NS(WalkerControl=lambda **kw:kw,Vector3D=lambda **kw:NS(**kw))
    obj._worker_phase='CROSSING'
    obj._crossing_worker_target_location=NS(x=10,y=0,z=0)
    obj._crossing_worker_start_location=NS(x=0,y=0,z=0)
    obj._crossing_worker_start_elapsed_s=0
    obj._worker_event={'crossing_behavior':{'speed_mps':1.8}}
    obj._crossing_worker_config={'start_s_m':3330}
    lane=NS(lane_width=3.5,transform=NS(location=NS(x=5,y=0),rotation=NS(yaw=90)))
    obj._map=NS(route_waypoint=lambda anchor:lane)
    controls=[]
    obj._crossing_worker=NS(is_alive=True,get_location=lambda:NS(x=x,y=0,z=0),
                            get_transform=lambda:NS(location=NS(x=x,y=0,z=0),rotation=NS(yaw=0)),
                            bounding_box=NS(location=NS(x=0,y=0),rotation=NS(yaw=0),extent=NS(x=.3,y=.3)),
                            apply_control=controls.append)
    return obj,controls


def test_elapsed_time_does_not_clear_stationary_worker():
    obj,controls=runtime()
    obj._update_worker_crossing(ego_route_s_m=3255,elapsed_s=60)
    assert obj._worker_phase=='CROSSING'
    assert controls[-1]['speed']==1.8
    assert controls[-1]['direction'].x==1


def test_observed_arrival_stops_worker_without_teleport():
    obj,controls=runtime(9.9)
    obj._update_worker_crossing(ego_route_s_m=3255,elapsed_s=1)
    assert obj._worker_phase=='YIELDED_CLEAR'
    assert controls[-1]['speed']==0


def test_arrival_inside_route_lane_does_not_report_clearance():
    obj,controls=runtime(9.9)
    obj._map.route_waypoint=lambda anchor:NS(lane_width=3.5,
        transform=NS(location=NS(x=10,y=0),rotation=NS(yaw=90)))
    with pytest.raises(RuntimeError,match='destination still occupies'):
        obj._update_worker_crossing(ego_route_s_m=3255,elapsed_s=1)
    assert obj._worker_phase=='CROSSING'
    assert controls[-1]['speed']==0


def test_arrival_without_route_geometry_cannot_report_clearance():
    obj,_=runtime(9.9)
    obj._map=NS()
    with pytest.raises(RuntimeError,match='actual ego route'):
        obj._update_worker_crossing(ego_route_s_m=3255,elapsed_s=1)
    assert obj._worker_phase=='CROSSING'


def test_missing_worker_cannot_be_replaced_during_assessment():
    obj,_=runtime()
    obj._crossing_worker.is_alive=False
    with pytest.raises(RuntimeError,match='identity cannot be replaced'):
        obj._update_worker_crossing(ego_route_s_m=3255,elapsed_s=1)


def test_event_boundary_does_not_force_clearance():
    obj,_=runtime()
    with pytest.raises(RuntimeError,match='did not clear'):
        obj.on_resolve({'id':'scene3_temporary_pedestrian'},route_s_m=3425,simulation_frame=10,elapsed_s=10)
    assert obj._worker_phase=='CROSSING'
